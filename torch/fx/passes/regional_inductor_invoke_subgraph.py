# mypy: allow-untyped-defs

import logging
from typing import Any

import torch
from torch.fx._compatibility import compatibility
from torch.fx.passes.regional_inductor import (
    _dummy_wrapper,
    _partition_by_supported_nodes,
)


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor_invoke_subgraph"]


def _compile_submod(gm, prefix):
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # Get compile configs from annotation
            compile_config = None
            compile_fn = None
            decompositions = None
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "nested_region_config" in custom:
                        compile_config = custom["nested_region_config"]
                        if node.meta.get("partitioner_tag") == "is_forward":
                            compile_fn = compile_config.fw_compiler
                        else:
                            compile_fn = compile_config.bw_compiler
                        decompositions = compile_config.decompositions
                        break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                compile_config,
            )

            options: dict[str, Any] = {}
            options["decompositions"] = decompositions

            compiled_fn = torch._inductor.standalone_compile(
                submod,
                fake_inputs,
                dynamic_shapes="from_tracing_context",
                aot=True,
                options=options,
                compile_fn=compile_fn,
            )
            assert isinstance(compiled_fn, AOTCompiledArtifact)
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node):
    return (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and node.meta["custom"].get("nested_region_config", None)
        and node.meta["custom"]["nested_region_config"].fw_compiler
        and node.meta.get("partitioner_tag") == "is_forward"
    ) or (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and node.meta["custom"].get("nested_region_config", None)
        and node.meta["custom"]["nested_region_config"].bw_compiler
        and node.meta.get("partitioner_tag") == "is_backward"
    )


def _compile_invoke_subgraph_nodes_with_inductor(gm):
    from torch.fx.passes.operator_support import OperatorSupport

    found_marked_node = False
    for node in gm.graph.nodes:
        if _needs_inductor_compile(node):
            found_marked_node = True
            break

    if not found_marked_node:
        logger.info("No inductor marked nodes found")
        return gm

    class InductorMarkedNodes(OperatorSupport):
        def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
            return _needs_inductor_compile(node)

    marked_nodes = InductorMarkedNodes()
    gm = _partition_by_supported_nodes(gm, marked_nodes, "__marked_inductor_submod")
    gm = _compile_submod(gm, "__marked_inductor_submod")
    return gm


def _recursive_compile_invoke_subgraph_nodes(gm):
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we dont do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_invoke_subgraph_nodes(submod)

    return _compile_invoke_subgraph_nodes_with_inductor(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor_invoke_subgraph(gm, *example_args):
    """
    Compile invoke_subgraph nodes if they have custom compiler specified
    in node.meta["nested_region_config"].bw_compiler or fw_compiler
    """
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _recursive_compile_invoke_subgraph_nodes(gm)
