"""Prepare an isolated metadata-builder patch; never modify the base image."""
from pathlib import Path
import sys


def patch(source: str) -> str:
    old = '''        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,'''
    new = '''        for group in groups:
            # Experimental: metadata must use the actual layer cache dtype,
            # not the target MLA dtype inherited by a quantized draft group.
            group_layers = get_layers_from_vllm_config(
                vllm_config, AttentionLayerBase, group.layer_names
            )
            cache_dtypes = {
                layer.kv_cache_dtype for layer in group_layers.values()
                if hasattr(layer, "kv_cache_dtype")
            }
            builder_config = vllm_config
            if len(cache_dtypes) == 1:
                cache_dtype = next(iter(cache_dtypes))
                if cache_dtype != vllm_config.cache_config.cache_dtype:
                    builder_config = replace(
                        vllm_config,
                        cache_config=replace(
                            vllm_config.cache_config, cache_dtype=cache_dtype
                        ),
                    )
                    logger.info("Trial builder dtype=%s layers=%s", cache_dtype, group.layer_names)
            elif len(cache_dtypes) > 1:
                raise ValueError("Mixed cache dtypes in an attention group")
            group.create_metadata_builders(
                vllm_config=builder_config,'''
    if source.count(old) != 1:
        raise ValueError("Unexpected attention-builder source; refusing patch")
    result = source.replace(old, new)
    compile(result, "attn_utils.py", "exec")
    return result


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:])
    with destination.open("x") as output:
        output.write(patch(source.read_text()))
