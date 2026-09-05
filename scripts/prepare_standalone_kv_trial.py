"""Select existing standalone draft-cache allocation in an isolated source copy."""
from pathlib import Path
import sys


def patch(source: str, block: int = 64) -> str:
    if block not in (64, 128, 256):
        raise ValueError("Trial block must divide the 1792-token MLA block")
    old = '''                    block_size=compact_block,
                    page_size_padded=mla_page,'''
    new = '''                    block_size=compact_block,
                    page_size_padded=None,'''
    if source.count(old) != 1:
        raise ValueError("Unexpected padded draft-cache source")
    result = source.replace(old, new).replace(
        '"DFlash2 drafter KV: padded slot-share block=%d "',
        '"DFlash2 drafter KV: TRIAL standalone block=%d "',
    )
    if source.count("            compact_block = 64") != 1:
        raise ValueError("Unexpected compact block declaration")
    result = result.replace("            compact_block = 64", f"            compact_block = {block}")
    compile(result, "kv_cache_utils.py", "exec")
    return result


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:3])
    with destination.open("x") as output:
        output.write(patch(source.read_text(), int(sys.argv[3]) if len(sys.argv) > 3 else 64))
