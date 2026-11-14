"""GGUF metadata reader for llama.cpp model files.

Parses GGUF binary format to extract model architecture, layers, context,
and other metadata without loading tensors.
"""

import glob
import os
import re
import struct

# Preferred mmproj quantization types (best first)
MMPROJ_PREFERENCE = ["f16", "bf16", "q8_0", "f32"]


# GGUF type → struct format map
_FMT = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def gguf_meta(path):
    """Read the key/value metadata of a GGUF file (no tensors)."""
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        f.read(4)  # version
        _, nkv = struct.unpack("<QQ", f.read(16))

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                at, n = struct.unpack("<IQ", f.read(12))
                if at == 8:
                    if n > 256:
                        for _ in range(n):
                            (ln,) = struct.unpack("<Q", f.read(8))
                            f.seek(ln, 1)
                        return f"<{n} strings>"
                    return [rstr() for _ in range(n)]
                if at in _FMT:
                    size = struct.calcsize(_FMT[at])
                    if n > 256:
                        f.seek(size * n, 1)
                        return f"<{n} items>"
                    return [struct.unpack(_FMT[at], f.read(size))[0] for _ in range(n)]
                return [rval(at) for _ in range(n)]
            return struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]

        for _ in range(nkv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            meta[key] = rval(t)
    return meta


def model_files(path):
    """Expand sharded GGUF filenames (e.g. *-00001-of-00005.gguf → all shards)."""
    if not re.search(r"-\d{5}-of-\d{5}\.gguf$", path):
        return [path]
    return sorted(glob.glob(re.sub(r"-\d{5}-of-", "-*-of-", path)))


def is_mmproj(path):
    """Check if a path is an mmproj (multimodal projection) file."""
    return "mmproj" in os.path.basename(path).lower()


def is_extra_part(path):
    """Check if a path is a non-first shard of a sharded GGUF file."""
    return bool(re.search(r"-(\d{5})-of-\d{5}\.gguf$", path)) and not re.search(
        r"-00001-of-\d{5}\.gguf$", path
    )


def model_info(path):
    """Extract model architecture, layers, experts, context, and size from GGUF metadata."""
    meta = gguf_meta(path)
    arch = meta.get("general.architecture", "?")
    g = lambda k, d=0: meta.get(f"{arch}.{k}", d)
    experts = g("expert_count", 0) or 0
    kv = g("attention.head_count_kv", 0)
    if isinstance(kv, list):  # hybrid models: 0 = no KV cache in that layer
        attn_layers = sum(1 for x in kv if x)
        kv_heads = max(kv) if kv else 0
    else:
        attn_layers, kv_heads = None, kv
    size = sum(os.path.getsize(p) for p in model_files(path))
    return {
        "name": meta.get("general.name", os.path.basename(path)),
        "arch": arch,
        "layers": int(g("block_count", 0) or 0),
        "native_ctx": int(g("context_length", 0) or 0),
        "experts": int(experts),
        "experts_used": int(g("expert_used_count", 0) or 0),
        "kv_heads": kv_heads,
        "attn_layers": attn_layers,
        "file_type": meta.get("general.file_type", ""),
        "size_mib": round(size / 1048576),
        "type": "moe" if experts else "dense",
    }


def find_mmproj(model_path):
    """Best mmproj file in the folder of the model (F16 first), or ''."""
    files = [
        p for p in glob.glob(os.path.join(os.path.dirname(model_path), "*.gguf")) if is_mmproj(p)
    ]
    if not files:
        return ""

    def rank(p):
        b = os.path.basename(p).lower()
        for i, q in enumerate(MMPROJ_PREFERENCE):
            if re.search(rf"[-_.]{q}[-_.]", b + "."):
                return i
        return len(MMPROJ_PREFERENCE)

    return sorted(files, key=lambda p: (rank(p), p))[0]


__all__ = [
    "MMPROJ_PREFERENCE",
    "gguf_meta",
    "model_files",
    "is_mmproj",
    "is_extra_part",
    "model_info",
    "find_mmproj",
]
