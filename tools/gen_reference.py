#!/usr/bin/env python3
# Reference (oracle) for sml-safetensors' golden tests.
# Run:  python3 tools/gen_reference.py
#
# Builds a tiny .safetensors blob BY HAND from the published spec (independent
# of the SML reader): 8-byte little-endian u64 header length, a JSON header,
# then the concatenated little-endian tensor buffers. Emits the exact file bytes
# as an SML `Word8Vector.fromList [...]` plus the expected shapes and decoded
# reals. The SML reader must reproduce these. Uses only the stdlib (struct/json),
# so bf16 is computed by hand (round-to-nearest-even of the top 16 bits of f32),
# giving a decoder-independent oracle.
import struct, json

def f32_bytes(xs):
    return b"".join(struct.pack("<f", x) for x in xs)

def f16_bytes(xs):
    return b"".join(struct.pack("<e", x) for x in xs)   # IEEE half

def f32_to_bf16_u16(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    bias = 0x7FFF + ((u >> 16) & 1)                      # round to nearest even
    return (u + bias) >> 16 & 0xFFFF

def bf16_bytes(xs):
    return b"".join(struct.pack("<H", f32_to_bf16_u16(x)) for x in xs)

def bf16_value(x):                                       # what bf16 decodes back to
    return struct.unpack("<f", struct.pack("<I", f32_to_bf16_u16(x) << 16))[0]

def f16_value(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]

# ---- tiny fixture: three tensors exercising F32 / F16 / BF16 + __metadata__ ----
w = [1.5, -2.5, 3.25, -4.0]     # F32  shape [2,2]
b = [0.5, -0.25, 2.0]           # F16  shape [3]
e = [1.0, 0.1]                  # BF16 shape [2]  (0.1 is inexact -> real oracle)
n = [7, -3]                     # I32  shape [2]  (non-float: get must reject it)

wbuf, bbuf, ebuf = f32_bytes(w), f16_bytes(b), bf16_bytes(e)
nbuf = b"".join(struct.pack("<i", x) for x in n)
data = wbuf + bbuf + ebuf + nbuf
o = 0
def rng(buf):
    global o
    r = [o, o + len(buf)]; o += len(buf); return r
header = {
    "w": {"dtype": "F32",  "shape": [2, 2], "data_offsets": rng(wbuf)},
    "b": {"dtype": "F16",  "shape": [3],    "data_offsets": rng(bbuf)},
    "e": {"dtype": "BF16", "shape": [2],    "data_offsets": rng(ebuf)},
    "n": {"dtype": "I32",  "shape": [2],    "data_offsets": rng(nbuf)},
    "__metadata__": {"format": "pt"},
}
hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
blob = struct.pack("<Q", len(hjson)) + hjson + data

def sml_bytes(bs):
    return "Word8Vector.fromList [\n      " + ",".join("0wx%02X" % x for x in bs) + "]"

def sml_reals(xs):
    return "[" + ", ".join(("~%.10g" % -v) if v < 0 else ("%.10g" % v) for v in xs) + "]"

print("(* ==== paste into test/test.sml ==== *)")
print("val BLOB : Word8Vector.vector =\n    " + sml_bytes(blob))
print()
print("(* names (header order): w, b, e, n  (+ __metadata__ excluded from names) *)")
print("(* metadata: [(\"format\",\"pt\")] ; n is I32 -> get must raise *)")
print('W_F32  shape [2,2] =', sml_reals(w))
print('B_F16  shape [3]   =', sml_reals([f16_value(x) for x in b]))
print('E_BF16 shape [2]   =', sml_reals([bf16_value(x) for x in e]))
print()
print("header size N =", len(hjson), "| data bytes =", len(data), "| file bytes =", len(blob))
