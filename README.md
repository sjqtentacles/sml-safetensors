# sml-safetensors

[![CI](https://github.com/sjqtentacles/sml-safetensors/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-safetensors/actions/workflows/ci.yml)

A reader for the [safetensors](https://github.com/huggingface/safetensors)
weight-container format — the modern standard for model weights — in pure
Standard ML.

The format is simple: an 8-byte little-endian `u64` header length `N`; then `N`
bytes of JSON header mapping each tensor name to `{dtype, shape,
data_offsets:[begin,end]}` (plus an optional `__metadata__` string map); then the
raw little-endian, row-major tensor buffers, with `data_offsets` measured from
the start of that data section.

This module is **pure** — bytes in, tensors out, no file IO (a caller reads the
file with `BinIO` and hands over the `Word8Vector.vector`). Floating-point
payloads (`F16`/`BF16`/`F32`/`F64`) are dequantized to SML `real` via
[`sml-float`](https://github.com/sjqtentacles/sml-float), so decoding is
**byte-identical under MLton and Poly/ML**. Every malformed input (short file,
bad header JSON, offsets out of range, size mismatch, unknown dtype) raises
`Safetensors msg` with a readable message — never a raw `Subscript`/`Overflow`.

Reusable for reading *any* safetensors model (GPT-2, BERT, …).

## Installation

```
smlpkg add github.com/sjqtentacles/sml-safetensors
smlpkg sync
```

Depends on [`sml-json`](https://github.com/sjqtentacles/sml-json) and
[`sml-float`](https://github.com/sjqtentacles/sml-float) (fetched by `smlpkg sync`).

## Usage

```sml
(* caller does the IO; the library is pure *)
val bytes = let val s = BinIO.openIn "model.safetensors"
            in BinIO.inputAll s before BinIO.closeIn s end
val model = Safetensors.parse bytes

val ns = Safetensors.names model                 (* tensor names, header order *)
val {dtype, shape, data} = Safetensors.get model "wte.weight"
(* data : real vector, row-major, `product shape` elements *)
```

## API (`signature SAFETENSORS`)

```sml
datatype dtype = F64 | F32 | F16 | BF16 | I64 | I32 | I16 | I8 | U8 | BOOL
type model
exception Safetensors of string

val parse     : Word8Vector.vector -> model
val names     : model -> string list
val metadata  : model -> (string * string) list
val contains  : model -> string -> bool
val dtypeOf   : model -> string -> dtype
val shapeOf   : model -> string -> int list
val get       : model -> string -> {dtype:dtype, shape:int list, data:real vector}
val dtypeName : dtype -> string
val dtypeSize : dtype -> int
```

`get` dequantizes the four float widths to `real`; the integer/bool dtypes are
reported by `dtypeOf`/`shapeOf` but `get` raises `Safetensors` for them (no lossy
real conversion is implied).

## Example

`make example` builds and runs [`examples/demo.sml`](examples/demo.sml), which
hand-builds a tiny two-tensor safetensors image in memory (F32 and F16
tensors plus a `__metadata__` entry) and exercises `parse`, `names`,
`metadata`, `contains`, `dtypeOf`, `shapeOf`, `get` (dequantizing both float
widths), `dtypeName`, `dtypeSize`, and the `Safetensors` error path on a
truncated blob (output is byte-identical under MLton and Poly/ML):

```
safetensors: hand-built image with tensors 'w' (F32 2x2) and 'b' (F16 len 2)
  names        = [w, b]
  metadata     = [note=demo]
  contains w/z = true / false
  dtypeOf w    = F32 (4 B/elem)
  dtypeOf b    = F16 (2 B/elem)
  shapeOf w    = [2,2]
  get w: data  = [1.0000, 2.0000, ~1.0000, 0.5000]
  get b: data  = [1.0000, ~1.0000]
  parse of truncated blob: caught Safetensors "truncated: no 8-byte header length"
```

## Testing

```
make test       # MLton
make test-poly  # Poly/ML
```

18 assertions, green on MLton and Poly/ML with byte-identical output. The golden
`BLOB` is a **real safetensors image built by an independent Python serializer**
(`tools/gen_reference.py`, spec-faithful, stdlib-only — bf16 computed by hand) so
the oracle does not share code with the SML reader; the test asserts exact shapes
and epsilon-close float payloads. CI needs no Python — only the SML and the
committed constants ship.

## License

MIT
