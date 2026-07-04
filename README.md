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
