(* safetensors.sig -- a reader for the safetensors weight-container format
   (https://github.com/huggingface/safetensors), in pure Standard ML.

   The format is: an 8-byte little-endian u64 header length N; then N bytes of
   JSON header mapping each tensor name to `{dtype, shape, data_offsets:[b,e]}`
   (plus an optional `__metadata__` string map); then the raw little-endian,
   row-major tensor buffers, with `data_offsets` measured from the start of that
   data section.

   This module is PURE: bytes in, tensors out, no file IO (a caller reads the
   file with `BinIO` and hands over the `Word8Vector.vector`). Floating-point
   payloads (F16/BF16/F32/F64) are dequantized to SML `real` via `sml-float`, so
   decoding is byte-identical under MLton and Poly/ML. Malformed input (short
   file, bad JSON header, offsets out of range, size mismatch, unknown dtype)
   raises `Safetensors msg` with a human-readable message -- never a raw
   `Subscript`/`Overflow`. Reusable for reading any safetensors model. *)

signature SAFETENSORS =
sig
  (* The safetensors dtype tags. `get` dequantizes the four float widths to
     `real`; the integer/bool tags are reported by `dtypeOf` but `get` raises
     `Safetensors` for them (no lossy real conversion is implied). *)
  datatype dtype =
      F64 | F32 | F16 | BF16
    | I64 | I32 | I16 | I8 | U8 | BOOL

  type model

  (* Raised for any malformed input or unsupported request, with a message. *)
  exception Safetensors of string

  (* Parse a whole safetensors image. Validates the header length, the JSON
     header, and every tensor's offsets/size against the data section. *)
  val parse : Word8Vector.vector -> model

  (* Tensor names, in header order (`__metadata__` is not a tensor name). *)
  val names : model -> string list

  (* The `__metadata__` string map (empty if absent), in header order. *)
  val metadata : model -> (string * string) list

  val contains : model -> string -> bool
  val dtypeOf  : model -> string -> dtype       (* raises if unknown name *)
  val shapeOf  : model -> string -> int list    (* raises if unknown name *)

  (* Fetch a tensor, dequantizing its buffer to a row-major `real vector`.
     `dtype` is the on-disk tag; `data` has `product shape` elements. Raises
     `Safetensors` for an unknown name or a non-float dtype. *)
  val get : model -> string -> {dtype : dtype, shape : int list, data : real vector}

  (* The on-disk tag name, e.g. `dtypeName F32 = "F32"`. *)
  val dtypeName : dtype -> string

  (* Bytes per element of a dtype (e.g. `F32` = 4, `BF16` = 2, `BOOL` = 1). *)
  val dtypeSize : dtype -> int
end
