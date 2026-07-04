(* float.sig -- IEEE-754 <-> bytes codec in pure Standard ML.

   Decodes and encodes IEEE-754 floating-point values to/from a
   `Word8Vector.vector` of raw bytes, for reading model weights and binary
   formats (safetensors, GGUF/npy, CBOR/msgpack floats, ...).

   Four widths: `f16` (half), `bf16` (bfloat16, the top 16 bits of an f32),
   `f32` (single), `f64` (double). Two byte orders: little-endian (`*Le`, the
   default for safetensors/x86) and big-endian (`*Be`, CBOR/network order).

   Decoding is exact and uses only integer arithmetic + `Math.pow`, so it is
   byte-identical under MLton and Poly/ML. Subnormals, signed zero, +/-inf and
   NaN are all handled. `decode*` raises `Subscript` if the bytes are not
   present at the given offset.

   Encoding rounds an SML `real` (an IEEE double) to the target width toward
   nearest, ties handled by the Basis; `f16`/`bf16` encoders saturate to +/-inf
   on overflow. `encode* (decode* (bytes)) = bytes` for every finite,
   representable value (round-trip); `decode* (encode* x)` reproduces `x` up to
   the target width's precision. *)

signature FLOAT =
sig
  (* ---- decode: (bytes, offset) -> real ---- *)
  (* little-endian *)
  val decodeF16Le  : Word8Vector.vector * int -> real   (* 2 bytes *)
  val decodeBf16Le : Word8Vector.vector * int -> real   (* 2 bytes *)
  val decodeF32Le  : Word8Vector.vector * int -> real   (* 4 bytes *)
  val decodeF64Le  : Word8Vector.vector * int -> real   (* 8 bytes *)
  (* big-endian *)
  val decodeF16Be  : Word8Vector.vector * int -> real
  val decodeBf16Be : Word8Vector.vector * int -> real
  val decodeF32Be  : Word8Vector.vector * int -> real
  val decodeF64Be  : Word8Vector.vector * int -> real

  (* ---- encode: real -> fresh Word8Vector ---- *)
  val encodeF16Le  : real -> Word8Vector.vector
  val encodeBf16Le : real -> Word8Vector.vector
  val encodeF32Le  : real -> Word8Vector.vector
  val encodeF64Le  : real -> Word8Vector.vector
  val encodeF16Be  : real -> Word8Vector.vector
  val encodeBf16Be : real -> Word8Vector.vector
  val encodeF32Be  : real -> Word8Vector.vector
  val encodeF64Be  : real -> Word8Vector.vector
end
