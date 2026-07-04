(* Tests for sml-safetensors. The BLOB is a real safetensors image built BY HAND
   from the published spec by an independent Python serializer (tools/gen_reference.py:
   8-byte LE header length, JSON header, then little-endian row-major buffers). The
   expected shapes are exact ints; decoded float payloads are compared within an
   epsilon (so the suite output is byte-identical across MLton and Poly/ML). The
   fixture carries F32/F16/BF16 tensors, an I32 tensor (which `get` must reject),
   and a `__metadata__` map. *)

structure Tests =
struct
  open Harness
  structure S = Safetensors

  (* --- golden blob + expected values (from tools/gen_reference.py) --- *)
  val BLOB : Word8Vector.vector =
      Word8Vector.fromList [
      0wxFE,0wx00,0wx00,0wx00,0wx00,0wx00,0wx00,0wx00,0wx7B,0wx22,0wx77,0wx22,0wx3A,0wx7B,0wx22,0wx64,0wx74,0wx79,0wx70,0wx65,0wx22,0wx3A,0wx22,0wx46,0wx33,0wx32,0wx22,0wx2C,0wx22,0wx73,0wx68,0wx61,0wx70,0wx65,0wx22,0wx3A,0wx5B,0wx32,0wx2C,0wx32,0wx5D,0wx2C,0wx22,0wx64,0wx61,0wx74,0wx61,0wx5F,0wx6F,0wx66,0wx66,0wx73,0wx65,0wx74,0wx73,0wx22,0wx3A,0wx5B,0wx30,0wx2C,0wx31,0wx36,0wx5D,0wx7D,0wx2C,0wx22,0wx62,0wx22,0wx3A,0wx7B,0wx22,0wx64,0wx74,0wx79,0wx70,0wx65,0wx22,0wx3A,0wx22,0wx46,0wx31,0wx36,0wx22,0wx2C,0wx22,0wx73,0wx68,0wx61,0wx70,0wx65,0wx22,0wx3A,0wx5B,0wx33,0wx5D,0wx2C,0wx22,0wx64,0wx61,0wx74,0wx61,0wx5F,0wx6F,0wx66,0wx66,0wx73,0wx65,0wx74,0wx73,0wx22,0wx3A,0wx5B,0wx31,0wx36,0wx2C,0wx32,0wx32,0wx5D,0wx7D,0wx2C,0wx22,0wx65,0wx22,0wx3A,0wx7B,0wx22,0wx64,0wx74,0wx79,0wx70,0wx65,0wx22,0wx3A,0wx22,0wx42,0wx46,0wx31,0wx36,0wx22,0wx2C,0wx22,0wx73,0wx68,0wx61,0wx70,0wx65,0wx22,0wx3A,0wx5B,0wx32,0wx5D,0wx2C,0wx22,0wx64,0wx61,0wx74,0wx61,0wx5F,0wx6F,0wx66,0wx66,0wx73,0wx65,0wx74,0wx73,0wx22,0wx3A,0wx5B,0wx32,0wx32,0wx2C,0wx32,0wx36,0wx5D,0wx7D,0wx2C,0wx22,0wx6E,0wx22,0wx3A,0wx7B,0wx22,0wx64,0wx74,0wx79,0wx70,0wx65,0wx22,0wx3A,0wx22,0wx49,0wx33,0wx32,0wx22,0wx2C,0wx22,0wx73,0wx68,0wx61,0wx70,0wx65,0wx22,0wx3A,0wx5B,0wx32,0wx5D,0wx2C,0wx22,0wx64,0wx61,0wx74,0wx61,0wx5F,0wx6F,0wx66,0wx66,0wx73,0wx65,0wx74,0wx73,0wx22,0wx3A,0wx5B,0wx32,0wx36,0wx2C,0wx33,0wx34,0wx5D,0wx7D,0wx2C,0wx22,0wx5F,0wx5F,0wx6D,0wx65,0wx74,0wx61,0wx64,0wx61,0wx74,0wx61,0wx5F,0wx5F,0wx22,0wx3A,0wx7B,0wx22,0wx66,0wx6F,0wx72,0wx6D,0wx61,0wx74,0wx22,0wx3A,0wx22,0wx70,0wx74,0wx22,0wx7D,0wx7D,0wx00,0wx00,0wxC0,0wx3F,0wx00,0wx00,0wx20,0wxC0,0wx00,0wx00,0wx50,0wx40,0wx00,0wx00,0wx80,0wxC0,0wx00,0wx38,0wx00,0wxB4,0wx00,0wx40,0wx80,0wx3F,0wxCD,0wx3D,0wx07,0wx00,0wx00,0wx00,0wxFD,0wxFF,0wxFF,0wxFF]

  val W_F32  = [1.5, ~2.5, 3.25, ~4.0]     (* shape [2,2] *)
  val B_F16  = [0.5, ~0.25, 2.0]           (* shape [3]   *)
  val E_BF16 = [1.0, 0.1000976562]         (* shape [2]   (bf16 of 0.1) *)

  fun vecEq eps (xs, v) =
      length xs = Vector.length v andalso
      let fun go (i, []) = true
            | go (i, x :: r) = Real.abs (x - Vector.sub (v, i)) <= eps andalso go (i + 1, r)
      in go (0, xs) end

  fun run () =
    let
      val m = S.parse BLOB

      val () = section "parse: names / metadata / contains"
      val () = checkStringList "names (header order, no __metadata__)"
                 (["w","b","e","n"], S.names m)
      val () = check "metadata format=pt" (S.metadata m = [("format","pt")])
      val () = check "contains w"   (S.contains m "w")
      val () = check "not contains z" (not (S.contains m "z"))

      val () = section "dtypeOf / shapeOf"
      val () = checkString "dtypeOf w = F32"  ("F32",  S.dtypeName (S.dtypeOf m "w"))
      val () = checkString "dtypeOf b = F16"  ("F16",  S.dtypeName (S.dtypeOf m "b"))
      val () = checkString "dtypeOf e = BF16" ("BF16", S.dtypeName (S.dtypeOf m "e"))
      val () = checkString "dtypeOf n = I32"  ("I32",  S.dtypeName (S.dtypeOf m "n"))
      val () = checkIntList "shapeOf w = [2,2]" ([2,2], S.shapeOf m "w")
      val () = checkIntList "shapeOf b = [3]"   ([3],   S.shapeOf m "b")
      val () = checkIntList "shapeOf e = [2]"   ([2],   S.shapeOf m "e")

      val () = section "get: F32 / F16 / BF16 dequantize"
      val () = check "get w (F32) values"  (vecEq 1E~6 (W_F32,  #data (S.get m "w")))
      val () = check "get b (F16) values"  (vecEq 1E~6 (B_F16,  #data (S.get m "b")))
      val () = check "get e (BF16) values" (vecEq 1E~6 (E_BF16, #data (S.get m "e")))
      val () = checkIntList "get w shape" ([2,2], #shape (S.get m "w"))

      val () = section "documented errors (Safetensors, never raw Subscript)"
      val () = checkRaises "parse short file"
                 (fn () => S.parse (Word8Vector.fromList [0w1,0w2,0w3]))
      val () = checkRaises "get unknown name"    (fn () => S.get m "nope")
      val () = checkRaises "get non-float (I32)"  (fn () => S.get m "n")
    in
      Harness.run ()
    end
end
