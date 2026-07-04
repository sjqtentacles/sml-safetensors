(* float.sml -- IEEE-754 <-> bytes codec.

   Decoding uses only integer arithmetic + Math.pow on the sign/exponent/
   mantissa fields (the sml-cbor/msgpack approach), so it is exact and
   byte-identical under MLton and Poly/ML. Encoding extracts the mantissa/
   exponent with Real.toManExp / Real.toLargeInt and assembles the bit pattern
   as an IntInf, split into bytes with div/mod (never IntInf.~>>, whose sign
   handling differs across compilers). *)

structure Float :> FLOAT =
struct
  fun ob (v, i) = Word8.toInt (Word8Vector.sub (v, i))
  val nan : real = 0.0 / 0.0

  (* ---- decode from big-endian field bytes (b0 is most significant) ---- *)

  fun f16FromBytes (b0, b1) =
      let val sign = if b0 >= 128 then ~1.0 else 1.0
          val exp  = (b0 mod 128) div 4
          val mant = (b0 mod 4) * 256 + b1
      in
        if exp = 0 then sign * Real.fromInt mant * Math.pow (2.0, ~24.0)
        else if exp = 31 then (if mant = 0 then sign * Real.posInf else nan)
        else sign * (1.0 + Real.fromInt mant / 1024.0) * Math.pow (2.0, Real.fromInt (exp - 15))
      end

  fun bf16FromBytes (b0, b1) =
      let val sign = if b0 >= 128 then ~1.0 else 1.0
          val exp  = (b0 mod 128) * 2 + b1 div 128
          val mant = b1 mod 128
      in
        if exp = 0 then sign * Real.fromInt mant * Math.pow (2.0, ~133.0)
        else if exp = 255 then (if mant = 0 then sign * Real.posInf else nan)
        else sign * (1.0 + Real.fromInt mant / 128.0) * Math.pow (2.0, Real.fromInt (exp - 127))
      end

  fun f32FromBytes (b0, b1, b2, b3) =
      let val sign = if b0 >= 128 then ~1.0 else 1.0
          val exp  = (b0 mod 128) * 2 + b1 div 128
          val mant = (b1 mod 128) * 65536 + b2 * 256 + b3
      in
        if exp = 0 then sign * Real.fromInt mant * Math.pow (2.0, ~149.0)
        else if exp = 255 then (if mant = 0 then sign * Real.posInf else nan)
        else sign * (1.0 + Real.fromInt mant / 8388608.0) * Math.pow (2.0, Real.fromInt (exp - 127))
      end

  fun f64FromBytes (b0, b1, b2, b3, b4, b5, b6, b7) =
      let val sign = if b0 >= 128 then ~1.0 else 1.0
          val exp  = (b0 mod 128) * 16 + b1 div 16
          val mHi  = (b1 mod 16) * 65536 + b2 * 256 + b3
          val mLo  = b4 * 16777216 + b5 * 65536 + b6 * 256 + b7
          val mant = Real.fromInt mHi * 4294967296.0 + Real.fromInt mLo
      in
        if exp = 0 then sign * mant * Math.pow (2.0, ~1074.0)
        else if exp = 2047 then (if mHi = 0 andalso mLo = 0 then sign * Real.posInf else nan)
        else sign * (1.0 + mant / 4503599627370496.0) * Math.pow (2.0, Real.fromInt (exp - 1023))
      end

  fun decodeF16Be (v, i)  = f16FromBytes (ob (v, i), ob (v, i+1))
  fun decodeF16Le (v, i)  = f16FromBytes (ob (v, i+1), ob (v, i))
  fun decodeBf16Be (v, i) = bf16FromBytes (ob (v, i), ob (v, i+1))
  fun decodeBf16Le (v, i) = bf16FromBytes (ob (v, i+1), ob (v, i))
  fun decodeF32Be (v, i)  = f32FromBytes (ob (v, i), ob (v, i+1), ob (v, i+2), ob (v, i+3))
  fun decodeF32Le (v, i)  = f32FromBytes (ob (v, i+3), ob (v, i+2), ob (v, i+1), ob (v, i))
  fun decodeF64Be (v, i)  =
      f64FromBytes (ob (v,i), ob (v,i+1), ob (v,i+2), ob (v,i+3),
                    ob (v,i+4), ob (v,i+5), ob (v,i+6), ob (v,i+7))
  fun decodeF64Le (v, i)  =
      f64FromBytes (ob (v,i+7), ob (v,i+6), ob (v,i+5), ob (v,i+4),
                    ob (v,i+3), ob (v,i+2), ob (v,i+1), ob (v,i))

  (* ---- encode ----
     Assemble the IEEE bit pattern as a non-negative IntInf, then split into
     big-endian bytes with div/mod. Handles zero, inf, NaN, subnormal, and
     overflow-to-inf; rounds the mantissa to nearest (with carry into the
     exponent at the boundary). *)

  fun p2 n = IntInf.pow (2, n)

  fun encodeBits (expBits, mantBits, bias) (r : real) : IntInf.int =
      let
        val mantPow = p2 mantBits                 (* 2^mantBits *)
        val maxExp  = IntInf.toInt (p2 expBits) - 1  (* all-ones exponent field *)
        val signBit = if Real.signBit r then p2 (expBits + mantBits) else 0
        val expField = fn e => IntInf.fromInt e * mantPow
      in
        if Real.isNan r then signBit + expField maxExp + IntInf.div (mantPow, 2)
        else if not (Real.isFinite r) then signBit + expField maxExp
        else if Real.== (r, 0.0) then signBit
        else
          let
            val {man, exp} = Real.toManExp (Real.abs r)  (* |r| = man*2^exp, 0.5<=man<1 *)
            val storedExp = (exp - 1) + bias
          in
            if storedExp >= maxExp then signBit + expField maxExp     (* overflow -> inf *)
            else if storedExp <= 0 then
              (* subnormal / underflow: mantissa = round(|r| / 2^(1-bias-mantBits)) *)
              let val scaled = Real.abs r / Math.pow (2.0, Real.fromInt (1 - bias - mantBits))
                  val m = Real.toLargeInt IEEEReal.TO_NEAREST scaled
              in signBit + m end
            else
              let val frac = man * 2.0 - 1.0    (* in [0,1) *)
                  val m = Real.toLargeInt IEEEReal.TO_NEAREST (frac * Real.fromLargeInt mantPow)
                  val (se, mf) = if m >= mantPow then (storedExp + 1, m - mantPow)
                                 else (storedExp, m)
              in
                if se >= maxExp then signBit + expField maxExp
                else signBit + expField se + mf
              end
          end
      end

  fun bytesBE (bits, nbytes) : Word8Vector.vector =
      Word8Vector.tabulate (nbytes, fn i =>
        Word8.fromInt (IntInf.toInt
          (IntInf.mod (IntInf.div (bits, p2 (8 * (nbytes - 1 - i))), 256))))

  fun bytesLE (bits, nbytes) : Word8Vector.vector =
      Word8Vector.tabulate (nbytes, fn i =>
        Word8.fromInt (IntInf.toInt
          (IntInf.mod (IntInf.div (bits, p2 (8 * i)), 256))))

  val f16bits  = encodeBits (5, 10, 15)
  val bf16bits = encodeBits (8, 7, 127)
  val f32bits  = encodeBits (8, 23, 127)
  val f64bits  = encodeBits (11, 52, 1023)

  fun encodeF16Be r  = bytesBE (f16bits r, 2)
  fun encodeF16Le r  = bytesLE (f16bits r, 2)
  fun encodeBf16Be r = bytesBE (bf16bits r, 2)
  fun encodeBf16Le r = bytesLE (bf16bits r, 2)
  fun encodeF32Be r  = bytesBE (f32bits r, 4)
  fun encodeF32Le r  = bytesLE (f32bits r, 4)
  fun encodeF64Be r  = bytesBE (f64bits r, 8)
  fun encodeF64Le r  = bytesLE (f64bits r, 8)
end
