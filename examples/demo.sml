(* demo.sml - hand-builds a tiny two-tensor safetensors image in memory (no
   file IO, no fixture) and exercises the SAFETENSORS reader API: parse,
   names, metadata, contains, dtypeOf, shapeOf, get (F32 + F16 dequantize),
   dtypeName, dtypeSize, and the documented `Safetensors` error on malformed
   input. Deterministic: every byte below is a literal. *)

structure S = Safetensors

fun fmtReal x =
  let val x = if Real.== (x, 0.0) then 0.0 else x
  in Real.fmt (StringCvt.FIX (SOME 4)) x end

fun fmtReals xs = "[" ^ String.concatWith ", " (List.map fmtReal xs) ^ "]"

(* JSON header for two tensors: "w" (F32, shape [2,2]) and "b" (F16, shape
   [2]), plus a __metadata__ map -- exactly the layout safetensors.sig
   documents. *)
val header =
  "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}," ^
  "\"b\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[16,20]}," ^
  "\"__metadata__\":{\"note\":\"demo\"}}"

(* 8-byte little-endian u64 header length, computed (not hand-counted). *)
fun u64le n =
  let
    val b0 = n mod 256  val n1 = n div 256
    val b1 = n1 mod 256 val n2 = n1 div 256
    val b2 = n2 mod 256 val n3 = n2 div 256
    val b3 = n3 mod 256
  in [b0, b1, b2, b3, 0, 0, 0, 0] end

(* Raw little-endian tensor bytes (standard IEEE-754 bit patterns):
   w = [1.0, 2.0, ~1.0, 0.5] as F32 (4 bytes/elem);
   b = [1.0, ~1.0] as F16 (2 bytes/elem). *)
val wBytes = [0,0,128,63, 0,0,0,64, 0,0,128,191, 0,0,0,63]
val bBytes = [0,60, 0,188]

val headerBytes = List.map (Word8.fromInt o Char.ord) (String.explode header)

val blob =
  Word8Vector.fromList
    (List.map Word8.fromInt (u64le (String.size header))
     @ headerBytes
     @ List.map Word8.fromInt (wBytes @ bBytes))

val () = print "safetensors: hand-built image with tensors 'w' (F32 2x2) and 'b' (F16 len 2)\n"
val m = S.parse blob
val () = print ("  names        = [" ^ String.concatWith ", " (S.names m) ^ "]\n")
val () = print ("  metadata     = ["
                ^ String.concatWith ", " (List.map (fn (k, v) => k ^ "=" ^ v) (S.metadata m))
                ^ "]\n")
val () = print ("  contains w/z = " ^ Bool.toString (S.contains m "w")
                ^ " / " ^ Bool.toString (S.contains m "z") ^ "\n")
val () = print ("  dtypeOf w    = " ^ S.dtypeName (S.dtypeOf m "w")
                ^ " (" ^ Int.toString (S.dtypeSize (S.dtypeOf m "w")) ^ " B/elem)\n")
val () = print ("  dtypeOf b    = " ^ S.dtypeName (S.dtypeOf m "b")
                ^ " (" ^ Int.toString (S.dtypeSize (S.dtypeOf m "b")) ^ " B/elem)\n")
val () = print ("  shapeOf w    = [" ^ String.concatWith "," (List.map Int.toString (S.shapeOf m "w")) ^ "]\n")

val w = S.get m "w"
val () = print ("  get w: data  = " ^ fmtReals (Vector.foldr (op ::) [] (#data w)) ^ "\n")
val b = S.get m "b"
val () = print ("  get b: data  = " ^ fmtReals (Vector.foldr (op ::) [] (#data b)) ^ "\n")

val () =
  (S.parse (Word8Vector.fromList [0w1, 0w2, 0w3]);
   print "  parse of truncated blob: (no exception -- unexpected)\n")
  handle S.Safetensors msg => print ("  parse of truncated blob: caught Safetensors \"" ^ msg ^ "\"\n")
