(* safetensors.sml -- a pure reader for the safetensors weight-container format.

   Layout: 8-byte little-endian u64 header length N; N bytes of JSON header
   (name -> {dtype, shape, data_offsets:[b,e]}, plus optional __metadata__);
   then the little-endian, row-major tensor buffers, with data_offsets measured
   from the start of that data section. Float payloads dequantize via sml-float,
   so decoding is byte-identical under MLton and Poly/ML. All malformed input is
   funnelled to `Safetensors msg` (never a raw Subscript/Overflow). *)

structure Safetensors :> SAFETENSORS =
struct
  structure J = Json
  structure W = Word8Vector

  datatype dtype =
      F64 | F32 | F16 | BF16
    | I64 | I32 | I16 | I8 | U8 | BOOL

  exception Safetensors of string

  fun dtypeName F64 = "F64" | dtypeName F32 = "F32" | dtypeName F16 = "F16"
    | dtypeName BF16 = "BF16" | dtypeName I64 = "I64" | dtypeName I32 = "I32"
    | dtypeName I16 = "I16" | dtypeName I8 = "I8" | dtypeName U8 = "U8"
    | dtypeName BOOL = "BOOL"

  fun dtypeSize F64 = 8 | dtypeSize F32 = 4 | dtypeSize F16 = 2 | dtypeSize BF16 = 2
    | dtypeSize I64 = 8 | dtypeSize I32 = 4 | dtypeSize I16 = 2 | dtypeSize I8 = 1
    | dtypeSize U8 = 1 | dtypeSize BOOL = 1

  fun parseDtype "F64" = F64  | parseDtype "F32" = F32 | parseDtype "F16" = F16
    | parseDtype "BF16" = BF16 | parseDtype "I64" = I64 | parseDtype "I32" = I32
    | parseDtype "I16" = I16  | parseDtype "I8" = I8   | parseDtype "U8" = U8
    | parseDtype "BOOL" = BOOL
    | parseDtype s = raise Safetensors ("unknown dtype: " ^ s)

  (* off/len are relative to the start of the data section. *)
  type entry = { dtype : dtype, shape : int list, off : int, len : int }
  type model = { data : W.vector, dataStart : int,
                 entries : (string * entry) list, meta : (string * string) list }

  fun product xs = List.foldl op* 1 xs
  fun u8 (v, i) = Word8.toInt (W.sub (v, i))

  (* 8-byte little-endian unsigned -> IntInf (no width assumptions). *)
  fun readU64Le (v, off) =
      let fun go (k, acc) =
              if k < 0 then acc
              else go (k - 1, acc * 256 + IntInf.fromInt (u8 (v, off + k)))
      in go (7, 0 : IntInf.int) end

  (* ---- JSON header accessors (each failure is a Safetensors message) ---- *)
  fun asObj (J.JObj fs) = fs
    | asObj _ = raise Safetensors "header: expected an object"
  fun asStr (J.JStr s) = s
    | asStr _ = raise Safetensors "header: expected a string"
  fun asIntv (J.JInt n) =
        (IntInf.toInt n handle Overflow => raise Safetensors "header: integer out of range")
    | asIntv _ = raise Safetensors "header: expected an integer"
  fun asIntList (J.JArr xs) = List.map asIntv xs
    | asIntList _ = raise Safetensors "header: expected an array"
  fun field fs k =
      case List.find (fn (k', _) => k' = k) fs of
          SOME (_, v) => v
        | NONE => raise Safetensors ("header: missing field \"" ^ k ^ "\"")

  fun parse v =
      let
        val total = W.length v
        val () = if total < 8 then raise Safetensors "truncated: no 8-byte header length" else ()
        val n = IntInf.toInt (readU64Le (v, 0))
                handle Overflow => raise Safetensors "header length too large for this platform"
        val () = if n < 0 orelse 8 + n > total
                 then raise Safetensors "truncated: header length exceeds file" else ()
        val dataStart = 8 + n
        val dataLen = total - dataStart
        val headerStr = Byte.unpackStringVec (Word8VectorSlice.slice (v, 8, SOME n))
        val members =
            asObj (case J.parseJson headerStr of
                       CharParsec.Ok j => j
                     | CharParsec.Err e =>
                         raise Safetensors ("bad header JSON: " ^ CharParsec.errorToString e))

        fun buildEntry (name, value) =
            let val fs = asObj value
                val dt = parseDtype (asStr (field fs "dtype"))
                val shape = asIntList (field fs "shape")
                val (b, e) =
                    case asIntList (field fs "data_offsets") of
                        [b, e] => (b, e)
                      | _ => raise Safetensors
                               ("tensor \"" ^ name ^ "\": data_offsets must be [begin,end]")
                val () = if b < 0 orelse e < b orelse e > dataLen
                         then raise Safetensors ("tensor \"" ^ name ^ "\": offsets out of range")
                         else ()
                val () = if e - b <> product shape * dtypeSize dt
                         then raise Safetensors
                                ("tensor \"" ^ name ^ "\": byte span <> product(shape) * sizeof(dtype)")
                         else ()
            in (name, { dtype = dt, shape = shape, off = b, len = e - b }) end

        fun walk ([], ents, meta) = (rev ents, rev meta)
          | walk ((name, value) :: rest, ents, meta) =
              if name = "__metadata__"
              then walk (rest, ents,
                         List.revAppend (List.map (fn (k, x) => (k, asStr x)) (asObj value), meta))
              else walk (rest, buildEntry (name, value) :: ents, meta)

        val (entries, meta) = walk (members, [], [])
      in
        { data = v, dataStart = dataStart, entries = entries, meta = meta }
      end

  fun names (m : model) = List.map #1 (#entries m)
  fun metadata (m : model) = #meta m
  fun contains (m : model) name = List.exists (fn (n, _) => n = name) (#entries m)
  fun findEntry (m : model) name =
      case List.find (fn (n, _) => n = name) (#entries m) of
          SOME (_, e) => e
        | NONE => raise Safetensors ("no such tensor: " ^ name)
  fun dtypeOf m name = #dtype (findEntry m name)
  fun shapeOf m name = #shape (findEntry m name)

  fun decoderFor F64  = Float.decodeF64Le
    | decoderFor F32  = Float.decodeF32Le
    | decoderFor F16  = Float.decodeF16Le
    | decoderFor BF16 = Float.decodeBf16Le
    | decoderFor dt   = raise Safetensors ("get: dtype " ^ dtypeName dt ^ " is not floating-point")

  fun get (m : model) name =
      let val e = findEntry m name
          val dt = #dtype e
          val dec = decoderFor dt          (* raises for non-float before decoding *)
          val esz = dtypeSize dt
          val base = #dataStart m + #off e
          val v = #data m
          val data = Vector.tabulate (#len e div esz, fn i => dec (v, base + i * esz))
      in { dtype = dt, shape = #shape e, data = data } end
end
