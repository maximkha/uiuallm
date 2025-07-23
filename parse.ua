KVs ← °map

# Splits a array into two parts: x[:n] and x[n:]
# prefix suffix ? n array
Yank ← ⊃↙↘
# given an array of bytes parses a u64
ParseU₆₄  ← /+×ⁿ:256⇡⧻.Yank8
TakeRange ← ↘:↙⊙:°⊟⇌

# Parses a FP 16 number from a high and low byte. **Only works for non-nan and inf
# numbers.**
# Float ? HighByte LowByte
# The functions used inside are pervasive and effictively makes this whole function
# pervasive.
ParseF₁₆ ← (
  ⊃(-×32≥32.⌊÷4|+×256◿4|ⁿ:¯1≥128)
  ⊃(ⁿ:2↥¯14-15|÷1024+×1024≠0)
  ××
)

# Takes a file path and parses the SafeTensors format with FP16 tensors.
# Returns a map from tensor names to their values.
# TensorMap ? FilePath
ParseSafeTensorsF₁₆ ← (
  &fo
  &rb ∞                   # due to a bug in uiua, we need to read the whole file rather than parse it as a stream
  °json°utf₈Yank ParseU₆₄ # get the json metadata
  ⊙::KVs⊙□                # get the key-value pairs from the metadata
  &p"Parsing Tensors"
  ⊞(
    ⊙:⊸get"data_offsets"°□ # get the byte offsets into the data section of the file
    TakeRange∩°□           # take the bytes for the tensor
    ↯∞_2                   # reshape into pairs of bytes (high and low)
    :°⊟⍉                   # split into high and low byte arrays (since FP16 parser is pervasive)
    ParseF₁₆
    ↯°□get"shape": # reshape the parsed data into the shape specified in the metadata
    &pf"."         # provide feedback to user that parsing is in progress
    □              # box it up!
  )
  map: # construct the map from tensor name to values
  &pf "\nDone loading safetensors!\n"
)

# currently importing functions with subscripts is broken, so we use the following workaround:
ParseSafeTensorsF ← ParseSafeTensorsF₁₆
