~ "parse.ua"
  ~ ParseSafeTensorsF
ParseSafeTensorsF₁₆ ← ParseSafeTensorsF
~ "model.ua"
  ~ DecoderBlock EmbeddingBlock OutputRMS UnEmbeddingBlock

DefaultPath ← "/Users/maxkhanov/Desktop/js/qwen_np/converted16.safetensors"

# weightpath ? 
ParseArgs ← (
  &args
  ⍤"Too many arguments!\nUsage: uiua run llm.ua <optional path>"≤2⧻.
  ⨬(⍤"Impossible!"0|DefaultPath|°□⊏1)⧻.
)

# map ? map
AppendModelInfo ← (
  insert "theta" 1000000
  insert "head_dim" 64
  insert "nhead" 14
  insert "kv_num" 2
  insert "decoders" 24
)

ModelWeightMap ← AppendModelInfo ParseSafeTensorsF₁₆ ParseArgs

# token ? input wmap
QwenModel ← (
  ⟜(&p⊂:" tokens )"⊂"Running model forward ( "°⋕⧻)
  ⊸EmbeddingBlock
  ∧(⊸DecoderBlock &pf".")⇡24
  ¤⊣ # don't decode tokens that are not being generated
  ⊸OutputRMS
  UnEmbeddingBlock
  &pf"\n"
)

TestTokens  ← [3838 374 220 20 15 15 10 18 1939 16141 25]
NUMTOKENGEN ← 16

ModelWeightMap
⍥⊸(⊂⟜(⊣≡₁(⊢⍖) QwenModel)) NUMTOKENGEN TestTokens
⊙◌
&p "Inference completed!"
