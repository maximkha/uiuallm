Uget ← °□(⍣get(⍤⊙0⊂"Key not found: "))

Eps ← 1E-6

# TODO: check if we can replace this with a more efficient implementation
TMM ← ⍉⊞(/+×)

EmbeddingBlock ← (
  ⊙(Uget"model.embed_tokens.weight") # fetch embedding weights
  ⊏                                  # select token embeddings
)

# Fetches weights from the model map for a decoder layer. It tacks them onto the stack.
FetchWeights ← (
  :⊂:"."⊂"model.layers."°⋕
  ⊙⊃(
    # RMS weights
    Uget⊂:"input_layernorm.weight"
    # attention weights
  | Uget⊂:"self_attn.k_proj.weight"
  | Uget⊂:"self_attn.q_proj.weight"
  | Uget⊂:"self_attn.v_proj.weight"
    # attention biases
  | Uget⊂:"self_attn.k_proj.bias"
  | Uget⊂:"self_attn.q_proj.bias"
  | Uget⊂:"self_attn.v_proj.bias"
    # rotary embedding data
  | Uget⊙⋅∘"head_dim"
  | Uget⊙⋅∘"theta"
    # final weights for attention (no biases!)
  | Uget⊂:"self_attn.o_proj.weight"
    # Pre MLP norm
  | Uget⊂:"post_attention_layernorm.weight"
    # MLP weights
  | Uget⊂:"mlp.gate_proj.weight"
  | Uget⊂:"mlp.up_proj.weight"
  | Uget⊂:"mlp.down_proj.weight"
  )
)

# Repeats a tensor column-wise
HTile ← ≡♭⤸1↯2

Rotate ← ≡₁⍜(°⊟↯2_∞)(¯:)

# sincos ? seqlen dim theta
RotaryEmbeddingSinCos ← (
  ⊓(⇡|÷:1ⁿ÷:×2⇡÷2.) # compute positions and column inverse frequencies
  ⊞×                # multiply the positions with the inverse frequencies
  ∩∿+η.             # compute cos and sin values
  ∩HTile            # tile the cos and sin values to match the input dimension
  ⊟:                # join the sin and cos values (this will be useful for applying them)
)

# kq ? kq sincos
ApplyRotaryKQ ← (
  ⊙(□≡₁↯1)        # expand sincos to △[2 seqlen 1 dim] and box it
  ⊞⍚(+⍜⊟×Rotate.) # apply rotary embeddings to kq
)

RMSNorm ← (
  ÷√+Eps⍜⍉(÷:⊃/+⧻)ⁿ2. # row-wise RMS norm
  ⍜⍉×                 # apply the norm weights
)

# reshapes to [seqlen ∞ hdim]
# kqv ? kqv hdim
ReshapeProjKQV ← ⊞⍚≡(↯:⊙(⊂∞))⊙□

# masks upper triangular part of a matrix with ¯∞
CMask   ← +-⊸÷:1⊞≤.⇡⧻.
Softmax ← ÷⍜⍉/+.ⁿ:e
SiLU    ← ÷+1ⁿ:e⊸¯

# x ? k q v
ComputeAttnHead ← (
  ⊙⟜(√⧻₁) # compute scale factor for attention
  TMM     # qk products
  ÷:      # apply the scale factor computed earlier
  CMask   # apply the causal mask
  -:⟜⍜⍉/↥ # subtract the max value from each row to improve numerical stability
  Softmax # softmax for each row
  TMM:⊙⍉  # apply the attention weights to the v matrix
)

# tk q tv ? k q v
ReshapeGroupKV ← (
  □÷◡∩⧻₁               # compute group size and box it
  ⍜⊙(⊟∩□⊙:)⊞⍚(≡₃♭₂≡₁↯) # reshape k v
  ⊙⊙°□                 # unclean, but needed to unbox v
)

# kqv ? x kmat qmat vmat kbias qbias vbias hdim
ProjectKQV ← (
  □              # box input tensor
  ⊙∩(⊂⊂∩₃□)      # convert k q v mat & biases to boxes
  ⊞⍚TMM:         # do KQV MMs
  ⍚⍜⍉+           # add KQV biases
  ReshapeProjKQV # reshape kqv to △[seqlen ∞ hdim]
)

# x ? x gate up down
MLP ← (
  ∩₃□         # box input and gate and up matrices
  ⍜⊙⊟(⊞⍚TMM:) # do MMs for gate and up
  ∩°□         # unbox the gate and up results
  ×SiLU       # apply gating
  TMM:        # do the final down projection
)

# x ? x kmat qmat vmat kbias qbias vbias hdim theta omat
GQAttention ← (
  ⊃(⊸ProjectKQV|⧻) # project KQV and yield hdim seqlen
  ⊙(RotaryEmbeddingSinCos:)
  ⊙:⊃(↙2|°□⊣) # kq sincos v ? kqv sincos
  ApplyRotaryKQ
  ∩₃°□°⊟                 # k q v ? kq v
  ReshapeGroupKV         # repeats k and v to match q
  ⍜∩₃⤸₁≡₂ComputeAttnHead # for each head, compute the attention operation
  ≡₂♭                    # flatten heads into single dimension
  TMM:                   # apply the final linear projection
)

DecoderBlock ← (
  FetchWeights
  +⟜(GQAttention RMSNorm)
  +⟜(MLP RMSNorm)
)

OutputRMS ← (
  ⊙(Uget"model.norm.weight") # fetch the final RMS norm weights
  RMSNorm
)

UnEmbeddingBlock ← (
  ⊙(Uget"model.embed_tokens.weight") # fetch the unembedding weights
  TMM:
)
