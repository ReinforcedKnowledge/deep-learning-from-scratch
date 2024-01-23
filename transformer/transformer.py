import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scaled_dot_product_attention(Q, K, V, mask=None):
  scaled_dot_product = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
  if mask is not None:
    scaled_dot_product = scaled_dot_product + mask
  attention_scores = F.softmax(scaled_dot_product, dim=-1)
  return torch.matmul(attention_scores, V)

class MultiHeadedAttention(nn.Module):
  def __init__(self, d_model=512, h=8):
    super(MultiHeadedAttention, self).__init__()
    self.d_model = d_model
    self.h = h
    self.d_k = d_model // h

    # Using single linear layer for each query, key and value
    self.query_linear = nn.Linear(d_model, d_model)
    self.key_linear = nn.Linear(d_model, d_model)
    self.value_linear = nn.Linear(d_model, d_model)

    self.projection_layer = nn.Linear(h * self.d_k, d_model)

  def forward(self, Q, K, V, mask=None):
    batch_size = Q.size(0)

    # Apply the linear layers and split into h heads
    queries = self.query_linear(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    keys = self.key_linear(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    values = self.value_linear(V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

    # Apply scaled dot product attention
    x = scaled_dot_product_attention(queries, keys, values, mask)

    # Concatenate heads and put through final linear layer
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    x = self.projection_layer(x)

    return x
  
class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model=512, d_ff=2048):
    super(PositionWiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    return self.linear2(x)
  
class Encoder(nn.Module):
  def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
    super(Encoder, self).__init__()
    self.multi_headed_attention = MultiHeadedAttention(d_model, h)
    self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x, mask=None):
    # MHA
    mha_output = self.dropout(self.multi_headed_attention(x, x, x, mask))
    # Add + Norm
    x = self.layernorm1(x + mha_output)
    # FF
    ff_output = self.dropout(self.position_wise_feed_forward(x))
    # Add + Norm
    x = self.layernorm2(x + ff_output)
    return x
  
class Decoder(nn.Module):
  def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
    super(Decoder, self).__init__()
    self.masked_multi_headed_attention = MultiHeadedAttention(d_model, h)
    self.multi_headed_attention = MultiHeadedAttention(d_model, h)
    self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.layernorm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x, decoder_mask, encoder_stack_output, memory_mask):
    # MMHA
    mmha_output = self.dropout(self.masked_multi_headed_attention(x, x, x, decoder_mask))
    # Add + Norm
    x = self.layernorm1(x + mmha_output)
    # MHA
    mha_output = self.dropout(self.multi_headed_attention(x, encoder_stack_output, encoder_stack_output, memory_mask))
    # Add + Norm
    x = self.layernorm1(x + mha_output)
    # FF
    ff_output = self.dropout(self.position_wise_feed_forward(x))
    # Add + Norm
    x = self.layernorm2(x + ff_output)
    return x
  
def compute_positional_encoding(max_input_tokens, d_model):
  positional_encoding = torch.zeros((max_input_tokens, d_model))
  positions = torch.arange(max_input_tokens)
  indices = torch.arange(d_model // 2)
  
  positional_encoding[:, ::2] = torch.sin(positions[:, None] / (10000 ** (2 * indices / d_model)))
  positional_encoding[:, 1::2] = torch.cos(positions[:, None] / (10000 ** (2 * indices / d_model)))
  return positional_encoding

class EmbeddingsComponent(nn.Module):
  def __init__(self, d_model, vocab, pe, dropout=0.1):
    super(EmbeddingsComponent, self).__init__()
    self.d_model = d_model
    self.positional_encoding = pe
    self.embed_layer = nn.Embedding(vocab, d_model)
    self.dropout = nn.Dropout(dropout)

  @property
  def embeddings_matrix(self):
    return self.embed_layer.weight

  def forward(self, x):
    x = self.embed_layer(x) * math.sqrt(self.d_model)
    x += self.positional_encoding[:x.size(1), :]
    return self.dropout(x)
  
# class Transformer(nn.Module):
#   def __init__(self, input_embedder, target_embedder, N, d_model, target_vocab):
#     super(Transformer, self).__init__()
#     self.N = N

#     self.input_embedder = input_embedder
#     self.target_embedder = target_embedder

#     self.encoders = nn.ModuleList([Encoder() for i in range(N)])
#     self.decoders = nn.ModuleList(
#         [Decoder() for i in range(N)]
#     )

#     self.target_projection = nn.Linear(d_model, target_vocab)
#     self.target_projection.weight = self.target_embedder.embed_layer.weight

#   def encode(self, enc_input, enc_mask):
#     embedded_input = self.input_embedder(enc_input)
#     encoder_output = self.encoders[0](embedded_input, enc_mask)
#     for i in range(1, self.N):
#       encoder_output = self.encoders[i](encoder_output, enc_mask)
#     return encoder_output

#   def decode(self, dec_input, dec_mask, enc_output, mem_mask):
#     embedded_target = self.target_embedder(dec_input)
#     decoder_output = self.decoders[0](embedded_target, dec_mask, enc_output, mem_mask)
#     for i in range(1, self.N):
#       decoder_output = self.decoders[i](decoder_output, dec_mask, enc_output, mem_mask)
#     return decoder_output

#   def forward(self, x, input_mask, y, target_mask, memory_mask):
#     x = self.input_embedder(x)
#     y = self.target_embedder(y)

#     x = self.encoders[0](x, input_mask)
#     for i in range(1, self.N):
#       x = self.encoders[i](x, input_mask)

#     y = self.decoders[0](y, target_mask, x, memory_mask)
#     for i in range(1, self.N):
#       y = self.decoders[i](y, target_mask, x, memory_mask)

#     y = self.target_projection(y)
#     y = F.log_softmax(y, dim=-1)
#     return y
  
class Transformer(nn.Module):
  def __init__(self, vocab_size, N, d_model, h, d_ff, max_input_tokens=4096):
    super(Transformer, self).__init__()
    self.N = N

    pe = compute_positional_encoding(max_input_tokens, d_model)
    self.embedder = EmbeddingsComponent(d_model, vocab_size, pe)

    self.encoders = nn.ModuleList([Encoder(d_model, h, d_ff) for i in range(N)])
    self.decoders = nn.ModuleList(
        [Decoder(d_model, h, d_ff) for i in range(N)]
    )

    self.target_projection = nn.Linear(d_model, vocab_size, bias=False)
    self.target_projection.weight = self.embedder.embeddings_matrix

  def encode(self, enc_input, enc_mask):
    embedded_input = self.embedder(enc_input)
    encoder_output = self.encoders[0](embedded_input, enc_mask)
    for i in range(1, self.N):
      encoder_output = self.encoders[i](encoder_output, enc_mask)
    return encoder_output

  def decode(self, dec_input, dec_mask, enc_output, mem_mask):
    embedded_target = self.embedder(dec_input)
    decoder_output = self.decoders[0](embedded_target, dec_mask, enc_output, mem_mask)
    for i in range(1, self.N):
      decoder_output = self.decoders[i](decoder_output, dec_mask, enc_output, mem_mask)
    return decoder_output

  def forward(self, x, input_mask, y, target_mask, memory_mask):
    x = self.embedder(x)
    y = self.embedder(y)

    x = self.encoders[0](x, input_mask)
    for i in range(1, self.N):
      x = self.encoders[i](x, input_mask)

    y = self.decoders[0](y, target_mask, x, memory_mask)
    for i in range(1, self.N):
      y = self.decoders[i](y, target_mask, x, memory_mask)

    y = self.target_projection(y)
    return y