import torch

def create_lookahead_mask(sentences, Nheads=8):
  batch_size, sequence_length = sentences.shape
  mask = torch.ones(sequence_length, sequence_length).triu(diagonal=1)
  mask = mask.masked_fill(mask == 1, -float('inf'))
  mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, Nheads, -1, -1)
  return mask

def create_padding_mask(src_sentences, tgt_sentences, pad=0, Nheads=8):
    _, src_seq_length = src_sentences.shape
    tgt_batch_size, tgt_seq_length = tgt_sentences.shape

    memory_mask = torch.zeros(tgt_batch_size, Nheads, tgt_seq_length, src_seq_length)

    # Create masks for positions where src_sentences and tgt_sentences are equal to pad
    src_pad_mask = src_sentences == pad

    # Expand the src_pad_mask to match the size (batch_size, 1, 1, src_seq_length)
    src_pad_mask_expanded = src_pad_mask.unsqueeze(1).unsqueeze(2)
    src_pad_mask_expanded = src_pad_mask_expanded.expand(tgt_batch_size, Nheads, tgt_seq_length, src_seq_length)

    # Apply the mask to the memory mask
    memory_mask.masked_fill_(src_pad_mask_expanded, -float('inf'))

    return memory_mask

def create_src_masks(src_sentences, pad, Nheads=8):
  return create_padding_mask(src_sentences, src_sentences, pad, Nheads)

def create_tgt_masks(tgt_sentences, pad, Nheads=8):
  lookahead_mask = create_lookahead_mask(tgt_sentences, Nheads)
  padding_mask = create_padding_mask(tgt_sentences, tgt_sentences, pad, Nheads)
  return lookahead_mask + padding_mask

def create_memory_masks(src_sentences, tgt_sentences, pad, Nheads=8):
  return create_padding_mask(src_sentences, tgt_sentences, pad, Nheads)