import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, tgt_vocab_len, smoothing=0.0, padding_index=None):
        super(LabelSmoothing, self).__init__()
        self.n_classes = tgt_vocab_len
        self.smoothing = smoothing
        self.padding_index = padding_index

    def forward(self, logits, target):
        bsz, tgt_seqlen, _ = logits.size()

        # Create smoothed target distribution
        with torch.no_grad():
            target_smoothed = torch.full_like(logits, fill_value=self.smoothing / (self.n_classes - 1))
            target_expanded = target.unsqueeze(2)
            target_smoothed.scatter_(2, target_expanded, 1.0 - self.smoothing)

        # Create a mask to ignore the padding tokens in the target
        if self.padding_index is not None:
            mask = target != self.padding_index
            num_non_padding_tokens = mask.sum()
            mask = mask.unsqueeze(-1).expand_as(logits)
        else:
            num_non_padding_tokens = bsz * tgt_seqlen
            mask = torch.ones_like(logits)

        # Apply log softmax on logits
        log_probs = F.log_softmax(logits, dim=2)

        # Compute the cross-entropy loss over non-padding tokens only
        loss = -torch.sum(log_probs * target_smoothed * mask) / num_non_padding_tokens

        return loss
    
def lr_func(dmodel, step_num, warmup_steps=4000):
  if step_num == 0:
    step_num = 1
  return (dmodel ** (-0.5)) * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))

def nmt_loss(model_output, ground_truth, labelsmoothing):
   return labelsmoothing(model_output, ground_truth)

