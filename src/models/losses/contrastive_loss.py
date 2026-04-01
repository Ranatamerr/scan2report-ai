import torch
import torch.nn.functional as F

def contrastive_loss(vision_emb, text_emb):
    """
    vision_emb: [B, D]
    text_emb:   [B, D]
    """

    # normalize vectors
    vision_emb = F.normalize(vision_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # similarity matrix
    logits = vision_emb @ text_emb.T  # [B, B]

    # correct pairs = diagonal
    labels = torch.arange(logits.size(0)).to(logits.device)

    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)

    return (loss_v2t + loss_t2v) / 2