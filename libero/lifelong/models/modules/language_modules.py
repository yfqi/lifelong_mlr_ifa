"""
This file contains modules that encode language task embeddings.
"""
import torch.nn as nn
from transformers import CLIPTextModel


class IdentityEncoder(nn.Module):
    """
    Dummy encoder that directly outputs the pretrained task embedding
    """

    def __init__(self, dummy=True):
        super().__init__()

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = data["task_emb"]  # (B, L, H)
        return h


class MLPEncoder(nn.Module):
    """
    Encode task embedding

    h = f(e), where
        e: pretrained task embedding from large model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(data["task_emb"])  # (B, H)
        return h

class CLIPTextEncoder(nn.Module):
    def __init__(self, text_model_name="openai/clip-vit-base-patch16", output_size=64,adaption=False, **kwargs):
        super().__init__()
        self.clip = CLIPTextModel.from_pretrained(text_model_name)
        if not adaption:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        return: (B, output_size)
        """
        text_outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_outputs.pooler_output  # [CLS] token
        return pooled