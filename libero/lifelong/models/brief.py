import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *

from libero.lifelong.utils import (
    torch_load_model,

)

    
class BriefPolicy(BasePolicy):

    def __init__(self, cfg, shape_meta):

        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.training = cfg.training

        self.init = 1

        for param in self.parameters():
            param.requires_grad = False
        
        self.temporal_transformer = OfficialGPT2Decoder(
            embed_size=512,
            num_layers=6,
            num_heads=8,
            dropout=0.15,
            max_position_embeddings=32,
            train = True

        )

        self.policy_head = GMMHead(input_size=512, output_size=7, 
                                           num_modes=policy_cfg.num_gmm_modes, min_std=1e-4,)

        if self.init == 1:
            self.init = 0
            self.load_state_dict(torch_load_model(cfg.pretrain)[0],strict=False)


    def forward(self, data):

        x = self.temporal_encode(data)    # (B, T, E)
        last_feat = x[:, -1, :]        # (B, E)
        if self.training:
            return self.policy_head(last_feat), last_feat


    def temporal_encode(self, x):
        B, T, M, E = x.shape

        x = TensorUtils.join_dimensions(x, 1, 2)
        N = T * M

        out = self.temporal_transformer(x)  # (B, N, E)

        out = out.view(B, T, M, E)
        return out[:, :, 0, :]     # (B, T, E)


    def reset(self):
        """
        Reset the latent queue at the start of each new episode/trajectory.
        """
        self.latent_queue = []