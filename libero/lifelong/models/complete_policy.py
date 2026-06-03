import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.modules.transformer_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from libero.lifelong.utils import (
    torch_load_model,

)
    
class CompletePolicy(BasePolicy):

    def __init__(self, cfg, shape_meta):

        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.cfg = cfg
        self.training = cfg.training

        self.init = 1

        ##### 1. Vision Encoder (Frozen CLIP ViT) #####
        # We assume rgb_modules.CLIPVisionEncoder wraps a pretrained, frozen CLIP ViT
        # self.image_encoders = nn.ModuleDict()
        self.image_proj = nn.ModuleDict()    # ← 新增这一行

        embed_size = policy_cfg.embed_size  # latent dimension E-------------------------------------------?????????????????????????????????????????????

        self.film_scale = nn.ModuleDict()
        self.film_shift = nn.ModuleDict()

        lora_cfg = LoraConfig(
            r=cfg.policy.rank,           
            lora_alpha=cfg.policy.alpha,  
            target_modules=["q_proj", "v_proj"],
        )
        for name, shape in shape_meta["all_shapes"].items():
            if "rgb" in name:
                break

        # shape is (C, H, W)
        kwargs = policy_cfg.image_encoder.network_kwargs.copy()
        kwargs["input_shape"] = shape
        kwargs["output_size"] = embed_size
        # CLIPVisionEncoder should freeze its weights internally
        self.image_encoders = get_peft_model(CLIPVisionEncoder(**kwargs), lora_cfg)

        self.film_scale = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

        self.film_shift = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

        proj = nn.Linear(768, embed_size) ##########################-------------------------------------------------------?????????????????????????
        self.image_proj = proj
 
        self.rgbkey = {'agentview_rgb':0, 'eye_in_hand_rgb':0}


        ##### 2. Language Encoder (Frozen CLIP Text) #####
        lang_kwargs = policy_cfg.language_encoder.network_kwargs.copy()
        lang_kwargs["output_size"] = embed_size
        # CLIPTextEncoder should freeze its weights internally
        self.language_encoder = get_peft_model(CLIPTextEncoder(**lang_kwargs), lora_cfg)


        ##### 3. State Encoder (MLP → FiLM hidden) #####

        state_dim = 0
        for key in shape_meta["all_shapes"]:
            if "state" in key:
                shape = shape_meta["all_shapes"][key]
                state_dim += shape[0] if isinstance(shape, (list, tuple)) else shape

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, policy_cfg.film_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(policy_cfg.film_hidden_size, embed_size),
        )


        self.state_scale = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

        self.state_shift = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

        self.max_seq_len = policy_cfg.max_seq_len

        for param in self.parameters():
            param.requires_grad = False
        
        self.temporal_transformer = OfficialGPT2Decoder(
            embed_size=embed_size,
            num_layers=6,
            num_heads=8,
            dropout=0.15,
            max_position_embeddings=32,
            train = self.cfg.training
        )

        ac_dim = shape_meta['ac_dim']
        self.policy_head = GMMHead(input_size=embed_size, output_size=ac_dim, 
                                           num_modes=policy_cfg.num_gmm_modes, min_std=1e-4,)

        if self.init == 1:
            self.init = 0
            self.load_state_dict(torch_load_model(cfg.pretrain)[0])

    def forward(self, data):

        # 1. Spatial‐temporal feature encoding
        x1 = self.spatial_encode(data)  # (B, T, num_modalities, E)
        return x1
        x = self.temporal_encode(x1)    # (B, T, E)

        # 2. Policy head only needs the last time step’s embedding
        last_feat = x[:, -1, :]        # (B, E)
        if self.training:

            return self.policy_head(last_feat)

    def spatial_encode(self, data):

        # batch and sequence dims

        B, T = next(iter(data['obs'].values())).shape[:2]
        

        if 'instr_token' in data:
            instr = data['instr_token']  # dict with input_ids & attention_mask
            lang_emb = self.language_encoder(**instr)           # (B, E)
  
        elif 'task_emb' in data and data['task_emb'] is not None:
            instr = data['task_emb']
            mask = data['masks'].squeeze(1)
            lang_emb = self.language_encoder(instr,mask)           # (B, E)

        else:
            raise KeyError("Expected 'instr_token' or 'task_emb' in data")

        E = lang_emb.size(-1)
        lang_tok = lang_emb.unsqueeze(1).unsqueeze(2).expand(-1, T, 1, -1)  # (B, T, 1, E)

        # flatten repeated lang_emb for FiLM inputs
        instr_rep = lang_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, E)
        instr_flat = instr_rep.contiguous().view(B * T, E)   

        vis_toks = []
        # for name, encoder in self.image_encoders.items():
        for name, shape in self.rgbkey.items():
            if "rgb" in name:
                imgs = data['obs'][name]                             # (B, T, C, H, W)
                imgs_flat = imgs.view(B * T, *imgs.shape[2:])
                if self.cfg.aug:
                    dev = next(self.parameters()).device
                    imgs_flat = imgs_flat.to(dev, non_blocking=True)

                    if imgs_flat.ndim == 4 and imgs_flat.shape[1] not in (1, 3) and imgs_flat.shape[-1] in (1, 3):
                        imgs_flat = imgs_flat.permute(0, 3, 1, 2)

                    if imgs_flat.dtype not in (torch.float16, torch.float32):
                        imgs_flat = imgs_flat.float()
                    imgs_flat = imgs_flat.contiguous()

                v_feat = self.image_encoders(imgs_flat, langs=None)             # (B*T, E)
                v_feat = self.image_proj(v_feat)              # (B*T, 512)
                v_feat = v_feat.view(B, T, 1, E)                    # (B, T, 1, E)

                sc = self.film_scale(instr_flat)
                sh = self.film_shift(instr_flat)

                v_mod = (1 + sc).view(B, T, 1, E) * v_feat + sh.view(B, T, 1, E)

                vis_toks.append(v_mod)
        vision_mod = torch.cat(vis_toks, dim=2)               # (B, T, V, E)

        state_list = []

        for key in ['joint_states', 'gripper_states']:
            if key in data['obs']:
                state_list.append(data['obs'][key])  # each (B, T, D_i)
        # concatenate along feature dim
        state = torch.cat(state_list, dim=-1) if len(state_list) > 1 else state_list[0]        
        s_flat = state.view(B * T, -1)                        # (B*T, state_dim)
        s_feat = self.state_encoder(s_flat)                   # (B*T, E)
        s_feat = s_feat.view(B, T, 1, E)                      # (B, T, 1, E)

        ssc = self.state_scale(instr_flat)
        ssh = self.state_shift(instr_flat)

        state_mod = (1 + ssc).view(B, T, 1, E) * s_feat + ssh.view(B, T, 1, E)

        if self.cfg.policy.use_language is False:
            encoded = torch.cat([v_feat, vision_mod, state_mod], dim=2)  # (B, T, M, E)
        elif self.cfg.policy.use_state is False:
            add_vision_mod = (1 + ssc).view(B, T, 1, E) * v_feat + ssh.view(B, T, 1, E)
            encoded = torch.cat([lang_tok, vision_mod, add_vision_mod], dim=2)  # (B, T, M, E)
        else:
            encoded = torch.cat([lang_tok, vision_mod, state_mod], dim=2)  # (B, T, M, E)
        return encoded

    def temporal_encode(self, x):
        B, T, M, E = x.shape
        x = TensorUtils.join_dimensions(x, 1, 2)
        out = self.temporal_transformer(x)  # (B, N, E)
        out = out.view(B, T, M, E)
        return out[:, :, 0, :]     # (B, T, E)

    def get_action(self, data):
        """
        Online inference: maintain a queue of past embeddings, then predict next action.
        Args:
          data: dict containing single‐step observations (unsqueezed along T=1)
        Returns:
          action_np: numpy array (B, ac_dim)
        """
        self.eval()
        with torch.no_grad():
            # Preprocess input (e.g., normalization) if BasePolicy defines it
            data = self.preprocess_input(data, train_mode=False)
            # Spatial encode for this single time step; treat T=1
            x_spatial = self.spatial_encode(data)  # (B, 1, M, E)

            # Temporal encode
            x_temp = self.temporal_encode(x_spatial)    # (B, T_cur, E)  1 1 512
            # Take last time step
            last_emb = x_temp[:, -1, :]             # (B, E)
            # Action distribution
            dist = self.policy_head(last_emb,train=False)
            action = dist.sample().detach().cpu()
            return  action.view(action.shape[0], -1).numpy()

    def reset(self):
        """
        Reset the latent queue at the start of each new episode/trajectory.
        """
        self.latent_queue = []


