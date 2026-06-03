import numpy as np
import torch
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.lifelong.algos.base import Sequential
from libero.lifelong.utils import *
from torch.nn.parallel import DistributedDataParallel as DDP


class Basic(Sequential):
    """
    The Elastic Weight Consolidation policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.checkpoint = None
        self.fish = None

    def get_params(self):
        return torch.cat([p.reshape(-1) for p in self.policy.parameters()])

    def get_grads(self):
        return torch.cat(
            [
                p.grad.reshape(-1)
                if p.grad is not None
                else torch.zeros_like(p).reshape(-1)
                for p in self.policy.parameters()
            ]
        )

    def observe(self, data, taskid, idx, save_feature=False,saving_dir=None):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        if save_feature is True:
            if isinstance(self.policy, DDP):
                self.policy.module.compute_loss(data, taskid, idx, save_feature=save_feature,saving_dir=saving_dir)
            else:
                self.policy.compute_loss(data, taskid, idx, save_feature=save_feature,saving_dir=saving_dir)
            return        

