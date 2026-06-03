import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from tqdm import tqdm

class multitaskpretraining(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.cfg = cfg

    def learn_all_tasks(self, datasets, benchmark, result_summary):
        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_model.pth"
        )
        all_tasks = list(range(benchmark.n_tasks))

        train_sampler = DistributedSampler(
            dataset=concat_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=train_sampler,
            persistent_workers=True,
        )
        self.start_task(-1,train_dataloader)

        prev_success_rate = -1.0

        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        # start training
        for epoch in range(0, self.cfg.n_epochs):

            train_sampler.set_epoch(epoch)  # 保证每个 epoch shuffle 不同
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch")
            t0 = time.time()

            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(pbar):
                loss = self.observe(data)
                training_loss += loss
                avg_loss = training_loss / (idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

            training_loss /= len(train_dataloader)
                
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )

            if epoch % 3 == 0:  # evaluate BC loss

                t0 = time.time()
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"ep{epoch}.pth"
                )
                if not dist.is_initialized() or dist.get_rank() == 0:

                    torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                    losses.append(training_loss)


                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rate = 0.0
                successes.append(success_rate)

                cumulated_counter += 1.0

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter

