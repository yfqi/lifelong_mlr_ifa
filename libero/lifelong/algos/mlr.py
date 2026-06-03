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
import itertools


import os, glob, random, shutil, hashlib, re
from pathlib import Path
from typing import List, Sequence, Dict, Tuple

from tqdm import tqdm
import math

import glob
from torch.utils.data import Dataset, DataLoader, DistributedSampler

BUFFER_CAPACITY = 59660000  #A buffer big enough

TASK_PREFIX_RE = re.compile(r'^t(\d+)__')


class FileTrajectoryDataset(Dataset):
    def __init__(self, file_paths, map_location="cpu"):
        self.file_paths = list(file_paths)
        if not self.file_paths:
            raise RuntimeError("No .pt files found.")
        self.map_location = map_location

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def detach_to_cpu(x):
        if torch.is_tensor(x):
            return x.detach().cpu()
        elif isinstance(x, dict):
            return {k: FileTrajectoryDataset.detach_to_cpu(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [FileTrajectoryDataset.detach_to_cpu(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(FileTrajectoryDataset.detach_to_cpu(v) for v in x)
        else:
            return x

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        middle = parse_path(path)

        sample = torch.load(path, map_location=self.map_location)

        if "actions" not in sample or "feature" not in sample:
            raise KeyError(f"{path} missing 'actions' or 'feature'")

        sample = self.detach_to_cpu(sample)

        return sample, middle

# =========================
# Some tools
# =========================

def parse_path(path):
    filename = os.path.basename(path)
    
    if filename.startswith("t"):
        match = re.match(r"t(\d+)", filename)
        if match:
            return match.group(1)
    else:
        return os.path.split(os.path.dirname(path))[-1]  # "0"

def _list_pt(dir_path: str, patterns: Sequence[str]=("*.pt",)) -> List[str]:
    if not os.path.isdir(dir_path): return []
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(dir_path, pat)))
    paths.sort()
    return paths

def _sample(paths: List[str], k: int, rng: random.Random) -> List[str]:
    if len(paths) <= k: return list(paths)
    return rng.sample(paths, k)

def _is_rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)

def _unique_dest(dst_dir: str, task_id: int, src_path: str) -> str:

    base = os.path.basename(src_path)
    stem, ext = os.path.splitext(base)
    pref = f"t{task_id}__{stem}"
    cand = os.path.join(dst_dir, pref + ext)
    if not os.path.exists(cand): return cand
    h = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:8]
    cand = os.path.join(dst_dir, f"{pref}__{h}{ext}")
    if not os.path.exists(cand): return cand
    i = 1
    while True:
        cand2 = os.path.join(dst_dir, f"{pref}__{h}__{i}{ext}")
        if not os.path.exists(cand2): return cand2
        i += 1

def _safe_copy(src: str, dst_dir: str, task_id: int) -> str:
    dst = _unique_dest(dst_dir, task_id, src)
    shutil.copy2(src, dst)
    return dst

def _group_buffer_by_task(buffer_dir: str) -> Dict[int, List[str]]:
    groups: Dict[int, List[str]] = {}
    for p in _list_pt(buffer_dir):
        m = TASK_PREFIX_RE.match(os.path.basename(p))
        tid = int(m.group(1)) if m else -1
        groups.setdefault(tid, []).append(p)
    return groups


def build_dataset_for_task(base_dir: str, task_id: int, buffer_dir: str, seed: int = 42, start=6) -> FileTrajectoryDataset:
    rng = random.Random(seed)
    task_dir = os.path.join(base_dir, str(task_id))
    task_paths = _list_pt(task_dir)
    if not task_paths:
        raise FileNotFoundError(f"No .pt files in {task_dir}")
    if task_id == start: #############################################$$$$$$$$$$$$
        mix_paths = task_paths
    else:
        buf_paths = _list_pt(buffer_dir)
        mix_paths = (buf_paths if buf_paths else []) + task_paths
    rng.shuffle(mix_paths)
    return FileTrajectoryDataset(mix_paths)

def update_buffer_equal_quota_10( 
    base_dir: str,
    task_id: int,
    buffer_dir: str,
    *,
    capacity: int = BUFFER_CAPACITY,
    seed: int = 42,
    drop_policy: str = "random",  # or "oldest"
    rate=0.5
) -> Tuple[int, int, int, int, Dict[int, int]]:
    """
    Strategy: buffer capacity = `capacity`, and the current task is task `task_id` (0-indexed).

    - Identify the set of previous tasks (`prev_task_ids`) already stored in the buffer
      by parsing the `t{tid}__` prefix. Samples without a valid prefix are assigned
      to a special task group `-1`.
    - Total number of tasks = number of known previous tasks + 1 (the current task).
    - Per-task quota: `per_task_quota = capacity // num_tasks`.
    - If the buffer does not exceed capacity:
        - Append only a 50% subset of samples from the current task.
        - Existing samples are not removed.
    - If the buffer exceeds capacity:
        - First, trim each previous task to its quota.
        - Remove all samples from the unknown group (`-1`).
        - Then add up to `per_task_quota` samples from the 10% subset of the current task.
    - If a task contains fewer samples than its quota, all of its samples are kept.
      No additional samples are added to fill the quota, so the final buffer size
      may be smaller than `capacity`.

    Returns:
        tuple:
            (
                final_total_count,
                num_added_this_round,
                num_removed_this_round,
                per_task_quota,
                final_counts_per_task: dict
            )
    """

    if not _is_rank0():
        groups = _group_buffer_by_task(buffer_dir)
        final_count = sum(len(v) for v in groups.values())
        return (final_count, 0, 0, 0, {k: len(v) for k, v in groups.items()})

    rng = random.Random(seed)
    Path(buffer_dir).mkdir(parents=True, exist_ok=True)

    curr_dir = os.path.join(base_dir, str(task_id))
    curr_all = _list_pt(curr_dir)
    if not curr_all:
        raise FileNotFoundError(f"No .pt files in {curr_dir}")

    ten_pct = max(1, math.ceil(len(curr_all) * rate))
    curr_10pct = _sample(curr_all, ten_pct, rng)


    groups = _group_buffer_by_task(buffer_dir)
    known_prev_ids = sorted([tid for tid in groups.keys() if tid >= 0])
    num_tasks = len(known_prev_ids) + 1
    per_quota = capacity // num_tasks if num_tasks > 0 else capacity

    prev_total = sum(len(v) for v in groups.values())
    if prev_total + len(curr_10pct) <= capacity:
        added = 0
        for src in curr_10pct:
            _safe_copy(src, buffer_dir, task_id=task_id)
            added += 1
        groups = _group_buffer_by_task(buffer_dir)
        final_total = sum(len(v) for v in groups.values())
        per_counts = {k: len(v) for k, v in groups.items()}
        return (final_total, added, 0, per_quota, per_counts)

    dropped = 0
    for tid in list(groups.keys()):
        limit = 0 if tid < 0 else per_quota
        files = groups[tid]
        if len(files) <= limit:
            continue
        if drop_policy == "oldest":
            files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))  # 最旧先删
        else:
            files_sorted = list(files)
            rng.shuffle(files_sorted)
        to_drop = files_sorted[limit:]
        for p in to_drop:
            try:
                Path(p).unlink()
                dropped += 1
            except FileNotFoundError:
                pass
        groups[tid] = files_sorted[:limit]

    add_limit = min(per_quota, len(curr_10pct))
    picked = _sample(curr_10pct, add_limit, rng)
    added = 0
    for src in picked:
        _safe_copy(src, buffer_dir, task_id=task_id)
        added += 1

    groups = _group_buffer_by_task(buffer_dir)
    final_total = sum(len(v) for v in groups.values())
    per_counts = {k: len(v) for k, v in groups.items()}
    return (final_total, added, dropped, per_quota, per_counts)

########################################################################################################

def _load_feature_from_pt(pt_path, language_idx=0, agent_idx=1):
    """
    Load one .pt feature file.

    Expected actual feature shape:
        [1, 8, 4, 512]

    Meaning:
        1   = batch
        8   = timestep/window
        4   = modality
        512 = feature dim

    Return:
        lang_feat:  [1, 512]
        agent_feat: [8, 512]
    """
    data = torch.load(pt_path, map_location="cpu")

    if isinstance(data, dict):
        feat = data["feature"]
    else:
        feat = data

    if not torch.is_tensor(feat):
        raise TypeError(f"feature in {pt_path} is not a tensor, got {type(feat)}")

    feat = feat.detach().float().cpu()

    # [1, 8, 4, 512] -> [8, 4, 512]
    if feat.dim() == 4 and feat.shape[0] == 1:
        feat = feat.squeeze(0)

    if feat.dim() != 3:
        raise ValueError(f"Unexpected feature shape in {pt_path}: {feat.shape}")

    # Case 1: real layout [T, M, E], e.g. [8, 4, 512]
    if feat.shape[1] == 4:
        lang_feat = feat[0, language_idx, :].unsqueeze(0)  # [1, 512]
        agent_feat = feat[:, agent_idx, :]                 # [T, 512]

    # Case 2: fallback layout [M, T, E], e.g. [4, 8, 512]
    elif feat.shape[0] == 4:
        lang_feat = feat[language_idx, 0, :].unsqueeze(0)   # [1, 512]
        agent_feat = feat[agent_idx, :, :]                  # [T, 512]

    else:
        raise ValueError(
            f"Cannot infer feature layout for {pt_path}, shape={feat.shape}. "
            "Expected [8,4,512] or [4,8,512]."
        )

    return lang_feat, agent_feat


def _concat_task_features(feats):
    """
    feats: list of tensors, each tensor can be [1, 512] or [T, 512]

    Return:
        [N_total, 512]
    """
    if len(feats) == 0:
        return None

    clean_feats = []

    for x in feats:
        if x is None:
            continue

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(f"Expected feature shape [N, D], got {x.shape}")

        clean_feats.append(x)

    if len(clean_feats) == 0:
        return None

    return torch.cat(clean_feats, dim=0)


def _pairwise_cosine_mean(feats_i, feats_j, eps=1e-8):
    """
    Strict pairwise cosine mean:

        Sim(T_i, T_j)
        = 1 / (Ni * Nj) * sum_i sum_j cos(h_i, h_j)

    Args:
        feats_i: [Ni, D]
        feats_j: [Nj, D]

    Return:
        scalar float
    """
    if feats_i is None or feats_j is None:
        return float("nan")

    if feats_i.numel() == 0 or feats_j.numel() == 0:
        return float("nan")

    feats_i = feats_i.float()
    feats_j = feats_j.float()

    if feats_i.dim() == 1:
        feats_i = feats_i.unsqueeze(0)

    if feats_j.dim() == 1:
        feats_j = feats_j.unsqueeze(0)

    feats_i = F.normalize(feats_i, dim=-1, eps=eps)
    feats_j = F.normalize(feats_j, dim=-1, eps=eps)

    sim_matrix = feats_i @ feats_j.T

    return sim_matrix.mean().item()


def collect_current_task_features(
    task_id,
    current_feat_root="/work/fyu/cl/robot/feature",
    language_idx=0,
    agent_idx=1,
):
    """
    Current task features are stored in:
        current_feat_root/{task_id}/*.pt

    Return:
        lang_feats:  list of [1, 512]
        agent_feats: list of [T, 512]
    """
    task_dir = os.path.join(current_feat_root, str(task_id))
    pt_files = sorted(glob.glob(os.path.join(task_dir, "*.pt")))

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"No .pt files found for current task {task_id}: {task_dir}"
        )

    lang_feats = []
    agent_feats = []

    for pt in pt_files:
        lang, agent = _load_feature_from_pt(
            pt,
            language_idx=language_idx,
            agent_idx=agent_idx,
        )

        lang_feats.append(lang)
        agent_feats.append(agent)

    return lang_feats, agent_feats


def collect_replay_task_features(
    replay_dir="/work",
    max_old_task_id=None,
    language_idx=0,
    agent_idx=1,
):
    """
    Replay files are named like:
        t6__949.pt

    where 6 means task 6.

    Return:
        replay_features = {
            6: {
                "language": [tensor, tensor, ...],
                "agent": [tensor, tensor, ...],
            },
            7: ...
        }
    """
    pt_files = sorted(glob.glob(os.path.join(replay_dir, "*.pt")))

    replay_features = {}

    pattern = re.compile(r"t(\d+)__.*\.pt$")

    for pt in pt_files:
        name = os.path.basename(pt)
        m = pattern.match(name)

        if m is None:
            continue

        old_task_id = int(m.group(1))

        if max_old_task_id is not None and old_task_id >= max_old_task_id:
            continue

        lang, agent = _load_feature_from_pt(
            pt,
            language_idx=language_idx,
            agent_idx=agent_idx,
        )

        if old_task_id not in replay_features:
            replay_features[old_task_id] = {
                "language": [],
                "agent": [],
            }

        replay_features[old_task_id]["language"].append(lang)
        replay_features[old_task_id]["agent"].append(agent)

    return replay_features


def build_ifa_similarity_matrix(
    current_task_id,
    current_feat_root="/work",
    replay_dir="/work",
    language_idx=0,
    agent_idx=1,
    top_ratio=0.5,
):
    """
    Build IFA similarity information.

    Selection logic:
        1. Compute similarities for all task pairs, excluding diagonal.
           Example:
               task_ids = [6, 7, 8, 9]
               candidate_pairs = [(6,7), (6,8), (6,9), (7,8), (7,9), (8,9)]

        2. Select top ceil(50%) pairs by language similarity.

        3. Select top ceil(50%) pairs by agent-view similarity.

        4. Take intersection.

        5. Only keep pairs that contain current_task_id.

    Return:
        ifa_info = {
            "task_ids": task_ids,
            "sim_language": sim_language,
            "sim_agent": sim_agent,
            "selected_pair_mask": selected_pair_mask,
            "selected_pairs": selected_pairs,
            "candidate_pairs": candidate_pairs,
            "lang_top_pairs": lang_top_pairs,
            "agent_top_pairs": agent_top_pairs,
        }
    """

    current_task_id = int(current_task_id)

    # ============================================================
    # 1. collect old replay features
    # ============================================================
    replay_features = collect_replay_task_features(
        replay_dir=replay_dir,
        max_old_task_id=current_task_id,
        language_idx=language_idx,
        agent_idx=agent_idx,
    )

    # ============================================================
    # 2. collect current task features
    # ============================================================
    cur_lang_feats, cur_agent_feats = collect_current_task_features(
        task_id=current_task_id,
        current_feat_root=current_feat_root,
        language_idx=language_idx,
        agent_idx=agent_idx,
    )

    # ============================================================
    # 3. merge old + current features
    # ============================================================
    all_features = {}

    for tid, feats in replay_features.items():
        all_features[int(tid)] = feats

    all_features[current_task_id] = {
        "language": cur_lang_feats,
        "agent": cur_agent_feats,
    }

    task_ids = sorted(all_features.keys())
    num_tasks = len(task_ids)

    if current_task_id not in task_ids:
        raise RuntimeError(
            f"Current task {current_task_id} is missing from task_ids."
        )

    # ============================================================
    # 4. concatenate features per task
    # ============================================================
    task_features = {}

    for tid in task_ids:
        task_features[tid] = {
            "language": _concat_task_features(all_features[tid]["language"]),
            "agent": _concat_task_features(all_features[tid]["agent"]),
        }

    # ============================================================
    # 5. build pairwise similarity matrices
    # ============================================================
    sim_language = torch.zeros(num_tasks, num_tasks)
    sim_agent = torch.zeros(num_tasks, num_tasks)

    for i, ti in enumerate(task_ids):
        for j, tj in enumerate(task_ids):
            sim_language[i, j] = _pairwise_cosine_mean(
                task_features[ti]["language"],
                task_features[tj]["language"],
            )

            sim_agent[i, j] = _pairwise_cosine_mean(
                task_features[ti]["agent"],
                task_features[tj]["agent"],
            )

    # ============================================================
    # 6. construct all non-diagonal candidate pairs
    # ============================================================
    candidate_pairs = []
    lang_scores = []
    agent_scores = []

    for ti, tj in itertools.combinations(task_ids, 2):
        i = task_ids.index(ti)
        j = task_ids.index(tj)

        candidate_pairs.append((ti, tj))
        lang_scores.append(sim_language[i, j].item())
        agent_scores.append(sim_agent[i, j].item())

    # ============================================================
    # 7. select top ceil(50%) by language and agent-view
    # ============================================================
    selected_pairs = []
    selected_pair_mask = torch.zeros(num_tasks, num_tasks, dtype=torch.bool)

    lang_top_pairs = []
    agent_top_pairs = []

    if len(candidate_pairs) > 0:
        lang_scores_t = torch.tensor(lang_scores)
        agent_scores_t = torch.tensor(agent_scores)

        k = max(1, math.ceil(len(candidate_pairs) * top_ratio))

        lang_top_idx = torch.topk(
            lang_scores_t,
            k=k,
            largest=True,
        ).indices.tolist()

        agent_top_idx = torch.topk(
            agent_scores_t,
            k=k,
            largest=True,
        ).indices.tolist()

        lang_top_pairs = [
            (candidate_pairs[idx], lang_scores[idx])
            for idx in lang_top_idx
        ]

        agent_top_pairs = [
            (candidate_pairs[idx], agent_scores[idx])
            for idx in agent_top_idx
        ]

        selected_indices = sorted(
            list(set(lang_top_idx).intersection(set(agent_top_idx)))
        )

        # 只保留包含当前任务的 pair
        for idx in selected_indices:
            ti, tj = candidate_pairs[idx]

            if current_task_id not in (ti, tj):
                continue

            selected_pairs.append((ti, tj))

            i = task_ids.index(ti)
            j = task_ids.index(tj)

            selected_pair_mask[i, j] = True
            selected_pair_mask[j, i] = True

    ifa_info = {
        "task_ids": task_ids,
        "sim_language": sim_language,
        "sim_agent": sim_agent,
        "selected_pair_mask": selected_pair_mask,
        "selected_pairs": selected_pairs,

        # debug info
        "candidate_pairs": candidate_pairs,
        "lang_top_pairs": lang_top_pairs,
        "agent_top_pairs": agent_top_pairs,
    }

    return ifa_info

class MLR(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.cfg = cfg

    def learn_all_tasks(self, datasets, benchmark, result_summary,current_task_id):

        ifa_info = build_ifa_similarity_matrix(
        current_task_id=current_task_id,
        current_feat_root=self.cfg.feature_dir,
        replay_dir=self.cfg.buffer_dir,
        language_idx=0,
        agent_idx=1,
        top_ratio=0.5,
        )

        base_dir = self.cfg.feature_dir
        buffer_dir = self.cfg.buffer_dir

        dataset = build_dataset_for_task(base_dir, current_task_id, buffer_dir, self.cfg.seed, start=self.cfg.start_task)
        print("Train set size:", len(dataset))

        # learn on all tasks, only used in multitask learning

        train_sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=train_sampler,
            persistent_workers=False,
        )
        self.start_task(-1,train_dataloader)

        cumulated_counter = 0.0
        successes = []
        losses = []

        # start training
        for epoch in range(0, self.cfg.train.n_epochs):
            train_sampler.set_epoch(epoch)
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch")
            t0 = time.time()
            self.policy.train()
            training_loss = 0.0
            for (idx, data) in enumerate(pbar):
                loss = self.observe(data,current_task_id,epoch,ifa_info=ifa_info)
                training_loss += loss
                avg_loss = training_loss / (idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

                training_loss /= len(train_dataloader)
                
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )

            if epoch % 5 == 0 and epoch > 69:  # evaluate BC loss

                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"mlrifa_{current_task_id}_model_ep{epoch}.pth"
                )
                if not dist.is_initialized() or dist.get_rank() == 0:

                    torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                    losses.append(training_loss)

                # Defaultly, we don't do evaluation during training.
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

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)

        total, added, dropped, quota, counts = update_buffer_equal_quota_10(
            base_dir, current_task_id, buffer_dir,
            capacity=BUFFER_CAPACITY, seed= self.cfg.seed, drop_policy="random",rate=self.cfg.past_data_rate
        )
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter