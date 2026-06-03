import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0, "/work/main")

import json
import multiprocessing
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@hydra.main(config_path="../configs", config_name="config_pretrain", version_base=None)
def main(hydra_cfg):

    dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    print(cfg.training)

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")

    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks
    n_manip_tasks = 6

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    # for i in range(n_manip_tasks):
    for i in range(n_manip_tasks):

        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
                demo_limit= 50,
                train = cfg.training,

            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(cfg, descriptions)
    # benchmark.set_task_embs(task_embs)

    gsz = 6
    datasets = []
    n_demos = []
    n_sequences = []
    for i in range(0, n_manip_tasks, gsz):
        dataset = GroupedTaskDataset(
            manip_datasets[i : i + gsz], task_embs['input_ids'][i : i + gsz], task_embs['attention_mask'][i : i + gsz],
        )
        datasets.append(dataset)
        n_demos.extend([x.n_demos for x in dataset.sequence_datasets])
        n_sequences.extend(
            [x.total_num_sequences for x in dataset.sequence_datasets]
        )

    n_tasks = n_manip_tasks // gsz  # number of lifelong learning tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        for j in range(gsz):
            print(f"        {benchmark.get_task(i*gsz+j).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    if cfg.use_wandb:
        wandb.init(project="libero", config=cfg)
        wandb.run.name = cfg.experiment_name

    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # loss confusion matrix
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # success confusion matrix
        "L_fwd": np.zeros((n_manip_tasks,)),  # loss AUC, how fast the agent learns
        "S_fwd": np.zeros((n_manip_tasks,)),  # success AUC, how fast the agent succeeds
    }

    if cfg.eval.save_sim_states:
        # for saving the evaluate simulation states, so we can replay them later
        for k in range(n_manip_tasks):
            for p in range(k + 1):  # for testing task p when the agent learns to task k
                result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
            for e in range(
                cfg.train.n_epochs + 1
            ):  # for testing task k at the e-th epoch when the agent learns on task k
                if e % cfg.eval.eval_every == 0:
                    result_summary[f"k{k}_e{e//cfg.eval.eval_every}"] = [
                        [] for _ in range(cfg.eval.n_eval)
                    ]

    # define lifelong algorithm
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)

    policy = algo.policy.to(local_rank)

    policy = DDP(policy, device_ids=[local_rank], output_device=local_rank)

    algo.policy = policy

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)
    if cfg.lifelong.algo == "multitask" or cfg.lifelong.algo == "Multitask":

        algo.train()
        s_fwd, l_fwd = algo.learn_all_tasks(datasets, benchmark, result_summary)
        result_summary["L_fwd"][-1] = l_fwd
        result_summary["S_fwd"][-1] = s_fwd

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # if multiprocessing.get_start_method(allow_none=True) != "spawn":  
    #     multiprocessing.set_start_method("spawn", force=True)
    main()
