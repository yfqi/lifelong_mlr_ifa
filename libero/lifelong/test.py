import os

import multiprocessing as mp
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
os.environ["EGL_PLATFORM"] = "surfaceless"
mp.set_start_method("spawn", force=True)  

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0, "/work")

import json
import multiprocessing
import pprint
import time
from pathlib import Path
from omegaconf import OmegaConf

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
from collections import OrderedDict

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = os.getenv("CUDA_VISIBLE_DEVICES","0").split(",")[0]
os.environ["EGL_PLATFORM"] = "surfaceless"
sys.path.insert(0, "/home/fyu/mlr_ifa")

# os.environ["PYTHONHASHSEED"] = "0"
def set_global_seed(seed:int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from pathlib import Path
def log_succ(S, i, j, pp, logfile="/home/ct_005.txt"):
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    line = f"[task {i} epoch {j:02d} and {pp} succ.] " + " | ".join(f"{x:.2f}" for x in S)

    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")

@hydra.main(config_path="../configs", config_name="test", version_base=None)
def main(hydra_cfg):

    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
  
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

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None
    print(n_manip_tasks)
    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
                demo_limit= cfg.data.demo_limit,
                train = cfg.training,
            )

        except Exception as e:

            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(i)
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    gsz = cfg.data.task_group_size
    if gsz == 1:  # each manipulation task is its own lifelong learning task
        datasets = [
            SequenceVLDataset(ds, emb, mask) for (ds, emb, mask) in zip(manip_datasets, task_embs['input_ids'], task_embs['attention_mask'])
        ]
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:  # group gsz manipulation tasks into a lifelong task, currently not used
        assert (
            n_manip_tasks % gsz == 0
        ), f"[error] task_group_size does not divide n_tasks"
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

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)
        a = [cfg.seed]
        for see in a:
            for pp in range(cfg.start_task,cfg.end_task,1):   # 7 → 0
                set_global_seed(see)
                cfg.seed=see
                for j in range(70,100,5):
                    for i in range(cfg.start_task,pp+1):
                        base_dir = cfg.epoch_dir

                        filename = "mlrifa_{}_model_ep{}.pth".format(pp,j)

                        fullpath = os.path.join(base_dir, filename)

                        while not os.path.exists(fullpath):
                            continue
                        if os.path.exists(fullpath):
                            print(fullpath)
                            algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)

                            checkpoint = torch.load(fullpath, map_location='cpu')

                            new_state_dict = OrderedDict((k[7:], v) for k, v in checkpoint["state_dict"].items())

                            algo.policy.load_state_dict(new_state_dict, strict=False)

                            algo.eval()

                            t2 = time.time()
                            S = evaluate_success(
                                cfg=cfg,
                                algo=algo,
                                benchmark=benchmark,
                                task_ids=[i],
                                result_summary=result_summary if cfg.eval.save_sim_states else None,
                                see = see
                            )

                            log_succ(S, i, j,pp,logfile=cfg.test_log_file)  

        
if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # if multiprocessing.get_start_method(allow_none=True) != "spawn":  
    #     multiprocessing.set_start_method("spawn", force=True)
    main()
