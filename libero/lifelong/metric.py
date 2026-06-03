import copy
import gc
import numpy as np
import os

import multiprocessing as mp
os.environ["MUJOCO_GL"] = "egl"
# 处理 CUDA_VISIBLE_DEVICES 重映射：对进程而言可见列表的第0张就是 "0"
os.environ["MUJOCO_EGL_DEVICE_ID"] = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
# 一些服务器上需要无 surface EGL
os.environ["EGL_PLATFORM"] = "surfaceless"
# 关键：避免 fork 带来的 GL 状态问题
mp.set_start_method("spawn", force=True)   # <- 关键：在任何 torch/robosuite 导入之前import torch.nn.functional as F


import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch

from torch.utils.data import DataLoader

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.utils import *
import imageio, h5py

# torch.backends.cudnn.enabled = False         # <- 让这行生效
# torch.backends.cudnn.benchmark = False
# torch.set_grad_enabled(False)                # 评测无需梯度更稳



def raw_obs_to_tensor_obs(obs, intr, mask, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "obs": {},
        "task_emb": intr.repeat(env_num, 1),
        "masks": mask.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    
    return data

def evaluate_one_task_success(
    cfg, algo, task, instr, mask, task_id, sim_states=None, task_str="", render = True,see=None
):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    # render = cfg.eval.render
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id)

        algo.eval()
        env_num =  10
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        # Try to handle the frame buffer issue

        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        ####################################################################################################################
        num_success = 0
        for i in range(eval_loop_num):
            seed = see + i
            env.seed(seed)
            env.reset()
            env.seed(seed) 
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            algo.reset()
            obs = env.set_init_state(init_states_)
            history_buffer = []
           
            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            frames = []
            with torch.no_grad():

                while steps < cfg.eval.max_steps:
                    steps += 1

                    data = raw_obs_to_tensor_obs(obs, instr, mask, cfg)

                    history_buffer.append(data)
                    if len(history_buffer) > 8:
                        history_buffer.pop(0)

                    sliding_window = history_buffer.copy()


                    def stack_trajectory(data_buffer):
                        assert len(data_buffer) > 0
                        result = {
                            "obs": {},
                            "task_emb": data_buffer[0]["task_emb"],  # 不变
                            "masks": data_buffer[0]["masks"],  # 不变
                        }
                        obs_keys = data_buffer[0]["obs"].keys()
                        for key in obs_keys:
                            result["obs"][key] = torch.stack(
                                [step["obs"][key] for step in data_buffer],
                                dim=1  # 这里就是 B,T,C,H,W
                            )
                        return result

                    batch_data = stack_trajectory(sliding_window)

                    actions = algo.policy.get_action(batch_data)

                    obs, reward, done, info = env.step(actions)

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]

                    if all(dones):
                        break

                # a new form of success record
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval:
                        num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate


def evaluate_success(cfg, algo, benchmark, task_ids, result_summary=None,see=None):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        instr, mask = benchmark.get_task_emb(i)
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(
            cfg, algo, task_i, instr, mask , i, sim_states=curr_summary, task_str=task_str,see = see
        )
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


@torch.no_grad()
def evaluate_loss(cfg, algo, benchmark, datasets):
    """
    Evaluate the loss on all datasets.
    """
    algo.eval()
    losses = []
    for i, dataset in enumerate(datasets):
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id=i)

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            shuffle=False,
        )
        test_loss = 0
        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=cfg.device)
            )
            loss = algo.policy.compute_loss(data)
            test_loss += loss.item()
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
