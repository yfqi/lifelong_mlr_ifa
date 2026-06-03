import abc
import os
import glob
import random
import torch

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if self.name == "libero_90":
            # 只保留任务名以 "KITCHEN" 开头的任务
            # tasks = [task for task in tasks if task.name.startswith("KITCHEN")]
            

            KITCHEN_DESCRIPTIONS = [

                # "put the wine bottle on the wine rack",
                # "put the wine bottle on the wine rack",

                # "close the top drawer of the cabinet and put the black bowl on top of it",  
                # "close the top drawer of the cabinet and put the black bowl on top of it",                
              
                # "turn off the stove",
            ]

            KITCHEN_DESCRIPTIONS = [
                "close the top drawer of the cabinet", ###################################################################################
                "close the top drawer of the cabinet and put the black bowl on top of it",
                "put the black bowl in the top drawer of the cabinet",
                "put the butter at the back in the top drawer of the cabinet and close it",
                "put the butter at the front in the top drawer of the cabinet and close it",
                "put the chocolate pudding in the top drawer of the cabinet and close it",
                "open the bottom drawer of the cabinet",
                "open the top drawer of the cabinet",
                "open the top drawer of the cabinet and put the bowl in it",
                "put the black bowl on the plate",
                "put the black bowl on top of the cabinet",
                "open the top drawer of the cabinet",
                "put the black bowl at the back on the plate",
                "put the black bowl at the front on the plate",
                "put the middle black bowl on the plate",
                "put the middle black bowl on top of the cabinet",
                "stack the black bowl at the front on the black bowl in the middle",
                "stack the middle black bowl on the back black bowl",
                "put the frying pan on the stove",###########################################################################################
                "put the moka pot on the stove",############################################################################
                "turn on the stove",################################################################################
                "turn on the stove and put the frying pan on it",
                "close the bottom drawer of the cabinet",
                "close the bottom drawer of the cabinet and open the top drawer",
                "put the black bowl in the bottom drawer of the cabinet",
                "put the black bowl on top of the cabinet",
                "put the wine bottle in the bottom drawer of the cabinet",
                "put the wine bottle on the wine rack",
                "close the top drawer of the cabinet",#########################################################################
                "put the black bowl in the top drawer of the cabinet",
                "put the black bowl on the plate",
                "put the black bowl on top of the cabinet",
                "put the ketchup in the top drawer of the cabinet",
                "close the microwave",
                "put the yellow and white mug to the front of the white mug",
                "open the microwave",
                "put the white bowl on the plate",
                "put the white bowl to the right of the plate",
                "put the right moka pot on the stove",
                "turn off the stove"
            ]

            # lowercase for robust matching
            target_descs = [desc.lower() for desc in KITCHEN_DESCRIPTIONS]
            matched_tasks = []

            used_tasks = set()

            # for target in target_descs:
            #     for task in tasks:
            #         if target == task.language.lower():
            #             matched_tasks.append(task)
            #             used_tasks.add(task)


            for target in target_descs:
                found = False
                for task in tasks:
                    if task in used_tasks:
                        continue
                    if target == task.language.lower():
                        matched_tasks.append(task)
                        used_tasks.add(task)
                        found = True
                        break
                    if not found:
                        raise ValueError(f"❌ 未找到与 '{target}' 完全匹配且未被使用的任务，请检查任务集语言字段")

            self.n_tasks = len(matched_tasks)
            self.tasks = matched_tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)


    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        instr = self.task_embs['input_ids'][i]
        mask = self.task_embs['attention_mask'][i]
        return instr, mask

    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()
