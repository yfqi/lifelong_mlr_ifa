from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.models.bc_vilt_policy import BCViLTPolicy
from libero.lifelong.models.complete_policy import CompletePolicy
from libero.lifelong.models.brief import BriefPolicy

from libero.lifelong.models.base_policy import get_policy_class, get_policy_list
POLICY_REGISTRY = {
    "bc_rnn_policy": BCRNNPolicy,
    "bc_transformer_policy": BCTransformerPolicy,
    "bc_vilt_policy": BCViLTPolicy,
    "BriefPolicy": BriefPolicy,
    "CompletePolicy": CompletePolicy,
}

def get_policy_class(name):
    if name == 'BCTransformerPolicy':
        name = 'bc_transformer_policy'
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Policy class with name {name} not found in registry")
    return POLICY_REGISTRY[name]

def get_policy_list():
    return list(POLICY_REGISTRY.keys())