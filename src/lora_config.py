# src/lora_config.py
from peft import LoraConfig, TaskType

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def make_default_lora_config(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
