"""Shared experiment definitions and defaults."""

from dataclasses import asdict, dataclass


MODEL_NAME = "sentence-transformers/LaBSE"
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
CLASS_NAMES = ["negative", "neutral", "positive"]
NUM_CLASSES = len(CLASS_NAMES)

# The final channel in ZuCo's retained order is Cz, the flat reference channel.
ZUCO_REFERENCE_CHANNEL_INDEX = 104
ZUCO_REFERENCE_CHANNEL_LABEL = "Cz"

DEFAULT_SETUPS = [
    "text_frozen",
    "text_finetune",
    "eeg_only",
    "concat_finetune",
    "gated_finetune",
    "gated_shuffled_finetune",
    "gated_noise_finetune",
]

VALID_SETUPS = [
    "text_frozen",
    "text_finetune",
    "eeg_only",
    "concat_frozen",
    "concat_finetune",
    "gated_frozen",
    "gated_finetune",
    "gated_shuffled_frozen",
    "gated_shuffled_finetune",
    "gated_noise_frozen",
    "gated_noise_finetune",
    "gated_zero_frozen",
    "gated_zero_finetune",
]


@dataclass(frozen=True)
class Setup:
    name: str
    fusion: str
    text_mode: str
    eeg_control: str = "aligned"

    @property
    def uses_text(self):
        return self.fusion != "eeg"

    @property
    def uses_eeg(self):
        return self.fusion != "text"

    @property
    def zero_eeg_contribution(self):
        return self.fusion == "gated" and self.eeg_control == "zero"


def parse_setup(name):
    """Turn a public setup name into explicit model choices."""
    if name not in VALID_SETUPS:
        raise ValueError(f"unknown setup {name!r}; choose from {VALID_SETUPS}")
    if name == "eeg_only":
        return Setup(name=name, fusion="eeg", text_mode="none")

    text_mode = "finetune" if name.endswith("finetune") else "frozen"
    if name.startswith("text_"):
        return Setup(name=name, fusion="text", text_mode=text_mode)
    fusion = "concat" if name.startswith("concat_") else "gated"
    if "_shuffled_" in name:
        control = "shuffled"
    elif "_noise_" in name:
        control = "noise"
    elif "_zero_" in name:
        control = "zero"
    else:
        control = "aligned"
    return Setup(name=name, fusion=fusion, text_mode=text_mode, eeg_control=control)


@dataclass
class TrainConfig:
    model_name: str = MODEL_NAME
    max_length: int = 64
    n_folds: int = 5
    val_size: float = 0.15
    epochs: int = 12
    patience: int = 4
    batch_size: int = 16
    encoder_lr: float = 2e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    dropout: float = 0.3
    text_dim: int = 128
    channel_dim: int = 32
    eeg_dim: int = 64
    num_workers: int = 2

    def to_dict(self):
        return asdict(self)
