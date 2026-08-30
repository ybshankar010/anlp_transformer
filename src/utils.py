import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    name: str = "C1"

    #data config
    train_split_ratio: float = 0.7
    test_split_ratio: float = 0.2
    validation_split_ratio: float = 0.1
    seed: int = 42
    max_src_len: int = 1024
    max_target_len: int = 256

    #tokenizer config
    src_pad_id: int = 0
    target_pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    tokenizer_type: str = "subword"

    #model
    d_model: int = 128
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    ffn_dim: int = 512
    dropout: float = 0.1

    #abalation study
    positional_encoding: str = "sinusoidal"
    attention_type: str = "mha"
    norm_type: str = "layernorm"

    #training
    batch_size: int = 2
    learning_rate: float = 3e-4
    epochs: int = 20
    checkpoint_dir: str = "checkpoints/c1"
    wandb_project: str = "anlp-assignment1"
    hf_repo_id: str = "ybs010/anlp-assignment1-c1"
    upload_to_hf: bool = True

EXPERIMENTS = [ExperimentConfig(),
               ExperimentConfig(name="C2",positional_encoding="rope"),
               ExperimentConfig(name="C3",attention_type="gqa"),
               ExperimentConfig(name="C4",norm_type="rmsnorm"),
               ExperimentConfig(name="C5",tokenizer_type="blt"),]


def print_experiment_configs():

    for experiment in EXPERIMENTS:
        logger.info("Experiment Name = %s, tokenizer = %s, positional encoding = %s, attention type = %s, Normalization function = %s ",
                    experiment.name,experiment.tokenizer_type,experiment.positional_encoding,experiment.attention_type,experiment.norm_type)


def get_lines_from_file_path(file_path):

    try:
        with (open(file=file_path) as f):
            content = f.readlines()

        return content
    except Exception as exec:
        logger.error("Can't read the file ",exc_info=exec)
        return []
