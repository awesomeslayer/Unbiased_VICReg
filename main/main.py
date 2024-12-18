from src.checkpoints import setup_logging, log_config
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train_evaluate import train_evaluate
import numpy as np


@hydra.main(config_path="..", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    base_checkpoint_dir = f"./results/{cfg.loss}/{cfg.probe}"
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    if cfg.probe == "online":
        cfg.num_eval_epochs = cfg.num_epochs

    for batch_size in cfg.batch_sizes:
        cfg.batch_size = batch_size
        cfg.checkpoint_dir = os.path.join(base_checkpoint_dir, str(cfg.batch_size))

        if cfg.scale_lr_batched:
            cfg.max_lr_vicreg = float(cfg.max_lr_vicreg * np.sqrt(batch_size / 256))
            cfg.final_lr_schedule_value = float(cfg.final_lr_schedule_value* np.sqrt(batch_size / 256))
        if cfg.batch_size_sharing:
            cfg.batch_size_evaluate = batch_size  # same bs not const for eval
            if cfg.scale_lr_batched:
                cfg.max_lr_linear = batch_size * cfg.max_lr_linear / 256

        logger = setup_logging(cfg.checkpoint_dir)
        logger.info(f"Checkpoint directory: {cfg.checkpoint_dir}")

        log_config(cfg, logger)

        logger.info(f"Running with batch_size={batch_size}")
        train_evaluate(cfg, logger)


if __name__ == "__main__":
    main()
