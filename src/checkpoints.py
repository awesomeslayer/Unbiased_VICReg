import os
import torch
import logging
from omegaconf import DictConfig
import logging
import os


def setup_logging(checkpoint_dir: str) -> logging.Logger:
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = logging.getLogger("vicreg")

    logger.setLevel(logging.INFO)

    # Disable propagation to the root logger (prevents logging in ./main.log)
    # logger.propagate = False

    log_file_path = os.path.join(checkpoint_dir, "main.log")
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)

    return logger


def log_config(cfg: DictConfig, logger: logging.Logger):
    logger.info("Configuration:")
    for key, value in cfg.items():
        logger.info(f"{key}: {value}")


def load_checkpoint(model, optimizer, checkpoint_dir, prefix="vicreg"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pt")

    if not os.path.exists(checkpoint_path):
        return 0

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"] + 1


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, prefix="vicreg"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
