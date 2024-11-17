import os
import hydra
from omegaconf import DictConfig
from train_evaluate import train_evaluate
from checkpoints import log_message, log_config, setup_logger

@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):    
    base_checkpoint_dir = f"./{cfg.loss}/{cfg.probe}"
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    for batch_size in cfg.batch_sizes:
        cfg.batch_size = batch_size
        cfg.checkpoint_dir = os.path.join(base_checkpoint_dir, str(cfg.batch_size))
        print(cfg.checkpoint_dir)
        
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        
        log_file = setup_logger(cfg.checkpoint_dir)
        log_config(cfg, log_file)
        
        log_message(f"Running with batch_size={batch_size}", log_file)
        train_evaluate(cfg, log_file)
        

if __name__ == "__main__":
    main()