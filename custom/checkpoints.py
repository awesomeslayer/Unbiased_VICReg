import os
import datetime
import torch

def setup_logger(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, 'log.txt')
    return log_file

def log_message(message, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')

def log_config(config, log_file):
    """Log the configuration parameters to the log file."""
    log_message("Launching with the following config parameters:", log_file)
    for key, value in config.items():
        log_message(f"{key}: {value}", log_file)

def load_checkpoint(model, optimizer, checkpoint_dir, prefix="vicreg"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_latest.pt')
    
    if not os.path.exists(checkpoint_path):
        return 0
    
    # Use weights_only=True to avoid FutureWarning
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'] + 1

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, prefix="vicreg"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_latest.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
