# VICReg Implementation

This repository contains an implementation of VICReg (Variance-Invariance-Covariance Regularization) with both biased and unbiased loss variants for self-supervised learning on the CIFAR-10 dataset.

## Project Structure

```
.
├── config.yaml           # Configuration file
├── main
│   └── main.py           # Main execution script
├── noxfile.py            # Development automation
├── src
│    ├── checkpoints.py   # Checkpoint management
│    ├── datasets_setup.py # Dataset preparation
│    ├── probing.py       # Evaluation probing
│    ├── train_evaluate.py # Training and evaluation
│    └── VICReg.py        # VICReg implementation
└── results               # Results directory
```

## Requirements

- Python 3.9.16
- Dependencies will be managed through nox sessions

## Configuration

The project uses a `config.yaml` file to manage hyperparameters and training settings. Below is a detailed description of the configuration parameters:

```yaml
hydra:
  run:
    dir: .

sim_coeff: 25.0                # Coefficient for the invariance term in VICReg loss
std_coeff: 25.0                # Coefficient for the variance term in VICReg loss
cov_coeff: 1                   # Coefficient for the covariance term in VICReg loss

batch_sizes: [256, 128, 64, 32, 16, 8]  # Available batch sizes for testing
num_epochs: 100                         # Number of training epochs
max_lr_vicreg: 30                       # Maximum learning rate for VICReg training
momentum: 0.9                           # Momentum for LARS optimizer and scheduler
weight_decay: 1e-4                      # Weight decay for LARS optimizer
final_lr_schedule_value: 0.002           # Final learning rate value for scheduler
warmup_epochs: 10                       # Number of warmup epochs

batch_size_evaluate: 64                  # Evaluation batch size
num_eval_epochs: 100                    # Number of evaluation epochs
max_lr_linear: 30.0                     # Maximum learning rate for linear probe training
linear_momentum: 0.9                     # Momentum for SGD optimizer and scheduler
linear_weight_decay: 0.0                 # Weight decay for SGD optimizer

backbone: "resnet18"                    # Backbone architecture (resnet18/resnet50)
num_layers: 3                           # Number of layers in the projection head
projection_head_dims: [512, 2048]       # Dimensions of the projection head (first must match ResNet output)
probe: "linear"                         # Probe type (linear/online)
loss: "unbiased"                        # Loss type (unbiased/biased)
batch_size_sharing: False                # Whether to use the same batch size for linear evaluation
```

## Usage

### Development Setup

The project uses nox for automation. Available sessions:

1. **Running the Training with Installing Requirements:**

```bash
nox -s install_requirements
nox -s run
```

2. **Linting and Style:**

```bash
# Run flake8
nox -s lint

# Format code with black
nox -s black

# Sort imports
nox -s isort

# Run pylint
nox -s pylint
```

### Running with Different Configurations

To test different batch sizes or switch between biased/unbiased loss:

1. Modify the `config.yaml` file:
   - Change `batch_size` to one of the values in `batch_sizes`
   - Set `loss` to either "biased" or "unbiased"

2. Run the training:

```bash
python -m main.main
```

### TensorBoard Visualization

Monitor training progress using TensorBoard:

1. Launch TensorBoard:

```bash
tensorboard --logdir=results/
```

2. Access the dashboard at `http://localhost:6006`

Key metrics available in TensorBoard:
- Training loss
- Linear probe accuracy

## Checkpoints and Logs

Checkpoints and logs are saved in the directory specified by `<loss>/<probe>/<batch_size>/` in the config. The default path is "./checkpoints".

## Features

- Support for both ResNet18 and ResNet50 backbones
- Flexible projection head configuration
- Linear and online probing options
- Configurable loss functions (biased/unbiased)
- Batch size experimentation support
- Automated development workflow with nox

## Notes

- The implementation supports various batch sizes to study their impact on training
- Both biased and unbiased loss variants are implemented for comparison
- Linear probing is performed every `num_eval_epochs` epochs
- Checkpoints include both the backbone and projection head states