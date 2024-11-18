# Unbiased VICReg Implementation

This repository contains an implementation of VICReg (Variance-Invariance-Covariance Regularization) with both biased and unbiased loss variants for self-supervised learning on CIFAR-10 dataset.

## Project Structure
```
.
├── config.yaml           # Configuration file
├── main
│   └── main.py          # Main execution script
├── noxfile.py           # Development automation
└── src
    ├── checkpoints.py   # Checkpoint management
    ├── datasets_setup.py # Dataset preparation
    ├── probing.py       # Evaluation probing
    ├── train_evaluate.py # Training and evaluation
    └── VICReg.py        # VICReg implementation
```

## Requirements

- Python 3.9.16
- Dependencies will be managed through nox sessions

## Configuration

The project uses a `config.yaml` file to manage hyperparameters and training settings:

```yaml
batch_sizes: [256, 128, 64, 32, 16, 8]  # Available batch sizes for testing
batch_size: 256                          # Default training batch size
batch_size_evaluate: 64                  # Evaluation batch size
num_epochs: 100                          # Number of training epochs
num_eval_epochs: 30                      # Number of evaluation epochs
sim_coeff: 25.0                         # Similarity loss coefficient
std_coeff: 25.0                         # Standard deviation loss coefficient
cov_coeff: 1                            # Covariance loss coefficient
lr_vicreg: 2e-4                         # VICReg learning rate
lr_linear: 2e-2                         # Linear probe learning rate
backbone: "resnet18"                    # Backbone architecture (resnet18/resnet50)
projection_head_dims: [512, 512, 512]   # Projection head dimensions
probe: "linear"                         # Probe type (linear/online)
loss: "unbiased"                        # Loss type (biased/unbiased)
batch_size_sharing: False               # Same batch_size on linear eval (True/False)
```

## Usage

### Development Setup

The project uses nox for automation. Available sessions:

1. Running the Training with installing req-s:
```bash
nox -s install_requirements

nox -s run
```
2. Linting and Style:
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

## Checkpoints and logs

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
