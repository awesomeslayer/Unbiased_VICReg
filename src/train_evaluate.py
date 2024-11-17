import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from lightly.loss import VICRegLoss
from lightly.transforms.vicreg_transform import VICRegTransform
import logging

from src.checkpoints import load_checkpoint
from src.VICReg import VICReg, UnbiasedVICRegLoss 
from src.datasets_setup import CIFAR10TripleView
from src.probing import online_probe, linear_probe

def train_evaluate(args, logger: logging.Logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}/tensorboard_logs")
    
    model, optimizer, linear, linear_optimizer, vicreg_loss, train_loader, \
        test_loader, start_epoch, vicreg_start = setup_experiment(args, writer, device, logger)
    
    logger.info(f"Beginning train + evaluate for {args.probe} probing")

    if args.probe == 'online':
        online_probe(start_epoch, writer, model, linear, 
                 device, train_loader, test_loader, vicreg_loss, logger, optimizer, linear_optimizer, args)
    elif args.probe == 'linear':
        linear_probe(start_epoch, writer, model, linear, vicreg_start, 
                 device, train_loader, test_loader, vicreg_loss, logger, optimizer, linear_optimizer, args)
    else:
        logger.error(f"Unknown type of probing: {args.probe}. Use online/linear instead.")
    
    return model, linear

def write_pictures(writer, train_loader, device, model, logger: logging.Logger):
    logger.info("Writing visualization data to TensorBoard")
    try:
        # Load the first batch
        batch = next(iter(train_loader))
        x, x0, _, y = batch  # x is original, x0 is augmented, y are labels

        # Select 4 samples from the batch for visualization
        num_samples = 4
        x_vis = x[:num_samples]  # Original images
        x0_vis = x0[0][:num_samples]  # First augmentation of the images
        labels_vis = y[:num_samples]  # Corresponding labels

        writer.add_images('Original Images', x_vis, 0)
        writer.add_images('Augmented Images', x0_vis, 0)

        for i in range(num_samples):
            writer.add_text(f'Label_{i}', f'Label: {labels_vis[i].item()}', 0)

        writer.add_graph(model, x_vis.to(device))
        logger.info("Successfully wrote visualization data to TensorBoard")
        return True
    except Exception as e:
        logger.error(f"Error writing visualization data: {str(e)}")
        return False    

def setup_experiment(args, writer, device, logger: logging.Logger):
    logger.info("Setting up experiment...")
    
    # Setup backbone
    if args.backbone == "resnet18":
        logger.info("Using ResNet18 backbone")
        resnet = torchvision.models.resnet18()
    elif args.backbone == "resnet50":
        logger.info("Using ResNet50 backbone")
        resnet = torchvision.models.resnet50()
    else:
        logger.error(f"Unknown backbone architecture: {args.backbone}")
        raise ValueError(f"Unknown backbone: {args.backbone}")
    
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone, args.projection_head_dims)
    model.to(device)
    linear = nn.Linear(args.projection_head_dims[-1], 10).to(device)    
    
    # Setup loss function
    if args.loss == 'biased':
        logger.info("Using biased VICReg loss")
        vicreg_loss = VICRegLoss(lambda_param=args.lambda_param, mu_param=args.mu_param, nu_param=args.nu_param)
    elif args.loss == 'unbiased':
        logger.info("Using unbiased VICReg loss")
        vicreg_loss = UnbiasedVICRegLoss(lambda_param=args.lambda_param, mu_param=args.mu_param, nu_param=args.nu_param)
    else:
        logger.error(f"Unknown loss type: {args.loss}")
        raise ValueError(f"Unknown loss type: {args.loss}")
   
    # Setup datasets and dataloaders
    logger.info("Setting up datasets and dataloaders")
    transform = VICRegTransform(input_size=32)
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        drop_last=True, 
        num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=8
    )
    logger.info(f"Created dataloaders with batch size {args.batch_size}")

    # Setup optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_vicreg, weight_decay=1e-6)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr_linear, weight_decay=1e-6)
    logger.info(f"Created optimizers with learning rates: vicreg={args.lr_vicreg}, linear={args.lr_linear}")

    # Load checkpoints
    vicreg_start = load_checkpoint(model, optimizer, args.checkpoint_dir, "vicreg")
    linear_start = load_checkpoint(linear, linear_optimizer, args.checkpoint_dir, "linear")
    logger.info(f"Loaded checkpoints: vicreg_epoch={vicreg_start}, linear_epoch={linear_start}")

    start_epoch = vicreg_start if vicreg_start == linear_start else 0
    logger.info(f"Starting from epoch {start_epoch}")
    
    write_pictures(writer, train_loader, device, model, logger)
    
    return model, optimizer, linear, linear_optimizer, vicreg_loss, train_loader, test_loader, start_epoch, vicreg_start