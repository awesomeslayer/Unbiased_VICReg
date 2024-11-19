import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from lightly.loss import VICRegLoss
import logging
import math

from src.checkpoints import load_checkpoint
from src.VICReg import VICReg, UnbiasedVICRegLoss
from src.datasets_setup import exCIFAR10
from src.probing import online_probe, linear_probe
from src.lars import LARS

def train_evaluate(args, logger: logging.Logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}/tensorboard_logs")

    (
        model,
        optimizer,
        scheduler,
        linear,
        linear_optimizer,
        linear_scheduler,
        vicreg_loss,
        linear_loss,
        train_loader_vicreg,
        train_loader_linear,
        test_loader,
        start_epoch,
        vicreg_start,
        linear_start,
    ) = setup_experiment(args, writer, device, logger)

    logger.info(f"Beginning train + evaluate for {args.probe} probing")

    if args.probe == "online":
        online_probe(
            start_epoch,
            writer,
            model,
            linear,
            device,
            train_loader_vicreg,
            test_loader,
            vicreg_loss,
            linear_loss,
            logger,
            optimizer,
            scheduler,
            linear_optimizer,
            linear_scheduler,
            args,
        )
    elif args.probe == "linear":
        linear_probe(
            writer,
            model,
            linear,
            vicreg_start,
            linear_start,
            device,
            train_loader_vicreg,
            train_loader_linear,
            test_loader,
            vicreg_loss,
            linear_loss,
            logger,
            optimizer,
            scheduler,
            linear_optimizer,
            linear_scheduler,
            args,
        )
    else:
        logger.error(
            f"Unknown type of probing: {args.probe}. Use online/linear instead."
        )

    return model, linear


def write_pictures(writer, train_loader, device, model, logger: logging.Logger):
    logger.info("Writing visualization data to TensorBoard")
    try:
        batch = next(iter(train_loader))
        x, _, x0, _, y = batch

        num_samples = 4
        x_vis = x[:num_samples]
        x0_vis = x0[0][:num_samples]
        labels_vis = y[:num_samples]

        writer.add_images("Original Images", x_vis, 0)
        writer.add_images("Augmented Images", x0_vis, 0)

        for i in range(num_samples):
            writer.add_text(f"Label_{i}", f"Label: {labels_vis[i].item()}", 0)

        writer.add_graph(model, x_vis.to(device))
        logger.info("Successfully wrote visualization data to TensorBoard")
        return True
    except Exception as e:
        logger.error(f"Error writing visualization data: {str(e)}")
        return False


def setup_experiment(args, writer, device, logger: logging.Logger):
    logger.info("Setting up experiment...")

    if args.backbone == "resnet18":
        logger.info("Using ResNet18 backbone")
        backbone = torchvision.models.resnet18()
    elif args.backbone == "resnet50":
        logger.info("Using ResNet50 backbone")
        backbone = torchvision.models.resnet50()
    else:
        logger.error(f"Unknown backbone architecture: {args.backbone}")
        raise ValueError(f"Unknown backbone: {args.backbone}")

    model = VICReg(backbone, args.projection_head_dims)
    model.to(device)
    linear = nn.Linear(args.projection_head_dims[0], 10).to(device)

    if args.loss == "biased":
        logger.info("Using biased VICReg loss")
        vicreg_loss = VICRegLoss(
            lambda_param=args.sim_coeff,
            mu_param=args.std_coeff,
            nu_param=args.cov_coeff,
        )
    elif args.loss == "unbiased":
        logger.info("Using unbiased VICReg loss")
        vicreg_loss = UnbiasedVICRegLoss(
            sim_coeff=args.sim_coeff,
            cov_coeff=args.cov_coeff,
        )
    else:
        logger.error(f"Unknown loss type: {args.loss}")
        raise ValueError(f"Unknown loss type: {args.loss}")

    linear_loss = nn.CrossEntropyLoss()
    logger.info("Setting up datasets and dataloaders")
    train_dataset = exCIFAR10("data/", train=True, download=True)
    test_dataset = exCIFAR10("data/", train=False, download=True)

    train_loader_vicreg = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    train_loader_linear = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_evaluate,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_evaluate,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    logger.info(
        f"Created dataloaders with batch size {args.batch_size} and evaluate {args.batch_size_evaluate}"
    )

    optimizer = LARS(
            model.parameters(),
            lr=args.max_lr_vicreg,
            momentum = args.momentum,
            weight_decay = args.weight_decay,
            warmup_epochs = args.warmup_epochs,
            max_epoch = args.num_epochs
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.num_epochs,
            eta_min=args.final_lr_schedule_value,
        )
    
    linear_optimizer = torch.optim.SGD(
            linear.parameters(),
            lr=args.max_lr_linear,
            momentum=args.linear_momentum,
            weight_decay=args.linear_weight_decay,
        )

    
    milestones = [math.floor(args.num_eval_epochs * 0.6), math.floor(args.num_eval_epochs * 0.8)]
    
    linear_scheduler = torch.optim.lr_scheduler.MultiStepLR(linear_optimizer, milestones)

    logger.info(
        f"Created optimizers with learning rates: vicreg={args.max_lr_vicreg}, linear={args.max_lr_linear}"
    )

    vicreg_start = load_checkpoint(
        model, optimizer, scheduler, args.checkpoint_dir, "vicreg"
    )
    linear_start = load_checkpoint(
        linear, linear_optimizer, linear_scheduler, args.checkpoint_dir, "linear"
    )
    logger.info(
        f"Loaded checkpoints: vicreg_epoch={vicreg_start}, linear_epoch={linear_start}"
    )

    start_epoch = vicreg_start if vicreg_start == linear_start else 0
    logger.info(f"Starting from epoch vicreg_start:{vicreg_start}")

    write_pictures(writer, train_loader_vicreg, device, model, logger)

    return (
        model,
        optimizer,
        scheduler,
        linear,
        linear_optimizer,
        linear_scheduler,
        vicreg_loss,
        linear_loss,
        train_loader_vicreg,
        train_loader_linear,
        test_loader,
        start_epoch,
        vicreg_start,
        linear_start,
    )
