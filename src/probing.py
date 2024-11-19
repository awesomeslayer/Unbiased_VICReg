from src.checkpoints import save_checkpoint
import torch
from torch import nn
from lightly.loss.vicreg_loss import invariance_loss, variance_loss, covariance_loss


def linear_probe(
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
):
    if vicreg_start < args.num_epochs:
        logger.info(
            f"Continuing VICReg training from epoch {vicreg_start} to {args.num_epochs}"
        )
        for epoch in range(vicreg_start, args.num_epochs):
            model.train()
            total_loss = 0
            total_inv_loss = 0
            total_var_loss = 0
            total_cov_loss = 0
            for batch in train_loader_vicreg:
                for param in model.parameters():
                    param.requires_grad = True
                for param in linear.parameters():
                    param.requires_grad = False

                _, _, x0, x1, _ = batch
                x0, x1 = x0[0].to(device), x1[0].to(device)
                z0, z1 = model(x0), model(x1)

                inv_loss = invariance_loss(z0, z1)
                var_loss = variance_loss(z0, z1)
                cov_loss = covariance_loss(z0, z1)
                loss = inv_loss + var_loss + cov_loss

                total_loss += loss.detach()
                total_inv_loss += inv_loss.detach()
                total_var_loss += var_loss.detach()
                total_cov_loss += cov_loss.detach()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader_vicreg)
            avg_inv_loss = total_inv_loss / len(train_loader_vicreg)
            avg_var_loss = total_var_loss / len(train_loader_vicreg)
            avg_cov_loss = total_cov_loss / len(train_loader_vicreg)

            logger.info(f"Epoch: {epoch:>02}, {args.loss}VICReg loss: {avg_loss:.5f}")
            logger.info(f"Epoch: {epoch:>02}, Invariance loss: {avg_inv_loss:.5f}")
            logger.info(f"Epoch: {epoch:>02}, Variance loss: {avg_var_loss:.5f}")
            logger.info(f"Epoch: {epoch:>02}, Covariance loss: {avg_cov_loss:.5f}")

            writer.add_scalar(f"{args.loss}VICReg_loss/train", avg_loss.item(), epoch)
            writer.add_scalar("Invariance_loss/train", avg_inv_loss.item(), epoch)
            writer.add_scalar("Variance_loss/train", avg_var_loss.item(), epoch)
            writer.add_scalar("Covariance_loss/train", avg_cov_loss.item(), epoch)

            save_checkpoint(
                model, optimizer, scheduler, epoch, args.checkpoint_dir, prefix="vicreg"
            )

            current_lr_optimizer = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch: {epoch:>02}, Optimizer LR: {current_lr_optimizer:.8f}")

    else:
        logger.info(f"VICReg training already completed on {vicreg_start} epoch")

    logger.info(f"linear_start {linear_start}, vicreg_start {vicreg_start}")
    if linear_start < args.num_eval_epochs:
        logger.info(
            f"Starting linear evaluation from {linear_start}/{args.num_eval_epochs} epochs"
        )

        for epoch in range(linear_start, args.num_eval_epochs):
            model.eval()
            linear.train()

            train_loss = 0
            correct = 0
            total = 0

            for batch in train_loader_linear:
                for param in model.parameters():
                    param.requires_grad = False
                for param in linear.parameters():
                    param.requires_grad = True

                _, x, _, _, y = batch

                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    features = model.backbone(x).flatten(start_dim=1)

                linear_optimizer.zero_grad()
                outputs = linear(features)
                loss = linear_loss(outputs, y)
                loss.backward()
                linear_optimizer.step()
                linear_scheduler.step()

                train_loss += loss.detach()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            train_accuracy = 100.0 * correct / total
            train_loss = train_loss / len(train_loader_linear)

            logger.info(
                f"Epoch: {epoch:>02}, Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%"
            )

            writer.add_scalar("Train_loss", train_loss, epoch)
            writer.add_scalar("Train_accuracy", train_accuracy, epoch)
            save_checkpoint(
                linear,
                linear_optimizer,
                linear_scheduler,
                epoch,
                args.checkpoint_dir,
                "linear",
            )

            # Print current learning rates
            current_lr_linear_optimizer = linear_optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch: {epoch:>02}, Linear Optimizer LR: {current_lr_linear_optimizer:.8f}"
            )

            model.eval()
            linear.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in test_loader:
                    x, _, _, _, y = batch
                    x, y = x.to(device), y.to(device)

                    features = model.backbone(x).flatten(start_dim=1)
                    outputs = linear(features)
                    loss = nn.CrossEntropyLoss()(outputs, y)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

            test_accuracy = 100.0 * correct / total
            test_loss = test_loss / len(test_loader)

            logger.info(
                f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%"
            )

            writer.add_scalar("Test_loss", test_loss, epoch)
            writer.add_scalar("Test_accuracy", test_accuracy, epoch)

    else:
        logger.info(
            f"Train exists: {linear_start}/{args.num_eval_epochs} epochs, evaluate on train:"
        )

        model.eval()
        linear.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                x, _, _, _, y = batch
                x, y = x.to(device), y.to(device)

                features = model.backbone(x).flatten(start_dim=1)
                outputs = linear(features)
                loss = nn.CrossEntropyLoss()(outputs, y)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        test_accuracy = 100.0 * correct / total
        test_loss = test_loss / len(test_loader)

        logger.info(
            f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%"
        )

        writer.add_scalar("Test_loss", test_loss, epoch)
        writer.add_scalar("Test_accuracy", test_accuracy, epoch)

    writer.close()

    return True


def online_probe(
    start_epoch,
    writer,
    model,
    linear,
    device,
    train_loader,
    test_loader,
    vicreg_loss,
    linear_loss,
    logger,
    optimizer,
    scheduler,
    linear_optimizer,
    linear_scheduler,
    args,
):

    if start_epoch < args.num_epochs:
        logger.info(
            f"Continuing training from epoch {start_epoch} to {args.num_epochs}"
        )
        for epoch in range(start_epoch, args.num_epochs):
            total_loss = 0
            total_inv_loss = 0
            total_var_loss = 0
            total_cov_loss = 0
            train_loss = 0
            correct = 0
            total = 0

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                for param in model.parameters():
                    param.requires_grad = True
                for param in linear.parameters():
                    param.requires_grad = False

                _, x, x0, x1, y = batch
                x0, x1 = x0[0].to(device), x1[0].to(device)
                x, y = x.to(device), y.to(device)

                z0, z1 = model(x0), model(x1)

                inv_loss = invariance_loss(z0, z1)
                var_loss = variance_loss(z0, z1)
                cov_loss = covariance_loss(z0, z1)
                loss = inv_loss + var_loss + cov_loss

                total_loss += loss.detach()
                total_inv_loss += inv_loss.detach()
                total_var_loss += var_loss.detach()
                total_cov_loss += cov_loss.detach()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                for param in model.parameters():
                    param.requires_grad = False
                for param in linear.parameters():
                    param.requires_grad = True

                features = model.backbone(x).flatten(start_dim=1)

                outputs = linear(features)
                loss = linear_loss(outputs, y)
                train_loss += loss.detach()
                loss.backward()
                linear_optimizer.step()
                linear_scheduler.step()
                linear_optimizer.zero_grad()

                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            avg_loss = total_loss / len(train_loader)
            avg_inv_loss = total_inv_loss / len(train_loader)
            avg_var_loss = total_var_loss / len(train_loader)
            avg_cov_loss = total_cov_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total
            train_loss = train_loss / len(train_loader)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar(f"{args.loss}VICReg Loss/train", avg_loss, epoch)
            writer.add_scalar("Invariance_loss/train", avg_inv_loss.item(), epoch)
            writer.add_scalar("Variance_loss/train", avg_var_loss.item(), epoch)
            writer.add_scalar("Covariance_loss/train", avg_cov_loss.item(), epoch)

            logger.info(
                f"Epoch: {epoch:>02}, {args.loss}VICReg loss: {avg_loss:.5f}, "
                f"Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%"
            )
            logger.info(f"Epoch: {epoch:>02}, Invariance loss: {avg_inv_loss:.5f}")
            logger.info(f"Epoch: {epoch:>02}, Variance loss: {avg_var_loss:.5f}")
            logger.info(f"Epoch: {epoch:>02}, Covariance loss: {avg_cov_loss:.5f}")

            save_checkpoint(
                model, optimizer, scheduler, epoch, args.checkpoint_dir, "vicreg"
            )
            save_checkpoint(
                linear,
                linear_optimizer,
                linear_scheduler,
                epoch,
                args.checkpoint_dir,
                "linear",
            )

            current_lr_optimizer = optimizer.param_groups[0]["lr"]
            current_lr_linear_optimizer = linear_optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch: {epoch:>02}, Optimizer LR: {current_lr_optimizer:.8f}, Linear Optimizer LR: {current_lr_linear_optimizer:.8f}"
            )

            model.eval()
            linear.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in test_loader:
                    x, _, _, _, y = batch
                    x, y = x.to(device), y.to(device)
                    features = model.backbone(x).flatten(start_dim=1)
                    outputs = linear(features)
                    loss = nn.CrossEntropyLoss()(outputs, y)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

            test_accuracy = 100.0 * correct / total
            test_loss = test_loss / len(test_loader)

            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/test", test_accuracy, epoch)

            logger.info(
                f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%"
            )

    else:
        logger.info(f"Training already completed on {start_epoch} epoch")

    writer.close()
