from src.checkpoints import save_checkpoint
import torch
from torch import nn
import logging


def linear_probe(
    start_epoch,
    writer,
    model,
    linear,
    vicreg_start,
    device,
    train_loader,
    test_loader,
    vicreg_loss,
    logger,
    optimizer,
    linear_optimizer,
    args,
):
    if start_epoch < args.num_epochs:
        logger.info(
            f"Continuing VICReg training from epoch {vicreg_start} to {args.num_epochs}"
        )
        model.train()

        for epoch in range(vicreg_start, args.num_epochs):
            total_loss = 0

            for batch in train_loader:
                _, x0, x1, _ = batch
                x0, x1 = x0[0].to(device), x1[0].to(device)
                z0, z1 = model(x0), model(x1)
                loss = vicreg_loss(z0, z1)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}")

            # TensorBoard logging for training loss
            writer.add_scalar("VICReg_loss/train", avg_loss.item(), epoch)

            save_checkpoint(
                model, optimizer, epoch, args.checkpoint_dir, prefix="vicreg"
            )
    else:
        logger.info(f"VICReg training already completed on {vicreg_start} epoch")

    # Linear evaluation
    logger.info(f"Starting linear evaluation for {args.num_eval_epochs} epochs")

    for epoch in range(args.num_eval_epochs):
        model.eval()
        linear.train()

        # Training loop
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x, _, _, y = batch
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                features = model.backbone(x).flatten(start_dim=1)

            linear_optimizer.zero_grad()
            outputs = linear(features)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            linear_optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_accuracy = 100.0 * correct / total
        train_loss = train_loss / len(train_loader)

        logger.info(
            f"Epoch: {epoch:>02}, Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%"
        )

        # TensorBoard logging for train loss and accuracy
        writer.add_scalar("Train_loss", train_loss, epoch)
        writer.add_scalar("Train_accuracy", train_accuracy, epoch)

        # Evaluation loop
        model.eval()
        linear.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                x, _, _, y = batch
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

        # TensorBoard logging for test loss and accuracy
        writer.add_scalar("Test_loss", test_loss, epoch)
        writer.add_scalar("Test_accuracy", test_accuracy, epoch)

    # Close the TensorBoard writer
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
    logger,
    optimizer,
    linear_optimizer,
    args,
):

    if start_epoch < args.num_epochs:
        logger.info(
            f"Continuing training from epoch {start_epoch} to {args.num_epochs}"
        )
        model.train()
        linear.train()

        for epoch in range(start_epoch, args.num_epochs):
            total_loss = 0
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                x, x0, x1, y = batch
                x0 = x0[0]
                x1 = x1[0]
                x0, x1 = x0.to(device), x1.to(device)
                z0, z1 = model(x0), model(x1)
                loss = vicreg_loss(z0, z1)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    features = model.backbone(x).flatten(start_dim=1)

                linear_optimizer.zero_grad()
                outputs = linear(features)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()
                linear_optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            avg_loss = total_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total
            train_loss = train_loss / len(train_loader)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("VICReg Loss/train", avg_loss, epoch)

            logger.info(
                f"Epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}, "
                f"Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%"
            )

            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, "vicreg")
            save_checkpoint(
                linear, linear_optimizer, epoch, args.checkpoint_dir, "linear"
            )

            model.eval()
            linear.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in test_loader:
                    x, _, _, y = batch
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
