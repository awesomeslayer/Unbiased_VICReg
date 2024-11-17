
from checkpoints import log_message, save_checkpoint
import torch
from torch import nn

def online_probe(start_epoch,writer, model, linear, 
                 device, train_loader, test_loader, vicreg_loss, log_file, optimizer, linear_optimizer, args):
    
    if start_epoch < args.num_epochs:
        log_message(f"Continuing training from epoch {start_epoch} to {args.num_epochs}", log_file)
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
            train_accuracy = 100. * correct / total
            train_loss = train_loss / len(train_loader)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('VICReg Loss/train', avg_loss, epoch)

            log_message(f"Epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}, "
                    f"Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%", log_file)

            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, "vicreg")
            save_checkpoint(linear, linear_optimizer, epoch, args.checkpoint_dir, "linear")

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
            
            test_accuracy = 100. * correct / total
            test_loss = test_loss / len(test_loader)

            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)

            log_message(f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%", log_file)

    else:
        log_message(f"Training already completed on {start_epoch} epoch", log_file)

    writer.close()