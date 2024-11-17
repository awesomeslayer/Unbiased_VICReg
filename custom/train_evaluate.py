import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from lightly.loss import VICRegLoss
from lightly.transforms.vicreg_transform import VICRegTransform

from checkpoints import log_message, load_checkpoint, save_checkpoint
from VICReg import VICReg, UnbiasedVICRegLoss 
from datasets_setup import CIFAR10TripleView

def train_evaluate(args, log_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Using device: {device}", log_file)
    
    model, optimizer, linear, linear_optimizer, vicreg_loss, \
        train_loader, test_loader, start_epoch, vicreg_start, linear_start = setup_experiment(args, writer, device)

    writer = SummaryWriter(log_dir=args.checkpoint_dir)
    
    if args.probe == 'online':
        online_probe(start_epoch, writer, model, linear, 
                 device, train_loader, test_loader, vicreg_loss, log_file, optimizer, linear_optimizer, args)
    elif args.probe == 'linear':
        linear_probe(start_epoch, writer, model, linear, vicreg_start, 
                 device, train_loader, test_loader, vicreg_loss, log_file, optimizer, linear_optimizer, args)
    #elif args.probe == 'online_mixed':
        #log_message(f"TODO not realised yet")
    else:
        log_message(f"Unknown type of probing, use online/linear/online_mixed instead.")
    
    return model, linear

def write_pictures(writer, train_loader, device, model, args):
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

    log_message(f"Begining train + evaluate for {args.probe} probing:")

    return True    

def setup_experiment(args, writer, device):
    if args.backbone == "resnet18":
        resnet = torchvision.models.resnet18()
    elif args.backbone == "resnet50":
        resnet = torchvision.models.resnet50()
    
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VICReg(backbone, args.projection_head_dims)
    model.to(device)
    linear = nn.Linear(args.projection_head_dims[-1], 10).to(device)    
  
    if args.loss == 'biased':
        vicreg_loss = VICRegLoss(lambda_param=args.lambda_param, mu_param=args.mu_param, nu_param=args.nu_param)
    elif args.loss == 'unbiased':
        vicreg_loss = UnbiasedVICRegLoss(lambda_param=args.lambda_param, mu_param=args.mu_param, nu_param=args.nu_param)
   
    transform = VICRegTransform(input_size=32)
    train_dataset = CIFAR10TripleView("data/", transform, train=True, download=True)
    test_dataset = CIFAR10TripleView("data/", transform, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                             shuffle=True, drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False, drop_last=False, num_workers=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_vicreg, weight_decay=1e-6)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr_linear, weight_decay=1e-6)

    vicreg_start = load_checkpoint(model, optimizer, args.checkpoint_dir, "vicreg")
    linear_start = load_checkpoint(linear, linear_optimizer, args.checkpoint_dir, "linear")

    start_epoch = vicreg_start if vicreg_start == linear_start else 0
    
    write_pictures(writer, train_loader, device, model, args)
    
    return model, optimizer, linear, linear_optimizer, vicreg_loss, train_loader, test_loader, start_epoch

def linear_probe(start_epoch, writer, model, linear, vicreg_start, 
                 device, train_loader, test_loader, vicreg_loss, log_file, optimizer, linear_optimizer, args):
    if start_epoch < args.num_epochs:
        log_message(f"Continuing VICReg training from epoch {vicreg_start} to {args.num_epochs}", log_file)
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
            log_message(f"Epoch: {epoch:>02}, VICReg loss: {avg_loss:.5f}", log_file)

            # TensorBoard logging for training loss
            writer.add_scalar('VICReg_loss/train', avg_loss.item(), epoch)

            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir, prefix="vicreg")
    else:
        log_message(f"VICReg training already completed on {vicreg_start} epoch", log_file)

    # Linear evaluation
    log_message(f"Starting linear evaluation for {args.num_eval_epochs} epochs", log_file)

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

        train_accuracy = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        log_message(f"Epoch: {epoch:>02}, Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.2f}%", log_file)

        # TensorBoard logging for train loss and accuracy
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Train_accuracy', train_accuracy, epoch)

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

        test_accuracy = 100. * correct / total
        test_loss = test_loss / len(test_loader)

        log_message(f"Epoch: {epoch:>02}, Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.2f}%", log_file)

        # TensorBoard logging for test loss and accuracy
        writer.add_scalar('Test_loss', test_loss, epoch)
        writer.add_scalar('Test_accuracy', test_accuracy, epoch)

    # Close the TensorBoard writer
    writer.close()

    return True


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