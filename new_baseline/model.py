from typing import Dict, Any, Tuple

from sklearn.metrics import f1_score
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from loss import KL_distillation_loss

def copy_pretrained_weight(model, model_name, freeze_backbone=False):
    """copy pretrained weight to model
    Args:
        model : The model whose weight ​​are to be updated
        model_name : model name
        freeze_backbone (bool, optional): Whether to freeze the weight of the backbone, Defaults to False.
    """
    pretrained_model = getattr(
        torchvision.models, model_name
    )(pretrained=True)
    
    with torch.no_grad():
        for params, pretrained_params in zip(model.parameters(), pretrained_model.parameters()):
            if params.shape == pretrained_params.shape:
                params.data = pretrained_params.data.clone().detach()

                if freeze_backbone:
                    params.requires_grad = False


def train_model(model: nn.Module,
                train_config: Dict[str, Any],
                train_dataloader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler,
                model_path: str,
                val_dataloader,
                device: torch.device = "cpu",):
                
    """Train model.

    Args:
        model: model to train
        train_config: train config dict
        train_dataloader: data loader module which is a iterator that returns (data, labels)
        criterion: loss fn
        optimizer: optimizer
        scheduler: learning rate scheduler
        model_path: absolute path to save model's state dict
        val_dataloader: dataloader for validation
        device: device on which run model
    """
    scaler = (
        torch.cuda.amp.GradScaler() if train_config["FP16"] and device != torch.device("cpu") else None
    )

    best_test_acc = -1.0
    best_test_f1 = -1.0
    num_classes = train_config['NUM_CLASSES']
    label_list = [i for i in range(num_classes)]

    for epoch in range(train_config['EPOCHS']):
        running_loss, correct, total = 0.0, 0, 0
        preds, gt = [], []
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        model.train()
        for batch, (data, labels) in pbar:
            data, labels = data.to(device), labels.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)

            if isinstance(criterion, KL_distillation_loss):
                loss = criterion(outputs, labels, data)
            else:
                loss = criterion(outputs, labels)

            optimizer.zero_grad()

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()

            running_loss += loss.item()
            pbar.update()
            train_f1_score = f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0) 
            pbar.set_description(
                f"Train: [{epoch + 1:03d}] "
                f"Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {train_f1_score:.2f}"
            )
            
        pbar.close()

        _, test_f1, test_acc = test(
            model=model, 
            test_dataloader=val_dataloader, 
            num_classes=train_config['NUM_CLASSES'],
            scaler=scaler
        )

        if best_test_f1 > test_f1:
            continue
        best_test_acc = test_acc
        best_test_f1 = test_f1
    
        print(f"Model saved. Current best test f1: {best_test_f1:.3f}")

        torch.save(model.state_dict(), model_path)

@torch.no_grad()
def test(model: nn.Module, 
    test_dataloader, 
    num_classes,
    scaler
    ) -> Tuple[float, float, float]:
    """Test model.

    Args:
        test_dataloader: test data loader module which is a iterator that returns (data, labels)
        num_classes : number of classes
        scaler : torch cuda amp GradScaler

    Returns:
        loss, f1, accuracy
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    running_loss = 0.0
    preds = []
    gt = []
    correct = 0
    total = 0

    label_list = [i for i in range(num_classes)]

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    model.to(device)
    model.eval()
    for batch, (data, labels) in pbar:
        data, labels = data.to(device), labels.to(device)

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(data)
        else:
            outputs = model(data)
        outputs = torch.squeeze(outputs)
        running_loss += nn.functional.cross_entropy(outputs, labels).item()

        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        preds += pred.to("cpu").tolist()
        gt += labels.to("cpu").tolist()
        pbar.update()
        pbar.set_description(
            f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
            f"Acc: {(correct / total) * 100:.2f}% "
            f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
        )
    loss = running_loss / len(test_dataloader)
    accuracy = correct / total
    f1 = f1_score(
        y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
    )
    return loss, f1, accuracy