"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union

from adamp import AdamP
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import torchvision
from tqdm import tqdm
import wandb
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from optuna.samplers import TPESampler
# optuna.logging.disable_default_handler()
# sampler = SkoptSampler(
#     skopt_kwargs={'n_random_starts':5,
#                   'acq_func':'EI',
#                   'acq_func_kwargs': {'xi':0.02}})
from src.utils.torch_utils import save_model


def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch

def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(dataset, torchvision.datasets.vision.VisionDataset):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError

    
        
class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        
        
    def objective(self, trial):

        #Optimizer
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD","AdamW","AdamP"])
        lr = trial.suggest_uniform("lr", 1e-5, 1e-1)
        wdecay = trial.suggest_uniform("wdecay", 1e-5, 1e-1)
        momentum = trial.suggest_uniform("momentum", 0, 1e-3)

        if optimizer_name == "AdamP":
            op_optimizer = AdamP(self.model.parameters(),lr=lr, betas=(0.9, 0.999), weight_decay=wdecay)
        elif optimizer_name == "AdamW" or optimizer_name == "Adam":
            op_optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, weight_decay = wdecay)
        else:
            op_optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, weight_decay = wdecay, momentum = momentum)

        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(self.train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        for epoch in range(15):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                op_optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(op_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    op_optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            pbar.close()

            _, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=self.val_dataloader
            )
            trial.report(test_f1, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            

            
                
        return best_test_f1
        
        
    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        #Wandb======================================================================================
        wandb.init(project='opt-sweep', entity='jy1559')
        wandb.run.name = self.model_path
        wandb.run.save()
        #===========================================================================================
        
        
        #Optuna=====================================================================================
#         study = optuna.create_study(direction='maximize',
#                                     sampler=TPESampler(),
#                                     pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
#                                 )
#         study.optimize(self.objective, n_trials=30)
        
#         pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
#         complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

#         print("Study statistics: ")
#         print("  Number of finished trials: ", len(study.trials))
#         print("  Number of pruned trials: ", len(pruned_trials))
#         print("  Number of complete trials: ", len(complete_trials))

#         print("Best trial:")
#         trial = study.best_trial

#         print("  Value: ", trial.value)

#         print("  Params: ")
#         for key, value in trial.params.items():
#             print("    {}: {}".format(key, value))
            
        
#         optimizer_name = trial.params['optimizer']
#         lr = trial.params['lr']
#         wdecay = trial.params['wdecay']
#         momentum = trial.params['momentum']
#         if optimizer_name == "AdamP":
#             self.optimizer = AdamP(self.model.parameters(),lr=lr, betas=(0.9, 0.999), weight_decay=wdecay)
#         elif optimizer_name == "AdamW" or optimizer_name == "Adam":
#             self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, weight_decay = wdecay)
#         else:
#             self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, weight_decay = wdecay, momentum = momentum)
        #===========================================================================================
        
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]
        
        wandb.watch(self.model)
        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )  
                wandb.log(dict(self.optimizer.state_dict()['param_groups'][0], **{"epoch":epoch+1}))
                wandb.log({"epoch": epoch+1, "Loss":(running_loss / (batch + 1)),"Acc":(correct / total) * 100,"F1(macro)":f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)})
            pbar.close()
            _, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )
            wandb.log({"epoch": epoch+1,"test_acc": test_acc,"test_f1": test_f1})
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            wandb.log({"epoch": epoch+1,"best_test_acc": best_test_acc,"best_test_f1": best_test_f1})
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

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


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
