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
        
        
    def train(
        self,
        train_dataloader_list: list,
        val_dataloader_list: list,
        n_epoch: int,
        n_fold: int,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        self.train_dataloader_list = train_dataloader_list
        self.val_dataloader_list = val_dataloader_list
        
        #Wandb======================================================================================
        wandb.init(project='opt-kfold', entity='jy1559')
        wandb.run.name = self.model_path
        wandb.run.save()
        #===========================================================================================
        
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader_list[0].dataset)
        label_list = [i for i in range(num_classes)]
        
        wandb.watch(self.model)
        for epoch in range(n_epoch):
            state_dict = self.model.state_dict()
            state_dict_list = []
            acc_list = []
            f1_list = []
            for fold in range(n_fold):
                self.model.load_state_dict(state_dict)
                running_loss, correct, total = 0.0, 0, 0
                preds, gt = [], []
                
                
                leng = len(train_dataloader_list[fold])
                pbar = tqdm(enumerate(train_dataloader_list[fold]), total=len(train_dataloader_list[fold]))
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
                        f"Train: [{epoch + 1:03d}], Fold: {fold} "
                        f"Loss: {(running_loss / (batch + 1)):.3f}, "
                        f"Acc: {(correct / total) * 100:.2f}% "
                        f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                    )
                    wandb.log(self.optimizer.state_dict()['param_groups'][0])
                    wandb.log({"epoch": epoch, f"Loss_{fold}":(running_loss / (batch + 1)),
                               f"Acc_{fold}":(correct / total) * 100,
                               "foldstep": epoch*leng+batch,
                               f"F1(macro)_{fold}":f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)})
                    wandb.log({"epoch": epoch, f"Loss":(running_loss / (batch + 1)),
                               f"Acc":(correct / total) * 100,
                               "foldstep": epoch*leng+batch,
                               f"F1(macro)":f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)})
                pbar.close()
                
                _, test_f1, test_acc = self.test(
                    model=self.model, test_dataloader=val_dataloader_list[fold]
                )
                acc_list.append(test_acc)
                f1_list.append(test_f1)
                wandb.log({"epoch": epoch, f"test_acc_{fold}": test_acc,
                           f"test_f1_{fold}": test_f1})
                wandb.log({"epoch": epoch, f"test_acc": test_acc,
                           f"test_f1": test_f1})
                state_dict_list.append(self.model.state_dict())
#                 for x in list(state_dict.keys()): 
#                     if(x in state_dict_fold): state_dict_fold[x] += self.model.state_dict()[x]
#                     else: state_dict_fold[x] = self.model.state_dict()[x]
                
            test_f1 = max(f1_list)
            wandb.log({"epoch": epoch, "test_f1_mean": test_f1})
#             for x in list(state_dict.keys()): 
#                 state_dict_fold[x] = state_dict_fold[x].float()/n_fold
#             self.model.load_state_dict(state_dict_fold)
            self.model.load_state_dict(state_dict_list[f1_list.index(max(f1_list))])
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            wandb.log({"epoch": epoch,"best_test_acc": best_test_acc,"best_test_f1": best_test_f1})
            
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
