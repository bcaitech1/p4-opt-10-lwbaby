"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader_sweep import create_dataloader
from src.cosine import CosineAnnealingWarmupRestarts
from src.loss import CustomCriterion
from src.model import Model
from src.trainer_sweep import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info
from adamp import AdamP
import wandb

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    sweep_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.model.to(device)
    
    #============================================================================
    #Pretrained weight
    checkpoint = torch.load('./exp/2021-05-26_05-24-43/best.pt')
    model_instance.model.load_state_dict(checkpoint)
    #==============================================================================
    
    #============================================================================
    #sweep config
    if sweep_config['batch_size']: data_config['BATCH_SIZE'] = sweep_config['batch_size']
    if sweep_config['lr']: data_config['INIT_LR'] = sweep_config['lr']
    if sweep_config['wd']: data_config['WEIGHT_DECAY'] = sweep_config['wd']
    if sweep_config['momentum']: data_config['MOMENTUM'] = sweep_config['momentum']
    if sweep_config['epoch']: data_config['EPOCHS'] = sweep_config['epoch']
    #===========================================================================
    
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    
    #===============================================================================
    # Create optimizer, scheduler, criterion
    if sweep_config['optimizer'] == "AdamP":
        optimizer = AdamP(model_instance.model.parameters(),
                            lr=data_config["INIT_LR"], 
                            betas=(0.9, 0.999),
                            weight_decay=data_config['WEIGHT_DECAY'])
    elif sweep_config['optimizer'] == "AdamW" or sweep_config['optimizer'] == "Adam":
        optimizer = getattr(optim, sweep_config['optimizer'])(model_instance.model.parameters(), 
                                                      lr=data_config["INIT_LR"], 
                                                      weight_decay = data_config['WEIGHT_DECAY'])
    elif not sweep_config['optimizer'] == None:
        optimizer = getattr(optim, sweep_config['optimizer'])(model_instance.model.parameters(), 
                                                      lr=data_config["INIT_LR"], 
                                                      weight_decay = data_config['WEIGHT_DECAY'], 
                                                      momentum = data_config['MOMENTUM'])
    else:
        optimizer = getattr(optim, 'AdamW')(model_instance.model.parameters(), 
                          lr=data_config["INIT_LR"], 
                          betas=(0.9, 0.999), 
                          weight_decay=1e-2)
        
   
        
    #===============================================================================
    #optimizer = torch.optim.AdamW(model_instance.model.parameters(), lr=data_config["INIT_LR"], betas=(0.9, 0.999), weight_decay=1e-2)
    print(optimizer.state_dict()['param_groups'][0])
#     optimizer = torch.optim.SGD(
#         model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
#     )
    
    
    if sweep_config['cosine']:
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer=optimizer, 
#             T_0=sweep_config['T0'], 
#             T_mult=sweep_config['Tm'], 
#             eta_min=sweep_config['etam']
#         )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=sweep_config['T0'],
            cycle_mult=sweep_config['Tm'],
            max_lr=data_config["INIT_LR"],
            min_lr=sweep_config['etam'],
            warmup_steps=sweep_config['warmup'],
            gamma=sweep_config['gamma']
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=data_config["INIT_LR"],
            steps_per_epoch=len(train_dl),
            epochs=data_config["EPOCHS"],
            pct_start=0.05,
        )
        
        
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="configs/model/mobilenetv3.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--optimizer", default=None, type=str, help="optimizer config to sweep"
    )
    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate config for seep"
    )
    parser.add_argument(
        "--wd", default=None, type=float, help="weight decay config for sweep"
    )
    parser.add_argument(
        "--batch_size", default=None, type=int, help="batch size config for sweep"
    )
    parser.add_argument(
        "--momentum", default=None, type=float, help="momentum config for sweep"
    )
    parser.add_argument(
        "--epoch", default=None, type=int, help="epoch config for sweep"
    )
    parser.add_argument(
        "--cosine", default=None, type=bool, help="cosine?"
    )
    parser.add_argument(
        "--T0", default=None, type=int, help="T0 config for sweep"
    )
    parser.add_argument(
        "--Tm", default=None, type=float, help="Tm config for sweep"
    )
    parser.add_argument(
        "--etam", default=None, type=float, help="etamin config for sweep"
    )
    parser.add_argument(
        "--warmup", default=None, type=int, help="warmup config for sweep"
    )
    parser.add_argument(
        "--gamma", default=None, type=float, help="gamma config for sweep"
    )
    args = parser.parse_args()
    
    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)
    sweep_config = {"optimizer":args.optimizer, 
                    "lr":args.lr, 
                    "wd":args.wd, 
                    "batch_size":args.batch_size, 
                    "momentum":args.momentum,
                    "epoch": args.epoch,
                    "cosine": args.T0,
                    "T0": args.T0,
                    "Tm": args.Tm,
                    "etam": args.etam,
                    "warmup": args.warmup,
                    "gamma": args.gamma
                   }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        sweep_config=sweep_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
