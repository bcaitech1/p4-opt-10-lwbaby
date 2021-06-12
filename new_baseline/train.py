from typing import Dict, Any, Tuple, Union

import argparse
import os
from datetime import datetime 
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from utils import read_yaml, calc_macs
from model import train_model, copy_pretrained_weight
import loss
from dataloader import create_dataloader

def train(train_config: Dict[str, Any],
        log_dir: str,
        device: torch.device):

    # torchvision model
    model = getattr(
        torchvision.models, train_config['MODEL_NAME']
    )(num_classes=train_config['NUM_CLASSES'])

    # log train config
    with open(os.path.join(log_dir, 'train_config.yml'), 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False)

    # load model weight
    model_path = os.path.join(log_dir, 'best.pt')

    if train_config['PRETRAINED']:
        copy_pretrained_weight(model, train_config['MODEL_NAME'], train_config['FREEZE_BACKBONE'])

    elif os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))

    model.to(device)

    # calculation macs
    macs = calc_macs(model, (3, train_config["IMG_SIZE"], train_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    # dataloader
    train_dl, val_dl, _ = create_dataloader(train_config)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if train_config['FREEZE_BACKBONE']:
        optimizer = optim.AdamW( 
            [ param for param in model.parameters() if param.requires_grad == True ], 
            lr=train_config["INIT_LR"]
        )
    else:
        optimizer = optim.AdamW( 
            model.parameters(), lr=train_config["INIT_LR"]
        )

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=1, eta_min=0)

    train_model(
        model=model,
        train_config=train_config,
        train_dataloader=train_dl,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        val_dataloader=val_dl,
        model_path=model_path
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        '--train_name', default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), type=str, help='train name'
    )
    parser.add_argument(
        "--train_config", default="train_config/example.yaml", type=str, help="train config"
    )

    args = parser.parse_args()
   
    train_config = read_yaml(cfg=args.train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join('exp', args.train_name)
    os.makedirs(log_dir, exist_ok=True)

    train(
        train_config=train_config,
        log_dir=log_dir,
        device=device,
    )