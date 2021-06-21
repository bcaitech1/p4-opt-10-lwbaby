import os
from PIL import Image
from PIL import ImageFile
import random
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms, models
import custom_transforms

import optuna

class TacoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        valid_images = [".jpg",".gif",".png",".tga"]
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.images = []
        self.labels = []
        for i, c in enumerate(self.classes):
            files_path = os.path.join(root_dir, c)
            files = os.listdir(files_path)
            files = random.sample(files, int(len(files) / 10))
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                self.images.append(Image.open(os.path.join(files_path, f)))
                self.labels.append(i)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.transform(self.images[idx]), self.labels[idx])

    def get_num_classes(self):
        return len(self.classes)

class Objective(object):
    def __init__(self, train_path, test_path, criterion, scheduler, optimizer, model, img_size):
        self.train_path = train_path
        self.test_path = test_path
        self.img_size = img_size
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.MEAN_V = (0.4914, 0.4822, 0.4465)
        self.STD_V = (0.2470, 0.2435, 0.2616)
   
    def __call__(self, trial):
        operators = ["Identity", "AutoContrast", "Equalize", "Rotate", "Solarize", "Color", "Posterize",
                        "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY"]
        suggestions = [trial.suggest_categorical('use_' + o, [True, False]) for o in operators]
        suggested_list = [o for s, o in zip(suggestions, operators) if s]
        print(suggested_list)

        use_SquarePad = trial.suggest_categorical('use_SquarePad', [True, False])
        use_Cutout = trial.suggest_categorical('use_Cutout', [True, False])
        
        train_transform = []
        if use_SquarePad:
            train_transform.append(custom_transforms.SquarePad())
        train_transform.append(transforms.Resize((self.img_size, self.img_size)))
        train_transform.append(custom_transforms.RandAugmentation(suggested_list))
        train_transform.append(transforms.RandomHorizontalFlip())
        if use_Cutout:
            train_transform.append(custom_transforms.SequentialAugmentation([("Cutout", 0.8, 9)]))
        train_transform.append(transforms.ToTensor())
        train_transform.append(transforms.Normalize(self.MEAN_V, self.STD_V))
        train_transform = transforms.Compose(train_transform)

        test_transform = []
        if use_SquarePad:
            test_transform.append(custom_transforms.SquarePad())
        test_transform.append(transforms.Resize((self.img_size, self.img_size)))
        test_transform.append(transforms.ToTensor())
        test_transform.append(transforms.Normalize(self.MEAN_V, self.STD_V))
        test_transform = transforms.Compose(test_transform)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_dataset = TacoDataset(self.train_path, train_transform)
        test_dataset = TacoDataset(self.test_path, test_transform)

        train_loader = DataLoader(dataset=train_dataset, 
                                  pin_memory=torch.cuda.is_available(), 
                                  shuffle=True, 
                                  batch_size=BATCH_SIZE, 
                                  num_workers=4, 
                                  drop_last=True)

        test_loader = DataLoader(dataset=test_dataset, 
                                pin_memory=torch.cuda.is_available(), 
                                shuffle=False, 
                                batch_size=BATCH_SIZE, 
                                num_workers=4)

        best_f1 = -1
        num_classes = train_dataset.get_num_classes()
        label_list = [i for i in range(num_classes)]
        for e in range(EPOCHS):
            train_correct = 0
            train_total = 0
            for b in tqdm(train_loader):
                input = b[0].to(device)
                target = b[1].to(device)

                pred = self.model(input)
                loss = self.criterion(pred, target)

                pred = torch.argmax(pred, dim=1)
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)

                self.optimizer.zero_grad()    
                loss.backward()
                self.optimizer.step()

            gt = []
            preds = []
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for b in test_loader:
                    input = b[0].to(device)
                    target = b[1].to(device)

                    pred = self.model(input)
                    pred = torch.argmax(pred, dim=1)

                    test_correct += (pred == target).sum().item()
                    test_total += target.size(0)

                    preds += pred.to("cpu").tolist()
                    gt += target.to("cpu").tolist()

            test_f1 = f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0)
            print(f'train_acc: {train_correct / train_total * 100:.3f}, test_acc: {test_correct / test_total * 100:.3f}, test_f1: {test_f1:.3f}')
            if test_f1 > best_f1:
                best_f1 = test_f1

        return best_f1

def find_policies(n_trials, train_path, test_path, criterion, scheduler, optimizer, model, img_size):
    study = optuna.create_study(direction="maximize", study_name="autoaug", load_if_exists=True)
    study.optimize(Objective(train_path, test_path, criterion, scheduler, optimizer, model, img_size), n_trials=n_trials)
    print(study.best_trial.params)

train_path = './data/train'
test_path = './data/val'
EPOCHS = 10
BATCH_SIZE = 256

model = models.shufflenet_v2_x0_5(pretrained=True)
model = nn.Sequential(model.conv1,
                    model.maxpool,
                    model.stage2,
                    model.stage3,
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(96, 9))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.001,
        steps_per_epoch=20,
        epochs=EPOCHS,
        pct_start=0.05,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
test = find_policies(1, train_path, test_path, criterion, scheduler, optimizer, model, 32)