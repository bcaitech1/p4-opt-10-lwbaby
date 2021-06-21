import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7, num_classes=9):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


class F1_loss_and_softmax(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7, alpha=0.5):
        super(F1_loss_and_softmax, self).__init__()
        self.F1_loss = F1_Loss(epsilon=epsilon, num_classes=num_classes)
        self.CE_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        f1_loss_value = self.F1_loss(y_pred, y_true)
        ce_loss_value = self.CE_loss(y_pred, y_true)
        return (1-self.alpha) * f1_loss_value + \
                self.alpha * ce_loss_value


class KL_distillation_loss(nn.Module):
    def __init__(self, teacher_model, 
                student_loss,
                alpha=0.9, 
                tau=3):
        super(KL_distillation_loss, self).__init__()
        self.teacher_model = teacher_model
        self.teacher_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_model.eval()
        
        self.student_loss = student_loss
        self.alpha=alpha
        self.tau=tau

        self.running_student_loss = 0
        self.running_distillation_loss = 0
    
    def forward(self, y_pred, y_true, input_data):
        # calc student loss
        student_outputs = F.log_softmax(y_pred/self.tau, dim=1)
        student_loss = self.student_loss(y_pred, y_true)
        self.running_student_loss += student_loss

        # get teacher model's output
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_data)

        teacher_outputs = torch.squeeze(teacher_outputs)
        teacher_outputs = F.softmax(teacher_outputs/self.tau, dim=1)

        # calc distillation_loss
        distillation_loss = F.kl_div(input=student_outputs, 
                                    target=teacher_outputs, 
                                    reduction='batchmean')
        self.running_distillation_loss += distillation_loss

        total_loss = (1-self.alpha) * student_loss + \
                      self.alpha * self.tau * self.tau * distillation_loss
        
        return total_loss

    def reset_running_loss(self):
        self.running_student_loss = 0
        self.running_distillation_loss = 0
