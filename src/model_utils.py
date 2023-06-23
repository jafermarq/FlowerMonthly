from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.common import ndarrays_to_parameters

def model_as_ndarrays(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# borrowed from Pytorch quickstart example
def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()

def train_with_kd(net, teacher, kd_config, trainloader, optim, epochs, device:str):
    """Train network on the training set using KD."""

    alpha = kd_config.alpha
    temp = kd_config.temperature
    def kd_loss(student_output, labels, teacher_outputs):
        # KD loss (borrowing from https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/model/net.py#L100)
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/temp, dim=1),
                                F.softmax(teacher_outputs/temp, dim=1)) * (alpha * temp**2) + F.cross_entropy(student_output, labels) * (1. - alpha)

    net.train()
    teacher.eval()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            s_out = net(images)

            # pass same batch throught teacher model
            with torch.no_grad():
                t_out = teacher(images)
            loss = kd_loss(s_out, labels, t_out)
            loss.backward()
            optim.step()

# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
