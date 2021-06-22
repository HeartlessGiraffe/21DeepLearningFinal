import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import models
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resnet Test')
    parser.add_argument('-m', '--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
    args = parser.parse_args()
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),])
    testset = datasets.CIFAR10(root='./data', train=False, download=False,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = './trained_model/model.pth'

    net = models.__dict__[args.model](num_classes=10).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(path, map_location=torch.device(device))['state_dict'])
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        top1 = 0
        top5 = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            _, maxk = torch.topk(outputs.data, 5, dim=-1)
            total += targets.size(0)
            test_labels = targets.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

            top1 += (test_labels == maxk[:, 0:1]).sum().item()
            top5 += (test_labels == maxk).sum().item()

        acc1 = 100. * top1 / total
        acc5 = 100. * top5 / total

        print('Accuracy of the network on total {} test images: @top1={}%; @top{}={}%'.format(total,
                                                                                                  100 * top1 / total, 5,
                                                                                                  100 * top5 / total))


