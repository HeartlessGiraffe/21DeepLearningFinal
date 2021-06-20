import argparse
import csv
import os
from datetime import datetime
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.argv = ['']
del sys
# 设置参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training Cutmix')
parser.add_argument('--dataset', default='CIFAR-100', type=str, help='using data set')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=1234, type=int, help='random seed')
parser.add_argument('--batch-size', default=50, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# 设置随机种子
if args.seed != 0:
    print('Set random seed:', args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True

# 数据载入和处理
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'CIFAR-10':
    trainset = datasets.CIFAR10(root='./data', train=True, download=False,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=False,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == 'CIFAR-100':
    trainset = datasets.CIFAR100(root='./data', train=True, download=False,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR100(root='./data', train=False, download=False,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    num_classes = 100

# 调控学习率每30个epoch除以10
def adjust_learning_rate(optimizer, epoch):
    lr = 0.1 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# cutmix主体函数
def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    # lambda ~ Beta(alpha, alpha)服从beta分布
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    # 在batch中随机找一副图像
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    # y_a为原图像label，y_b为用于mix的图像label
    y_a, y_b = y, y[index]
    # 生成替换区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # 进行替换
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # 根据实际被替换区域的面积占比修正lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    # W，H分别为图像的宽和高，以cifar-10为例，W = H = 32
    W = size[2]
    H = size[3]
    # 替换区域的w和h
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 计算替换区域的位置
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# cutmix损失函数，原图像label和替换图像label的交叉熵损失函数的线性组合
def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':

    print('==> Building model..')
    # 生成网络
    net = models.__dict__[args.model](num_classes=num_classes)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')
    if not os.path.isdir('results'):
        os.mkdir('results')
    # logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
    #            + str(args.seed) + '.csv')
    # 基础交叉熵损失函数，对于不进行cutmix的训练集和全部测试集，仍使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器使用SGD
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.decay)
    best_acc1 = 0
    best_acc5 = 0
    # 计算实验耗时
    start_time = datetime.now()
    print(net)
    # 创建tensorboard文件夹
    writer_train_loss = SummaryWriter('./runs' + args.dataset + args.model + '/train/' + 'cutmix')
    writer_test_acc = SummaryWriter('./runs' + args.dataset + args.model + '/test/' + 'cutmix')
    writer_test_acc5 = SummaryWriter('./runs' + args.dataset + args.model + '/test_top5/' + 'cutmix')

    # 训练
    for epoch in range(start_epoch, args.epoch):
        print('\nEpoch: %d' % (epoch + 1))
        adjust_learning_rate(optimizer, epoch + 1)
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            r = np.random.rand(1)
            # 随机对图像进行cutmix
            if args.alpha > 0 and r < args.cutmix_prob:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets,
                                                               args.alpha, use_cuda)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))
                outputs = net(inputs)
                loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().float()
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                            100.*correct/total, correct, total))
            if batch_idx % 100 == 99:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (batch_idx + 1), train_loss / (batch_idx + 1), 100. * correct / total))
            writer_train_loss.add_scalar('training loss',
                                             train_loss / (batch_idx + 1),
                                             epoch * len(trainloader) + batch_idx)
        # 测试，不计算梯度提升运算效率
        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            top1 = 0
            top5 = 0
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, maxk = torch.topk(outputs.data, 5, dim=-1)
                total += targets.size(0)
                # _, predicted = torch.max(outputs.data, 1)
                test_labels = targets.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

                top1 += (test_labels == maxk[:, 0:1]).sum().item()
                top5 += (test_labels == maxk).sum().item()

                # progress_bar(batch_idx, len(testloader),
                #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_idx + 1), 100. * correct / total,
                #                 correct, total))
            acc1 = 100. * top1 / total
            acc5 = 100. * top5 / total
            # 记录最优准确率
            best_acc1 = max(best_acc1, acc1)
            best_acc5 = max(best_acc5, acc5)

            # 每个epoch向tensorboard中写入一次测试集准确率
            writer_test_acc.add_scalar('test accuracy',
                                       top1 / total,
                                       epoch + 1)
            writer_test_acc5.add_scalar('test accuracy top5',
                                        top5 / total,
                                        epoch + 1)
            print('Accuracy of the network on total {} test images: @top1={}%; @top{}={}%'.format(total,
                                                                                                  100 * top1 / total, 5,
                                                                                                  100 * top5 / total))

            # print('Test\'s accuracy is: %.3f%%' % (100 * correct / total))
        checkpoint_path = './checkpoint/model_{}'.format(args.dataset + args.model + '_random_erasing')
        if epoch == args.epoch - 1:
            torch.save({'state_dict': net.state_dict()},
                       checkpoint_path + '.pth')
            print('Model trained to epoch {} has been saved.'.format(epoch + 1))
        # 计算结束时间
    end_time = datetime.now()
    run_time = end_time - start_time

    print('Train has finished, total epoch is %d' % args.epoch)
    print(run_time)
    print('Best accuracy is: %.3f%%' % best_acc1)
    print('Best accuracy5 is: %.3f%%' % best_acc5)
    logname = ('results/log_' + str(args.seed) + '.csv')

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['dataset:', args.dataset, 'model:', args.model, 'arg_method:', 'cutmix', 'epoch:', args.epoch,
             'batch_size',
             args.batch_size, 'best accuracy1:', best_acc1, 'best accuracy5:', best_acc5])

