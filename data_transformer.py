import numpy as np
import torch
import copy
def cutmix_data(x, alpha=1.0):
    x = copy.deepcopy(x)
    # lambda ~ Beta(alpha, alpha)服从beta分布
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    # 在batch中随机找一副图像
    index = torch.randperm(batch_size)
    print(lam)
    print(index)
    # 生成替换区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # 进行替换
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    return x

def cutout_data(x, holes=1):
    x = copy.deepcopy(x)
    h = x.size(2)
    w = x.size(3)
    length = 16
    mask = np.ones((h, w), np.float32)

    for n in range(holes):
        yy = np.random.randint(h)
        xx = np.random.randint(w)

        y1 = np.clip(yy - length // 2, 0, h)
        y2 = np.clip(yy + length // 2, 0, h)
        x1 = np.clip(xx - length // 2, 0, w)
        x2 = np.clip(xx + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(x)
    x = x * mask
    return x

def mixup_data(x, alpha=1.0):
    x = copy.deepcopy(x)
    # lambda ~ Beta(alpha, alpha)服从beta分布
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    print(lam)
    print(index)
    # 直接mixup两幅图的像素点
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

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