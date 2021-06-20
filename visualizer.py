import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from show_data import imshow
import torchvision
from data_transformer import cutmix_data
from data_transformer import cutout_data
from data_transformer import mixup_data
import argparse
def main():
    random_erasing = transforms.RandomErasing(
            p=1,
            scale=(0.05, 0.33),
            ratio=(0.3, 3.3),
            value='random')
    transform_map = {
        'cutmix':cutmix_data,
        'cutout':cutout_data,
        'mixup':mixup_data,
        'random_erasing':random_erasing
    }
    batch_size = 4
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=False,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    data_iter = iter(trainloader)
    images, labels = data_iter.next()
    images = images
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    names = ' '.join('%4s' % classes[labels[j]] for j in range(batch_size))
    if args.transformer != 'all_together':
        img = torchvision.utils.make_grid(images)
        img2 = torchvision.utils.make_grid(transform_map[args.transformer](images))
        imshow([
            (img, ''.join(['Original: ', names])),
            (img2, ''.join([f'Transformed by {args.transformer}: ', names]))])
    if args.transformer == 'all_together':
        # img1 = torchvision.utils.make_grid(images)
        img2 = torchvision.utils.make_grid(cutmix_data(images))
        img3 = torchvision.utils.make_grid(mixup_data(images))
        img4 = torchvision.utils.make_grid(cutout_data(images))
        img5 = torchvision.utils.make_grid(random_erasing(images))
        imshow([img2, img3, img4, img5])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizer')
    parser.add_argument(
        '-t', 
        '--transformer', 
        default='mixup', 
        choices=['mixup', 'cutmix', 'random_erasing', 'cutout', 'all_together'], 
        help='Transform method, select from {mixup, cutmix, random_erasing, cutout, all_together}')
    parser.add_argument(
        '-s',
        '--seed',
        default=None,
        type=int,
        help='Random seed, defalut is None'
    )
    args = parser.parse_args()
    main()

