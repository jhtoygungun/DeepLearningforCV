import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import os

from model import AlexNet


def get_data():
    """
    data iteration
    """
    data_transform = {
        "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(32 ,scale=(0.64, 1.1), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "valid": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )
    }

    date_root = '../../datasets/'

    if os.path.isdir(date_root + 'cifar-10-batches-py'):
        download_flag = False
    else:
        download_flag = True

    train_dataset = datasets.CIFAR10(root=date_root, train=True, transform=data_transform["train"], download=download_flag)
    valid_dataset = datasets.CIFAR10(root=date_root, train=False, transform=data_transform["valid"], download=download_flag)

    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, valid_dataloader

def get_model(device, num_classes):
    model = AlexNet(num_classes).to(device)
    return model

def train(device, num_epochs, num_classes, lr, wd):
    num_classes = num_classes # CIFAR-10
    net = get_model(device, num_classes)
    train_iter, valid_iter = get_data()

    loss_fun = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=wd)

    save_path = './model/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    temp_correct, temp_loss = 0, 0
    for epoch in range(num_epochs):
        net.train()
        for batch_idx, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            y_hat = net(x)
            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()

        label_hat = torch.argmax(y_hat, dim=1)
        temp_correct += (label_hat == y).max()
        temp_loss += loss

        print(f'epoch:{epoch + 1}, batch:{batch_idx}, loss')

    valid(net, valid_iter, device)

def valid(net, valid_iter, device):
    net.eval()



if __name__ == '__main__':
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:{}".format(device))

    num_epochs, lr, wd = 20, 2e-4, 5e-4
    num_classes = 10

    train(device, num_epochs, num_classes, lr, wd)