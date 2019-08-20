import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def load_CIFAR10():
    # C10+
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                                          ])

    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10(root='/data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_set, batch_size=64, num_workers=4)

    test_set = datasets.CIFAR10(root='/data', train=False, transform=transforms_test, download=True)
    test_loader = DataLoader(test_set, batch_size=100, num_workers=4)

    return train_loader, test_loader
