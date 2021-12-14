from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import  DataLoader

def get_transforms(mode='train'):
    if mode == 'train':
        data_transforms = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023, 0.1994, 0.2010])
        ])
    return data_transforms

def get_dataset(name='cifar10',mode='train'):
    if name == 'cifar10':
        # dataset = datasets.CIFAR10(mode=mode,transform=transforms)
        data_transforms = get_transforms(mode=mode)
        dataset = datasets.CIFAR10(root='data/',train=(mode=='train'),transform=data_transforms,download=True)
    return dataset

def get_dataloader(dataset,batch_size=128,mode='train'):
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=0)
    return dataloader