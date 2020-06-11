from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import model, criterion, optimizer
from model_trainer import ModelTrainer


_tasks = transforms.Compose([
    transforms.ToTensor()
])

mnist = MNIST("data", download=True, train=True, transform=_tasks)

split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)

dataloaders = {}
dataloaders['train'] = trainloader
dataloaders['val'] = validloader

# print(next(iter(trainloader))[0].shape)

mt = ModelTrainer(checkpoint_path='/Users/paramshah/Documents/Param/github/my_repos/pytorch-model-trainer/development/artifacts', verbose = False, epoch_save_freq = 5)

# mt.train_model(dataloaders, model, criterion, optimizer, epochs=5)

mt.continue_training(dataloaders, experiment_number = 2, from_epoch = 11, update_optimizer = True, till_epoch=20)