from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np, torch
from matplotlib import pyplot as plt
from PIL import Image
import os


class params:
  
    output = 'out'
    ep = 24
    bs = 64
    lower_lr = 1e-5
    upper_lr = 1e-4

class MyData(Dataset):

    def __init__(self, data, transform=None):
        super(MyData, self).__init__()

        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data['imgs'])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        ret = {'label': self.data['label'][idx], 'imgs': self.transform(Image.fromarray(self.data['imgs'][idx])) if self.transform is not None else self.data['img'][idx]}
        return ret
    
def makeDataLoader(data_train, data_test, batch_size, crop_size=100):

  transform = transforms.Compose([
      
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0), std=(255, 255, 255)),
      ])

  transform_train = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0), std=(255, 255, 255))
      ])

  train_loader = DataLoader(MyData(data_train, transform=transform_train),
                                            batch_size=batch_size,
                                            shuffle=True)

  dev_loader = DataLoader(MyData(data_test, transform=transform),
                                          batch_size=batch_size,
                                          shuffle=False)
  
  return train_loader, dev_loader


def plot_training(history, output, measure='error'):
  
  plt.plot(history[measure])
  plt.plot(history['dev_' + measure])
  plt.legend(['train', 'dev'], loc='upper left')
  plt.ylabel(measure)
  plt.xlabel('Epoch')
  if measure == 'loss':
    x = np.argmin(history['dev_loss'])
  else: x = np.argmax(history['dev_f1'])

  plt.plot(x,history['dev_' + measure][x], marker="o", color="red")
  plt.savefig(os.path.join(output, f'train_history.png'))
  plt.show()