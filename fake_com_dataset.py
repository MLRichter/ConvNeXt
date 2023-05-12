import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = np.array(self.data.iloc[:, 0]) # extract the first column as labels
        self.features = np.array(self.data.iloc[:, 1:]) # extract the remaining columns as features
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        features = self.features[idx].reshape(3, 20, 20) # reshape the features into a 20x20 tensor
        return torch.from_numpy(features), label


if __name__ == '__main__':
    # Example usage:
    dataset = MyDataset('train.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    for batch_idx, (features, labels) in enumerate(dataloader):
        # do something with the batch of 20x20 features and labels
        print(features.shape, labels.shape)
        img: torch.Tensor = features[0]
        img = img.to("cpu")
        from skimage.io import imshow, show, imsave
        imsave("test.jpg", np.swapaxes(img.numpy(), 0, 2))
        break
