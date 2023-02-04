import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class FundusDataset(Dataset):
    def __init__(self, directory, patient_id, transform=None):
        self.directory = directory
        self.patient_id = patient_id
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if f.startswith(patient_id)]
        self.target_image = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = plt.imread(os.path.join(self.directory, image_file))
        if self.transform:
            image = self.transform(image)

        if self.target_image is None:
            self.target_image = image

        return image, self.target_image

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 32)
        self.fc2 = nn.Linear(32, 3 * 2)

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 2, 3)
        return x

def train(model, dataset, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataset):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, running_loss / len(dataset)))

def main():
    directory = 'fundus_images'
    patient_id = '001'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = FundusDataset(directory, patient_id, transform)
    model = STN()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10
    train(model, dataset, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()

