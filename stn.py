import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class STN(nn.Module):
    """
    Spatial Transformer Network (STN)
    """
    def __init__(self, in_channels):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        theta = x.view(-1, 2, 3)
        return theta

class FundusDataset(torch.utils.data.Dataset):
    """
    Dataset for retinal fundus images
    """
    def __init__(self, directory, patient_id, transform=None):
        self.directory = directory
        self.patient_id = patient_id
        self.transform = transform
        self.files = self.get_files()
        self.images = self.get_images()

    def get_files(self):
        """
        Get all fundus image files for a given patient
        """
        pattern = f"^{self.patient_id}_.*_fundus_.*_(L|R)\.png$"
        files = [f for f in os.listdir(self.directory) if re.match(pattern, f)]
        return files

    def get_images(self):
        """
        Get all fundus images for a given patient
        """
        images = [cv2.imread(os.path.join(self.directory, f), cv2.IMREAD_GRAYSCALE) for f in self.files]
        return images

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def train(model, dataset, criterion, optimizer, num_epochs):
    """
    Train the model
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for epoch in range(num_epochs):
        for i, image in enumerate(dataloader):
            theta = model(image)
            loss = criterion(theta, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

def main():
    """
    Main function
    """
    directory = "path/to/fundus/images"
    patient_id = "123456"
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FundusDataset(directory, patient_id, transform)
    model = STN(1)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    train(model, dataset, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()

