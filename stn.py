import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from run_registration import *

class FundusDataset(Dataset):
    def __init__(self, image_list, target_image):
        self.image_list = image_list
        self.target_image = target_image

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        image = torch.from_numpy(image).float()
        return image, self.target_image

def voxelmorph_loss_2d(source, target, source_weight=1, target_weight=1, smoothness_weight=0.001):
    def gradient(x):
        d_dx = x[:, :-1, :-1] - x[:, 1:, :-1]
        d_dy = x[:, :-1, :-1] - x[:, :-1, 1:]
        return d_dx, d_dy

    def gradient_penalty(x):
        d_dx, d_dy = gradient(x)
        return (d_dx.abs().mean() + d_dy.abs().mean()) * smoothness_weight
    
    reconstruction_loss = (source - target).abs().mean() * target_weight
    smoothness_penalty = gradient_penalty(target)
    return reconstruction_loss + smoothness_penalty

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 188 * 188, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 188 * 188)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

def train(epoch):
    model.train()
    batch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.reshape(output.shape[0], 768, 768), target)
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        avg_loss = batch_loss / len(train_loader)
        print('Train Epoch: {}, Average Loss: {:.6f}'.format(epoch, avg_loss))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print("CUDA is available on this system!")
    else: print("CUDA is not available on this system.")
    model = Net().to(device)

    # Load your list of Numpy arrays of training images
    training_images, template = retrieve_images()
    template_image = torch.from_numpy(template).float()

    # Create the dataset
    dataset = FundusDataset(training_images, template_image)

    # Create the data loader
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = voxelmorph_loss_2d

    for epoch in range(1, 5 + 1):
        model.train()
        batch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.reshape(output.shape[0], 768, 768), target)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            avg_loss = batch_loss / len(train_loader)
            print('Train Epoch: {}, Average Loss: {:.6f}'.format(epoch, avg_loss))

if __name__ == "__main__":
    main()

