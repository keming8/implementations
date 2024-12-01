import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image

# Define hyperparameters
batch_size = 32
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dims = 100
learning_rate = 2e-4

# Load the dataset. Use MNIST for this

ds = load_dataset("ylecun/mnist")
# image = ds['train'][10]['image']
# print(type(image))

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class TransformDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]['image'], self.dataset[idx]['label']
 
        if self.transform:
            image = self.transform(image)
        return image, label
    
# MNIST dataset consists of 28x28 images
train_dataset = TransformDataset(ds['train'], transform)
test_dataset = TransformDataset(ds['test'], transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class SimpleGeneratorBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class SimpleDiscriminatorBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class Generator(nn.Module):

    def __init__(self, input_dim, output_dim, block):
        super().__init__()
        self.generator_block = block
        self.model = nn.Sequential(
            self.generator_block(input_dim, 128),
            self.generator_block(128, 256),
            self.generator_block(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x


# Discriminator takes image as input, classifies whether it's from the dataset or not
class Discriminator(nn.Module):

    def __init__(self, input_dim, output_dim, block):
        super().__init__()
        self.discriminator_block = block
        self.model = nn.Sequential(
            self.discriminator_block(input_dim, 512),
            self.discriminator_block(512, 256),
            self.discriminator_block(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)

        return x

class GAN():
    
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=learning_rate)
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=learning_rate)



# Create blocks

generator = Generator(100, 28*28, SimpleGeneratorBlock)
discriminator = Discriminator(28*28, 1, SimpleDiscriminatorBlock)
model = GAN(generator, discriminator)

criterion = nn.BCELoss()

for epoch in range(epochs):
    for i, (image, _) in enumerate(train_loader):
        real_images = image.to(device)

        # Create labels
        real_labels = torch.ones((batch_size, 1)).to(device)

        # Create generated samples
        fake_images = model.generator(torch.randn((batch_size, input_dims)).to(device))
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # Put all samples together
        all_images = torch.cat((real_images, fake_images))
        all_labels = torch.cat((real_labels, fake_labels))
        
        # Train discriminator
        model.d_optimizer.zero_grad()
        d_output = model.discriminator(all_images)
        d_loss = criterion(d_output, all_labels)
        d_loss.backward()
        model.d_optimizer.step()

        # Data to train generator
        latent_fake_images = torch.randn((batch_size, input_dims)).to(device)

        # Train generator
        model.g_optimizer.zero_grad()
        fake_images = model.generator(latent_fake_images)
        fake_output = model.discriminator(fake_images)

        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        model.g_optimizer.step()


    print(f"Epoch [{epoch+1}/{epochs}], "
        f"D Loss: {d_loss.item():.4f}, "
        f"G Loss: {g_loss.item():.4f}")
