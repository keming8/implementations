import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from basic_gan_model import Generator, Discriminator, SimpleGeneratorBlock, SimpleDiscriminatorBlock
# from deep_convolutional_gan_model import Generator, Discriminator, ConvDiscriminatorBlock, ConvGeneratorBlock

# Set manual seed
manual_seed = 1000
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True)

# Define hyperparameters
batch_size = 32
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dims = 100
learning_rate = 2e-4

# Load the dataset. Use MNIST for this

ds = load_dataset("ylecun/mnist")


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

def check_transformation(ds, ind=10):
    image, label = ds[ind]

    image = image * 0.5 + 0.5 
    image = image.squeeze(0) 

    plt.imshow(image.numpy(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create blocks

generator = Generator(100, 28*28, SimpleGeneratorBlock)
discriminator = Discriminator(28*28, 1, SimpleDiscriminatorBlock)
# generator = Generator(100, 28*28, ConvGeneratorBlock)
# discriminator = Discriminator(28*28, 1, ConvDiscriminatorBlock)


criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# Training to be split into two parts
# Part 1 updates the discriminator
# Part 2 updates the generator
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


generator.train()
discriminator.train()
for epoch in range(epochs):
    for i, (image, _) in enumerate(train_loader):
        real_images = image.to(device)

        # Create labels
        real_labels = torch.ones((batch_size, 1)).to(device)
        # real_labels = torch.ones((len(image), 1)).to(device)

        # Create generated samples
        fake_images = generator(torch.randn((batch_size, input_dims)).to(device))
        fake_labels = torch.zeros((batch_size, 1)).to(device)
        # fake_labels = torch.zeros((len(image), 1)).to(device)


        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images)

        # Put all samples together
        # all_images = torch.cat((real_images, fake_images))
        # all_labels = torch.cat((real_labels, fake_labels))
        
        # Train discriminator
        d_optimizer.zero_grad()

        d_real_loss = criterion(real_output, real_labels)
        d_fake_loss = criterion(fake_output, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        for _ in range(5):
            # Data to train generator
            latent_fake_images = torch.randn((batch_size, input_dims)).to(device)
            # latent_fake_images = torch.randn((len(image), input_dims)).to(device)

            # Train generator
            g_optimizer.zero_grad()
            fake_images = generator(latent_fake_images)
            fake_output = discriminator(fake_images)

            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()
        

    print(f"Epoch [{epoch+1}/{epochs}], "
        f"D Loss: {d_loss.item():.4f}, "
        f"G Loss: {g_loss.item():.4f}")

    # with torch.no_grad():
    #     test_noise = torch.randn(16, input_dims).to(device)
    #     generated_images = generator(test_noise)
    #     grid = torchvision.utils.make_grid(generated_images, nrow=4, normalize=True)
    #     plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    #     plt.title(f"Epoch {epoch+1}")
    #     plt.show()
