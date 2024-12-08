import torch.nn as nn
import torch

class ConvGeneratorBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class ConvDiscriminatorBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
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
            self.generator_block(128, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        # x = self.model(x)
        # x = x.view(x.size(0), 1, 28, 28)
        x = self.model(x.view(x.size(0), x.size(1), 1, 1))
        return x


# Discriminator takes image as input, classifies whether it's from the dataset or not
class Discriminator(nn.Module):

    def __init__(self, input_dim, output_dim, block):
        super().__init__()
        self.discriminator_block = block
        self.model = nn.Sequential(
            self.discriminator_block(input_dim, 64),
            self.discriminator_block(64, 128),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)

        return x
