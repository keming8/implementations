import torch.nn as nn
import torch

class SimpleGeneratorBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.ReLU()
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class SimpleDiscriminatorBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.ReLU()
            nn.LeakyReLU()
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
            nn.Dropout(0.1),
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
            # nn.Dropout(0.1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)

        return x
