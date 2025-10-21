import torch
import torch.nn as nn
import torch.nn.functional as F

# FCNN
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # Flatten image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Basic CNN 
# with two conv layers, two pools, two fully connected layers.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Deeper CNN
# with four conv layers, batch normalization for stability, dropout for regularization
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # convolution filter, feature extraction  
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # normalize activations, stabilize 
        self.bn1 = nn.BatchNorm2d(16)
        # downsample size, compress information
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder (3 conv layers)
        self.encoder = nn.Sequential(
            # # 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Flattened encoder output: 128 * 4 * 4 = 2048
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        # Decoder input layer
        self.fc_decode = nn.Linear(latent_dim, 2048)

        # Decoder
        self.decoder = nn.Sequential(
            # 128x4x4 -> 64x8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 64x8x8 -> 32x16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 32x16x16 -> 3x32x32
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            # Ensure pixel values between 0 - 1
            nn.Sigmoid() 
        )
    
    # Encode step
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    # Reparameterization
    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decode step
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.decoder(x)
        return x

    # Foward pass (full vae pipeline)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
# Loss function for VAE
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence: regularization for latent space
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)



# GAN
# for judging 28x28 grayscale images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 1x28x28 -> 64x14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 64x14x14 -> 128x7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        # x shape: (B, 1, 28, 28)
        x = self.act1(self.conv1(x))
        # x shape: (B, 128, 7, 7)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        # Output probability between 0 = fake, and 1 = real
        return torch.sigmoid(self.fc(x))  


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        # Expand z_dim to 128*7*7
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        # Smooth and normalize outputs
        self.bn0 = nn.BatchNorm1d(128 * 7 * 7)
        # Upsample to 14x14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Upsample to 28x28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z shape: (B, 100)
        # (B, 128*7*7)
        x = F.relu(self.bn0(self.fc(z)))
        # Reshape to (B, 128, 7, 7)
        x = x.view(-1, 128, 7, 7)
        # Reshape to (B, 64, 14, 14)
        x = F.relu(self.bn1(self.deconv1(x)))
        # Reshape to (B, 1, 28, 28)
        x = self.tanh(self.deconv2(x))
        return x


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        fake = self.generator(z)
        validity = self.discriminator(fake)
        return fake, validity




# Return a model instance based on the given model_name
# Supported models: 'FCNN', 'CNN', 'EnhancedCNN', 'VAE', 'GAN'
def get_model(model_name):
    if model_name == "FCNN":
        return FCNN()
    elif model_name == "CNN":
        return SimpleCNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    elif model_name == "VAE":
        return VAE()
    elif model_name == "GAN":
        return GAN()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
