import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ---------------------------------
# FCNN
# ---------------------------------
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

# ---------------------------------
# Basic CNN 
# ---------------------------------
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

# ---------------------------------
# Deeper CNN
# ---------------------------------
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

# ---------------------------------
# VAE
# ---------------------------------
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


# ---------------------------------
# GAN
# ---------------------------------

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
    






# ---------------------------------
# Diffusion Model (DDPM)
# ---------------------------------
# Sinusoidal Embedding
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


# Diffusion Schedule
def cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


# UNet Building Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips):
        x = self.block(x)
        skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels

        # Initial conv (extract features)
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)

        # Embedding of noise variance
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        # Downsampling path
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)

        # Bottleneck (center)
        self.mid1 = ResidualBlock(512, 512)
        self.mid2 = ResidualBlock(512, 512)

        # Upsampling path
        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)

        # Final projection to RGB
        self.final = nn.Conv2d(64, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)

        # Embed noise variance
        noise_emb = self.embedding(noise_variances)
        noise_emb = F.interpolate(
            noise_emb.permute(0, 3, 1, 2),
            size=(self.image_size, self.image_size),
            mode="nearest"
        )
        x = torch.cat([x, self.embedding_proj(noise_emb)], dim=1)

        # Down path
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        # Middle
        x = self.mid1(x)
        x = self.mid2(x)

        # Up path
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)








# Diffusion Model Wrapper
class Diffusion(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training=True):
        net = self.network if training else self.ema_network
        pred_noises = net(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)

        return loss.item()

    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)

        return loss.item()


    @torch.no_grad()
    def sample(self, n=16, device=None, steps=100, deterministic=False):
        """
        Generate samples via reverse diffusion.
        - steps: number of denoising steps (higher = better but slower)
        - deterministic: if True, DDIM-like (no noise); if False, DDPM-like (adds noise each step)
        """
        self.eval()
        device = device or next(self.parameters()).device

        # Infer image size/channels from the backbone
        img_size = getattr(self.network, "image_size", 32)
        num_channels = getattr(self.network, "num_channels", 3)

        # Start from Gaussian noise
        x = torch.randn(n, num_channels, img_size, img_size, device=device)

        # Time schedule from 1 -> 0
        t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            # broadcast scalar t_seq[i] to [n,1,1,1]
            t_cur = torch.full((n, 1, 1, 1), t_seq[i], device=device)
            t_next = torch.full((n, 1, 1, 1), t_seq[i + 1], device=device)

            # Rates for current and next time
            noise_cur, signal_cur = self.schedule_fn(t_cur)
            noise_next, signal_next = self.schedule_fn(t_next)

            # Predict clean image x0 from current noisy x
            _, pred_x0 = self.denoise(x, noise_cur, signal_cur, training=False)

            if deterministic:
                # DDIM-like update (no added noise)
                x = signal_next * pred_x0
            else:
                # DDPM-like update (inject next-step noise)
                eps = torch.randn_like(x)
                x = signal_next * pred_x0 + noise_next * eps

        # Map back to [0, 1]
        return self.denormalize(x)







# ---------------------------------
# Energy-Based Model (EBM)
# ---------------------------------

import random
import numpy as np

def swish(x):
    return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    """Neural network that approximates the energy function E(x, w)."""
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)

def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """
    Langevin dynamics sampler for EBM — patched version.
    Ensures gradients are always tracked even if called from no_grad().
    """
    device = next(nn_energy_model.parameters()).device
    inp_imgs = inp_imgs.clone().detach().to(device)

    for _ in range(steps):
        # Make sure we’re inside an autograd-enabled context
        with torch.enable_grad():
            inp_imgs.requires_grad_(True)
            energy = nn_energy_model(inp_imgs)

            # Compute gradients of summed energy w.r.t inputs
            grads = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=inp_imgs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]

        # Langevin update
        inp_imgs = inp_imgs - step_size * grads
        inp_imgs = inp_imgs + noise_std * torch.randn_like(inp_imgs)
        inp_imgs = inp_imgs.clamp(-1.0, 1.0)

        # Detach before next iteration to avoid graph explosion
        inp_imgs = inp_imgs.detach()

    return inp_imgs




class Buffer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Initialize buffer with random images
        self.examples = [torch.rand((1, 1, 32, 32), device=self.device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(128, 0.05)
        new_rand_imgs = torch.rand((n_new, 1, 32, 32), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        with torch.enable_grad():
            new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs

class Metric:
    def __init__(self):
        self.reset()

    def update(self, val):
        self.total += val.item()
        self.count += 1

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0

class EBM(nn.Module):
    def __init__(self, model, alpha=0.1, steps=30, step_size=1.0, noise=0.005, device='cuda'):
        super().__init__()
        self.device = device
        self.model = model
        self.buffer = Buffer(self.model, device=device)

        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

        # metrics
        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result()
        }

    def reset_metrics(self):
        for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                  self.real_out_metric, self.fake_out_metric]:
            m.reset()

    def train_step(self, real_imgs, optimizer):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(self.steps, self.step_size, self.noise)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).to(self.device)
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        self.loss_metric.update(loss)
        self.reg_loss_metric.update(reg_loss)
        self.cdiv_loss_metric.update(cdiv_loss)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return self.metrics()

    def test_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        fake_imgs = torch.rand((batch_size, 1, 32, 32), device=self.device) * 2 - 1
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

        with torch.no_grad():
            out_scores = self.model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
            cdiv = real_out.mean() - fake_out.mean()

        self.cdiv_loss_metric.update(cdiv)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return {
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result()
        }

    @torch.no_grad()
    def sample(self, n=16):
        """
        Generate n samples using the trained Energy-Based Model (EBM).
        Uses Langevin dynamics via the energy function and replay buffer.
        """
        # Start from random noise in [-1, 1]
        noise_imgs = torch.rand((n, 1, 32, 32), device=self.device) * 2 - 1

        # Generate samples via Langevin dynamics
        samples = generate_samples(
            self.model,
            noise_imgs,
            steps=self.steps,
            step_size=self.step_size,
            noise_std=self.noise
        )

        return samples






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
    elif model_name == "Diffusion":
        unet = UNet(image_size=32, num_channels=3)
        return Diffusion(unet, cosine_diffusion_schedule)
    elif model_name == "EBM":
        energy_net = EnergyModel()
        return EBM(energy_net, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError(f"Unknown model name: {model_name}")
