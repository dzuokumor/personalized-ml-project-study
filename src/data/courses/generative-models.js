export const generativemodels = {
  id: 'generative-models',
  title: 'Generative Models',
  description: 'Learn to create AI systems that generate new content',
  category: 'Advanced',
  difficulty: 'Advanced',
  duration: '6 hours',
  lessons: [
    {
      id: 'autoencoders',
      title: 'Autoencoders',
      duration: '55 min',
      concepts: ['encoder', 'decoder', 'latent space', 'reconstruction', 'bottleneck'],
      content: [
        {
          type: 'heading',
          text: 'Learning Compressed Representations'
        },
        {
          type: 'text',
          text: 'Autoencoders learn to compress data into a lower-dimensional representation, then reconstruct it. The "bottleneck" forces the network to learn the most important features. They are the foundation of many generative models.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Autoencoder Architecture</text>
            <rect x="30" y="60" width="60" height="100" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="60" y="115" text-anchor="middle" font-size="10" fill="#4f46e5">Input</text>
            <text x="60" y="130" text-anchor="middle" font-size="8" fill="#64748b">784 dim</text>
            <polygon points="110,80 110,140 170,100 170,120" fill="#dbeafe" stroke="#3b82f6"/>
            <text x="140" y="115" text-anchor="middle" font-size="9" fill="#1d4ed8">Encoder</text>
            <rect x="190" y="85" width="40" height="50" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="4"/>
            <text x="210" y="115" text-anchor="middle" font-size="9" fill="#92400e">Latent</text>
            <text x="210" y="128" text-anchor="middle" font-size="7" fill="#64748b">32 dim</text>
            <polygon points="250,100 250,120 310,80 310,140" fill="#dcfce7" stroke="#22c55e"/>
            <text x="280" y="115" text-anchor="middle" font-size="9" fill="#15803d">Decoder</text>
            <rect x="330" y="60" width="60" height="100" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="360" y="115" text-anchor="middle" font-size="10" fill="#4f46e5">Output</text>
            <text x="360" y="130" text-anchor="middle" font-size="8" fill="#64748b">784 dim</text>
            <text x="210" y="175" text-anchor="middle" font-size="10" fill="#64748b">Bottleneck forces compression</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'The Autoencoder Objective'
        },
        {
          type: 'text',
          text: 'The autoencoder minimizes reconstruction loss—the difference between input and output. For images, this is typically MSE or binary cross-entropy.'
        },
        {
          type: 'formula',
          latex: '\\mathcal{L} = ||x - \\hat{x}||^2 = ||x - D(E(x))||^2'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

model = Autoencoder()
x = torch.randn(32, 784)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Latent shape: {model.encode(x).shape}")`
        },
        {
          type: 'subheading',
          text: 'Types of Autoencoders'
        },
        {
          type: 'table',
          headers: ['Type', 'Modification', 'Use Case'],
          rows: [
            ['Vanilla', 'Basic encoder-decoder', 'Dimensionality reduction'],
            ['Sparse', 'Sparsity constraint on latent', 'Feature learning'],
            ['Denoising', 'Add noise to input, reconstruct clean', 'Robust representations'],
            ['Contractive', 'Penalize Jacobian of encoder', 'Stable representations'],
            ['Variational', 'Probabilistic latent space', 'Generation (next lesson)']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Autoencoders vs PCA',
          text: 'Linear autoencoders learn the same subspace as PCA. The power of autoencoders comes from non-linear activations, which can learn complex manifolds that PCA cannot capture.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why is the bottleneck layer important in autoencoders?',
          options: [
            'It speeds up training',
            'It forces the network to learn compressed, essential features',
            'It reduces memory usage',
            'It improves accuracy on classification'
          ],
          correct: 1,
          explanation: 'The bottleneck (latent layer with fewer dimensions than input) forces the autoencoder to learn a compressed representation that captures the most important features needed for reconstruction.'
        }
      ]
    },
    {
      id: 'vaes',
      title: 'Variational Autoencoders',
      duration: '65 min',
      concepts: ['probabilistic encoding', 'KL divergence', 'reparameterization trick', 'latent sampling'],
      content: [
        {
          type: 'heading',
          text: 'Autoencoders That Generate'
        },
        {
          type: 'text',
          text: 'VAEs transform autoencoders into generative models. Instead of encoding to a single point, they encode to a probability distribution. This structured latent space allows sampling and generation of new, realistic data.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">VAE: Encoding to Distribution</text>
            <rect x="40" y="70" width="80" height="80" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="80" y="115" text-anchor="middle" font-size="10" fill="#4f46e5">Encoder</text>
            <rect x="160" y="60" width="50" height="40" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="185" y="85" text-anchor="middle" font-size="9" fill="#1d4ed8">μ</text>
            <rect x="160" y="120" width="50" height="40" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="185" y="145" text-anchor="middle" font-size="9" fill="#15803d">σ</text>
            <line x1="120" y1="95" x2="160" y2="80" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="120" y1="125" x2="160" y2="140" stroke="#6366f1" stroke-width="1.5"/>
            <ellipse cx="270" cy="110" rx="30" ry="40" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
            <text x="270" y="115" text-anchor="middle" font-size="9" fill="#92400e">z~N(μ,σ)</text>
            <line x1="210" y1="80" x2="245" y2="95" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="210" y1="140" x2="245" y2="125" stroke="#6366f1" stroke-width="1.5"/>
            <rect x="340" y="70" width="80" height="80" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="380" y="115" text-anchor="middle" font-size="10" fill="#7c3aed">Decoder</text>
            <line x1="300" y1="110" x2="340" y2="110" stroke="#6366f1" stroke-width="1.5"/>
            <text x="270" y="180" text-anchor="middle" font-size="10" fill="#64748b">Sample from distribution</text>
            <text x="270" y="200" text-anchor="middle" font-size="9" fill="#64748b">z = μ + σ × ε, where ε ~ N(0,1)</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'The VAE Loss Function'
        },
        {
          type: 'text',
          text: 'VAEs optimize two terms: reconstruction loss (how well we recreate input) and KL divergence (how close our learned distribution is to a standard normal).'
        },
        {
          type: 'formula',
          latex: '\\mathcal{L} = \\mathbb{E}[\\log p(x|z)] - D_{KL}(q(z|x) || p(z))'
        },
        {
          type: 'text',
          text: 'The KL term acts as a regularizer, preventing the encoder from just memorizing data. It pushes the latent space toward a structured, smooth distribution.'
        },
        {
          type: 'formula',
          latex: 'D_{KL} = -\\frac{1}{2}\\sum_{j=1}^{J}(1 + \\log(\\sigma_j^2) - \\mu_j^2 - \\sigma_j^2)'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

vae = VAE()
x = torch.rand(32, 784)
recon, mu, logvar = vae(x)
loss = vae_loss(recon, x, mu, logvar)
print(f"VAE loss: {loss.item():.2f}")`
        },
        {
          type: 'subheading',
          text: 'The Reparameterization Trick'
        },
        {
          type: 'text',
          text: 'Sampling is not differentiable. The reparameterization trick rewrites z = μ + σ×ε where ε~N(0,1). Now gradients can flow through μ and σ while randomness comes from ε.'
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'VAE Applications',
          text: 'VAEs excel at learning smooth latent spaces useful for interpolation, disentanglement, and semi-supervised learning. They produce blurrier images than GANs but have stable training.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the purpose of the KL divergence term in the VAE loss?',
          options: [
            'To improve reconstruction quality',
            'To regularize the latent space toward a standard normal distribution',
            'To speed up training',
            'To reduce model size'
          ],
          correct: 1,
          explanation: 'KL divergence pushes the learned latent distribution q(z|x) toward the prior p(z)=N(0,1). This creates a smooth, structured latent space where we can sample and interpolate meaningfully.'
        }
      ]
    },
    {
      id: 'gan-fundamentals',
      title: 'GAN Fundamentals',
      duration: '60 min',
      concepts: ['generator', 'discriminator', 'adversarial training', 'minimax game'],
      content: [
        {
          type: 'heading',
          text: 'Generative Adversarial Networks'
        },
        {
          type: 'text',
          text: 'GANs pit two networks against each other: a Generator that creates fake data and a Discriminator that tries to distinguish real from fake. Through this adversarial game, the Generator learns to produce increasingly realistic outputs.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowGAN" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">GAN Architecture</text>
            <rect x="30" y="80" width="60" height="50" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="60" y="110" text-anchor="middle" font-size="9" fill="#92400e">Noise z</text>
            <rect x="120" y="70" width="100" height="70" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="170" y="110" text-anchor="middle" font-size="11" font-weight="bold" fill="#4f46e5">Generator</text>
            <rect x="250" y="80" width="60" height="50" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="280" y="105" text-anchor="middle" font-size="9" fill="#1d4ed8">Fake</text>
            <text x="280" y="118" text-anchor="middle" font-size="9" fill="#1d4ed8">Image</text>
            <rect x="340" y="50" width="100" height="110" fill="#dcfce7" stroke="#22c55e" stroke-width="2" rx="6"/>
            <text x="390" y="110" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">Discriminator</text>
            <rect x="250" y="160" width="60" height="35" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="280" y="182" text-anchor="middle" font-size="9" fill="#7c3aed">Real Data</text>
            <line x1="90" y1="105" x2="120" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowGAN)"/>
            <line x1="220" y1="105" x2="250" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowGAN)"/>
            <line x1="310" y1="105" x2="340" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowGAN)"/>
            <line x1="310" y1="178" x2="340" y2="140" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowGAN)"/>
            <text x="460" y="85" font-size="10" fill="#22c55e">Real?</text>
            <text x="460" y="135" font-size="10" fill="#ef4444">Fake?</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'The Minimax Objective'
        },
        {
          type: 'text',
          text: 'The Generator minimizes while the Discriminator maximizes the same objective:'
        },
        {
          type: 'formula',
          latex: '\\min_G \\max_D \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]'
        },
        {
          type: 'text',
          text: 'D(x) is the probability that x is real. The Discriminator wants D(real)→1 and D(fake)→0. The Generator wants D(G(z))→1 (fool the Discriminator).'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

latent_dim = 100
G = Generator(latent_dim)
D = Discriminator()

z = torch.randn(32, latent_dim)
fake_images = G(z)
d_output = D(fake_images)
print(f"Generated images shape: {fake_images.shape}")
print(f"Discriminator output shape: {d_output.shape}")`
        },
        {
          type: 'subheading',
          text: 'Training Loop'
        },
        {
          type: 'code',
          language: 'python',
          code: `def train_gan(G, D, dataloader, epochs, latent_dim, device):
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            d_optimizer.zero_grad()
            d_real = D(real_images)
            d_real_loss = criterion(d_real, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = G(z)
            d_fake = D(fake_images.detach())
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            d_fake = D(fake_images)
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            g_optimizer.step()`
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'Training Instability',
          text: 'GANs are notoriously hard to train. Mode collapse (Generator produces limited variety), vanishing gradients, and oscillating losses are common. Many techniques exist to stabilize training.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the Generators goal in a GAN?',
          options: [
            'Minimize the probability of real images being classified as real',
            'Generate images that fool the Discriminator into classifying them as real',
            'Maximize reconstruction loss',
            'Classify images accurately'
          ],
          correct: 1,
          explanation: 'The Generator wants to produce fake images that the Discriminator believes are real. It maximizes D(G(z))—the probability the Discriminator assigns "real" to generated images.'
        }
      ]
    },
    {
      id: 'gan-training',
      title: 'GAN Training and Challenges',
      duration: '55 min',
      concepts: ['mode collapse', 'training stability', 'WGAN', 'spectral normalization'],
      content: [
        {
          type: 'heading',
          text: 'Stabilizing GAN Training'
        },
        {
          type: 'text',
          text: 'GANs are powerful but finicky. Mode collapse, vanishing gradients, and training instability are common. Understanding these issues and their solutions is essential for successful GAN training.'
        },
        {
          type: 'subheading',
          text: 'Common Problems'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Mode Collapse</text>
            <text x="125" y="50" text-anchor="middle" font-size="11" fill="#64748b">Ideal Output</text>
            <circle cx="60" cy="100" r="15" fill="#6366f1"/>
            <circle cx="100" cy="80" r="15" fill="#22c55e"/>
            <circle cx="140" cy="110" r="15" fill="#f59e0b"/>
            <circle cx="180" cy="90" r="15" fill="#a855f7"/>
            <text x="375" y="50" text-anchor="middle" font-size="11" fill="#64748b">Mode Collapse</text>
            <circle cx="340" cy="95" r="15" fill="#ef4444"/>
            <circle cx="360" cy="95" r="15" fill="#ef4444"/>
            <circle cx="380" cy="95" r="15" fill="#ef4444"/>
            <circle cx="400" cy="95" r="15" fill="#ef4444"/>
            <text x="125" y="150" text-anchor="middle" font-size="9" fill="#64748b">Diverse outputs</text>
            <text x="375" y="150" text-anchor="middle" font-size="9" fill="#64748b">Same output repeated</text>
          </svg>`
        },
        {
          type: 'table',
          headers: ['Problem', 'Symptom', 'Cause'],
          rows: [
            ['Mode Collapse', 'Generator produces limited variety', 'G finds one output that fools D'],
            ['Vanishing Gradients', 'G stops improving', 'D becomes too good too fast'],
            ['Oscillation', 'Losses fluctuate wildly', 'G and D not reaching equilibrium'],
            ['Non-convergence', 'No improvement over time', 'Architecture or hyperparameter issues']
          ]
        },
        {
          type: 'subheading',
          text: 'Wasserstein GAN (WGAN)'
        },
        {
          type: 'text',
          text: 'WGAN uses the Wasserstein distance instead of JS divergence, providing meaningful gradients even when distributions do not overlap. The critic (not discriminator) outputs a score, not a probability.'
        },
        {
          type: 'formula',
          latex: 'W(P_r, P_g) = \\sup_{||f||_L \\leq 1} \\mathbb{E}_{x \\sim P_r}[f(x)] - \\mathbb{E}_{x \\sim P_g}[f(x)]'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn

class WGANCritic(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

def wgan_critic_loss(critic, real, fake):
    return -(torch.mean(critic(real)) - torch.mean(critic(fake)))

def wgan_generator_loss(critic, fake):
    return -torch.mean(critic(fake))

def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty`
        },
        {
          type: 'subheading',
          text: 'Training Tips'
        },
        {
          type: 'text',
          text: '**Architecture**: Use LeakyReLU (not ReLU) in discriminator. Use BatchNorm in generator (not discriminator for WGAN). Use strided convolutions instead of pooling.'
        },
        {
          type: 'text',
          text: '**Training**: Train D more than G (especially early). Use low learning rates (0.0002). Use Adam with β1=0.5. Label smoothing can help (real=0.9 instead of 1.0).'
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Modern GANs',
          text: 'StyleGAN, BigGAN, and StyleGAN2 achieve remarkable image quality through progressive growing, style-based generation, and careful regularization. They build on these fundamental stabilization techniques.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is mode collapse in GANs?',
          options: [
            'When the discriminator becomes too powerful',
            'When the generator produces only a limited variety of outputs',
            'When training takes too long',
            'When the loss becomes negative'
          ],
          correct: 1,
          explanation: 'Mode collapse occurs when the Generator finds a few outputs that consistently fool the Discriminator and stops exploring other modes of the data distribution, resulting in low output diversity.'
        }
      ]
    },
    {
      id: 'diffusion-models',
      title: 'Diffusion Models',
      duration: '65 min',
      concepts: ['forward process', 'reverse process', 'denoising', 'score matching'],
      content: [
        {
          type: 'heading',
          text: 'The New Kings of Image Generation'
        },
        {
          type: 'text',
          text: 'Diffusion models have surpassed GANs in image quality and diversity. They work by learning to reverse a gradual noising process: add noise step by step, then learn to remove it. DALL-E 2, Stable Diffusion, and Midjourney all use diffusion.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowDiff" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
              <marker id="arrowDiffRev" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#22c55e"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Diffusion Process</text>
            <rect x="30" y="60" width="60" height="60" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="60" y="95" text-anchor="middle" font-size="9" fill="#4f46e5">x₀</text>
            <text x="60" y="135" text-anchor="middle" font-size="8" fill="#64748b">Clean</text>
            <rect x="130" y="60" width="60" height="60" fill="#dbeafe" stroke="#3b82f6" rx="4">
              <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
            </rect>
            <text x="160" y="95" text-anchor="middle" font-size="9" fill="#1d4ed8">x_t</text>
            <text x="160" y="135" text-anchor="middle" font-size="8" fill="#64748b">Noisy</text>
            <rect x="230" y="60" width="60" height="60" fill="#fef3c7" stroke="#f59e0b" rx="4">
              <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
            </rect>
            <text x="260" y="95" text-anchor="middle" font-size="9" fill="#92400e">x_t+1</text>
            <text x="260" y="135" text-anchor="middle" font-size="8" fill="#64748b">Noisier</text>
            <rect x="330" y="60" width="60" height="60" fill="#fecaca" stroke="#ef4444" rx="4"/>
            <text x="360" y="95" text-anchor="middle" font-size="9" fill="#ef4444">x_T</text>
            <text x="360" y="135" text-anchor="middle" font-size="8" fill="#64748b">Noise</text>
            <line x1="90" y1="75" x2="130" y2="75" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowDiff)"/>
            <line x1="190" y1="75" x2="230" y2="75" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowDiff)"/>
            <line x1="290" y1="75" x2="330" y2="75" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowDiff)"/>
            <text x="210" y="55" text-anchor="middle" font-size="9" fill="#6366f1">Forward (add noise)</text>
            <line x1="330" y1="105" x2="290" y2="105" stroke="#22c55e" stroke-width="2" marker-end="url(#arrowDiffRev)"/>
            <line x1="230" y1="105" x2="190" y2="105" stroke="#22c55e" stroke-width="2" marker-end="url(#arrowDiffRev)"/>
            <line x1="130" y1="105" x2="90" y2="105" stroke="#22c55e" stroke-width="2" marker-end="url(#arrowDiffRev)"/>
            <text x="210" y="165" text-anchor="middle" font-size="9" fill="#22c55e">Reverse (denoise) - LEARNED</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Forward Process (Adding Noise)'
        },
        {
          type: 'text',
          text: 'The forward process gradually adds Gaussian noise over T steps until the data becomes pure noise:'
        },
        {
          type: 'formula',
          latex: 'q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t}x_{t-1}, \\beta_t I)'
        },
        {
          type: 'text',
          text: 'We can jump directly to any timestep t using the cumulative schedule:'
        },
        {
          type: 'formula',
          latex: 'q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t}x_0, (1-\\bar{\\alpha}_t)I)'
        },
        {
          type: 'subheading',
          text: 'Reverse Process (Denoising)'
        },
        {
          type: 'text',
          text: 'The model learns to reverse this process—predicting the noise added at each step, then subtracting it:'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn

class SimpleDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,))

class NoisePredictor(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )
        self.time_embed = nn.Embedding(1000, 64)

    def forward(self, x, t):
        return self.net(x)`
        },
        {
          type: 'subheading',
          text: 'Training Objective'
        },
        {
          type: 'text',
          text: 'The model predicts the noise ε that was added. Training minimizes the MSE between predicted and actual noise:'
        },
        {
          type: 'formula',
          latex: '\\mathcal{L} = \\mathbb{E}_{x_0, \\epsilon, t}[||\\epsilon - \\epsilon_\\theta(x_t, t)||^2]'
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Why Diffusion Works',
          text: 'Diffusion models have stable training (no adversarial dynamics), excellent mode coverage (no collapse), and high sample quality. The trade-off is slow generation (many denoising steps), though techniques like DDIM and consistency models address this.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does the neural network learn in a diffusion model?',
          options: [
            'To generate images directly from noise',
            'To predict the noise added at each timestep',
            'To classify real vs fake images',
            'To compress images'
          ],
          correct: 1,
          explanation: 'The model learns to predict the noise ε that was added to create x_t from x_{t-1}. By predicting and subtracting this noise step by step, it gradually recovers the clean image from pure noise.'
        }
      ]
    },
    {
      id: 'finetuning-llms',
      title: 'Fine-tuning LLMs',
      duration: '60 min',
      concepts: ['transfer learning', 'LoRA', 'PEFT', 'instruction tuning'],
      content: [
        {
          type: 'heading',
          text: 'Adapting Large Language Models'
        },
        {
          type: 'text',
          text: 'Fine-tuning adapts pre-trained LLMs to specific tasks or domains. Full fine-tuning is expensive, so parameter-efficient methods like LoRA enable adaptation with minimal resources.'
        },
        {
          type: 'subheading',
          text: 'Fine-tuning Approaches'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Fine-tuning Strategies</text>
            <rect x="30" y="50" width="130" height="100" fill="#fecaca" stroke="#ef4444" rx="6"/>
            <text x="95" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#dc2626">Full Fine-tune</text>
            <text x="95" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Update ALL params</text>
            <text x="95" y="115" text-anchor="middle" font-size="8" fill="#64748b">7B params = 28GB</text>
            <text x="95" y="135" text-anchor="middle" font-size="8" fill="#64748b">Expensive, powerful</text>
            <rect x="180" y="50" width="130" height="100" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="245" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">LoRA</text>
            <text x="245" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Low-rank adapters</text>
            <text x="245" y="115" text-anchor="middle" font-size="8" fill="#64748b">~0.1% of params</text>
            <text x="245" y="135" text-anchor="middle" font-size="8" fill="#64748b">Efficient, effective</text>
            <rect x="330" y="50" width="130" height="100" fill="#dbeafe" stroke="#3b82f6" rx="6"/>
            <text x="395" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1d4ed8">Prompt Tuning</text>
            <text x="395" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Learnable prompts</text>
            <text x="395" y="115" text-anchor="middle" font-size="8" fill="#64748b">~0.01% of params</text>
            <text x="395" y="135" text-anchor="middle" font-size="8" fill="#64748b">Very efficient</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'LoRA: Low-Rank Adaptation'
        },
        {
          type: 'text',
          text: 'LoRA freezes the pre-trained weights and adds small trainable matrices. Instead of updating W directly, it learns a low-rank decomposition ΔW = BA:'
        },
        {
          type: 'formula',
          latex: 'W_{new} = W_{frozen} + BA, \\quad B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times k}'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in peft_model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")`
        },
        {
          type: 'subheading',
          text: 'Instruction Tuning'
        },
        {
          type: 'text',
          text: 'Instruction tuning teaches models to follow instructions by training on (instruction, response) pairs. This transforms a completion model into an assistant.'
        },
        {
          type: 'code',
          language: 'python',
          code: `from datasets import load_dataset
from transformers import Trainer, TrainingArguments

dataset = load_dataset("databricks/dolly-15k")

def format_instruction(example):
    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]

    if context:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Context:\\n{context}\\n\\n### Response:\\n{response}"
    else:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}"

    return {"text": prompt}

formatted_dataset = dataset.map(format_instruction)

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True
)`
        },
        {
          type: 'table',
          headers: ['Method', 'Trainable Params', 'Memory', 'Quality'],
          rows: [
            ['Full fine-tune', '100%', 'Very high', 'Best'],
            ['LoRA', '0.1-1%', 'Low', 'Near full'],
            ['QLoRA', '0.1-1%', 'Very low', 'Good'],
            ['Prompt tuning', '0.01%', 'Minimal', 'Limited']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'When to Fine-tune',
          text: 'Fine-tune when: you have domain-specific data, need consistent style/format, or the base model struggles with your task. Consider prompt engineering first—it is cheaper and often sufficient.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What makes LoRA parameter-efficient?',
          options: [
            'It uses smaller models',
            'It trains only low-rank adapter matrices while freezing original weights',
            'It reduces the number of layers',
            'It uses quantization'
          ],
          correct: 1,
          explanation: 'LoRA freezes the pre-trained weights and only trains small low-rank matrices (BA). With rank r=8 and d=4096, this is 8×4096×2 = 65K params vs 4096×4096 = 16M params per layer—a 250x reduction.'
        }
      ]
    }
  ]
}
