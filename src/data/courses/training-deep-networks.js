export const trainingdeepnetworks = {
  id: 'training-deep-networks',
  title: 'Training Deep Networks',
  description: 'Master the art and science of training deep neural networks - optimization algorithms, regularization techniques, and debugging strategies',
  category: 'Deep Learning',
  difficulty: 'Intermediate',
  duration: '6 hours',
  lessons: [
    {
      id: 'optimization-algorithms',
      title: 'Optimization Algorithms',
      duration: '55 min',
      concepts: ['SGD', 'Momentum', 'Adam', 'Learning Rate'],
      content: [
        {
          type: 'heading',
          content: 'Beyond Vanilla Gradient Descent'
        },
        {
          type: 'text',
          content: `Vanilla gradient descent updates weights by subtracting the gradient scaled by learning rate:

w = w - η∇L

This simple approach has problems:
- Same learning rate for all parameters
- Gets stuck in ravines (narrow valleys)
- Slow convergence on flat surfaces
- Sensitive to learning rate choice

Modern optimizers fix these issues with clever modifications.`
        },
        {
          type: 'heading',
          content: 'Stochastic Gradient Descent (SGD)'
        },
        {
          type: 'text',
          content: `Instead of computing gradients on the entire dataset (batch gradient descent), SGD uses mini-batches:

- **Batch GD**: All data, one update per epoch (slow, stable)
- **SGD**: One sample, many updates (fast, noisy)
- **Mini-batch SGD**: Small batches, best of both worlds

Mini-batch SGD (typically batch size 32-256) is the standard in deep learning.`
        },
        {
          type: 'heading',
          content: 'Momentum'
        },
        {
          type: 'text',
          content: `Momentum accelerates SGD by adding "velocity" - remembering the direction of previous gradients.

Imagine a ball rolling down a hill. It builds up speed and can roll through small bumps without stopping.`
        },
        {
          type: 'formula',
          content: 'v = βv + (1-β)∇L\nw = w - ηv'
        },
        {
          type: 'text',
          content: `Where β (typically 0.9) controls how much history to remember. Higher β = more momentum.

**Benefits**:
- Accelerates through flat regions
- Dampens oscillations in ravines
- Can escape shallow local minima`
        },
        {
          type: 'visualization',
          title: 'SGD vs Momentum',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- SGD path -->
            <g transform="translate(30,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="600">SGD (Oscillates)</text>
              <rect x="10" y="10" width="120" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Contours -->
              <ellipse cx="90" cy="60" rx="15" ry="40" fill="none" stroke="#e2e8f0" stroke-width="1"/>
              <ellipse cx="90" cy="60" rx="25" ry="45" fill="none" stroke="#e2e8f0" stroke-width="1"/>

              <!-- Zig-zag path -->
              <path d="M30,30 L45,70 L35,50 L50,80 L40,60 L55,90 L45,70 L60,100 L50,80 L65,100 L55,85 L70,95 L60,85 L75,90 L65,80 L80,85 L75,75 L85,70 L80,65 L90,60"
                    fill="none" stroke="#ef4444" stroke-width="1.5"/>
              <circle cx="30" cy="30" r="3" fill="#ef4444"/>
              <circle cx="90" cy="60" r="3" fill="#10b981"/>
            </g>

            <!-- Momentum path -->
            <g transform="translate(210,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">Momentum (Smooth)</text>
              <rect x="10" y="10" width="120" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Contours -->
              <ellipse cx="90" cy="60" rx="15" ry="40" fill="none" stroke="#e2e8f0" stroke-width="1"/>
              <ellipse cx="90" cy="60" rx="25" ry="45" fill="none" stroke="#e2e8f0" stroke-width="1"/>

              <!-- Smooth path -->
              <path d="M30,30 Q45,50 50,55 Q60,60 70,60 Q80,60 90,60"
                    fill="none" stroke="#10b981" stroke-width="1.5"/>
              <circle cx="30" cy="30" r="3" fill="#10b981"/>
              <circle cx="90" cy="60" r="3" fill="#10b981"/>
            </g>

            <!-- Legend -->
            <text x="200" y="175" text-anchor="middle" font-size="9" fill="#64748b">Momentum smooths the path toward the minimum</text>
          </svg>`,
          caption: 'Momentum reduces oscillations and accelerates convergence'
        },
        {
          type: 'heading',
          content: 'RMSprop'
        },
        {
          type: 'text',
          content: `RMSprop adapts the learning rate for each parameter based on the magnitude of recent gradients.`
        },
        {
          type: 'formula',
          content: 's = βs + (1-β)(∇L)²\nw = w - η · ∇L / √(s + ε)'
        },
        {
          type: 'text',
          content: `Parameters with large gradients get smaller effective learning rates. Parameters with small gradients get larger effective learning rates.

This helps when different parameters need different learning rates.`
        },
        {
          type: 'heading',
          content: 'Adam (Adaptive Moment Estimation)'
        },
        {
          type: 'text',
          content: `Adam combines momentum AND adaptive learning rates. It's the default choice for most deep learning.`
        },
        {
          type: 'formula',
          content: 'm = β₁m + (1-β₁)∇L        (momentum)\ns = β₂s + (1-β₂)(∇L)²     (RMSprop)\nm̂ = m/(1-β₁ᵗ)             (bias correction)\nŝ = s/(1-β₂ᵗ)              (bias correction)\nw = w - η · m̂ / √(ŝ + ε)'
        },
        {
          type: 'text',
          content: `**Default hyperparameters** (usually work well):
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSprop)
- ε = 1e-8 (numerical stability)
- η = 0.001 (learning rate)`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch
import torch.optim as optim

model = MyNetwork()

optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()`
        },
        {
          type: 'heading',
          content: 'Comparing Optimizers'
        },
        {
          type: 'table',
          headers: ['Optimizer', 'When to Use', 'Key Feature'],
          rows: [
            ['SGD', 'Simple problems, lots of data', 'Simple, proven for generalization'],
            ['SGD + Momentum', 'General training', 'Faster than vanilla SGD'],
            ['Adam', 'Default choice', 'Adaptive + momentum, easy to tune'],
            ['AdamW', 'With weight decay', 'Proper L2 regularization'],
            ['SGD (fine-tune)', 'After Adam plateau', 'Often finds better minima']
          ],
          caption: 'Optimizer selection guide'
        },
        {
          type: 'keypoints',
          points: [
            'Mini-batch SGD balances speed and stability',
            'Momentum accelerates through flat regions and dampens oscillations',
            'Adaptive methods (RMSprop, Adam) adjust learning rates per parameter',
            'Adam is the default choice - combines momentum and adaptation',
            'Start with Adam, consider SGD for fine-tuning'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What problem does momentum solve?',
          options: [
            'Overfitting',
            'Vanishing gradients',
            'Oscillations in narrow valleys',
            'Memory usage'
          ],
          correct: 2,
          explanation: 'Momentum helps smooth out oscillations when gradients point in different directions across dimensions. It accumulates velocity in consistent directions and dampens oscillations.'
        },
        {
          type: 'multiple-choice',
          question: 'What does Adam combine?',
          options: [
            'Dropout and batch normalization',
            'Momentum and adaptive learning rates',
            'L1 and L2 regularization',
            'ReLU and sigmoid'
          ],
          correct: 1,
          explanation: 'Adam combines momentum (from SGD with momentum) and adaptive per-parameter learning rates (from RMSprop), plus bias correction.'
        }
      ]
    },
    {
      id: 'weight-initialization',
      title: 'Weight Initialization',
      duration: '40 min',
      concepts: ['Xavier', 'He', 'Initialization Effects'],
      content: [
        {
          type: 'heading',
          content: 'Why Initialization Matters'
        },
        {
          type: 'text',
          content: `Poor initialization can doom training before it starts:

**All zeros**: Every neuron computes the same thing, learns the same thing → "symmetry problem"

**Too small**: Activations shrink to zero through layers → vanishing gradients

**Too large**: Activations explode through layers → exploding gradients, NaN losses

Good initialization keeps activations and gradients in a reasonable range through all layers.`
        },
        {
          type: 'heading',
          content: 'The Variance Perspective'
        },
        {
          type: 'text',
          content: `Each layer transforms its input: z = Wx. If weights have variance σ² and input has n elements, then:

Var(z) = n × σ² × Var(x)

If n × σ² ≠ 1, variance grows or shrinks with depth. Deep networks need balanced initialization.`
        },
        {
          type: 'heading',
          content: 'Xavier/Glorot Initialization'
        },
        {
          type: 'text',
          content: `For layers with tanh or sigmoid activations:`
        },
        {
          type: 'formula',
          content: 'W ~ N(0, σ²)  where σ² = 2/(nᵢₙ + nₒᵤₜ)\nor\nW ~ Uniform(-a, a)  where a = √(6/(nᵢₙ + nₒᵤₜ))'
        },
        {
          type: 'text',
          content: `This keeps variance roughly constant through forward and backward passes.`
        },
        {
          type: 'heading',
          content: 'He (Kaiming) Initialization'
        },
        {
          type: 'text',
          content: `ReLU kills half the signal (negative values become 0), so we need more variance:`
        },
        {
          type: 'formula',
          content: 'W ~ N(0, σ²)  where σ² = 2/nᵢₙ'
        },
        {
          type: 'text',
          content: `He initialization is the default for networks with ReLU activations.`
        },
        {
          type: 'visualization',
          title: 'Activation Distribution Through Layers',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Poor init -->
            <g transform="translate(20,30)">
              <text x="75" y="0" text-anchor="middle" font-size="9" fill="#ef4444" font-weight="500">Poor Initialization</text>

              <rect x="0" y="10" width="50" height="60" fill="white" stroke="#e2e8f0" rx="2"/>
              <rect x="5" y="20" width="40" height="40" fill="#3b82f6" opacity="0.7"/>
              <text x="25" y="82" text-anchor="middle" font-size="7" fill="#64748b">L1</text>

              <rect x="60" y="10" width="50" height="60" fill="white" stroke="#e2e8f0" rx="2"/>
              <rect x="65" y="30" width="40" height="20" fill="#3b82f6" opacity="0.5"/>
              <text x="85" y="82" text-anchor="middle" font-size="7" fill="#64748b">L2</text>

              <rect x="120" y="10" width="50" height="60" fill="white" stroke="#e2e8f0" rx="2"/>
              <rect x="125" y="37" width="40" height="6" fill="#3b82f6" opacity="0.3"/>
              <text x="145" y="82" text-anchor="middle" font-size="7" fill="#64748b">L3</text>

              <text x="75" y="100" text-anchor="middle" font-size="8" fill="#ef4444">Activations vanish!</text>
            </g>

            <!-- Good init -->
            <g transform="translate(210,30)">
              <text x="75" y="0" text-anchor="middle" font-size="9" fill="#10b981" font-weight="500">He Initialization</text>

              <rect x="0" y="10" width="50" height="60" fill="white" stroke="#e2e8f0" rx="2"/>
              <rect x="5" y="20" width="40" height="40" fill="#10b981" opacity="0.7"/>
              <text x="25" y="82" text-anchor="middle" font-size="7" fill="#64748b">L1</text>

              <rect x="60" y="10" width="50" height="60" fill="white" stroke="#e2e8f0" rx="2"/>
              <rect x="65" y="22" width="40" height="36" fill="#10b981" opacity="0.7"/>
              <text x="85" y="82" text-anchor="middle" font-size="7" fill="#64748b">L2</text>

              <rect x="120" y="10" width="50" height="60" fill="white" stroke="#e2e8f0" rx="2"/>
              <rect x="125" y="24" width="40" height="32" fill="#10b981" opacity="0.7"/>
              <text x="145" y="82" text-anchor="middle" font-size="7" fill="#64748b">L3</text>

              <text x="75" y="100" text-anchor="middle" font-size="8" fill="#10b981">Stable activations!</text>
            </g>

            <!-- Legend -->
            <text x="200" y="150" text-anchor="middle" font-size="8" fill="#64748b">Bars show activation variance at each layer</text>
            <text x="200" y="170" text-anchor="middle" font-size="8" fill="#64748b">Good initialization keeps variance stable through depth</text>
          </svg>`,
          caption: 'He initialization maintains stable activation variance through deep networks'
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn

layer = nn.Linear(256, 128)

nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)`
        },
        {
          type: 'table',
          headers: ['Activation', 'Recommended Init', 'Formula'],
          rows: [
            ['Sigmoid, Tanh', 'Xavier/Glorot', 'σ² = 2/(n_in + n_out)'],
            ['ReLU, Leaky ReLU', 'He/Kaiming', 'σ² = 2/n_in'],
            ['SELU', 'LeCun', 'σ² = 1/n_in']
          ],
          caption: 'Match initialization to activation function'
        },
        {
          type: 'keypoints',
          points: [
            'Never initialize all weights to zero (symmetry problem)',
            'Xavier for sigmoid/tanh, He for ReLU',
            'Good initialization maintains stable gradients through depth',
            'Modern frameworks often set good defaults automatically',
            'Check activation statistics when debugging deep networks'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why is initializing all weights to zero a problem?',
          options: [
            'Training is too slow',
            'All neurons learn the same thing (symmetry)',
            'Memory usage is too high',
            'Gradients explode'
          ],
          correct: 1,
          explanation: 'With zero weights, all neurons in a layer produce identical outputs and receive identical gradients. They update identically and remain identical forever - the "symmetry problem."'
        },
        {
          type: 'multiple-choice',
          question: 'Which initialization should you use for ReLU networks?',
          options: [
            'All zeros',
            'Xavier/Glorot',
            'He/Kaiming',
            'Random uniform'
          ],
          correct: 2,
          explanation: 'He initialization is designed for ReLU, which kills half the signal (negative values become 0). It uses σ² = 2/n_in to compensate for this signal loss.'
        }
      ]
    },
    {
      id: 'batch-normalization',
      title: 'Batch Normalization',
      duration: '50 min',
      concepts: ['BatchNorm', 'Internal Covariate Shift', 'Inference'],
      content: [
        {
          type: 'heading',
          content: 'The Problem: Internal Covariate Shift'
        },
        {
          type: 'text',
          content: `As training progresses, the distribution of each layer's inputs changes. Each layer must constantly adapt to these shifting distributions. This is called **internal covariate shift**.

Batch normalization normalizes layer inputs, making training faster and more stable.`
        },
        {
          type: 'heading',
          content: 'How BatchNorm Works'
        },
        {
          type: 'text',
          content: `For a mini-batch of activations, BatchNorm:

1. Compute batch mean: μ = (1/m) Σxᵢ
2. Compute batch variance: σ² = (1/m) Σ(xᵢ - μ)²
3. Normalize: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
4. Scale and shift: yᵢ = γx̂ᵢ + β

The learnable parameters γ and β allow the network to undo normalization if needed.`
        },
        {
          type: 'visualization',
          title: 'Batch Normalization Pipeline',
          svg: `<svg viewBox="0 0 450 150" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="150" fill="#f8fafc"/>

            <!-- Input -->
            <rect x="20" y="50" width="60" height="50" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" rx="4"/>
            <text x="50" y="80" text-anchor="middle" font-size="10" fill="#475569">x</text>
            <text x="50" y="115" text-anchor="middle" font-size="8" fill="#64748b">Input</text>

            <!-- Arrow -->
            <path d="M85,75 L105,75" stroke="#64748b" stroke-width="1.5" marker-end="url(#bnarrow)"/>

            <!-- Normalize -->
            <rect x="110" y="50" width="80" height="50" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="4"/>
            <text x="150" y="72" text-anchor="middle" font-size="9" fill="#1e40af">Normalize</text>
            <text x="150" y="88" text-anchor="middle" font-size="8" fill="#3b82f6">(x-μ)/σ</text>
            <text x="150" y="115" text-anchor="middle" font-size="8" fill="#64748b">μ, σ from batch</text>

            <!-- Arrow -->
            <path d="M195,75 L215,75" stroke="#64748b" stroke-width="1.5" marker-end="url(#bnarrow)"/>

            <!-- Scale & Shift -->
            <rect x="220" y="50" width="80" height="50" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="4"/>
            <text x="260" y="72" text-anchor="middle" font-size="9" fill="#065f46">Scale & Shift</text>
            <text x="260" y="88" text-anchor="middle" font-size="8" fill="#10b981">γx̂ + β</text>
            <text x="260" y="115" text-anchor="middle" font-size="8" fill="#64748b">Learnable</text>

            <!-- Arrow -->
            <path d="M305,75 L325,75" stroke="#64748b" stroke-width="1.5" marker-end="url(#bnarrow)"/>

            <!-- Output -->
            <rect x="330" y="50" width="60" height="50" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="4"/>
            <text x="360" y="80" text-anchor="middle" font-size="10" fill="#dc2626">y</text>
            <text x="360" y="115" text-anchor="middle" font-size="8" fill="#64748b">Output</text>

            <defs>
              <marker id="bnarrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'BatchNorm normalizes then allows the network to learn optimal scale and shift'
        },
        {
          type: 'heading',
          content: 'Training vs Inference'
        },
        {
          type: 'text',
          content: `**Training**: Use batch statistics (μ, σ from current mini-batch)

**Inference**: Use running averages computed during training
- No mini-batch at inference time (might be single sample)
- Running mean and variance are tracked with exponential moving average

This is why you must call \`model.eval()\` before inference!`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

model.train()
for batch in train_loader:
    output = model(batch)

model.eval()
with torch.no_grad():
    predictions = model(test_data)`
        },
        {
          type: 'heading',
          content: 'Where to Place BatchNorm?'
        },
        {
          type: 'text',
          content: `**Original paper**: After linear/conv, before activation
Conv → BatchNorm → ReLU

**Alternative (often used)**: After activation
Conv → ReLU → BatchNorm

Both work. The original placement is more common. Be consistent throughout your network.`
        },
        {
          type: 'heading',
          content: 'Benefits of BatchNorm'
        },
        {
          type: 'text',
          content: `**Faster training**: Allows higher learning rates
**Better gradients**: Reduces vanishing/exploding gradients
**Regularization**: Adds noise (batch statistics vary), slight regularization effect
**Less sensitive**: More robust to weight initialization

BatchNorm is nearly universal in modern deep networks.`
        },
        {
          type: 'callout',
          variant: 'warning',
          content: 'BatchNorm behaves differently in training vs eval modes. Always call model.train() for training and model.eval() for inference. Forgetting this is a common bug!'
        },
        {
          type: 'heading',
          content: 'Layer Normalization Alternative'
        },
        {
          type: 'text',
          content: `BatchNorm normalizes across the batch. **Layer Normalization** normalizes across features for each sample:

- Better for RNNs and Transformers (sequence length varies)
- No difference between training and inference
- Works with batch size 1`
        },
        {
          type: 'keypoints',
          points: [
            'BatchNorm normalizes activations to stabilize training',
            'Uses batch statistics during training, running averages during inference',
            'Place after linear/conv layer, before or after activation',
            'Always call model.eval() before inference',
            'Use LayerNorm for transformers and variable-length sequences'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What statistics does BatchNorm use during inference?',
          options: [
            'Statistics from the current mini-batch',
            'Running averages computed during training',
            'Global dataset statistics',
            'No statistics - it is disabled'
          ],
          correct: 1,
          explanation: 'During inference, BatchNorm uses running mean and variance that were accumulated during training via exponential moving average. This ensures consistent behavior regardless of batch composition.'
        },
        {
          type: 'multiple-choice',
          question: 'Why are γ and β learnable parameters in BatchNorm?',
          options: [
            'To speed up computation',
            'To allow the network to undo normalization if needed',
            'To reduce memory usage',
            'To prevent overfitting'
          ],
          correct: 1,
          explanation: 'The learnable scale (γ) and shift (β) allow the network to recover the original distribution if that is optimal. The network can learn γ=σ and β=μ to undo the normalization completely.'
        }
      ]
    },
    {
      id: 'dropout-regularization',
      title: 'Dropout and Regularization',
      duration: '45 min',
      concepts: ['Dropout', 'L2 Regularization', 'Early Stopping'],
      content: [
        {
          type: 'heading',
          content: 'The Overfitting Problem'
        },
        {
          type: 'text',
          content: `Neural networks have millions of parameters - plenty of capacity to memorize training data instead of learning general patterns.

**Regularization** techniques prevent overfitting by constraining the model. Three main approaches:
1. **Explicit**: Add penalty to loss (L2 regularization)
2. **Implicit**: Modify training (dropout, early stopping)
3. **Data**: Augment training data`
        },
        {
          type: 'heading',
          content: 'Dropout'
        },
        {
          type: 'text',
          content: `During training, randomly "drop" (set to zero) a fraction of neurons. Each forward pass uses a different random subset of the network.

This prevents neurons from co-adapting - each neuron must learn to be useful on its own.

**Key insight**: Dropout is like training an ensemble of 2^n different networks (n = number of neurons) and averaging their predictions.`
        },
        {
          type: 'visualization',
          title: 'Dropout Effect',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Without dropout -->
            <g transform="translate(30,30)">
              <text x="60" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Without Dropout</text>

              <!-- Nodes -->
              <circle cx="20" cy="40" r="10" fill="#3b82f6"/>
              <circle cx="20" cy="80" r="10" fill="#3b82f6"/>
              <circle cx="60" cy="30" r="10" fill="#3b82f6"/>
              <circle cx="60" cy="60" r="10" fill="#3b82f6"/>
              <circle cx="60" cy="90" r="10" fill="#3b82f6"/>
              <circle cx="100" cy="60" r="10" fill="#3b82f6"/>

              <!-- All connections -->
              <g stroke="#94a3b8" stroke-width="1">
                <line x1="30" y1="40" x2="50" y2="30"/>
                <line x1="30" y1="40" x2="50" y2="60"/>
                <line x1="30" y1="40" x2="50" y2="90"/>
                <line x1="30" y1="80" x2="50" y2="30"/>
                <line x1="30" y1="80" x2="50" y2="60"/>
                <line x1="30" y1="80" x2="50" y2="90"/>
                <line x1="70" y1="30" x2="90" y2="60"/>
                <line x1="70" y1="60" x2="90" y2="60"/>
                <line x1="70" y1="90" x2="90" y2="60"/>
              </g>

              <text x="60" y="120" text-anchor="middle" font-size="8" fill="#64748b">All neurons active</text>
            </g>

            <!-- With dropout -->
            <g transform="translate(200,30)">
              <text x="60" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">With Dropout (p=0.5)</text>

              <!-- Active nodes -->
              <circle cx="20" cy="40" r="10" fill="#3b82f6"/>
              <circle cx="20" cy="80" r="10" fill="#3b82f6" opacity="0.2" stroke="#ef4444" stroke-width="2" stroke-dasharray="2"/>
              <circle cx="60" cy="30" r="10" fill="#3b82f6" opacity="0.2" stroke="#ef4444" stroke-width="2" stroke-dasharray="2"/>
              <circle cx="60" cy="60" r="10" fill="#3b82f6"/>
              <circle cx="60" cy="90" r="10" fill="#3b82f6"/>
              <circle cx="100" cy="60" r="10" fill="#3b82f6"/>

              <!-- Remaining connections -->
              <g stroke="#10b981" stroke-width="1.5">
                <line x1="30" y1="40" x2="50" y2="60"/>
                <line x1="30" y1="40" x2="50" y2="90"/>
                <line x1="70" y1="60" x2="90" y2="60"/>
                <line x1="70" y1="90" x2="90" y2="60"/>
              </g>

              <text x="60" y="120" text-anchor="middle" font-size="8" fill="#64748b">Random subset active</text>
            </g>

            <!-- Legend -->
            <g transform="translate(130,150)">
              <circle cx="0" cy="0" r="5" fill="#3b82f6" opacity="0.2" stroke="#ef4444" stroke-width="1" stroke-dasharray="2"/>
              <text x="10" y="4" font-size="8" fill="#64748b">Dropped neurons</text>
            </g>
          </svg>`,
          caption: 'Dropout randomly disables neurons during training'
        },
        {
          type: 'text',
          content: `**Training**: Apply dropout with probability p
**Inference**: Use all neurons, scale outputs by (1-p)

Or equivalently (inverted dropout): Scale training outputs by 1/(1-p), use all neurons at inference without scaling.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x`
        },
        {
          type: 'heading',
          content: 'L2 Regularization (Weight Decay)'
        },
        {
          type: 'text',
          content: `Add a penalty for large weights to the loss:`
        },
        {
          type: 'formula',
          content: 'L_total = L_original + λ Σ w²'
        },
        {
          type: 'text',
          content: `This encourages smaller weights, which leads to simpler models.

In optimizers, this is often implemented as "weight decay" - directly shrinking weights each update:`
        },
        {
          type: 'formula',
          content: 'w = w - η(∇L + λw) = (1 - ηλ)w - η∇L'
        },
        {
          type: 'code',
          language: 'python',
          content: `optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)`
        },
        {
          type: 'heading',
          content: 'Early Stopping'
        },
        {
          type: 'text',
          content: `Monitor validation loss during training. Stop when it stops improving.

Training loss keeps decreasing, but validation loss eventually increases (overfitting). Stop at the minimum validation loss.`
        },
        {
          type: 'code',
          language: 'python',
          content: `best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(torch.load('best_model.pt'))`
        },
        {
          type: 'heading',
          content: 'When to Use What'
        },
        {
          type: 'table',
          headers: ['Technique', 'When to Use', 'Typical Values'],
          rows: [
            ['Dropout', 'Dense layers, overfitting', 'p=0.2-0.5'],
            ['L2/Weight decay', 'Always a good default', 'λ=0.0001-0.01'],
            ['Early stopping', 'Always, monitor validation', 'patience=5-20'],
            ['Data augmentation', 'Images, limited data', 'Task-specific']
          ],
          caption: 'Regularization technique selection'
        },
        {
          type: 'keypoints',
          points: [
            'Dropout randomly zeros neurons, preventing co-adaptation',
            'Remember to call model.eval() to disable dropout at inference',
            'L2 regularization (weight decay) penalizes large weights',
            'Early stopping prevents overfitting by monitoring validation loss',
            'Combine multiple techniques for best results'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What should dropout probability be during inference?',
          options: [
            'Same as training',
            'Zero (dropout disabled)',
            'Doubled',
            'Random'
          ],
          correct: 1,
          explanation: 'During inference, dropout is disabled (all neurons active). With inverted dropout, no output scaling is needed. Call model.eval() to ensure dropout is disabled.'
        },
        {
          type: 'multiple-choice',
          question: 'What does early stopping use to decide when to stop?',
          options: [
            'Training loss',
            'Validation loss',
            'Number of epochs',
            'Learning rate'
          ],
          correct: 1,
          explanation: 'Early stopping monitors validation loss. When it stops improving (starts increasing), training is stopped to prevent overfitting. The model from the best validation epoch is kept.'
        }
      ]
    },
    {
      id: 'learning-rate-scheduling',
      title: 'Learning Rate Scheduling',
      duration: '40 min',
      concepts: ['Step Decay', 'Cosine Annealing', 'Warmup'],
      content: [
        {
          type: 'heading',
          content: 'Why Schedule Learning Rate?'
        },
        {
          type: 'text',
          content: `A fixed learning rate is a compromise:
- Too high: Training is unstable, may diverge
- Too low: Training is slow, may get stuck

**Learning rate scheduling** adjusts the rate during training:
- Start high for fast initial progress
- Reduce to fine-tune and stabilize`
        },
        {
          type: 'heading',
          content: 'Step Decay'
        },
        {
          type: 'text',
          content: `Reduce learning rate by a factor at specific epochs:`
        },
        {
          type: 'formula',
          content: 'lr = lr₀ × γ^(epoch // step_size)'
        },
        {
          type: 'text',
          content: `Example: Start at 0.1, multiply by 0.1 every 30 epochs
- Epochs 0-29: lr = 0.1
- Epochs 30-59: lr = 0.01
- Epochs 60+: lr = 0.001`
        },
        {
          type: 'visualization',
          title: 'Learning Rate Schedules',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="50" y1="160" x2="370" y2="160" stroke="#475569" stroke-width="1.5"/>
            <line x1="50" y1="160" x2="50" y2="30" stroke="#475569" stroke-width="1.5"/>
            <text x="210" y="185" text-anchor="middle" font-size="9" fill="#64748b">Epoch</text>
            <text x="30" y="95" font-size="9" fill="#64748b" transform="rotate(-90,30,95)">LR</text>

            <!-- Step decay (blue) -->
            <path d="M60,50 L140,50 L140,90 L220,90 L220,130 L300,130 L300,150 L360,150" fill="none" stroke="#3b82f6" stroke-width="2"/>

            <!-- Cosine (green) -->
            <path d="M60,50 Q120,50 150,70 Q200,110 250,135 Q300,150 360,155" fill="none" stroke="#10b981" stroke-width="2"/>

            <!-- Warmup + decay (orange) -->
            <path d="M60,150 L100,50 L100,50 Q180,50 230,100 Q280,140 360,155" fill="none" stroke="#f59e0b" stroke-width="2"/>

            <!-- Legend -->
            <g transform="translate(100,20)">
              <line x1="0" y1="0" x2="20" y2="0" stroke="#3b82f6" stroke-width="2"/>
              <text x="25" y="4" font-size="8" fill="#475569">Step Decay</text>

              <line x1="100" y1="0" x2="120" y2="0" stroke="#10b981" stroke-width="2"/>
              <text x="125" y="4" font-size="8" fill="#475569">Cosine</text>

              <line x1="180" y1="0" x2="200" y2="0" stroke="#f59e0b" stroke-width="2"/>
              <text x="205" y="4" font-size="8" fill="#475569">Warmup+Cosine</text>
            </g>
          </svg>`,
          caption: 'Different schedules for reducing learning rate over training'
        },
        {
          type: 'heading',
          content: 'Cosine Annealing'
        },
        {
          type: 'text',
          content: `Smoothly decrease learning rate following a cosine curve:`
        },
        {
          type: 'formula',
          content: 'lr = lr_min + (lr_max - lr_min) × (1 + cos(π × t/T)) / 2'
        },
        {
          type: 'text',
          content: `Benefits:
- Smooth transitions (no sudden drops)
- Spends more time at lower rates
- Often works better than step decay`
        },
        {
          type: 'heading',
          content: 'Learning Rate Warmup'
        },
        {
          type: 'text',
          content: `Start with a very small learning rate and gradually increase it over the first few epochs.

**Why?** At initialization, gradients can be large and unstable. High learning rates cause divergence. Warmup lets the model stabilize before using larger rates.

Especially important for:
- Transformers
- Large batch sizes
- Training from scratch`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

scheduler = CosineAnnealingLR(optimizer, T_max=100)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=len(train_loader)
)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f"LR: {scheduler.get_last_lr()}")`
        },
        {
          type: 'heading',
          content: 'One Cycle Policy'
        },
        {
          type: 'text',
          content: `A popular schedule that combines warmup and decay:
1. Warm up from low LR to max LR
2. Anneal from max LR to very low LR

Often allows faster training with higher learning rates.`
        },
        {
          type: 'table',
          headers: ['Schedule', 'Use Case', 'Key Parameter'],
          rows: [
            ['Step Decay', 'Simple, proven baseline', 'Drop factor (0.1), epochs per step'],
            ['Cosine', 'Smooth decay, good default', 'Total epochs'],
            ['OneCycle', 'Fast training', 'Max LR'],
            ['ReduceOnPlateau', 'Automatic adjustment', 'Patience, factor']
          ],
          caption: 'Learning rate schedule options'
        },
        {
          type: 'keypoints',
          points: [
            'Learning rate scheduling improves training dynamics',
            'Start high, decay over time is the general principle',
            'Warmup prevents instability at the start of training',
            'Cosine annealing is a smooth, effective default',
            'OneCycle can achieve faster convergence'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the purpose of learning rate warmup?',
          options: [
            'Speed up training',
            'Prevent instability at the start of training',
            'Increase final accuracy',
            'Reduce memory usage'
          ],
          correct: 1,
          explanation: 'At initialization, gradients can be large and unpredictable. Starting with a small learning rate and gradually increasing it lets the model stabilize before using larger, faster updates.'
        },
        {
          type: 'multiple-choice',
          question: 'Which schedule reduces learning rate smoothly without sudden drops?',
          options: [
            'Step decay',
            'Constant',
            'Cosine annealing',
            'Exponential'
          ],
          correct: 2,
          explanation: 'Cosine annealing smoothly reduces the learning rate following a cosine curve, without the sudden drops of step decay. This often leads to better convergence.'
        }
      ]
    },
    {
      id: 'vanishing-exploding-gradients',
      title: 'Vanishing and Exploding Gradients',
      duration: '45 min',
      concepts: ['Gradient Problems', 'Solutions', 'Gradient Clipping'],
      content: [
        {
          type: 'heading',
          content: 'The Depth Problem'
        },
        {
          type: 'text',
          content: `In deep networks, gradients are computed via the chain rule - multiplying many terms together. This multiplication can cause problems:

**Vanishing gradients**: If terms are < 1, product shrinks exponentially with depth
**Exploding gradients**: If terms are > 1, product grows exponentially with depth

Early layers barely learn (vanishing) or weights become NaN (exploding).`
        },
        {
          type: 'visualization',
          title: 'Gradient Flow Through Deep Networks',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Vanishing -->
            <g transform="translate(30,40)">
              <text x="70" y="0" text-anchor="middle" font-size="9" fill="#ef4444" font-weight="500">Vanishing Gradients</text>

              <!-- Bars shrinking -->
              <rect x="0" y="20" width="20" height="80" fill="#ef4444" opacity="0.3"/>
              <rect x="30" y="40" width="20" height="60" fill="#ef4444" opacity="0.4"/>
              <rect x="60" y="55" width="20" height="45" fill="#ef4444" opacity="0.5"/>
              <rect x="90" y="70" width="20" height="30" fill="#ef4444" opacity="0.6"/>
              <rect x="120" y="85" width="20" height="15" fill="#ef4444" opacity="0.8"/>

              <!-- Labels -->
              <text x="10" y="115" text-anchor="middle" font-size="7" fill="#64748b">L1</text>
              <text x="40" y="115" text-anchor="middle" font-size="7" fill="#64748b">L2</text>
              <text x="70" y="115" text-anchor="middle" font-size="7" fill="#64748b">L3</text>
              <text x="100" y="115" text-anchor="middle" font-size="7" fill="#64748b">L4</text>
              <text x="130" y="115" text-anchor="middle" font-size="7" fill="#64748b">L5</text>

              <text x="70" y="130" text-anchor="middle" font-size="8" fill="#ef4444">Gradients → 0</text>
            </g>

            <!-- Exploding -->
            <g transform="translate(220,40)">
              <text x="70" y="0" text-anchor="middle" font-size="9" fill="#f59e0b" font-weight="500">Exploding Gradients</text>

              <!-- Bars growing -->
              <rect x="0" y="85" width="20" height="15" fill="#f59e0b" opacity="0.3"/>
              <rect x="30" y="70" width="20" height="30" fill="#f59e0b" opacity="0.4"/>
              <rect x="60" y="50" width="20" height="50" fill="#f59e0b" opacity="0.5"/>
              <rect x="90" y="25" width="20" height="75" fill="#f59e0b" opacity="0.7"/>
              <rect x="120" y="5" width="20" height="95" fill="#f59e0b" opacity="0.9"/>

              <!-- Labels -->
              <text x="10" y="115" text-anchor="middle" font-size="7" fill="#64748b">L1</text>
              <text x="40" y="115" text-anchor="middle" font-size="7" fill="#64748b">L2</text>
              <text x="70" y="115" text-anchor="middle" font-size="7" fill="#64748b">L3</text>
              <text x="100" y="115" text-anchor="middle" font-size="7" fill="#64748b">L4</text>
              <text x="130" y="115" text-anchor="middle" font-size="7" fill="#64748b">L5</text>

              <text x="70" y="130" text-anchor="middle" font-size="8" fill="#f59e0b">Gradients → ∞</text>
            </g>

            <text x="200" y="165" text-anchor="middle" font-size="8" fill="#64748b">Gradient magnitude at each layer (backward pass)</text>
          </svg>`,
          caption: 'Gradients can vanish or explode as they flow through deep networks'
        },
        {
          type: 'heading',
          content: 'Causes of Vanishing Gradients'
        },
        {
          type: 'text',
          content: `**Sigmoid/Tanh activations**: Derivatives are < 1, and are very small for large inputs (saturation regions)

For sigmoid: σ'(x) = σ(x)(1-σ(x)), max value is 0.25

Multiply 0.25 × 0.25 × 0.25... through many layers → effectively zero.`
        },
        {
          type: 'heading',
          content: 'Solutions'
        },
        {
          type: 'text',
          content: `**1. ReLU activation**: Derivative is 1 for positive inputs
- Doesn't saturate in positive region
- But "dying ReLU" problem for negative inputs

**2. Proper initialization**: He initialization maintains variance

**3. Batch Normalization**: Keeps activations in good range

**4. Residual connections** (Skip connections): Allow gradients to flow directly`
        },
        {
          type: 'heading',
          content: 'Residual Connections (ResNet)'
        },
        {
          type: 'text',
          content: `Instead of learning f(x), learn f(x) + x. The identity shortcut allows gradients to flow directly through, bypassing problematic layers.`
        },
        {
          type: 'formula',
          content: 'y = F(x) + x'
        },
        {
          type: 'visualization',
          title: 'Residual Block',
          svg: `<svg viewBox="0 0 350 150" xmlns="http://www.w3.org/2000/svg">
            <rect width="350" height="150" fill="#f8fafc"/>

            <!-- Input -->
            <rect x="30" y="60" width="40" height="30" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" rx="4"/>
            <text x="50" y="80" text-anchor="middle" font-size="10" fill="#475569">x</text>

            <!-- Split -->
            <line x1="70" y1="75" x2="100" y2="75" stroke="#64748b" stroke-width="1.5"/>
            <circle cx="100" cy="75" r="3" fill="#64748b"/>

            <!-- Main path (F(x)) -->
            <line x1="100" y1="75" x2="100" y2="40" stroke="#64748b" stroke-width="1.5"/>
            <line x1="100" y1="40" x2="130" y2="40" stroke="#64748b" stroke-width="1.5"/>

            <rect x="130" y="25" width="60" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="4"/>
            <text x="160" y="45" text-anchor="middle" font-size="9" fill="#1e40af">Conv+BN</text>

            <line x1="190" y1="40" x2="220" y2="40" stroke="#64748b" stroke-width="1.5"/>

            <rect x="220" y="25" width="40" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="4"/>
            <text x="240" y="45" text-anchor="middle" font-size="9" fill="#065f46">ReLU</text>

            <line x1="260" y1="40" x2="280" y2="40" stroke="#64748b" stroke-width="1.5"/>
            <line x1="280" y1="40" x2="280" y2="75" stroke="#64748b" stroke-width="1.5"/>

            <!-- Skip connection -->
            <line x1="100" y1="75" x2="100" y2="110" stroke="#10b981" stroke-width="2"/>
            <line x1="100" y1="110" x2="280" y2="110" stroke="#10b981" stroke-width="2"/>
            <line x1="280" y1="110" x2="280" y2="75" stroke="#10b981" stroke-width="2"/>
            <text x="190" y="125" text-anchor="middle" font-size="8" fill="#10b981">Identity (skip)</text>

            <!-- Add -->
            <circle cx="280" cy="75" r="12" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
            <text x="280" y="80" text-anchor="middle" font-size="14" fill="#92400e">+</text>

            <!-- Output -->
            <line x1="292" y1="75" x2="310" y2="75" stroke="#64748b" stroke-width="1.5"/>
            <rect x="310" y="60" width="30" height="30" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="4"/>
            <text x="325" y="80" text-anchor="middle" font-size="10" fill="#dc2626">y</text>
          </svg>`,
          caption: 'Residual connection: y = F(x) + x allows direct gradient flow'
        },
        {
          type: 'heading',
          content: 'Gradient Clipping'
        },
        {
          type: 'text',
          content: `For exploding gradients, clip them to a maximum value before updating:`
        },
        {
          type: 'code',
          language: 'python',
          content: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()`
        },
        {
          type: 'heading',
          content: 'Diagnosing Gradient Problems'
        },
        {
          type: 'code',
          language: 'python',
          content: `def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-7:
                print(f"WARNING: {name} has near-zero gradient: {grad_norm}")
            elif grad_norm > 100:
                print(f"WARNING: {name} has large gradient: {grad_norm}")
            elif torch.isnan(param.grad).any():
                print(f"ERROR: {name} has NaN gradient!")

loss.backward()
check_gradients(model)`
        },
        {
          type: 'keypoints',
          points: [
            'Vanishing gradients: early layers dont learn (gradients too small)',
            'Exploding gradients: training diverges (gradients too large)',
            'Use ReLU, proper initialization, and batch normalization',
            'Residual connections allow gradients to bypass problematic layers',
            'Gradient clipping prevents exploding gradients'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What causes vanishing gradients with sigmoid activation?',
          options: [
            'Sigmoid outputs are always positive',
            'Sigmoid derivative is always less than 0.25',
            'Sigmoid is computationally expensive',
            'Sigmoid requires special initialization'
          ],
          correct: 1,
          explanation: 'Sigmoid\'s derivative σ\'(x) = σ(x)(1-σ(x)) has a maximum of 0.25. Multiplying many values less than 1 through layers causes gradients to shrink exponentially.'
        },
        {
          type: 'multiple-choice',
          question: 'How do residual connections help with gradient flow?',
          options: [
            'They make networks shallower',
            'They remove activation functions',
            'They provide a direct path for gradients to flow',
            'They increase the learning rate'
          ],
          correct: 2,
          explanation: 'Residual connections add an identity shortcut that allows gradients to flow directly through, bypassing layers where gradients might vanish or explode.'
        }
      ]
    },
    {
      id: 'debugging-neural-networks',
      title: 'Debugging Neural Networks',
      duration: '50 min',
      concepts: ['Common Issues', 'Debugging Strategies', 'Sanity Checks'],
      content: [
        {
          type: 'heading',
          content: 'Why Debugging NNs is Hard'
        },
        {
          type: 'text',
          content: `Neural networks fail silently. Unlike traditional code that crashes with errors, a poorly configured NN just produces bad results.

Common symptoms:
- Loss doesn't decrease
- Loss goes to NaN
- Model always predicts same class
- Training works, validation fails
- Results vary wildly between runs`
        },
        {
          type: 'heading',
          content: 'Sanity Checks'
        },
        {
          type: 'text',
          content: `Before training, verify your setup:

**1. Overfit a tiny batch**
Can the model memorize 10 examples? If not, something is wrong.

**2. Check initial loss**
Random weights should give expected loss:
- Binary CE with p=0.5: -log(0.5) ≈ 0.69
- 10-class softmax: -log(0.1) ≈ 2.3

**3. Verify gradients flow**
No NaN, no zeros, reasonable magnitudes.`
        },
        {
          type: 'code',
          language: 'python',
          content: `def sanity_check_overfit(model, train_loader, epochs=100):
    tiny_x, tiny_y = next(iter(train_loader))
    tiny_x, tiny_y = tiny_x[:10], tiny_y[:10]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(tiny_x)
        loss = criterion(output, tiny_y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            acc = (output.argmax(1) == tiny_y).float().mean()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.2f}")

    final_acc = (model(tiny_x).argmax(1) == tiny_y).float().mean()
    if final_acc < 0.99:
        print("WARNING: Could not overfit tiny batch!")
    else:
        print("OK: Model can overfit tiny batch")`
        },
        {
          type: 'heading',
          content: 'Common Issues and Solutions'
        },
        {
          type: 'table',
          headers: ['Symptom', 'Likely Cause', 'Solution'],
          rows: [
            ['Loss = NaN', 'Exploding gradients, learning rate too high', 'Lower LR, add gradient clipping'],
            ['Loss stuck high', 'Learning rate too low, bad init', 'Increase LR, check initialization'],
            ['Loss oscillates wildly', 'Learning rate too high', 'Lower learning rate'],
            ['Train good, val bad', 'Overfitting', 'More data, regularization, simpler model'],
            ['Same prediction always', 'Class imbalance, dead ReLUs', 'Balance classes, check activations'],
            ['Very slow convergence', 'Poor normalization', 'Normalize inputs, add BatchNorm']
          ],
          caption: 'Debugging neural network training issues'
        },
        {
          type: 'heading',
          content: 'Debugging Toolkit'
        },
        {
          type: 'code',
          language: 'python',
          content: `class DebugModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}

    def hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self.hook_fn(name))

    def check_activations(self):
        for name, act in self.activations.items():
            print(f"{name}:")
            print(f"  mean: {act.mean().item():.4f}")
            print(f"  std: {act.std().item():.4f}")
            print(f"  % zeros: {(act == 0).float().mean().item()*100:.1f}%")

            if act.std().item() < 0.01:
                print(f"  WARNING: Very low variance!")
            if (act == 0).float().mean() > 0.5:
                print(f"  WARNING: Many dead neurons!")`
        },
        {
          type: 'heading',
          content: 'Systematic Debugging Process'
        },
        {
          type: 'text',
          content: `**1. Simplify**
- Start with simplest possible model
- Use small dataset that should be easy
- Verify end-to-end before adding complexity

**2. Visualize**
- Plot loss curves (train and val)
- Visualize predictions vs actual
- Check activation distributions
- Monitor gradient magnitudes

**3. Test incrementally**
- Add one component at a time
- Verify each addition doesn't break things
- Keep track of what works

**4. Compare to baseline**
- Does a linear model do something?
- Does published architecture work?
- Are your metrics computed correctly?`
        },
        {
          type: 'heading',
          content: 'Logging and Monitoring'
        },
        {
          type: 'code',
          language: 'python',
          content: `from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

writer.close()`
        },
        {
          type: 'callout',
          variant: 'tip',
          content: 'When stuck, try the simplest thing first. Most issues are caused by simple bugs: wrong data preprocessing, forgetting model.eval(), mismatched dimensions, incorrect loss function.'
        },
        {
          type: 'keypoints',
          points: [
            'Always start with sanity checks: overfit tiny batch, check initial loss',
            'Monitor training with loss curves, not just final metrics',
            'Check activations and gradients to diagnose problems',
            'Start simple, add complexity incrementally',
            'Use logging (TensorBoard) to track experiments'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Your model\'s loss is NaN after a few iterations. What should you try first?',
          options: [
            'Add more layers',
            'Lower the learning rate',
            'Remove regularization',
            'Use a larger batch size'
          ],
          correct: 1,
          explanation: 'NaN loss typically indicates exploding gradients, often caused by a learning rate that is too high. Lowering the learning rate or adding gradient clipping usually fixes this.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the purpose of overfitting a tiny batch during debugging?',
          options: [
            'To find the best hyperparameters',
            'To verify the model can learn at all',
            'To speed up training',
            'To prevent overfitting'
          ],
          correct: 1,
          explanation: 'If a model cannot memorize 10 examples perfectly, something fundamental is wrong with the setup (data pipeline, loss function, architecture). This sanity check catches basic errors before wasting time on full training.'
        }
      ]
    }
  ]
}
