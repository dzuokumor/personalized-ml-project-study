export const neuralnetworks = {
  id: 'neural-networks',
  title: 'Neural Network Fundamentals',
  description: 'Build deep understanding of neural networks from the ground up - perceptrons, activation functions, forward/backward propagation, and implementing networks from scratch',
  category: 'Deep Learning',
  difficulty: 'Intermediate',
  duration: '7 hours',
  lessons: [
    {
      id: 'perceptron',
      title: 'The Perceptron',
      duration: '45 min',
      concepts: ['Perceptron', 'Linear Classifier', 'Decision Boundary'],
      content: [
        {
          type: 'heading',
          content: 'The Building Block of Neural Networks'
        },
        {
          type: 'text',
          content: `The perceptron, invented in 1958, is the simplest neural network - a single artificial neuron. It's inspired by biological neurons: receive inputs, process them, produce an output.

Despite its simplicity, the perceptron introduces concepts that underpin all modern deep learning: weights, bias, activation functions, and learning through error correction.`
        },
        {
          type: 'heading',
          content: 'How a Perceptron Works'
        },
        {
          type: 'text',
          content: `A perceptron takes multiple inputs, multiplies each by a weight, sums them up, adds a bias, and passes the result through an activation function:

1. **Weighted sum**: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
2. **Activation**: y = f(z)

For the classic perceptron, the activation is a step function: output 1 if z ≥ 0, else output 0.`
        },
        {
          type: 'visualization',
          title: 'Perceptron Architecture',
          svg: `<svg viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="220" fill="#f8fafc"/>

            <!-- Input nodes -->
            <circle cx="60" cy="50" r="20" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
            <text x="60" y="55" text-anchor="middle" font-size="12" fill="#475569">x₁</text>

            <circle cx="60" cy="110" r="20" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
            <text x="60" y="115" text-anchor="middle" font-size="12" fill="#475569">x₂</text>

            <circle cx="60" cy="170" r="20" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
            <text x="60" y="175" text-anchor="middle" font-size="12" fill="#475569">x₃</text>

            <!-- Bias -->
            <circle cx="170" cy="30" r="15" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
            <text x="170" y="35" text-anchor="middle" font-size="10" fill="#92400e">+1</text>

            <!-- Weights on connections -->
            <line x1="80" y1="50" x2="200" y2="105" stroke="#94a3b8" stroke-width="2"/>
            <text x="130" y="65" font-size="9" fill="#3b82f6">w₁</text>

            <line x1="80" y1="110" x2="200" y2="110" stroke="#94a3b8" stroke-width="2"/>
            <text x="140" y="103" font-size="9" fill="#3b82f6">w₂</text>

            <line x1="80" y1="170" x2="200" y2="115" stroke="#94a3b8" stroke-width="2"/>
            <text x="130" y="155" font-size="9" fill="#3b82f6">w₃</text>

            <line x1="170" y1="45" x2="200" y2="100" stroke="#f59e0b" stroke-width="2"/>
            <text x="180" y="65" font-size="9" fill="#f59e0b">b</text>

            <!-- Sum node -->
            <circle cx="220" cy="110" r="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
            <text x="220" y="115" text-anchor="middle" font-size="14" fill="#1e40af">Σ</text>

            <!-- Activation -->
            <rect x="280" y="90" width="50" height="40" fill="#d1fae5" stroke="#10b981" stroke-width="2" rx="4"/>
            <text x="305" y="115" text-anchor="middle" font-size="10" fill="#065f46">f(z)</text>

            <line x1="245" y1="110" x2="280" y2="110" stroke="#94a3b8" stroke-width="2"/>

            <!-- Output -->
            <line x1="330" y1="110" x2="380" y2="110" stroke="#94a3b8" stroke-width="2" marker-end="url(#outputarrow)"/>

            <circle cx="400" cy="110" r="20" fill="#10b981" stroke="#059669" stroke-width="2"/>
            <text x="400" y="115" text-anchor="middle" font-size="12" fill="white">y</text>

            <!-- Labels -->
            <text x="60" y="205" text-anchor="middle" font-size="9" fill="#64748b">Inputs</text>
            <text x="220" y="150" text-anchor="middle" font-size="9" fill="#64748b">Weighted Sum</text>
            <text x="305" y="145" text-anchor="middle" font-size="9" fill="#64748b">Activation</text>
            <text x="400" y="145" text-anchor="middle" font-size="9" fill="#64748b">Output</text>

            <defs>
              <marker id="outputarrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'The perceptron: inputs → weighted sum → activation → output'
        },
        {
          type: 'heading',
          content: 'The Mathematics'
        },
        {
          type: 'formula',
          content: 'z = Σᵢ wᵢxᵢ + b = w · x + b'
        },
        {
          type: 'formula',
          content: 'y = step(z) = { 1 if z ≥ 0, 0 otherwise }'
        },
        {
          type: 'text',
          content: `In vector notation, the weighted sum is just a dot product plus bias. This is a **linear operation** - the perceptron finds a linear decision boundary.`
        },
        {
          type: 'heading',
          content: 'Decision Boundaries'
        },
        {
          type: 'text',
          content: `The perceptron separates classes with a hyperplane. In 2D, it's a line. The weights determine the orientation, and the bias shifts it.`
        },
        {
          type: 'visualization',
          title: 'Perceptron Decision Boundary',
          svg: `<svg viewBox="0 0 350 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="350" height="220" fill="#f8fafc"/>

            <!-- Plot area -->
            <rect x="50" y="30" width="180" height="150" fill="white" stroke="#e2e8f0" rx="4"/>

            <!-- Axes -->
            <line x1="50" y1="180" x2="230" y2="180" stroke="#64748b" stroke-width="1"/>
            <line x1="50" y1="180" x2="50" y2="30" stroke="#64748b" stroke-width="1"/>
            <text x="240" y="183" font-size="9" fill="#64748b">x₁</text>
            <text x="45" y="25" font-size="9" fill="#64748b">x₂</text>

            <!-- Decision boundary (line) -->
            <line x1="50" y1="160" x2="220" y2="50" stroke="#10b981" stroke-width="2"/>

            <!-- Regions -->
            <text x="90" y="70" font-size="9" fill="#10b981">Class 1</text>
            <text x="160" y="160" font-size="9" fill="#3b82f6">Class 0</text>

            <!-- Data points -->
            <circle cx="80" cy="60" r="5" fill="#10b981"/>
            <circle cx="100" cy="80" r="5" fill="#10b981"/>
            <circle cx="70" cy="90" r="5" fill="#10b981"/>
            <circle cx="120" cy="55" r="5" fill="#10b981"/>
            <circle cx="90" cy="100" r="5" fill="#10b981"/>

            <circle cx="160" cy="140" r="5" fill="#3b82f6"/>
            <circle cx="180" cy="160" r="5" fill="#3b82f6"/>
            <circle cx="200" cy="130" r="5" fill="#3b82f6"/>
            <circle cx="170" cy="120" r="5" fill="#3b82f6"/>
            <circle cx="190" cy="150" r="5" fill="#3b82f6"/>

            <!-- Formula -->
            <text x="290" y="80" text-anchor="middle" font-size="10" fill="#475569">w · x + b = 0</text>
            <text x="290" y="100" text-anchor="middle" font-size="9" fill="#64748b">defines the</text>
            <text x="290" y="115" text-anchor="middle" font-size="9" fill="#64748b">decision boundary</text>
          </svg>`,
          caption: 'The perceptron learns a linear decision boundary'
        },
        {
          type: 'heading',
          content: 'The Perceptron Learning Algorithm'
        },
        {
          type: 'text',
          content: `The perceptron learns by adjusting weights when it makes mistakes:

1. Initialize weights randomly (or to zero)
2. For each training example:
   - Compute prediction: ŷ = step(w · x + b)
   - If prediction is wrong, update weights:
     - w ← w + η(y - ŷ)x
     - b ← b + η(y - ŷ)
3. Repeat until no errors (or max iterations)

Here, η (eta) is the learning rate - how big a step to take.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear >= 0 else 0

                if y_pred != y[idx]:
                    update = self.lr * (y[idx] - y_pred)
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear >= 0, 1, 0)`
        },
        {
          type: 'heading',
          content: 'The XOR Problem'
        },
        {
          type: 'text',
          content: `A single perceptron can only solve **linearly separable** problems. The famous XOR (exclusive or) problem showed this limitation:

XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0

No single line can separate the 1s from the 0s! This limitation led to the "AI winter" of the 1970s.

The solution? **Multiple layers of neurons** - which we'll explore next.`
        },
        {
          type: 'visualization',
          title: 'The XOR Problem',
          svg: `<svg viewBox="0 0 350 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="350" height="180" fill="#f8fafc"/>

            <!-- AND (solvable) -->
            <g transform="translate(30,20)">
              <text x="60" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">AND ✓</text>
              <rect x="10" y="10" width="100" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <circle cx="30" cy="90" r="6" fill="#3b82f6"/>
              <circle cx="30" cy="30" r="6" fill="#3b82f6"/>
              <circle cx="90" cy="90" r="6" fill="#3b82f6"/>
              <circle cx="90" cy="30" r="6" fill="#10b981"/>

              <line x1="15" y1="50" x2="105" y2="20" stroke="#10b981" stroke-width="2"/>
              <text x="60" y="125" text-anchor="middle" font-size="8" fill="#64748b">Linearly separable</text>
            </g>

            <!-- OR (solvable) -->
            <g transform="translate(140,20)">
              <text x="60" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">OR ✓</text>
              <rect x="10" y="10" width="100" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <circle cx="30" cy="90" r="6" fill="#3b82f6"/>
              <circle cx="30" cy="30" r="6" fill="#10b981"/>
              <circle cx="90" cy="90" r="6" fill="#10b981"/>
              <circle cx="90" cy="30" r="6" fill="#10b981"/>

              <line x1="15" y1="100" x2="105" y2="70" stroke="#10b981" stroke-width="2"/>
              <text x="60" y="125" text-anchor="middle" font-size="8" fill="#64748b">Linearly separable</text>
            </g>

            <!-- XOR (NOT solvable) -->
            <g transform="translate(250,20)">
              <text x="50" y="0" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="600">XOR ✗</text>
              <rect x="5" y="10" width="90" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <circle cx="25" cy="90" r="6" fill="#3b82f6"/>
              <circle cx="25" cy="30" r="6" fill="#10b981"/>
              <circle cx="75" cy="90" r="6" fill="#10b981"/>
              <circle cx="75" cy="30" r="6" fill="#3b82f6"/>

              <text x="50" y="65" text-anchor="middle" font-size="12" fill="#ef4444">?</text>
              <text x="50" y="125" text-anchor="middle" font-size="8" fill="#ef4444">NOT separable!</text>
            </g>

            <!-- Legend -->
            <g transform="translate(100,155)">
              <circle cx="0" cy="0" r="5" fill="#10b981"/>
              <text x="10" y="4" font-size="8" fill="#64748b">Output 1</text>
              <circle cx="80" cy="0" r="5" fill="#3b82f6"/>
              <text x="90" y="4" font-size="8" fill="#64748b">Output 0</text>
            </g>
          </svg>`,
          caption: 'XOR cannot be solved by a single perceptron - no line separates the classes'
        },
        {
          type: 'keypoints',
          points: [
            'The perceptron is the simplest neural network - a single artificial neuron',
            'It computes a weighted sum of inputs, adds bias, and applies activation',
            'The perceptron learning algorithm adjusts weights when predictions are wrong',
            'Single perceptrons can only solve linearly separable problems',
            'The XOR problem motivated the development of multi-layer networks'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the main limitation of a single perceptron?',
          options: [
            'It is too slow to train',
            'It can only solve linearly separable problems',
            'It requires too much data',
            'It cannot have a bias term'
          ],
          correct: 1,
          explanation: 'A single perceptron can only learn linear decision boundaries, so it cannot solve problems where classes are not linearly separable, like XOR.'
        },
        {
          type: 'multiple-choice',
          question: 'When does the perceptron update its weights?',
          options: [
            'After every training example',
            'Only when it makes a wrong prediction',
            'At the end of each epoch',
            'When the loss is below a threshold'
          ],
          correct: 1,
          explanation: 'The perceptron learning algorithm only updates weights when the prediction is incorrect. If the prediction is correct, no update is made.'
        }
      ]
    },
    {
      id: 'activation-functions',
      title: 'Activation Functions',
      duration: '50 min',
      concepts: ['Sigmoid', 'Tanh', 'ReLU', 'Softmax'],
      content: [
        {
          type: 'heading',
          content: 'Why Activation Functions?'
        },
        {
          type: 'text',
          content: `Without activation functions, a neural network would just be a series of linear transformations - which collapses to a single linear transformation. No matter how many layers, you'd get:

y = W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx

**Activation functions introduce non-linearity**, allowing networks to learn complex, non-linear patterns. They're what make deep learning powerful.`
        },
        {
          type: 'heading',
          content: 'Sigmoid (Logistic)'
        },
        {
          type: 'formula',
          content: 'σ(x) = 1 / (1 + e⁻ˣ)'
        },
        {
          type: 'text',
          content: `The sigmoid squashes any input to the range (0, 1). Historically important, used in logistic regression and early neural networks.

**Pros**: Smooth, differentiable, outputs interpretable as probabilities
**Cons**: Vanishing gradients for large |x|, not zero-centered`
        },
        {
          type: 'visualization',
          title: 'Activation Function Comparison',
          svg: `<svg viewBox="0 0 450 280" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="280" fill="#f8fafc"/>

            <!-- Sigmoid -->
            <g transform="translate(30,30)">
              <text x="80" y="0" text-anchor="middle" font-size="10" fill="#3b82f6" font-weight="600">Sigmoid</text>
              <rect x="10" y="10" width="140" height="90" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="20" y1="55" x2="140" y2="55" stroke="#94a3b8" stroke-width="1"/>
              <line x1="80" y1="20" x2="80" y2="90" stroke="#94a3b8" stroke-width="1"/>

              <!-- Sigmoid curve -->
              <path d="M25,85 Q40,80 55,70 Q70,55 80,55 Q90,55 95,45 Q110,30 135,22"
                    fill="none" stroke="#3b82f6" stroke-width="2"/>

              <!-- Labels -->
              <text x="145" y="58" font-size="8" fill="#64748b">0</text>
              <text x="142" y="25" font-size="8" fill="#64748b">1</text>
              <text x="20" y="105" font-size="8" fill="#64748b">Range: (0, 1)</text>
            </g>

            <!-- Tanh -->
            <g transform="translate(200,30)">
              <text x="80" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">Tanh</text>
              <rect x="10" y="10" width="140" height="90" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="20" y1="55" x2="140" y2="55" stroke="#94a3b8" stroke-width="1"/>
              <line x1="80" y1="20" x2="80" y2="90" stroke="#94a3b8" stroke-width="1"/>

              <!-- Tanh curve -->
              <path d="M25,88 Q40,85 55,75 Q70,55 80,55 Q90,55 95,35 Q110,25 135,22"
                    fill="none" stroke="#10b981" stroke-width="2"/>

              <!-- Labels -->
              <text x="145" y="58" font-size="8" fill="#64748b">0</text>
              <text x="140" y="25" font-size="8" fill="#64748b">+1</text>
              <text x="140" y="92" font-size="8" fill="#64748b">-1</text>
              <text x="20" y="105" font-size="8" fill="#64748b">Range: (-1, 1)</text>
            </g>

            <!-- ReLU -->
            <g transform="translate(30,155)">
              <text x="80" y="0" text-anchor="middle" font-size="10" fill="#f59e0b" font-weight="600">ReLU</text>
              <rect x="10" y="10" width="140" height="90" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="20" y1="70" x2="140" y2="70" stroke="#94a3b8" stroke-width="1"/>
              <line x1="80" y1="20" x2="80" y2="90" stroke="#94a3b8" stroke-width="1"/>

              <!-- ReLU (flat then linear) -->
              <line x1="25" y1="70" x2="80" y2="70" stroke="#f59e0b" stroke-width="2"/>
              <line x1="80" y1="70" x2="135" y2="25" stroke="#f59e0b" stroke-width="2"/>

              <text x="20" y="105" font-size="8" fill="#64748b">Range: [0, ∞)</text>
            </g>

            <!-- Leaky ReLU -->
            <g transform="translate(200,155)">
              <text x="80" y="0" text-anchor="middle" font-size="10" fill="#8b5cf6" font-weight="600">Leaky ReLU</text>
              <rect x="10" y="10" width="140" height="90" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="20" y1="70" x2="140" y2="70" stroke="#94a3b8" stroke-width="1"/>
              <line x1="80" y1="20" x2="80" y2="90" stroke="#94a3b8" stroke-width="1"/>

              <!-- Leaky ReLU -->
              <line x1="25" y1="75" x2="80" y2="70" stroke="#8b5cf6" stroke-width="2"/>
              <line x1="80" y1="70" x2="135" y2="25" stroke="#8b5cf6" stroke-width="2"/>

              <text x="45" y="82" font-size="7" fill="#8b5cf6">slope=0.01</text>
              <text x="20" y="105" font-size="8" fill="#64748b">Range: (-∞, ∞)</text>
            </g>
          </svg>`,
          caption: 'Common activation functions and their shapes'
        },
        {
          type: 'heading',
          content: 'Tanh (Hyperbolic Tangent)'
        },
        {
          type: 'formula',
          content: 'tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)'
        },
        {
          type: 'text',
          content: `Tanh is a scaled and shifted sigmoid that outputs (-1, 1). Being zero-centered makes optimization easier.

**Pros**: Zero-centered, stronger gradients than sigmoid
**Cons**: Still has vanishing gradient problem for large |x|`
        },
        {
          type: 'heading',
          content: 'ReLU (Rectified Linear Unit)'
        },
        {
          type: 'formula',
          content: 'ReLU(x) = max(0, x)'
        },
        {
          type: 'text',
          content: `ReLU is the default choice for modern neural networks. It's simple, fast, and works remarkably well.

**Pros**:
- No vanishing gradient for positive inputs
- Computationally efficient
- Induces sparsity (many neurons output 0)

**Cons**:
- "Dying ReLU" problem - neurons can get stuck at 0
- Not zero-centered
- Unbounded output`
        },
        {
          type: 'heading',
          content: 'Leaky ReLU and Variants'
        },
        {
          type: 'formula',
          content: 'LeakyReLU(x) = max(αx, x), typically α = 0.01'
        },
        {
          type: 'text',
          content: `Leaky ReLU fixes the dying ReLU problem by allowing a small gradient when x < 0.

**Variants**:
- **PReLU**: α is learned during training
- **ELU**: Smooth curve for x < 0, approaches -α
- **GELU**: Used in transformers, smooth approximation of ReLU`
        },
        {
          type: 'heading',
          content: 'Softmax for Multi-Class Classification'
        },
        {
          type: 'formula',
          content: 'softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ'
        },
        {
          type: 'text',
          content: `Softmax converts a vector of scores into probabilities that sum to 1. Used in the output layer for multi-class classification.

Each output represents the probability of that class.`
        },
        {
          type: 'visualization',
          title: 'Softmax Example',
          svg: `<svg viewBox="0 0 400 150" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="150" fill="#f8fafc"/>

            <!-- Input scores -->
            <g transform="translate(30,30)">
              <text x="40" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Logits (raw scores)</text>
              <rect x="0" y="10" width="80" height="80" fill="white" stroke="#e2e8f0" rx="4"/>
              <text x="40" y="40" text-anchor="middle" font-size="11" fill="#475569">2.0</text>
              <text x="40" y="58" text-anchor="middle" font-size="11" fill="#475569">1.0</text>
              <text x="40" y="76" text-anchor="middle" font-size="11" fill="#475569">0.1</text>
            </g>

            <!-- Arrow -->
            <g transform="translate(130,60)">
              <path d="M0,15 L50,15" stroke="#10b981" stroke-width="2" marker-end="url(#softmaxarrow)"/>
              <text x="25" y="8" text-anchor="middle" font-size="9" fill="#10b981">softmax</text>
            </g>

            <!-- Output probabilities -->
            <g transform="translate(200,30)">
              <text x="55" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Probabilities</text>
              <rect x="0" y="10" width="110" height="80" fill="white" stroke="#e2e8f0" rx="4"/>

              <rect x="10" y="25" width="50" height="12" fill="#10b981" rx="2"/>
              <text x="65" y="35" font-size="10" fill="#10b981">0.66</text>

              <rect x="10" y="45" width="25" height="12" fill="#3b82f6" rx="2"/>
              <text x="65" y="55" font-size="10" fill="#3b82f6">0.24</text>

              <rect x="10" y="65" width="10" height="12" fill="#f59e0b" rx="2"/>
              <text x="65" y="75" font-size="10" fill="#f59e0b">0.10</text>

              <text x="55" y="105" text-anchor="middle" font-size="9" fill="#64748b">Sum = 1.0</text>
            </g>

            <defs>
              <marker id="softmaxarrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#10b981"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Softmax converts scores to probabilities that sum to 1'
        },
        {
          type: 'heading',
          content: 'Choosing Activation Functions'
        },
        {
          type: 'table',
          headers: ['Layer', 'Task', 'Recommended'],
          rows: [
            ['Hidden layers', 'General', 'ReLU (or Leaky ReLU)'],
            ['Hidden layers', 'RNNs', 'Tanh or ReLU'],
            ['Output layer', 'Binary classification', 'Sigmoid'],
            ['Output layer', 'Multi-class classification', 'Softmax'],
            ['Output layer', 'Regression', 'Linear (no activation)']
          ],
          caption: 'Activation function selection guide'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

x = np.array([2.0, 1.0, 0.1])
print(f"Softmax: {softmax(x)}")
print(f"Sum: {softmax(x).sum()}")`
        },
        {
          type: 'keypoints',
          points: [
            'Activation functions introduce non-linearity, enabling complex patterns',
            'ReLU is the default for hidden layers in modern networks',
            'Sigmoid outputs (0,1), used for binary classification output',
            'Softmax outputs probabilities that sum to 1 for multi-class',
            'Leaky ReLU helps prevent "dying" neurons'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why are activation functions necessary in neural networks?',
          options: [
            'To speed up training',
            'To introduce non-linearity',
            'To reduce memory usage',
            'To normalize inputs'
          ],
          correct: 1,
          explanation: 'Without activation functions, multiple layers of linear transformations collapse to a single linear transformation. Activation functions introduce non-linearity, allowing networks to learn complex patterns.'
        },
        {
          type: 'multiple-choice',
          question: 'What activation function is typically used for the output layer of a multi-class classifier?',
          options: [
            'ReLU',
            'Sigmoid',
            'Softmax',
            'Tanh'
          ],
          correct: 2,
          explanation: 'Softmax converts raw scores (logits) into probabilities that sum to 1, making it ideal for multi-class classification where each class gets a probability.'
        }
      ]
    },
    {
      id: 'mlp',
      title: 'Multi-Layer Perceptrons',
      duration: '55 min',
      concepts: ['Hidden Layers', 'Universal Approximation', 'Network Architecture'],
      content: [
        {
          type: 'heading',
          content: 'From Perceptron to Deep Networks'
        },
        {
          type: 'text',
          content: `A Multi-Layer Perceptron (MLP) stacks multiple layers of neurons. By adding hidden layers between input and output, we can learn non-linear decision boundaries and solve problems like XOR.

The key insight: each layer transforms the data into a new representation, making the next layer's job easier.`
        },
        {
          type: 'heading',
          content: 'MLP Architecture'
        },
        {
          type: 'visualization',
          title: 'Multi-Layer Perceptron Structure',
          svg: `<svg viewBox="0 0 450 250" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="250" fill="#f8fafc"/>

            <!-- Input Layer -->
            <g transform="translate(50,0)">
              <text x="0" y="25" text-anchor="middle" font-size="9" fill="#64748b">Input</text>
              <circle cx="0" cy="60" r="15" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
              <text x="0" y="64" text-anchor="middle" font-size="10" fill="#475569">x₁</text>

              <circle cx="0" cy="110" r="15" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
              <text x="0" y="114" text-anchor="middle" font-size="10" fill="#475569">x₂</text>

              <circle cx="0" cy="160" r="15" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
              <text x="0" y="164" text-anchor="middle" font-size="10" fill="#475569">x₃</text>

              <circle cx="0" cy="210" r="15" fill="#e2e8f0" stroke="#64748b" stroke-width="2"/>
              <text x="0" y="214" text-anchor="middle" font-size="10" fill="#475569">x₄</text>
            </g>

            <!-- Hidden Layer 1 -->
            <g transform="translate(160,0)">
              <text x="0" y="25" text-anchor="middle" font-size="9" fill="#3b82f6">Hidden 1</text>
              <circle cx="0" cy="70" r="15" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
              <circle cx="0" cy="120" r="15" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
              <circle cx="0" cy="170" r="15" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
            </g>

            <!-- Hidden Layer 2 -->
            <g transform="translate(270,0)">
              <text x="0" y="25" text-anchor="middle" font-size="9" fill="#10b981">Hidden 2</text>
              <circle cx="0" cy="95" r="15" fill="#d1fae5" stroke="#10b981" stroke-width="2"/>
              <circle cx="0" cy="155" r="15" fill="#d1fae5" stroke="#10b981" stroke-width="2"/>
            </g>

            <!-- Output Layer -->
            <g transform="translate(380,0)">
              <text x="0" y="25" text-anchor="middle" font-size="9" fill="#ef4444">Output</text>
              <circle cx="0" cy="125" r="15" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
              <text x="0" y="129" text-anchor="middle" font-size="10" fill="#dc2626">y</text>
            </g>

            <!-- Connections Input to Hidden 1 -->
            <g stroke="#94a3b8" stroke-width="1" opacity="0.6">
              <line x1="65" y1="60" x2="145" y2="70"/>
              <line x1="65" y1="60" x2="145" y2="120"/>
              <line x1="65" y1="60" x2="145" y2="170"/>

              <line x1="65" y1="110" x2="145" y2="70"/>
              <line x1="65" y1="110" x2="145" y2="120"/>
              <line x1="65" y1="110" x2="145" y2="170"/>

              <line x1="65" y1="160" x2="145" y2="70"/>
              <line x1="65" y1="160" x2="145" y2="120"/>
              <line x1="65" y1="160" x2="145" y2="170"/>

              <line x1="65" y1="210" x2="145" y2="70"/>
              <line x1="65" y1="210" x2="145" y2="120"/>
              <line x1="65" y1="210" x2="145" y2="170"/>
            </g>

            <!-- Connections Hidden 1 to Hidden 2 -->
            <g stroke="#94a3b8" stroke-width="1" opacity="0.6">
              <line x1="175" y1="70" x2="255" y2="95"/>
              <line x1="175" y1="70" x2="255" y2="155"/>
              <line x1="175" y1="120" x2="255" y2="95"/>
              <line x1="175" y1="120" x2="255" y2="155"/>
              <line x1="175" y1="170" x2="255" y2="95"/>
              <line x1="175" y1="170" x2="255" y2="155"/>
            </g>

            <!-- Connections Hidden 2 to Output -->
            <g stroke="#94a3b8" stroke-width="1" opacity="0.6">
              <line x1="285" y1="95" x2="365" y2="125"/>
              <line x1="285" y1="155" x2="365" y2="125"/>
            </g>

            <!-- Labels -->
            <text x="110" y="240" font-size="8" fill="#64748b">W¹</text>
            <text x="215" y="240" font-size="8" fill="#64748b">W²</text>
            <text x="325" y="240" font-size="8" fill="#64748b">W³</text>
          </svg>`,
          caption: 'An MLP with 4 inputs, two hidden layers, and 1 output'
        },
        {
          type: 'heading',
          content: 'Layer-by-Layer Computation'
        },
        {
          type: 'text',
          content: `Each layer performs:
1. Linear transformation: z = Wx + b
2. Non-linear activation: a = f(z)

The output of one layer becomes the input to the next.`
        },
        {
          type: 'formula',
          content: 'h₁ = f(W₁x + b₁)  →  h₂ = f(W₂h₁ + b₂)  →  y = f(W₃h₂ + b₃)'
        },
        {
          type: 'heading',
          content: 'The Universal Approximation Theorem'
        },
        {
          type: 'text',
          content: `A remarkable result: an MLP with just **one hidden layer** and enough neurons can approximate any continuous function to arbitrary precision.

This doesn't mean one hidden layer is always best - deeper networks often learn better representations with fewer total parameters. But it shows the theoretical power of MLPs.`
        },
        {
          type: 'callout',
          variant: 'info',
          content: 'Universal approximation says MLPs CAN represent any function, but not that gradient descent will FIND that representation. Training remains the challenge.'
        },
        {
          type: 'heading',
          content: 'How Hidden Layers Transform Data'
        },
        {
          type: 'text',
          content: `Think of each hidden layer as learning a new representation of the data. For XOR:

- Input: original (x₁, x₂) coordinates
- Hidden layer: transforms to a space where classes ARE linearly separable
- Output layer: draws a simple line in this new space`
        },
        {
          type: 'visualization',
          title: 'Hidden Layer Learns New Representation',
          svg: `<svg viewBox="0 0 400 170" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="170" fill="#f8fafc"/>

            <!-- Original space -->
            <g transform="translate(30,20)">
              <text x="50" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Input Space</text>
              <rect x="5" y="10" width="90" height="90" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- XOR points -->
              <circle cx="25" cy="85" r="5" fill="#3b82f6"/>
              <circle cx="25" cy="25" r="5" fill="#10b981"/>
              <circle cx="75" cy="85" r="5" fill="#10b981"/>
              <circle cx="75" cy="25" r="5" fill="#3b82f6"/>

              <text x="50" y="115" text-anchor="middle" font-size="8" fill="#ef4444">Not separable</text>
            </g>

            <!-- Arrow -->
            <g transform="translate(140,55)">
              <path d="M0,15 L45,15" stroke="#64748b" stroke-width="2" marker-end="url(#transarrow)"/>
              <text x="22" y="8" text-anchor="middle" font-size="8" fill="#64748b">Hidden</text>
              <text x="22" y="38" text-anchor="middle" font-size="8" fill="#64748b">Layer</text>
            </g>

            <!-- Transformed space -->
            <g transform="translate(205,20)">
              <text x="70" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Hidden Layer Space</text>
              <rect x="10" y="10" width="120" height="90" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Transformed XOR points - now separable -->
              <circle cx="35" cy="75" r="5" fill="#3b82f6"/>
              <circle cx="95" cy="35" r="5" fill="#10b981"/>
              <circle cx="95" cy="75" r="5" fill="#10b981"/>
              <circle cx="35" cy="35" r="5" fill="#3b82f6"/>

              <!-- Decision boundary now possible -->
              <line x1="15" y1="55" x2="125" y2="55" stroke="#10b981" stroke-width="2"/>

              <text x="70" y="115" text-anchor="middle" font-size="8" fill="#10b981">Now separable!</text>
            </g>

            <defs>
              <marker id="transarrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'The hidden layer transforms XOR into a linearly separable problem'
        },
        {
          type: 'heading',
          content: 'Architecture Choices'
        },
        {
          type: 'text',
          content: `**Number of hidden layers**: Start simple. One or two hidden layers often work well. Add depth if needed.

**Neurons per layer**: Common patterns:
- Pyramid: decreasing neurons (128→64→32)
- Constant: same in each layer (64→64→64)
- Expanding then contracting (encoder-decoder)

**Rules of thumb**:
- Total parameters should be << number of training samples
- Wider layers can capture more complex features
- Deeper networks can learn hierarchical representations`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = MLP(input_size=4, hidden_sizes=[64, 32], output_size=3)
print(model)

x = torch.randn(1, 4)
output = model(x)
print(f"Output shape: {output.shape}")`
        },
        {
          type: 'keypoints',
          points: [
            'MLPs stack layers of neurons to learn complex patterns',
            'Hidden layers transform data into more useful representations',
            'The universal approximation theorem proves MLPs can represent any function',
            'Each layer applies: linear transformation → activation',
            'Start with simple architectures and add complexity as needed'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does the Universal Approximation Theorem state?',
          options: [
            'Neural networks always converge to the optimal solution',
            'An MLP with one hidden layer can approximate any continuous function',
            'Deeper networks are always better',
            'All activation functions are equivalent'
          ],
          correct: 1,
          explanation: 'The theorem states that an MLP with one hidden layer and enough neurons can approximate any continuous function to arbitrary accuracy. However, it does not guarantee that training will find this approximation.'
        },
        {
          type: 'multiple-choice',
          question: 'How does a hidden layer help solve the XOR problem?',
          options: [
            'It adds more weights',
            'It transforms the data into a linearly separable representation',
            'It memorizes the training examples',
            'It reduces overfitting'
          ],
          correct: 1,
          explanation: 'The hidden layer learns to transform the input space into a new representation where the XOR classes become linearly separable, allowing the output layer to separate them with a simple boundary.'
        }
      ]
    },
    {
      id: 'forward-propagation',
      title: 'Forward Propagation',
      duration: '45 min',
      concepts: ['Matrix Operations', 'Layer Outputs', 'Predictions'],
      content: [
        {
          type: 'heading',
          content: 'From Input to Output'
        },
        {
          type: 'text',
          content: `Forward propagation is how a neural network makes predictions. Data flows forward through the network, layer by layer, from input to output.

Each layer:
1. Receives input from the previous layer
2. Applies weights and bias (linear transformation)
3. Applies activation function (non-linearity)
4. Passes output to the next layer`
        },
        {
          type: 'visualization',
          title: 'Forward Propagation Flow',
          svg: `<svg viewBox="0 0 450 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="180" fill="#f8fafc"/>

            <!-- Input -->
            <g transform="translate(20,30)">
              <rect x="0" y="0" width="60" height="100" fill="#e2e8f0" stroke="#64748b" stroke-width="2" rx="4"/>
              <text x="30" y="55" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">x</text>
              <text x="30" y="115" text-anchor="middle" font-size="9" fill="#64748b">Input</text>
            </g>

            <!-- Linear 1 -->
            <g transform="translate(100,30)">
              <rect x="0" y="20" width="60" height="60" fill="#dbeafe" stroke="#3b82f6" stroke-width="2" rx="4"/>
              <text x="30" y="45" text-anchor="middle" font-size="9" fill="#1e40af">W¹x + b¹</text>
              <text x="30" y="60" text-anchor="middle" font-size="9" fill="#1e40af">= z¹</text>
              <text x="30" y="95" text-anchor="middle" font-size="8" fill="#3b82f6">Linear</text>
            </g>

            <!-- Activation 1 -->
            <g transform="translate(170,30)">
              <rect x="0" y="20" width="50" height="60" fill="#d1fae5" stroke="#10b981" stroke-width="2" rx="4"/>
              <text x="25" y="45" text-anchor="middle" font-size="9" fill="#065f46">f(z¹)</text>
              <text x="25" y="60" text-anchor="middle" font-size="9" fill="#065f46">= a¹</text>
              <text x="25" y="95" text-anchor="middle" font-size="8" fill="#10b981">ReLU</text>
            </g>

            <!-- Linear 2 -->
            <g transform="translate(240,30)">
              <rect x="0" y="20" width="60" height="60" fill="#dbeafe" stroke="#3b82f6" stroke-width="2" rx="4"/>
              <text x="30" y="45" text-anchor="middle" font-size="9" fill="#1e40af">W²a¹+ b²</text>
              <text x="30" y="60" text-anchor="middle" font-size="9" fill="#1e40af">= z²</text>
              <text x="30" y="95" text-anchor="middle" font-size="8" fill="#3b82f6">Linear</text>
            </g>

            <!-- Activation 2 -->
            <g transform="translate(310,30)">
              <rect x="0" y="20" width="50" height="60" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="4"/>
              <text x="25" y="45" text-anchor="middle" font-size="9" fill="#92400e">σ(z²)</text>
              <text x="25" y="60" text-anchor="middle" font-size="9" fill="#92400e">= ŷ</text>
              <text x="25" y="95" text-anchor="middle" font-size="8" fill="#f59e0b">Sigmoid</text>
            </g>

            <!-- Output -->
            <g transform="translate(375,30)">
              <rect x="0" y="0" width="55" height="100" fill="#fee2e2" stroke="#ef4444" stroke-width="2" rx="4"/>
              <text x="27" y="55" text-anchor="middle" font-size="11" fill="#dc2626" font-weight="500">ŷ</text>
              <text x="27" y="115" text-anchor="middle" font-size="9" fill="#64748b">Output</text>
            </g>

            <!-- Arrows -->
            <g fill="#64748b">
              <polygon points="85,80 95,75 95,85"/>
              <polygon points="165,80 175,75 175,85"/>
              <polygon points="225,80 235,75 235,85"/>
              <polygon points="305,80 315,75 315,85"/>
              <polygon points="365,80 375,75 375,85"/>
            </g>
          </svg>`,
          caption: 'Forward propagation: each layer applies linear transform then activation'
        },
        {
          type: 'heading',
          content: 'Matrix Form'
        },
        {
          type: 'text',
          content: `For efficiency, we process entire batches of inputs using matrix operations:`
        },
        {
          type: 'formula',
          content: 'Z¹ = X · W¹ + b¹  (batch_size × hidden_1)'
        },
        {
          type: 'formula',
          content: 'A¹ = ReLU(Z¹)'
        },
        {
          type: 'formula',
          content: 'Z² = A¹ · W² + b²  (batch_size × hidden_2)'
        },
        {
          type: 'formula',
          content: 'Ŷ = σ(Z²)  (batch_size × output_size)'
        },
        {
          type: 'heading',
          content: 'Dimension Tracking'
        },
        {
          type: 'text',
          content: `Keeping track of dimensions is crucial. For a network with input size n, hidden size h, and output size k:

- **X**: (batch_size, n) - input data
- **W¹**: (n, h) - first layer weights
- **b¹**: (h,) - first layer bias (broadcasts)
- **A¹**: (batch_size, h) - hidden activations
- **W²**: (h, k) - output layer weights
- **Ŷ**: (batch_size, k) - predictions`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    Y_hat = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2}
    return Y_hat, cache

np.random.seed(42)
n_input, n_hidden, n_output = 4, 8, 1
batch_size = 32

params = {
    'W1': np.random.randn(n_input, n_hidden) * 0.01,
    'b1': np.zeros(n_hidden),
    'W2': np.random.randn(n_hidden, n_output) * 0.01,
    'b2': np.zeros(n_output)
}

X = np.random.randn(batch_size, n_input)
Y_hat, cache = forward(X, params)
print(f"Input shape: {X.shape}")
print(f"Output shape: {Y_hat.shape}")`
        },
        {
          type: 'heading',
          content: 'Why Cache Intermediate Values?'
        },
        {
          type: 'text',
          content: `Notice we save Z1, A1, Z2 in a cache. These intermediate values are needed for **backpropagation** - computing gradients to update weights.

During training:
1. Forward pass: compute predictions, save intermediate values
2. Compute loss
3. Backward pass: use cached values to compute gradients
4. Update weights`
        },
        {
          type: 'keypoints',
          points: [
            'Forward propagation computes predictions by flowing data through layers',
            'Each layer: linear transform (Wx + b) then activation',
            'Use matrix operations for efficient batch processing',
            'Track dimensions carefully: (batch, features)',
            'Cache intermediate values for backpropagation'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the order of operations in a single layer during forward propagation?',
          options: [
            'Activation → Linear transformation',
            'Linear transformation → Activation',
            'Activation → Activation',
            'Linear transformation → Linear transformation'
          ],
          correct: 1,
          explanation: 'Each layer first applies a linear transformation (Wx + b), then passes the result through a non-linear activation function.'
        },
        {
          type: 'multiple-choice',
          question: 'Why do we cache intermediate values during forward propagation?',
          options: [
            'To save memory',
            'To make predictions faster',
            'For use in backpropagation',
            'To normalize the outputs'
          ],
          correct: 2,
          explanation: 'The intermediate values (Z and A for each layer) are needed during backpropagation to compute gradients. Without them, we cannot efficiently train the network.'
        }
      ]
    },
    {
      id: 'loss-functions',
      title: 'Loss Functions',
      duration: '45 min',
      concepts: ['Cross-Entropy', 'MSE', 'Loss Landscape'],
      content: [
        {
          type: 'heading',
          content: 'Measuring Prediction Quality'
        },
        {
          type: 'text',
          content: `The loss function (cost function, objective function) measures how wrong our predictions are. Training a neural network means minimizing this loss.

**Good loss functions**:
- Are differentiable (for gradient descent)
- Have meaningful gradients (point toward better solutions)
- Match the problem type (classification vs regression)`
        },
        {
          type: 'heading',
          content: 'Mean Squared Error (MSE)'
        },
        {
          type: 'formula',
          content: 'MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²'
        },
        {
          type: 'text',
          content: `Used for regression. Penalizes large errors heavily due to squaring.

**Gradient**: d(MSE)/dŷ = (2/n)(ŷ - y)

The gradient is proportional to the error - larger errors produce larger gradients, pushing harder toward the correct answer.`
        },
        {
          type: 'heading',
          content: 'Binary Cross-Entropy'
        },
        {
          type: 'formula',
          content: 'BCE = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]'
        },
        {
          type: 'text',
          content: `Used for binary classification with sigmoid output. Measures the "distance" between predicted probabilities and true labels.

When y=1: loss = -log(ŷ) → penalizes low confidence
When y=0: loss = -log(1-ŷ) → penalizes high confidence

Perfect prediction (ŷ=y) → loss = 0`
        },
        {
          type: 'visualization',
          title: 'Binary Cross-Entropy Intuition',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- y=1 case -->
            <g transform="translate(30,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">When y = 1</text>
              <rect x="10" y="10" width="120" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="25" y1="95" x2="120" y2="95" stroke="#94a3b8" stroke-width="1"/>
              <line x1="25" y1="95" x2="25" y2="20" stroke="#94a3b8" stroke-width="1"/>
              <text x="75" y="120" text-anchor="middle" font-size="8" fill="#64748b">ŷ (prediction)</text>
              <text x="15" y="60" font-size="8" fill="#64748b" transform="rotate(-90,15,60)">Loss</text>

              <!-- -log(ŷ) curve -->
              <path d="M30,20 Q40,40 50,55 Q70,75 90,85 Q100,90 115,92" fill="none" stroke="#10b981" stroke-width="2"/>

              <!-- Labels -->
              <text x="35" y="17" font-size="7" fill="#ef4444">High loss</text>
              <text x="95" y="105" font-size="7" fill="#10b981">Low loss</text>
              <text x="30" y="105" font-size="7" fill="#64748b">0</text>
              <text x="110" y="105" font-size="7" fill="#64748b">1</text>
            </g>

            <!-- y=0 case -->
            <g transform="translate(200,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#3b82f6" font-weight="600">When y = 0</text>
              <rect x="10" y="10" width="120" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="25" y1="95" x2="120" y2="95" stroke="#94a3b8" stroke-width="1"/>
              <line x1="25" y1="95" x2="25" y2="20" stroke="#94a3b8" stroke-width="1"/>
              <text x="75" y="120" text-anchor="middle" font-size="8" fill="#64748b">ŷ (prediction)</text>

              <!-- -log(1-ŷ) curve -->
              <path d="M30,92 Q50,90 70,85 Q90,75 100,55 Q110,40 115,20" fill="none" stroke="#3b82f6" stroke-width="2"/>

              <!-- Labels -->
              <text x="95" y="17" font-size="7" fill="#ef4444">High loss</text>
              <text x="35" y="105" font-size="7" fill="#10b981">Low loss</text>
              <text x="30" y="105" font-size="7" fill="#64748b">0</text>
              <text x="110" y="105" font-size="7" fill="#64748b">1</text>
            </g>

            <!-- Explanation -->
            <text x="200" y="175" text-anchor="middle" font-size="9" fill="#475569">Loss is low when prediction matches label, high when it disagrees</text>
          </svg>`,
          caption: 'BCE penalizes confident wrong predictions severely'
        },
        {
          type: 'heading',
          content: 'Categorical Cross-Entropy'
        },
        {
          type: 'formula',
          content: 'CCE = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)'
        },
        {
          type: 'text',
          content: `Used for multi-class classification with softmax output. Labels are one-hot encoded: [0, 0, 1, 0] means class 3.

Only the true class contributes to loss (other terms are 0 × log(ŷ) = 0).`
        },
        {
          type: 'heading',
          content: 'The Loss Landscape'
        },
        {
          type: 'text',
          content: `Imagine the loss function as a surface over weight space. Each point is a set of weights, and the height is the loss for those weights.

Training = finding the lowest point on this surface.

**Challenges**:
- Local minima (valleys that aren't the lowest)
- Saddle points (flat in some directions)
- Plateaus (large flat regions)
- Sharp valleys (unstable optima)`
        },
        {
          type: 'visualization',
          title: 'Loss Landscape Visualization',
          svg: `<svg viewBox="0 0 350 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="350" height="180" fill="#f8fafc"/>

            <!-- 3D-ish loss surface -->
            <ellipse cx="175" cy="90" rx="140" ry="60" fill="none" stroke="#e2e8f0" stroke-width="1"/>

            <!-- Contour-like curves -->
            <ellipse cx="100" cy="85" rx="40" ry="20" fill="none" stroke="#3b82f6" stroke-width="1.5" opacity="0.6"/>
            <ellipse cx="100" cy="85" rx="25" ry="12" fill="none" stroke="#3b82f6" stroke-width="1.5" opacity="0.7"/>
            <ellipse cx="100" cy="85" rx="10" ry="5" fill="none" stroke="#3b82f6" stroke-width="1.5" opacity="0.8"/>

            <ellipse cx="240" cy="95" rx="35" ry="18" fill="none" stroke="#10b981" stroke-width="1.5" opacity="0.6"/>
            <ellipse cx="240" cy="95" rx="20" ry="10" fill="none" stroke="#10b981" stroke-width="1.5" opacity="0.7"/>
            <ellipse cx="240" cy="95" rx="8" ry="4" fill="none" stroke="#10b981" stroke-width="1.5" opacity="0.8"/>

            <!-- Local minimum -->
            <circle cx="100" cy="85" r="4" fill="#3b82f6"/>
            <text x="100" y="115" text-anchor="middle" font-size="8" fill="#3b82f6">Local minimum</text>

            <!-- Global minimum -->
            <circle cx="240" cy="95" r="4" fill="#10b981"/>
            <text x="240" y="125" text-anchor="middle" font-size="8" fill="#10b981">Global minimum</text>

            <!-- Saddle point region -->
            <text x="175" cy="65" text-anchor="middle" font-size="8" fill="#f59e0b">Saddle point</text>
            <circle cx="175" cy="75" r="3" fill="#f59e0b"/>

            <!-- Axes labels -->
            <text x="175" y="165" text-anchor="middle" font-size="9" fill="#64748b">Weight Space</text>

            <!-- Title -->
            <text x="175" y="20" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Loss Landscape</text>
          </svg>`,
          caption: 'The loss landscape has multiple minima - gradient descent may find a local one'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])
print(f"BCE Loss: {binary_crossentropy(y_true, y_pred):.4f}")`
        },
        {
          type: 'heading',
          content: 'Choosing a Loss Function'
        },
        {
          type: 'table',
          headers: ['Problem Type', 'Loss Function', 'Output Activation'],
          rows: [
            ['Regression', 'MSE or MAE', 'Linear (none)'],
            ['Binary Classification', 'Binary Cross-Entropy', 'Sigmoid'],
            ['Multi-class Classification', 'Categorical Cross-Entropy', 'Softmax'],
            ['Multi-label Classification', 'Binary Cross-Entropy (per label)', 'Sigmoid (per output)']
          ],
          caption: 'Match loss function to problem type'
        },
        {
          type: 'keypoints',
          points: [
            'Loss functions measure how wrong predictions are',
            'MSE for regression, cross-entropy for classification',
            'Cross-entropy penalizes confident wrong predictions severely',
            'The loss landscape is complex with many local minima',
            'Match loss function with output activation and problem type'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Which loss function should you use for multi-class classification?',
          options: [
            'Mean Squared Error',
            'Binary Cross-Entropy',
            'Categorical Cross-Entropy',
            'Mean Absolute Error'
          ],
          correct: 2,
          explanation: 'Categorical cross-entropy is designed for multi-class classification with softmax output. It computes loss based on the predicted probability of the true class.'
        },
        {
          type: 'multiple-choice',
          question: 'What happens to binary cross-entropy loss when the prediction is very wrong (e.g., predicting 0.01 when true label is 1)?',
          options: [
            'Loss is very low',
            'Loss is moderate',
            'Loss is very high',
            'Loss is undefined'
          ],
          correct: 2,
          explanation: 'BCE includes -log(ŷ) when y=1. When ŷ=0.01, -log(0.01) ≈ 4.6, which is very high. BCE severely penalizes confident wrong predictions.'
        }
      ]
    },
    {
      id: 'backpropagation',
      title: 'Backpropagation',
      duration: '60 min',
      concepts: ['Chain Rule', 'Gradient Flow', 'Weight Updates'],
      content: [
        {
          type: 'heading',
          content: 'The Heart of Neural Network Training'
        },
        {
          type: 'text',
          content: `Backpropagation is the algorithm that makes training neural networks possible. It efficiently computes how much each weight contributed to the error, allowing us to update weights in the right direction.

The key insight: use the **chain rule** to propagate gradients backward through the network.`
        },
        {
          type: 'heading',
          content: 'The Chain Rule'
        },
        {
          type: 'text',
          content: `If y = f(g(x)), then:`
        },
        {
          type: 'formula',
          content: 'dy/dx = (dy/dg) × (dg/dx)'
        },
        {
          type: 'text',
          content: `This lets us compute gradients through composed functions - which is exactly what neural networks are!`
        },
        {
          type: 'visualization',
          title: 'Chain Rule in Neural Networks',
          svg: `<svg viewBox="0 0 450 150" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="150" fill="#f8fafc"/>

            <!-- Forward pass (top) -->
            <g transform="translate(20,25)">
              <text x="0" y="0" font-size="9" fill="#3b82f6">Forward:</text>

              <rect x="40" y="-12" width="40" height="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="60" y="5" text-anchor="middle" font-size="10" fill="#1e40af">x</text>

              <path d="M85,0 L105,0" stroke="#64748b" stroke-width="1.5" marker-end="url(#fwdarrow)"/>

              <rect x="110" y="-12" width="40" height="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="130" y="5" text-anchor="middle" font-size="10" fill="#1e40af">z</text>

              <path d="M155,0 L175,0" stroke="#64748b" stroke-width="1.5" marker-end="url(#fwdarrow)"/>

              <rect x="180" y="-12" width="40" height="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="200" y="5" text-anchor="middle" font-size="10" fill="#1e40af">a</text>

              <path d="M225,0 L245,0" stroke="#64748b" stroke-width="1.5" marker-end="url(#fwdarrow)"/>

              <rect x="250" y="-12" width="40" height="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="270" y="5" text-anchor="middle" font-size="10" fill="#1e40af">ŷ</text>

              <path d="M295,0 L315,0" stroke="#64748b" stroke-width="1.5" marker-end="url(#fwdarrow)"/>

              <rect x="320" y="-12" width="40" height="25" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="3"/>
              <text x="340" y="5" text-anchor="middle" font-size="10" fill="#dc2626">L</text>

              <!-- Labels -->
              <text x="97" y="25" text-anchor="middle" font-size="8" fill="#64748b">Wx+b</text>
              <text x="167" y="25" text-anchor="middle" font-size="8" fill="#64748b">ReLU</text>
              <text x="237" y="25" text-anchor="middle" font-size="8" fill="#64748b">Wx+b</text>
              <text x="307" y="25" text-anchor="middle" font-size="8" fill="#64748b">Loss</text>
            </g>

            <!-- Backward pass (bottom) -->
            <g transform="translate(20,100)">
              <text x="0" y="0" font-size="9" fill="#ef4444">Backward:</text>

              <rect x="40" y="-12" width="40" height="25" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="3"/>
              <text x="60" y="5" text-anchor="middle" font-size="9" fill="#dc2626">∂L/∂x</text>

              <path d="M85,0 L105,0" stroke="#ef4444" stroke-width="1.5" marker-start="url(#bwdarrow)"/>

              <rect x="110" y="-12" width="40" height="25" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="3"/>
              <text x="130" y="5" text-anchor="middle" font-size="9" fill="#dc2626">∂L/∂z</text>

              <path d="M155,0 L175,0" stroke="#ef4444" stroke-width="1.5" marker-start="url(#bwdarrow)"/>

              <rect x="180" y="-12" width="40" height="25" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="3"/>
              <text x="200" y="5" text-anchor="middle" font-size="9" fill="#dc2626">∂L/∂a</text>

              <path d="M225,0 L245,0" stroke="#ef4444" stroke-width="1.5" marker-start="url(#bwdarrow)"/>

              <rect x="250" y="-12" width="40" height="25" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="3"/>
              <text x="270" y="5" text-anchor="middle" font-size="9" fill="#dc2626">∂L/∂ŷ</text>

              <path d="M295,0 L315,0" stroke="#ef4444" stroke-width="1.5" marker-start="url(#bwdarrow)"/>

              <rect x="320" y="-12" width="40" height="25" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="3"/>
              <text x="340" y="5" text-anchor="middle" font-size="9" fill="#dc2626">1</text>
            </g>

            <defs>
              <marker id="fwdarrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
              </marker>
              <marker id="bwdarrow" markerWidth="8" markerHeight="8" refX="0" refY="3" orient="auto">
                <path d="M7,0 L7,6 L0,3 z" fill="#ef4444"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Gradients flow backward through the network using the chain rule'
        },
        {
          type: 'heading',
          content: 'Computing Gradients Step by Step'
        },
        {
          type: 'text',
          content: `For a simple network: x → (Wx + b) → ReLU → (Wy + c) → sigmoid → L

**Step 1**: Start at the loss
∂L/∂ŷ = (ŷ - y) / (ŷ(1-ŷ))  for BCE loss

**Step 2**: Through sigmoid
∂L/∂z₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ = ∂L/∂ŷ × ŷ(1-ŷ) = (ŷ - y)

**Step 3**: Through linear layer 2
∂L/∂W₂ = a₁ᵀ × ∂L/∂z₂
∂L/∂a₁ = ∂L/∂z₂ × W₂ᵀ

**Step 4**: Through ReLU
∂L/∂z₁ = ∂L/∂a₁ × (z₁ > 0)  (derivative is 1 if z₁ > 0, else 0)

**Step 5**: Through linear layer 1
∂L/∂W₁ = xᵀ × ∂L/∂z₁`
        },
        {
          type: 'heading',
          content: 'The Algorithm'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

def backward(Y, Y_hat, cache, params):
    m = Y.shape[0]
    Z1, A1, Z2 = cache['Z1'], cache['A1'], cache['Z2']
    W1, W2 = params['W1'], params['W2']

    dZ2 = Y_hat - Y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0)

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradients

def update_params(params, gradients, learning_rate):
    params['W1'] -= learning_rate * gradients['dW1']
    params['b1'] -= learning_rate * gradients['db1']
    params['W2'] -= learning_rate * gradients['dW2']
    params['b2'] -= learning_rate * gradients['db2']
    return params`
        },
        {
          type: 'heading',
          content: 'Gradient Flow Through Layers'
        },
        {
          type: 'text',
          content: `Each layer receives a gradient from above (∂L/∂output) and must compute:
1. **Gradient w.r.t. weights**: How to adjust this layer's weights
2. **Gradient w.r.t. input**: What gradient to pass to the layer below

This continues until we reach the input layer.`
        },
        {
          type: 'visualization',
          title: 'Gradient Flow',
          svg: `<svg viewBox="0 0 350 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="350" height="200" fill="#f8fafc"/>

            <!-- Layer box -->
            <rect x="100" y="60" width="150" height="80" fill="white" stroke="#3b82f6" stroke-width="2" rx="4"/>
            <text x="175" y="100" text-anchor="middle" font-size="11" fill="#1e40af" font-weight="500">Layer l</text>
            <text x="175" y="120" text-anchor="middle" font-size="9" fill="#64748b">a = f(Wx + b)</text>

            <!-- Input -->
            <rect x="20" y="85" width="60" height="30" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" rx="3"/>
            <text x="50" y="105" text-anchor="middle" font-size="10" fill="#475569">x</text>
            <path d="M80,100 L100,100" stroke="#64748b" stroke-width="1.5" marker-end="url(#gradarrow)"/>

            <!-- Output -->
            <rect x="270" y="85" width="60" height="30" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" rx="3"/>
            <text x="300" y="105" text-anchor="middle" font-size="10" fill="#475569">a</text>
            <path d="M250,100 L270,100" stroke="#64748b" stroke-width="1.5" marker-end="url(#gradarrow)"/>

            <!-- Gradient from above -->
            <path d="M300,75 L300,85" stroke="#ef4444" stroke-width="2" marker-end="url(#reddownarrow)"/>
            <text x="300" y="55" text-anchor="middle" font-size="9" fill="#ef4444">∂L/∂a</text>

            <!-- Gradient to below -->
            <path d="M50,125 L50,135" stroke="#ef4444" stroke-width="2" marker-end="url(#reddownarrow)"/>
            <text x="50" y="165" text-anchor="middle" font-size="9" fill="#ef4444">∂L/∂x</text>

            <!-- Gradient to weights -->
            <path d="M175,140 L175,170" stroke="#10b981" stroke-width="2" marker-end="url(#greendownarrow)"/>
            <text x="175" y="188" text-anchor="middle" font-size="9" fill="#10b981">∂L/∂W, ∂L/∂b</text>

            <defs>
              <marker id="gradarrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
              </marker>
              <marker id="reddownarrow" markerWidth="8" markerHeight="8" refX="3" refY="7" orient="auto">
                <path d="M0,0 L6,0 L3,7 z" fill="#ef4444"/>
              </marker>
              <marker id="greendownarrow" markerWidth="8" markerHeight="8" refX="3" refY="7" orient="auto">
                <path d="M0,0 L6,0 L3,7 z" fill="#10b981"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Each layer receives gradient from above, computes gradients for weights and passes gradient below'
        },
        {
          type: 'heading',
          content: 'Common Derivatives'
        },
        {
          type: 'table',
          headers: ['Function', 'Derivative'],
          rows: [
            ['Sigmoid: σ(x)', 'σ(x)(1 - σ(x))'],
            ['Tanh: tanh(x)', '1 - tanh²(x)'],
            ['ReLU: max(0, x)', '1 if x > 0, else 0'],
            ['Linear: Wx + b', '∂/∂W = xᵀ, ∂/∂x = W'],
            ['MSE: (y - ŷ)²', '2(ŷ - y)'],
            ['BCE: -y log(ŷ)', '(ŷ - y) / (ŷ(1-ŷ))']
          ],
          caption: 'Derivatives you need for backpropagation'
        },
        {
          type: 'callout',
          variant: 'tip',
          content: 'Modern frameworks like PyTorch and TensorFlow compute gradients automatically using automatic differentiation. Understanding backprop helps you debug and understand what is happening.'
        },
        {
          type: 'keypoints',
          points: [
            'Backpropagation uses the chain rule to compute gradients efficiently',
            'Gradients flow backward from loss to input',
            'Each layer computes gradients for its weights and passes gradients backward',
            'The algorithm is: forward pass → compute loss → backward pass → update weights',
            'Modern frameworks handle backprop automatically'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What mathematical principle enables backpropagation?',
          options: [
            'Law of large numbers',
            'Chain rule of calculus',
            'Central limit theorem',
            'Bayes theorem'
          ],
          correct: 1,
          explanation: 'Backpropagation uses the chain rule to compute gradients through composed functions. It allows us to efficiently compute how each weight affects the final loss.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the derivative of ReLU(x) when x > 0?',
          options: [
            '0',
            '1',
            'x',
            '-1'
          ],
          correct: 1,
          explanation: 'ReLU(x) = max(0, x). When x > 0, ReLU(x) = x, so the derivative is 1. When x ≤ 0, ReLU(x) = 0, so the derivative is 0.'
        }
      ]
    },
    {
      id: 'nn-from-scratch',
      title: 'Building a Neural Network from Scratch',
      duration: '60 min',
      concepts: ['Implementation', 'Training Loop', 'Debugging'],
      content: [
        {
          type: 'heading',
          content: 'Putting It All Together'
        },
        {
          type: 'text',
          content: `Now we'll implement a complete neural network from scratch using only NumPy. This will solidify your understanding of every component: initialization, forward prop, loss, backprop, and updates.

We'll build a network that can learn XOR - the problem that defeated single perceptrons.`
        },
        {
          type: 'heading',
          content: 'The Complete Implementation'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.params = {}
        self.num_layers = len(layer_sizes) - 1

        for l in range(1, len(layer_sizes)):
            self.params[f'W{l}'] = np.random.randn(
                layer_sizes[l-1], layer_sizes[l]
            ) * np.sqrt(2 / layer_sizes[l-1])
            self.params[f'b{l}'] = np.zeros(layer_sizes[l])

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def forward(self, X):
        self.cache = {'A0': X}

        A = X
        for l in range(1, self.num_layers + 1):
            Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
            self.cache[f'Z{l}'] = Z

            if l == self.num_layers:
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)
            self.cache[f'A{l}'] = A

        return A

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        epsilon = 1e-15
        Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
        loss = -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return loss

    def backward(self, Y, Y_hat):
        m = Y.shape[0]
        self.gradients = {}

        dA = Y_hat - Y

        for l in range(self.num_layers, 0, -1):
            if l == self.num_layers:
                dZ = dA
            else:
                dZ = dA * self.relu_derivative(self.cache[f'Z{l}'])

            A_prev = self.cache[f'A{l-1}']
            self.gradients[f'dW{l}'] = (1/m) * A_prev.T @ dZ
            self.gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=0)

            if l > 1:
                dA = dZ @ self.params[f'W{l}'].T

    def update(self, learning_rate):
        for l in range(1, self.num_layers + 1):
            self.params[f'W{l}'] -= learning_rate * self.gradients[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * self.gradients[f'db{l}']

    def train(self, X, Y, epochs, learning_rate, print_every=100):
        losses = []
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            losses.append(loss)

            self.backward(Y, Y_hat)
            self.update(learning_rate)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)`
        },
        {
          type: 'heading',
          content: 'Training on XOR'
        },
        {
          type: 'code',
          language: 'python',
          content: `X = np.array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])

Y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])

losses = nn.train(X, Y, epochs=5000, learning_rate=1.0, print_every=1000)

print("\\nPredictions:")
predictions = nn.predict(X)
for i in range(len(X)):
    print(f"{X[i]} -> {predictions[i][0]} (true: {Y[i][0]})")`
        },
        {
          type: 'heading',
          content: 'Debugging Tips'
        },
        {
          type: 'text',
          content: `**Loss not decreasing?**
- Check learning rate (try 0.01, 0.1, 1.0)
- Verify gradient computation
- Ensure proper initialization

**Gradients are NaN?**
- Add epsilon to log() calls
- Clip sigmoid input to avoid overflow
- Check for division by zero

**Loss stuck at high value?**
- Network may need more capacity (add neurons)
- Training may need more epochs
- Try different random seed

**Gradient checking** - verify gradients numerically:`
        },
        {
          type: 'code',
          language: 'python',
          content: `def gradient_check(nn, X, Y, epsilon=1e-7):
    nn.forward(X)
    nn.backward(Y, nn.cache[f'A{nn.num_layers}'])

    for key in nn.params:
        analytical = nn.gradients[f'd{key}']

        numerical = np.zeros_like(nn.params[key])
        it = np.nditer(nn.params[key], flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index

            original = nn.params[key][idx]

            nn.params[key][idx] = original + epsilon
            Y_hat_plus = nn.forward(X)
            loss_plus = nn.compute_loss(Y, Y_hat_plus)

            nn.params[key][idx] = original - epsilon
            Y_hat_minus = nn.forward(X)
            loss_minus = nn.compute_loss(Y, Y_hat_minus)

            numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            nn.params[key][idx] = original

            it.iternext()

        diff = np.max(np.abs(analytical - numerical))
        print(f"{key}: max difference = {diff:.2e}")`
        },
        {
          type: 'heading',
          content: 'What We Learned'
        },
        {
          type: 'visualization',
          title: 'Neural Network Training Loop',
          svg: `<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="220" fill="#f8fafc"/>

            <!-- Training loop circle -->
            <ellipse cx="200" cy="110" rx="120" ry="80" fill="none" stroke="#e2e8f0" stroke-width="2"/>

            <!-- Forward -->
            <rect x="160" y="25" width="80" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="2" rx="4"/>
            <text x="200" y="45" text-anchor="middle" font-size="10" fill="#1e40af">Forward</text>

            <!-- Loss -->
            <rect x="280" y="70" width="70" height="30" fill="#fee2e2" stroke="#ef4444" stroke-width="2" rx="4"/>
            <text x="315" y="90" text-anchor="middle" font-size="10" fill="#dc2626">Loss</text>

            <!-- Backward -->
            <rect x="160" y="155" width="80" height="30" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="4"/>
            <text x="200" y="175" text-anchor="middle" font-size="10" fill="#92400e">Backward</text>

            <!-- Update -->
            <rect x="50" y="70" width="70" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="2" rx="4"/>
            <text x="85" y="90" text-anchor="middle" font-size="10" fill="#065f46">Update</text>

            <!-- Arrows -->
            <path d="M240,40 Q300,40 315,70" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#loopArrow)"/>
            <path d="M315,100 Q315,155 240,170" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#loopArrow)"/>
            <path d="M160,170 Q85,170 85,100" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#loopArrow)"/>
            <path d="M85,70 Q85,40 160,40" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#loopArrow)"/>

            <!-- Center text -->
            <text x="200" y="105" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">Repeat until</text>
            <text x="200" y="120" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">convergence</text>

            <defs>
              <marker id="loopArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                <path d="M0,0 L0,6 L7,3 z" fill="#64748b"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'The training loop: forward → loss → backward → update → repeat'
        },
        {
          type: 'keypoints',
          points: [
            'Initialize weights with proper scaling (Xavier/He initialization)',
            'Forward propagation computes predictions layer by layer',
            'Loss measures how wrong the predictions are',
            'Backpropagation computes gradients for all parameters',
            'Update weights in the direction that reduces loss',
            'Use gradient checking to verify your implementation'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'In what order are the four main steps of training executed?',
          options: [
            'Backward → Forward → Update → Loss',
            'Forward → Loss → Backward → Update',
            'Update → Forward → Backward → Loss',
            'Loss → Forward → Update → Backward'
          ],
          correct: 1,
          explanation: 'The training loop is: Forward pass (compute predictions) → Loss (measure error) → Backward pass (compute gradients) → Update (adjust weights).'
        },
        {
          type: 'multiple-choice',
          question: 'What is gradient checking used for?',
          options: [
            'Making training faster',
            'Verifying that gradient computations are correct',
            'Choosing the learning rate',
            'Preventing overfitting'
          ],
          correct: 1,
          explanation: 'Gradient checking compares your analytical gradients (from backprop) against numerical gradients (computed by finite differences). If they match, your backprop implementation is likely correct.'
        }
      ]
    }
  ]
}
