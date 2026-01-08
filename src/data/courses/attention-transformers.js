export const attentiontransformers = {
  id: 'attention-transformers',
  title: 'Attention & Transformers',
  description: 'Master the architecture that revolutionized deep learning',
  category: 'Deep Learning',
  difficulty: 'Advanced',
  duration: '8 hours',
  lessons: [
    {
      id: 'attention-intuition',
      title: 'Attention Mechanism Intuition',
      duration: '55 min',
      concepts: ['attention', 'alignment', 'context vectors', 'soft attention'],
      content: [
        {
          type: 'heading',
          text: 'The Attention Revolution'
        },
        {
          type: 'text',
          text: 'Attention is perhaps the most important innovation in deep learning since backpropagation. The intuition is simple: when processing a sequence, not all elements are equally important. Attention lets the model focus on the relevant parts dynamically.'
        },
        {
          type: 'subheading',
          text: 'The Problem Attention Solves'
        },
        {
          type: 'text',
          text: 'In sequence-to-sequence models, the encoder compresses the entire input into a single fixed-size vector. For long sequences, this bottleneck loses information. Attention allows the decoder to look back at all encoder states, not just the final one.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Attention: Dynamic Focus</text>
            <rect x="50" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="80" y="70" text-anchor="middle" font-size="10" fill="#1e293b">The</text>
            <rect x="120" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="150" y="70" text-anchor="middle" font-size="10" fill="#1e293b">cat</text>
            <rect x="190" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="220" y="70" text-anchor="middle" font-size="10" fill="#1e293b">sat</text>
            <rect x="260" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="290" y="70" text-anchor="middle" font-size="10" fill="#1e293b">on</text>
            <rect x="330" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="360" y="70" text-anchor="middle" font-size="10" fill="#1e293b">mat</text>
            <rect x="200" y="160" width="100" height="35" fill="#dcfce7" stroke="#22c55e" stroke-width="2" rx="6"/>
            <text x="250" y="182" text-anchor="middle" font-size="11" fill="#15803d">Output: "sat"</text>
            <line x1="80" y1="80" x2="220" y2="160" stroke="#6366f1" stroke-width="1" opacity="0.2"/>
            <line x1="150" y1="80" x2="230" y2="160" stroke="#6366f1" stroke-width="1" opacity="0.3"/>
            <line x1="220" y1="80" x2="250" y2="160" stroke="#6366f1" stroke-width="4" opacity="0.9"/>
            <line x1="290" y1="80" x2="270" y2="160" stroke="#6366f1" stroke-width="1" opacity="0.2"/>
            <line x1="360" y1="80" x2="280" y2="160" stroke="#6366f1" stroke-width="1" opacity="0.1"/>
            <text x="80" y="110" text-anchor="middle" font-size="9" fill="#64748b">0.05</text>
            <text x="150" y="110" text-anchor="middle" font-size="9" fill="#64748b">0.10</text>
            <text x="220" y="110" text-anchor="middle" font-size="9" fill="#64748b">0.70</text>
            <text x="290" y="110" text-anchor="middle" font-size="9" fill="#64748b">0.10</text>
            <text x="360" y="110" text-anchor="middle" font-size="9" fill="#64748b">0.05</text>
            <text x="250" y="210" text-anchor="middle" font-size="10" fill="#64748b">Attention weights show which input tokens are most relevant</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Query, Key, Value'
        },
        {
          type: 'text',
          text: 'Attention uses three projections: Query (what am I looking for?), Key (what do I contain?), and Value (what do I provide if selected?). The attention score is computed between query and keys, then used to weight values.'
        },
        {
          type: 'formula',
          latex: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V'
        },
        {
          type: 'text',
          text: 'The division by √d_k prevents dot products from becoming too large, which would push softmax into regions with tiny gradients.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention(query, key, value):
    d_k = query.shape[-1]
    scores = query @ key.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ value
    return output, weights

seq_len, d_model = 5, 64
query = np.random.randn(1, d_model)
key = np.random.randn(seq_len, d_model)
value = np.random.randn(seq_len, d_model)

output, weights = attention(query, key, value)
print(f"Attention weights: {weights.round(3)}")
print(f"Weights sum to: {weights.sum():.4f}")`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Why Softmax?',
          text: 'Softmax ensures attention weights sum to 1, creating a probability distribution over input positions. This makes the output a weighted average of values, where weights reflect relevance.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What problem does attention solve in sequence-to-sequence models?',
          options: [
            'Making training faster',
            'Reducing model size',
            'Removing the fixed-size bottleneck between encoder and decoder',
            'Eliminating the need for recurrence'
          ],
          correct: 2,
          explanation: 'Without attention, the entire input sequence must be compressed into a single vector. Attention allows the decoder to access all encoder hidden states directly, removing this bottleneck.'
        }
      ]
    },
    {
      id: 'self-attention',
      title: 'Self-Attention Mathematics',
      duration: '65 min',
      concepts: ['self-attention', 'scaled dot-product', 'attention matrix', 'quadratic complexity'],
      content: [
        {
          type: 'heading',
          text: 'Self-Attention: Attending to Yourself'
        },
        {
          type: 'text',
          text: 'Self-attention is when a sequence attends to itself. Each position can attend to every other position (including itself), allowing the model to capture relationships between any two tokens regardless of distance.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 250" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Self-Attention Matrix</text>
            <g transform="translate(150, 40)">
              <rect x="0" y="40" width="40" height="40" fill="#dbeafe" stroke="#3b82f6"/>
              <rect x="40" y="40" width="40" height="40" fill="#93c5fd" stroke="#3b82f6"/>
              <rect x="80" y="40" width="40" height="40" fill="#dbeafe" stroke="#3b82f6"/>
              <rect x="120" y="40" width="40" height="40" fill="#eff6ff" stroke="#3b82f6"/>
              <rect x="0" y="80" width="40" height="40" fill="#93c5fd" stroke="#3b82f6"/>
              <rect x="40" y="80" width="40" height="40" fill="#3b82f6" stroke="#3b82f6"/>
              <rect x="80" y="80" width="40" height="40" fill="#93c5fd" stroke="#3b82f6"/>
              <rect x="120" y="80" width="40" height="40" fill="#dbeafe" stroke="#3b82f6"/>
              <rect x="0" y="120" width="40" height="40" fill="#dbeafe" stroke="#3b82f6"/>
              <rect x="40" y="120" width="40" height="40" fill="#93c5fd" stroke="#3b82f6"/>
              <rect x="80" y="120" width="40" height="40" fill="#3b82f6" stroke="#3b82f6"/>
              <rect x="120" y="120" width="40" height="40" fill="#93c5fd" stroke="#3b82f6"/>
              <rect x="0" y="160" width="40" height="40" fill="#eff6ff" stroke="#3b82f6"/>
              <rect x="40" y="160" width="40" height="40" fill="#dbeafe" stroke="#3b82f6"/>
              <rect x="80" y="160" width="40" height="40" fill="#93c5fd" stroke="#3b82f6"/>
              <rect x="120" y="160" width="40" height="40" fill="#dbeafe" stroke="#3b82f6"/>
              <text x="20" y="30" text-anchor="middle" font-size="10" fill="#1e293b">The</text>
              <text x="60" y="30" text-anchor="middle" font-size="10" fill="#1e293b">cat</text>
              <text x="100" y="30" text-anchor="middle" font-size="10" fill="#1e293b">sat</text>
              <text x="140" y="30" text-anchor="middle" font-size="10" fill="#1e293b">down</text>
              <text x="-15" y="65" text-anchor="middle" font-size="10" fill="#1e293b">The</text>
              <text x="-15" y="105" text-anchor="middle" font-size="10" fill="#1e293b">cat</text>
              <text x="-15" y="145" text-anchor="middle" font-size="10" fill="#1e293b">sat</text>
              <text x="-15" y="185" text-anchor="middle" font-size="10" fill="#1e293b">down</text>
            </g>
            <text x="250" y="235" text-anchor="middle" font-size="10" fill="#64748b">Darker = higher attention weight</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Step-by-Step Computation'
        },
        {
          type: 'text',
          text: 'Given input X with shape (seq_len, d_model), self-attention computes:'
        },
        {
          type: 'text',
          text: '**Step 1**: Project to Q, K, V using learned weight matrices'
        },
        {
          type: 'formula',
          latex: 'Q = XW^Q, \\quad K = XW^K, \\quad V = XW^V'
        },
        {
          type: 'text',
          text: '**Step 2**: Compute attention scores (compatibility between each query and all keys)'
        },
        {
          type: 'formula',
          latex: '\\text{scores} = \\frac{QK^T}{\\sqrt{d_k}}'
        },
        {
          type: 'text',
          text: '**Step 3**: Apply softmax to get weights, then compute weighted sum of values'
        },
        {
          type: 'formula',
          latex: '\\text{output} = \\text{softmax}(\\text{scores}) \\cdot V'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

class SelfAttention:
    def __init__(self, d_model, d_k):
        self.d_k = d_k
        self.wq = np.random.randn(d_model, d_k) * 0.1
        self.wk = np.random.randn(d_model, d_k) * 0.1
        self.wv = np.random.randn(d_model, d_k) * 0.1

    def forward(self, x, mask=None):
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        scores = q @ k.T / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask * -1e9

        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        output = weights @ v
        return output, weights

seq_len, d_model, d_k = 6, 64, 32
x = np.random.randn(seq_len, d_model)
attention = SelfAttention(d_model, d_k)
output, weights = attention.forward(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention matrix shape: {weights.shape}")`
        },
        {
          type: 'subheading',
          text: 'Computational Complexity'
        },
        {
          type: 'text',
          text: 'Self-attention has O(n²) complexity where n is sequence length. The attention matrix stores n×n scores. This becomes prohibitive for very long sequences (thousands of tokens), motivating efficient attention variants.'
        },
        {
          type: 'table',
          headers: ['Sequence Length', 'Attention Matrix Size', 'Memory (float32)'],
          rows: [
            ['512', '262,144', '1 MB'],
            ['2,048', '4,194,304', '16 MB'],
            ['8,192', '67,108,864', '256 MB'],
            ['32,768', '1,073,741,824', '4 GB']
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the computational complexity of self-attention with respect to sequence length n?',
          options: [
            'O(n)',
            'O(n log n)',
            'O(n²)',
            'O(n³)'
          ],
          correct: 2,
          explanation: 'Self-attention computes attention scores between every pair of positions, resulting in an n×n attention matrix. This quadratic complexity is why long sequences are challenging for standard Transformers.'
        }
      ]
    },
    {
      id: 'multi-head-attention',
      title: 'Multi-Head Attention',
      duration: '55 min',
      concepts: ['multiple heads', 'parallel attention', 'head concatenation', 'diverse representations'],
      content: [
        {
          type: 'heading',
          text: 'Multiple Attention Heads'
        },
        {
          type: 'text',
          text: 'A single attention head can only focus on one type of relationship at a time. Multi-head attention runs multiple attention operations in parallel, each learning different relationship patterns. One head might focus on syntax, another on semantics.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Multi-Head Attention</text>
            <rect x="200" y="40" width="100" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="250" y="60" text-anchor="middle" font-size="11" fill="#4f46e5">Input X</text>
            <rect x="60" y="100" width="70" height="35" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="95" y="122" text-anchor="middle" font-size="10" fill="#92400e">Head 1</text>
            <rect x="150" y="100" width="70" height="35" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="185" y="122" text-anchor="middle" font-size="10" fill="#1d4ed8">Head 2</text>
            <rect x="240" y="100" width="70" height="35" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="275" y="122" text-anchor="middle" font-size="10" fill="#15803d">Head 3</text>
            <rect x="330" y="100" width="70" height="35" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="365" y="122" text-anchor="middle" font-size="10" fill="#7c3aed">Head 4</text>
            <line x1="220" y1="70" x2="95" y2="100" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="240" y1="70" x2="185" y2="100" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="260" y1="70" x2="275" y2="100" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="280" y1="70" x2="365" y2="100" stroke="#6366f1" stroke-width="1.5"/>
            <rect x="140" y="160" width="120" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="200" y="180" text-anchor="middle" font-size="10" fill="#4f46e5">Concat + Linear</text>
            <line x1="95" y1="135" x2="170" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="185" y1="135" x2="190" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="275" y1="135" x2="210" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="365" y1="135" x2="230" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <rect x="290" y="160" width="100" height="30" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="340" y="180" text-anchor="middle" font-size="10" fill="#15803d">Output</text>
            <line x1="260" y1="175" x2="290" y2="175" stroke="#6366f1" stroke-width="1.5"/>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Multi-Head Attention Formula'
        },
        {
          type: 'formula',
          latex: '\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O'
        },
        {
          type: 'formula',
          latex: '\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)'
        },
        {
          type: 'text',
          text: 'Each head has its own projection matrices. If d_model=512 and we use 8 heads, each head operates in d_k = d_v = 512/8 = 64 dimensions. The concatenated output is projected back to d_model.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = np.random.randn(d_model, d_model) * 0.1
        self.wk = np.random.randn(d_model, d_model) * 0.1
        self.wv = np.random.randn(d_model, d_model) * 0.1
        self.wo = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        batch_size = 1
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)

    def forward(self, x, mask=None):
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask * -1e9

        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        attended = np.matmul(weights, v)
        attended = attended.transpose(1, 0, 2).reshape(-1, self.num_heads * self.d_k)

        output = attended @ self.wo
        return output, weights

mha = MultiHeadAttention(d_model=256, num_heads=8)
x = np.random.randn(10, 256)
output, attention_weights = mha.forward(x)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")`
        },
        {
          type: 'subheading',
          text: 'What Different Heads Learn'
        },
        {
          type: 'text',
          text: 'Research shows different heads specialize in different linguistic phenomena:'
        },
        {
          type: 'table',
          headers: ['Head Type', 'What It Captures', 'Example'],
          rows: [
            ['Positional', 'Nearby tokens', 'Adjacent word relationships'],
            ['Syntactic', 'Grammatical structure', 'Subject-verb agreement'],
            ['Semantic', 'Meaning relationships', 'Coreference resolution'],
            ['Rare pattern', 'Specific constructions', 'Idioms, named entities']
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why use multiple attention heads instead of one large head?',
          options: [
            'To reduce computation',
            'Each head can learn different types of relationships in parallel',
            'To avoid overfitting',
            'To reduce memory usage'
          ],
          correct: 1,
          explanation: 'Multiple heads allow the model to attend to information from different representation subspaces at different positions. One head might focus on syntax while another focuses on semantics.'
        }
      ]
    },
    {
      id: 'positional-encoding',
      title: 'Positional Encoding',
      duration: '50 min',
      concepts: ['position information', 'sinusoidal encoding', 'learned embeddings', 'relative positions'],
      content: [
        {
          type: 'heading',
          text: 'Adding Position Information'
        },
        {
          type: 'text',
          text: 'Self-attention is permutation invariant—it treats "dog bites man" and "man bites dog" identically because it has no notion of position. Positional encodings inject sequence order information into the model.'
        },
        {
          type: 'subheading',
          text: 'Sinusoidal Positional Encoding'
        },
        {
          type: 'text',
          text: 'The original Transformer uses sinusoidal functions of different frequencies:'
        },
        {
          type: 'formula',
          latex: 'PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)'
        },
        {
          type: 'formula',
          latex: 'PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)'
        },
        {
          type: 'text',
          text: 'Each dimension uses a different frequency. Lower dimensions change rapidly (capturing fine position), higher dimensions change slowly (capturing coarse position). This creates a unique pattern for each position.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Sinusoidal Positional Encoding Pattern</text>
            <line x1="50" y1="100" x2="450" y2="100" stroke="#e2e8f0" stroke-width="1"/>
            <path d="M 50 100 Q 75 60 100 100 Q 125 140 150 100 Q 175 60 200 100 Q 225 140 250 100 Q 275 60 300 100 Q 325 140 350 100 Q 375 60 400 100 Q 425 140 450 100" fill="none" stroke="#6366f1" stroke-width="2"/>
            <text x="470" y="105" font-size="9" fill="#6366f1">dim 0</text>
            <path d="M 50 100 Q 100 70 150 100 Q 200 130 250 100 Q 300 70 350 100 Q 400 130 450 100" fill="none" stroke="#f59e0b" stroke-width="2"/>
            <text x="470" y="85" font-size="9" fill="#f59e0b">dim 4</text>
            <path d="M 50 100 Q 150 75 250 100 Q 350 125 450 100" fill="none" stroke="#22c55e" stroke-width="2"/>
            <text x="470" y="125" font-size="9" fill="#22c55e">dim 8</text>
            <text x="50" y="170" font-size="10" fill="#64748b">Position 0</text>
            <text x="250" y="170" font-size="10" fill="#64748b">Position 50</text>
            <text x="430" y="170" font-size="10" fill="#64748b">Position 100</text>
            <text x="250" y="190" text-anchor="middle" font-size="10" fill="#64748b">Lower dimensions oscillate faster</text>
          </svg>`
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def sinusoidal_positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

pe = sinusoidal_positional_encoding(100, 256)
print(f"Positional encoding shape: {pe.shape}")

def relative_position_property(pe, pos1, pos2):
    offset = pos2 - pos1
    return np.dot(pe[pos1], pe[pos2])

print(f"PE[10] · PE[15] = {relative_position_property(pe, 10, 15):.4f}")
print(f"PE[20] · PE[25] = {relative_position_property(pe, 20, 25):.4f}")
print(f"PE[10] · PE[50] = {relative_position_property(pe, 10, 50):.4f}")`
        },
        {
          type: 'subheading',
          text: 'Learned vs Fixed Positional Encodings'
        },
        {
          type: 'table',
          headers: ['Type', 'Pros', 'Cons'],
          rows: [
            ['Sinusoidal', 'Generalizes to unseen lengths, no extra params', 'May not capture task-specific patterns'],
            ['Learned', 'Can adapt to task, often performs better', 'Limited to training length, more parameters'],
            ['Relative', 'Captures relative distance directly', 'More complex, slower']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Modern Approaches',
          text: 'Most modern models use learned positional embeddings (GPT, BERT) or rotary position embeddings (RoPE) which encode relative positions through rotation in the embedding space.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why do Transformers need positional encodings?',
          options: [
            'To speed up training',
            'Self-attention is position-agnostic without them',
            'To reduce model size',
            'To improve gradient flow'
          ],
          correct: 1,
          explanation: 'Self-attention computes the same output regardless of token order. Without positional encodings, the model cannot distinguish "the cat sat" from "sat cat the".'
        }
      ]
    },
    {
      id: 'transformer-architecture',
      title: 'The Transformer Architecture',
      duration: '70 min',
      concepts: ['encoder-decoder', 'layer normalization', 'feed-forward network', 'residual connections'],
      content: [
        {
          type: 'heading',
          text: 'The Complete Transformer'
        },
        {
          type: 'text',
          text: 'The Transformer, introduced in "Attention Is All You Need" (2017), combines self-attention with feed-forward networks, residual connections, and layer normalization into a powerful architecture that has become the foundation of modern NLP.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 350" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Transformer Encoder Block</text>
            <rect x="150" y="40" width="200" height="280" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" rx="8"/>
            <rect x="175" y="260" width="150" height="35" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="250" y="282" text-anchor="middle" font-size="10" fill="#4f46e5">Input + Pos Encoding</text>
            <rect x="175" y="200" width="150" height="35" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="250" y="222" text-anchor="middle" font-size="10" fill="#1d4ed8">Multi-Head Attention</text>
            <rect x="175" y="150" width="150" height="35" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="250" y="172" text-anchor="middle" font-size="10" fill="#15803d">Add & Norm</text>
            <rect x="175" y="100" width="150" height="35" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="250" y="122" text-anchor="middle" font-size="10" fill="#92400e">Feed Forward</text>
            <rect x="175" y="50" width="150" height="35" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="250" y="72" text-anchor="middle" font-size="10" fill="#15803d">Add & Norm</text>
            <line x1="250" y1="260" x2="250" y2="235" stroke="#6366f1" stroke-width="2"/>
            <line x1="250" y1="200" x2="250" y2="185" stroke="#6366f1" stroke-width="2"/>
            <line x1="250" y1="150" x2="250" y2="135" stroke="#6366f1" stroke-width="2"/>
            <line x1="250" y1="100" x2="250" y2="85" stroke="#6366f1" stroke-width="2"/>
            <path d="M 340 275 L 360 275 L 360 165 L 325 165" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,2"/>
            <text x="375" y="220" font-size="8" fill="#ef4444">Residual</text>
            <path d="M 340 115 L 370 115 L 370 65 L 325 65" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,2"/>
            <text x="385" y="90" font-size="8" fill="#ef4444">Residual</text>
            <text x="250" y="340" text-anchor="middle" font-size="10" fill="#64748b">×N layers stacked</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Key Components'
        },
        {
          type: 'text',
          text: '**Multi-Head Self-Attention**: Captures relationships between all positions in the sequence.'
        },
        {
          type: 'text',
          text: '**Feed-Forward Network**: Two linear layers with ReLU, applied independently to each position. Adds non-linearity and increases model capacity.'
        },
        {
          type: 'formula',
          latex: 'FFN(x) = \\max(0, xW_1 + b_1)W_2 + b_2'
        },
        {
          type: 'text',
          text: '**Residual Connections**: Add input to output of each sublayer. Enables training of very deep networks by providing gradient highways.'
        },
        {
          type: 'text',
          text: '**Layer Normalization**: Normalizes across features for each position. Stabilizes training and speeds convergence.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta

def feed_forward(x, w1, b1, w2, b2):
    hidden = np.maximum(0, x @ w1 + b1)
    return hidden @ w2 + b2

class TransformerEncoderLayer:
    def __init__(self, d_model, d_ff, num_heads):
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.w1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha.forward(x, mask)
        x = layer_norm(x + attn_output, self.gamma1, self.beta1)
        ff_output = feed_forward(x, self.w1, self.b1, self.w2, self.b2)
        x = layer_norm(x + ff_output, self.gamma2, self.beta2)
        return x

layer = TransformerEncoderLayer(d_model=256, d_ff=1024, num_heads=8)
x = np.random.randn(20, 256)
output = layer.forward(x)
print(f"Output shape: {output.shape}")`
        },
        {
          type: 'subheading',
          text: 'Encoder vs Decoder'
        },
        {
          type: 'table',
          headers: ['Component', 'Encoder', 'Decoder'],
          rows: [
            ['Self-attention', 'Bidirectional (sees all)', 'Masked (sees only past)'],
            ['Cross-attention', 'None', 'Attends to encoder output'],
            ['Use case', 'Understanding input', 'Generating output'],
            ['Example models', 'BERT, RoBERTa', 'GPT, LLaMA']
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the purpose of residual connections in Transformers?',
          options: [
            'To reduce model size',
            'To enable gradient flow through deep networks',
            'To speed up inference',
            'To add non-linearity'
          ],
          correct: 1,
          explanation: 'Residual connections add the input directly to the output, creating a "shortcut" for gradients. This prevents vanishing gradients and enables training of much deeper networks.'
        }
      ]
    },
    {
      id: 'bert-encoders',
      title: 'BERT and Encoder Models',
      duration: '55 min',
      concepts: ['bidirectional encoding', 'masked language modeling', 'pre-training', 'fine-tuning'],
      content: [
        {
          type: 'heading',
          text: 'BERT: Bidirectional Encoder Representations'
        },
        {
          type: 'text',
          text: 'BERT uses only the encoder part of the Transformer with a key innovation: bidirectional pre-training. Unlike GPT which reads left-to-right, BERT can look at context from both directions simultaneously, making it powerful for understanding tasks.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">BERT: Masked Language Modeling</text>
            <rect x="40" y="60" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="65" y="80" text-anchor="middle" font-size="9" fill="#1e293b">The</text>
            <rect x="100" y="60" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="125" y="80" text-anchor="middle" font-size="9" fill="#1e293b">cat</text>
            <rect x="160" y="60" width="50" height="30" fill="#fecaca" stroke="#ef4444" stroke-width="2" rx="4"/>
            <text x="185" y="80" text-anchor="middle" font-size="9" fill="#ef4444">[MASK]</text>
            <rect x="220" y="60" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="245" y="80" text-anchor="middle" font-size="9" fill="#1e293b">on</text>
            <rect x="280" y="60" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="305" y="80" text-anchor="middle" font-size="9" fill="#1e293b">the</text>
            <rect x="340" y="60" width="50" height="30" fill="#fecaca" stroke="#ef4444" stroke-width="2" rx="4"/>
            <text x="365" y="80" text-anchor="middle" font-size="9" fill="#ef4444">[MASK]</text>
            <rect x="400" y="60" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="425" y="80" text-anchor="middle" font-size="9" fill="#1e293b">.</text>
            <rect x="120" y="130" width="260" height="40" fill="#f8fafc" stroke="#6366f1" rx="6"/>
            <text x="250" y="155" text-anchor="middle" font-size="11" fill="#4f46e5">Transformer Encoder (12 layers)</text>
            <line x1="185" y1="90" x2="185" y2="130" stroke="#ef4444" stroke-width="2"/>
            <line x1="365" y1="90" x2="365" y2="130" stroke="#ef4444" stroke-width="2"/>
            <text x="185" y="185" text-anchor="middle" font-size="10" fill="#22c55e">Predict: "sat"</text>
            <text x="365" y="185" text-anchor="middle" font-size="10" fill="#22c55e">Predict: "mat"</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Pre-training Objectives'
        },
        {
          type: 'text',
          text: '**Masked Language Modeling (MLM)**: Randomly mask 15% of tokens and predict them. This forces bidirectional understanding—you need both left and right context to guess the masked word.'
        },
        {
          type: 'text',
          text: '**Next Sentence Prediction (NSP)**: Given two sentences, predict if the second follows the first in the original text. Helps with tasks requiring sentence-pair understanding (though later work showed this may not be essential).'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The cat [MASK] on the mat."
inputs = tokenizer(text, return_tensors="pt")
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits[0, mask_idx]
    top_tokens = torch.topk(predictions, 5)

print("Top predictions for [MASK]:")
for score, idx in zip(top_tokens.values, top_tokens.indices):
    token = tokenizer.decode([idx])
    print(f"  {token}: {score:.2f}")`
        },
        {
          type: 'subheading',
          text: 'Fine-tuning BERT'
        },
        {
          type: 'text',
          text: 'After pre-training on massive text corpora, BERT is fine-tuned for specific tasks by adding a task-specific head and training on labeled data:'
        },
        {
          type: 'table',
          headers: ['Task', 'Input Format', 'Output'],
          rows: [
            ['Classification', '[CLS] text [SEP]', 'Class from [CLS] embedding'],
            ['NER', '[CLS] tokens [SEP]', 'Label per token'],
            ['QA', '[CLS] question [SEP] context [SEP]', 'Start/end positions'],
            ['Similarity', '[CLS] sent1 [SEP] sent2 [SEP]', 'Similarity score']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'BERT Variants',
          text: 'RoBERTa removes NSP and trains longer. ALBERT shares parameters across layers. DistilBERT is a smaller, faster version. These variants improve efficiency or performance while keeping the core architecture.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What makes BERT bidirectional unlike GPT?',
          options: [
            'It has more layers',
            'It uses masked language modeling so each token sees both left and right context',
            'It trains on more data',
            'It has larger embeddings'
          ],
          correct: 1,
          explanation: 'GPT predicts the next token and can only see previous tokens. BERT masks random tokens and predicts them using context from both directions, enabling truly bidirectional understanding.'
        }
      ]
    },
    {
      id: 'gpt-decoders',
      title: 'GPT and Decoder Models',
      duration: '55 min',
      concepts: ['autoregressive', 'causal masking', 'text generation', 'in-context learning'],
      content: [
        {
          type: 'heading',
          text: 'GPT: Generative Pre-trained Transformer'
        },
        {
          type: 'text',
          text: 'GPT uses only the decoder part of the Transformer. It is autoregressive—it generates text one token at a time, each prediction conditioned only on previous tokens. This makes it excellent for generation but means it cannot "see the future" like BERT.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">GPT: Autoregressive Generation</text>
            <rect x="50" y="70" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="75" y="90" text-anchor="middle" font-size="9" fill="#1e293b">The</text>
            <rect x="110" y="70" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="135" y="90" text-anchor="middle" font-size="9" fill="#1e293b">cat</text>
            <rect x="170" y="70" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="195" y="90" text-anchor="middle" font-size="9" fill="#1e293b">sat</text>
            <rect x="230" y="70" width="50" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="255" y="90" text-anchor="middle" font-size="9" fill="#1e293b">on</text>
            <rect x="290" y="70" width="50" height="30" fill="#dcfce7" stroke="#22c55e" stroke-width="2" rx="4"/>
            <text x="315" y="90" text-anchor="middle" font-size="9" fill="#22c55e">the</text>
            <rect x="350" y="70" width="50" height="30" fill="#f3e8ff" stroke="#a855f7" stroke-width="2" stroke-dasharray="4,2" rx="4"/>
            <text x="375" y="90" text-anchor="middle" font-size="9" fill="#a855f7">???</text>
            <line x1="75" y1="100" x2="315" y2="130" stroke="#6366f1" stroke-width="1" opacity="0.3"/>
            <line x1="135" y1="100" x2="315" y2="130" stroke="#6366f1" stroke-width="1" opacity="0.4"/>
            <line x1="195" y1="100" x2="315" y2="130" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
            <line x1="255" y1="100" x2="315" y2="130" stroke="#6366f1" stroke-width="1" opacity="0.7"/>
            <text x="315" y="150" text-anchor="middle" font-size="10" fill="#64748b">Attends to all previous tokens</text>
            <line x1="375" y1="100" x2="375" y2="130" stroke="#a855f7" stroke-width="2" stroke-dasharray="4,2"/>
            <text x="375" y="150" text-anchor="middle" font-size="10" fill="#a855f7">Next prediction</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Causal (Masked) Self-Attention'
        },
        {
          type: 'text',
          text: 'GPT uses causal masking in self-attention: each position can only attend to earlier positions. This is implemented by setting attention scores to -∞ for future positions before softmax.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def causal_attention(query, key, value):
    seq_len = query.shape[0]
    d_k = query.shape[-1]

    scores = query @ key.T / np.sqrt(d_k)

    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores = scores - mask * 1e9

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)

    output = weights @ value
    return output, weights

seq_len, d_k = 5, 64
q = np.random.randn(seq_len, d_k)
k = np.random.randn(seq_len, d_k)
v = np.random.randn(seq_len, d_k)

output, weights = causal_attention(q, k, v)
print("Causal attention weights:")
print(weights.round(2))`
        },
        {
          type: 'subheading',
          text: 'In-Context Learning'
        },
        {
          type: 'text',
          text: 'GPT models exhibit emergent abilities at scale. Given examples in the prompt, they can perform new tasks without fine-tuning:'
        },
        {
          type: 'code',
          language: 'python',
          code: `prompt = """Translate English to French:

English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: What is your name?
French: Comment vous appelez-vous?

English: I love machine learning.
French:"""

# GPT completes: "J'adore l'apprentissage automatique."`
        },
        {
          type: 'table',
          headers: ['Model', 'Parameters', 'Key Innovation'],
          rows: [
            ['GPT-1', '117M', 'Showed pre-training + fine-tuning works'],
            ['GPT-2', '1.5B', 'Zero-shot task transfer'],
            ['GPT-3', '175B', 'Few-shot learning, emergent abilities'],
            ['GPT-4', '~1.7T (estimated)', 'Multimodal, advanced reasoning']
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does causal masking in GPT prevent?',
          options: [
            'Overfitting',
            'Attending to future tokens during training',
            'Gradient explosion',
            'Large model sizes'
          ],
          correct: 1,
          explanation: 'Causal masking ensures that when predicting token t, the model can only see tokens 1 through t-1. This maintains the autoregressive property necessary for text generation.'
        }
      ]
    },
    {
      id: 'modern-llms',
      title: 'Modern LLMs Overview',
      duration: '50 min',
      concepts: ['scaling laws', 'instruction tuning', 'RLHF', 'efficiency techniques'],
      content: [
        {
          type: 'heading',
          text: 'The Evolution of Large Language Models'
        },
        {
          type: 'text',
          text: 'The field has exploded since the original Transformer. Modern LLMs combine architectural improvements, massive scale, and training innovations to achieve remarkable capabilities across diverse tasks.'
        },
        {
          type: 'subheading',
          text: 'Scaling Laws'
        },
        {
          type: 'text',
          text: 'Research by Kaplan et al. and Hoffmann et al. showed that model performance follows predictable power laws with compute, data, and parameters. Chinchilla showed optimal allocation: for a given compute budget, train a smaller model on more data.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">LLM Evolution Timeline</text>
            <line x1="50" y1="100" x2="450" y2="100" stroke="#e2e8f0" stroke-width="2"/>
            <circle cx="80" cy="100" r="6" fill="#6366f1"/>
            <text x="80" y="130" text-anchor="middle" font-size="9" fill="#1e293b">GPT-1</text>
            <text x="80" y="145" text-anchor="middle" font-size="8" fill="#64748b">2018</text>
            <circle cx="140" cy="100" r="8" fill="#6366f1"/>
            <text x="140" y="130" text-anchor="middle" font-size="9" fill="#1e293b">BERT</text>
            <text x="140" y="145" text-anchor="middle" font-size="8" fill="#64748b">2018</text>
            <circle cx="200" cy="100" r="10" fill="#6366f1"/>
            <text x="200" y="130" text-anchor="middle" font-size="9" fill="#1e293b">GPT-2</text>
            <text x="200" y="145" text-anchor="middle" font-size="8" fill="#64748b">2019</text>
            <circle cx="260" cy="100" r="14" fill="#6366f1"/>
            <text x="260" y="130" text-anchor="middle" font-size="9" fill="#1e293b">GPT-3</text>
            <text x="260" y="145" text-anchor="middle" font-size="8" fill="#64748b">2020</text>
            <circle cx="320" cy="100" r="12" fill="#22c55e"/>
            <text x="320" y="130" text-anchor="middle" font-size="9" fill="#1e293b">LLaMA</text>
            <text x="320" y="145" text-anchor="middle" font-size="8" fill="#64748b">2023</text>
            <circle cx="380" cy="100" r="16" fill="#f59e0b"/>
            <text x="380" y="130" text-anchor="middle" font-size="9" fill="#1e293b">GPT-4</text>
            <text x="380" y="145" text-anchor="middle" font-size="8" fill="#64748b">2023</text>
            <circle cx="430" cy="100" r="13" fill="#a855f7"/>
            <text x="430" y="130" text-anchor="middle" font-size="9" fill="#1e293b">Claude</text>
            <text x="430" y="145" text-anchor="middle" font-size="8" fill="#64748b">2024</text>
            <text x="250" y="180" text-anchor="middle" font-size="10" fill="#64748b">Circle size ~ relative model capability</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Training Innovations'
        },
        {
          type: 'text',
          text: '**Instruction Tuning**: Fine-tune on instruction-following examples to make models better at understanding and executing user requests.'
        },
        {
          type: 'text',
          text: '**RLHF (Reinforcement Learning from Human Feedback)**: Train a reward model on human preferences, then use it to fine-tune the LLM via PPO. This aligns model outputs with human values.'
        },
        {
          type: 'text',
          text: '**Constitutional AI**: Define principles the model should follow, then use AI feedback (not just human) to train for safety and helpfulness.'
        },
        {
          type: 'subheading',
          text: 'Efficiency Techniques'
        },
        {
          type: 'table',
          headers: ['Technique', 'Description', 'Use Case'],
          rows: [
            ['Quantization', 'Reduce weight precision (32→8→4 bit)', 'Inference on consumer hardware'],
            ['LoRA', 'Train low-rank adapters, freeze base model', 'Efficient fine-tuning'],
            ['Flash Attention', 'Memory-efficient attention computation', 'Training with longer contexts'],
            ['KV Cache', 'Cache key/value for previous tokens', 'Faster autoregressive generation'],
            ['Mixture of Experts', 'Sparse activation of parameters', 'Scale without proportional compute']
          ]
        },
        {
          type: 'code',
          language: 'python',
          code: `# Example: Using a modern LLM with the transformers library
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

prompt = "[INST] Explain attention in transformers briefly. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Open vs Closed Models',
          text: 'The field is split between proprietary models (GPT-4, Claude) and open-weight models (LLaMA, Mistral, Falcon). Open models enable research and customization but may lag in capabilities.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is RLHF used for in modern LLMs?',
          options: [
            'Making models run faster',
            'Reducing model size',
            'Aligning model outputs with human preferences',
            'Increasing context length'
          ],
          correct: 2,
          explanation: 'RLHF (Reinforcement Learning from Human Feedback) trains models to produce outputs that humans prefer. A reward model learns human preferences, then guides the LLM toward more helpful, harmless, and honest outputs.'
        }
      ]
    }
  ]
}
