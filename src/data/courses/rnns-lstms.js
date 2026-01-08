export const rnnslstms = {
  id: 'rnns-lstms',
  title: 'RNNs & LSTMs',
  description: 'Master recurrent neural networks for sequential data processing',
  category: 'Deep Learning',
  difficulty: 'Advanced',
  duration: '7 hours',
  lessons: [
    {
      id: 'sequential-data',
      title: 'Sequential Data and Time Series',
      duration: '55 min',
      concepts: ['sequences', 'time series', 'temporal patterns', 'sequence modeling'],
      content: [
        {
          type: 'heading',
          text: 'Understanding Sequential Data'
        },
        {
          type: 'text',
          text: 'Sequential data is everywhere: text is a sequence of words, audio is a sequence of sound waves, stock prices are sequences of values over time. What makes sequential data special is that order matters—"dog bites man" means something very different from "man bites dog".'
        },
        {
          type: 'subheading',
          text: 'Why Standard Neural Networks Fail'
        },
        {
          type: 'text',
          text: 'Regular feedforward networks treat each input independently. They have no memory of previous inputs. For sequential data, we need networks that can remember what came before to understand what comes next.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowSeq" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Sequential vs Independent Processing</text>
            <rect x="30" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="60" y="70" text-anchor="middle" font-size="10" fill="#1e293b">The</text>
            <rect x="100" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="130" y="70" text-anchor="middle" font-size="10" fill="#1e293b">cat</text>
            <rect x="170" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="200" y="70" text-anchor="middle" font-size="10" fill="#1e293b">sat</text>
            <rect x="240" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="270" y="70" text-anchor="middle" font-size="10" fill="#1e293b">on</text>
            <rect x="310" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="340" y="70" text-anchor="middle" font-size="10" fill="#1e293b">the</text>
            <rect x="380" y="50" width="60" height="30" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="410" y="70" text-anchor="middle" font-size="10" fill="#1e293b">mat</text>
            <line x1="90" y1="65" x2="100" y2="65" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowSeq)"/>
            <line x1="160" y1="65" x2="170" y2="65" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowSeq)"/>
            <line x1="230" y1="65" x2="240" y2="65" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowSeq)"/>
            <line x1="300" y1="65" x2="310" y2="65" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowSeq)"/>
            <line x1="370" y1="65" x2="380" y2="65" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowSeq)"/>
            <text x="250" y="110" text-anchor="middle" font-size="11" fill="#059669">RNN: Each word processed with memory of previous words</text>
            <rect x="30" y="140" width="60" height="30" fill="#fecaca" stroke="#ef4444" rx="4"/>
            <text x="60" y="160" text-anchor="middle" font-size="10" fill="#1e293b">The</text>
            <rect x="100" y="140" width="60" height="30" fill="#fecaca" stroke="#ef4444" rx="4"/>
            <text x="130" y="160" text-anchor="middle" font-size="10" fill="#1e293b">cat</text>
            <rect x="170" y="140" width="60" height="30" fill="#fecaca" stroke="#ef4444" rx="4"/>
            <text x="200" y="160" text-anchor="middle" font-size="10" fill="#1e293b">sat</text>
            <text x="250" y="190" text-anchor="middle" font-size="11" fill="#dc2626">Feedforward: Each word processed independently</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Types of Sequence Problems'
        },
        {
          type: 'table',
          headers: ['Type', 'Input', 'Output', 'Example'],
          rows: [
            ['One-to-Many', 'Single', 'Sequence', 'Image captioning'],
            ['Many-to-One', 'Sequence', 'Single', 'Sentiment analysis'],
            ['Many-to-Many (sync)', 'Sequence', 'Sequence (same length)', 'POS tagging'],
            ['Many-to-Many (async)', 'Sequence', 'Sequence (different)', 'Translation']
          ]
        },
        {
          type: 'subheading',
          text: 'Time Series Characteristics'
        },
        {
          type: 'text',
          text: 'Time series data has unique properties: trend (long-term direction), seasonality (repeating patterns), and noise (random fluctuations). Understanding these helps design better models.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np
import matplotlib.pyplot as plt

def generate_time_series(n_steps=100):
    time = np.arange(n_steps)
    trend = 0.1 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 20)
    noise = np.random.randn(n_steps) * 2
    series = trend + seasonality + noise
    return series

series = generate_time_series(200)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(series, seq_length)
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why do feedforward networks struggle with sequential data?',
          options: [
            'They are too slow',
            'They have no memory of previous inputs',
            'They can only process numbers',
            'They require too much data'
          ],
          correct: 1,
          explanation: 'Feedforward networks treat each input independently with no mechanism to remember previous inputs, making them unsuitable for data where order and context matter.'
        }
      ]
    },
    {
      id: 'vanilla-rnns',
      title: 'Vanilla RNNs',
      duration: '60 min',
      concepts: ['recurrent connections', 'hidden state', 'weight sharing', 'unrolling'],
      content: [
        {
          type: 'heading',
          text: 'The Recurrent Neural Network'
        },
        {
          type: 'text',
          text: 'An RNN introduces a simple but powerful idea: let the network have a loop. At each time step, it takes the current input AND its previous hidden state to produce output. This hidden state acts as memory, carrying information from past inputs.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowRNN" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">RNN Cell: Folded vs Unfolded</text>
            <rect x="60" y="70" width="80" height="50" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="8"/>
            <text x="100" y="100" text-anchor="middle" font-size="12" font-weight="bold" fill="#4f46e5">RNN</text>
            <path d="M 100 70 Q 100 40 130 40 Q 160 40 160 70" fill="none" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
            <text x="130" y="35" text-anchor="middle" font-size="10" fill="#6366f1">h</text>
            <line x1="100" y1="150" x2="100" y2="120" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
            <text x="100" y="165" text-anchor="middle" font-size="11" fill="#1e293b">x_t</text>
            <line x1="100" y1="70" x2="100" y2="50" stroke="#6366f1" stroke-width="2"/>
            <text x="100" y="195" text-anchor="middle" font-size="10" fill="#64748b">Folded View</text>
            <g transform="translate(200, 0)">
              <rect x="20" y="70" width="50" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
              <text x="45" y="95" text-anchor="middle" font-size="10" fill="#4f46e5">RNN</text>
              <rect x="100" y="70" width="50" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
              <text x="125" y="95" text-anchor="middle" font-size="10" fill="#4f46e5">RNN</text>
              <rect x="180" y="70" width="50" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
              <text x="205" y="95" text-anchor="middle" font-size="10" fill="#4f46e5">RNN</text>
              <line x1="70" y1="90" x2="100" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <line x1="150" y1="90" x2="180" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <line x1="230" y1="90" x2="260" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <text x="85" y="82" text-anchor="middle" font-size="9" fill="#6366f1">h₀</text>
              <text x="165" y="82" text-anchor="middle" font-size="9" fill="#6366f1">h₁</text>
              <text x="245" y="82" text-anchor="middle" font-size="9" fill="#6366f1">h₂</text>
              <line x1="45" y1="140" x2="45" y2="110" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <line x1="125" y1="140" x2="125" y2="110" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <line x1="205" y1="140" x2="205" y2="110" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <text x="45" y="155" text-anchor="middle" font-size="10" fill="#1e293b">x₀</text>
              <text x="125" y="155" text-anchor="middle" font-size="10" fill="#1e293b">x₁</text>
              <text x="205" y="155" text-anchor="middle" font-size="10" fill="#1e293b">x₂</text>
              <line x1="45" y1="70" x2="45" y2="50" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <line x1="125" y1="70" x2="125" y2="50" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <line x1="205" y1="70" x2="205" y2="50" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowRNN)"/>
              <text x="45" y="42" text-anchor="middle" font-size="10" fill="#1e293b">y₀</text>
              <text x="125" y="42" text-anchor="middle" font-size="10" fill="#1e293b">y₁</text>
              <text x="205" y="42" text-anchor="middle" font-size="10" fill="#1e293b">y₂</text>
              <text x="125" y="195" text-anchor="middle" font-size="10" fill="#64748b">Unfolded Through Time</text>
            </g>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'The RNN Equations'
        },
        {
          type: 'text',
          text: 'At each time step t, the RNN computes a new hidden state by combining the previous hidden state with the current input:'
        },
        {
          type: 'formula',
          latex: 'h_t = \\tanh(W_{hh} \\cdot h_{t-1} + W_{xh} \\cdot x_t + b_h)'
        },
        {
          type: 'formula',
          latex: 'y_t = W_{hy} \\cdot h_t + b_y'
        },
        {
          type: 'text',
          text: 'W_hh connects previous hidden state to current, W_xh connects input to hidden state, and W_hy connects hidden state to output. These weights are shared across all time steps—this is key to handling variable-length sequences.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        scale = 0.01
        self.wxh = np.random.randn(hidden_size, input_size) * scale
        self.whh = np.random.randn(hidden_size, hidden_size) * scale
        self.why = np.random.randn(output_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.hidden_size = hidden_size

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.hidden_states = {-1: h.copy()}
        self.inputs = inputs
        outputs = []

        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
            y = self.why @ h + self.by
            self.hidden_states[t] = h
            outputs.append(y)

        return outputs, h

rnn = VanillaRNN(input_size=10, hidden_size=32, output_size=5)
sequence = [np.random.randn(10) for _ in range(20)]
outputs, final_hidden = rnn.forward(sequence)
print(f"Sequence length: {len(sequence)}")
print(f"Final hidden state shape: {final_hidden.shape}")`
        },
        {
          type: 'subheading',
          text: 'Weight Sharing Across Time'
        },
        {
          type: 'text',
          text: 'The same weights are used at every time step. This is what makes RNNs powerful: they can process sequences of any length with a fixed number of parameters, and they learn patterns that apply regardless of position in the sequence.'
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Parameter Efficiency',
          text: 'A feedforward network processing sequences of length 100 would need 100x more parameters. RNNs share weights across time, making them much more parameter-efficient.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does the hidden state h_t represent in an RNN?',
          options: [
            'The final output of the network',
            'A summary of all information seen up to time t',
            'The input at time t',
            'The learning rate'
          ],
          correct: 1,
          explanation: 'The hidden state acts as memory, encoding a summary of all the information the network has seen from the beginning of the sequence up to the current time step.'
        }
      ]
    },
    {
      id: 'bptt',
      title: 'Backpropagation Through Time',
      duration: '65 min',
      concepts: ['BPTT', 'gradient flow', 'truncated BPTT', 'computational graph'],
      content: [
        {
          type: 'heading',
          text: 'Training RNNs: Backpropagation Through Time'
        },
        {
          type: 'text',
          text: 'Training an RNN is like training a very deep network where the depth equals the sequence length. We "unroll" the RNN through time and apply backpropagation. This is called Backpropagation Through Time (BPTT).'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowBPTT" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
              <marker id="arrowGrad" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">BPTT: Gradients Flow Backward Through Time</text>
            <rect x="50" y="80" width="60" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="80" y="105" text-anchor="middle" font-size="11" fill="#4f46e5">h₁</text>
            <rect x="150" y="80" width="60" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="180" y="105" text-anchor="middle" font-size="11" fill="#4f46e5">h₂</text>
            <rect x="250" y="80" width="60" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="280" y="105" text-anchor="middle" font-size="11" fill="#4f46e5">h₃</text>
            <rect x="350" y="80" width="60" height="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="380" y="105" text-anchor="middle" font-size="11" fill="#4f46e5">h₄</text>
            <line x1="110" y1="100" x2="150" y2="100" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowBPTT)"/>
            <line x1="210" y1="100" x2="250" y2="100" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowBPTT)"/>
            <line x1="310" y1="100" x2="350" y2="100" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowBPTT)"/>
            <text x="130" y="95" text-anchor="middle" font-size="9" fill="#6366f1">W_hh</text>
            <text x="230" y="95" text-anchor="middle" font-size="9" fill="#6366f1">W_hh</text>
            <text x="330" y="95" text-anchor="middle" font-size="9" fill="#6366f1">W_hh</text>
            <text x="80" y="55" text-anchor="middle" font-size="10" fill="#1e293b">L₁</text>
            <text x="180" y="55" text-anchor="middle" font-size="10" fill="#1e293b">L₂</text>
            <text x="280" y="55" text-anchor="middle" font-size="10" fill="#1e293b">L₃</text>
            <text x="380" y="55" text-anchor="middle" font-size="10" fill="#1e293b">L₄</text>
            <line x1="80" y1="60" x2="80" y2="80" stroke="#94a3b8" stroke-width="1"/>
            <line x1="180" y1="60" x2="180" y2="80" stroke="#94a3b8" stroke-width="1"/>
            <line x1="280" y1="60" x2="280" y2="80" stroke="#94a3b8" stroke-width="1"/>
            <line x1="380" y1="60" x2="380" y2="80" stroke="#94a3b8" stroke-width="1"/>
            <path d="M 380 135 L 380 150 L 80 150 L 80 135" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="5,3"/>
            <line x1="350" y1="140" x2="310" y2="140" stroke="#ef4444" stroke-width="2" marker-end="url(#arrowGrad)"/>
            <line x1="250" y1="140" x2="210" y2="140" stroke="#ef4444" stroke-width="2" marker-end="url(#arrowGrad)"/>
            <line x1="150" y1="140" x2="110" y2="140" stroke="#ef4444" stroke-width="2" marker-end="url(#arrowGrad)"/>
            <text x="250" y="175" text-anchor="middle" font-size="11" fill="#ef4444">Gradients backpropagate through all time steps</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'The BPTT Algorithm'
        },
        {
          type: 'text',
          text: 'The gradient of the loss with respect to W_hh must account for its effect at every time step. The total gradient is the sum of gradients from all time steps:'
        },
        {
          type: 'formula',
          latex: '\\frac{\\partial L}{\\partial W_{hh}} = \\sum_{t=1}^{T} \\frac{\\partial L_t}{\\partial W_{hh}}'
        },
        {
          type: 'text',
          text: 'At each time step t, the gradient flows back through all previous time steps:'
        },
        {
          type: 'formula',
          latex: '\\frac{\\partial L_t}{\\partial h_k} = \\frac{\\partial L_t}{\\partial h_t} \\prod_{i=k+1}^{t} \\frac{\\partial h_i}{\\partial h_{i-1}}'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

class RNNWithBPTT:
    def __init__(self, input_size, hidden_size, output_size):
        self.wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.hidden_size = hidden_size

    def forward(self, inputs, targets):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.zeros((self.hidden_size, 1))
        loss = 0

        for t in range(len(inputs)):
            xs[t] = np.zeros((len(inputs[0]), 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.wxh @ xs[t] + self.whh @ hs[t-1] + self.bh)
            ys[t] = self.why @ hs[t] + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])

        self.cache = (xs, hs, ps)
        return loss, hs[len(inputs)-1]

    def backward(self, targets):
        xs, hs, ps = self.cache
        dwxh = np.zeros_like(self.wxh)
        dwhh = np.zeros_like(self.whh)
        dwhy = np.zeros_like(self.why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(targets))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1
            dwhy += dy @ hs[t].T
            dby += dy
            dh = self.why.T @ dy + dhnext
            dhraw = (1 - hs[t] ** 2) * dh
            dbh += dhraw
            dwxh += dhraw @ xs[t].T
            dwhh += dhraw @ hs[t-1].T
            dhnext = self.whh.T @ dhraw

        for dparam in [dwxh, dwhh, dwhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dwxh, dwhh, dwhy, dbh, dby`
        },
        {
          type: 'subheading',
          text: 'Truncated BPTT'
        },
        {
          type: 'text',
          text: 'For long sequences, full BPTT is computationally expensive and memory-intensive. Truncated BPTT limits how far back gradients flow, processing the sequence in chunks. This trades off long-term learning for computational efficiency.'
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'Gradient Clipping',
          text: 'BPTT can produce very large gradients. Always clip gradients to prevent numerical instability. A common range is [-5, 5] or scaling by global norm.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why is truncated BPTT used in practice?',
          options: [
            'It produces better gradients',
            'It reduces memory and computation for long sequences',
            'It eliminates the vanishing gradient problem',
            'It increases model accuracy'
          ],
          correct: 1,
          explanation: 'Truncated BPTT processes sequences in chunks, limiting how far back gradients flow. This reduces memory usage and computation time, making it practical to train on long sequences.'
        }
      ]
    },
    {
      id: 'vanishing-gradient-rnn',
      title: 'The Vanishing Gradient Problem',
      duration: '55 min',
      concepts: ['vanishing gradients', 'exploding gradients', 'long-term dependencies', 'gradient flow'],
      content: [
        {
          type: 'heading',
          text: 'Why Vanilla RNNs Struggle with Long Sequences'
        },
        {
          type: 'text',
          text: 'The fundamental problem with vanilla RNNs: when gradients flow backward through many time steps, they get repeatedly multiplied by the same weight matrix. If the largest eigenvalue is less than 1, gradients vanish. If greater than 1, they explode.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Gradient Magnitude Over Time Steps</text>
            <line x1="60" y1="160" x2="460" y2="160" stroke="#1e293b" stroke-width="2"/>
            <line x1="60" y1="160" x2="60" y2="40" stroke="#1e293b" stroke-width="2"/>
            <text x="250" y="185" text-anchor="middle" font-size="11" fill="#64748b">Time Steps (backward)</text>
            <text x="30" y="100" text-anchor="middle" font-size="11" fill="#64748b" transform="rotate(-90, 30, 100)">Gradient</text>
            <path d="M 60 150 Q 150 145 200 130 Q 280 100 350 50 Q 400 30 460 25" fill="none" stroke="#ef4444" stroke-width="2"/>
            <text x="420" y="45" font-size="10" fill="#ef4444">Exploding</text>
            <path d="M 60 80 Q 150 90 200 120 Q 280 145 350 155 Q 400 158 460 159" fill="none" stroke="#3b82f6" stroke-width="2"/>
            <text x="420" y="155" font-size="10" fill="#3b82f6">Vanishing</text>
            <line x1="60" y1="100" x2="460" y2="100" stroke="#10b981" stroke-width="2" stroke-dasharray="5,3"/>
            <text x="420" y="95" font-size="10" fill="#10b981">Ideal</text>
            <text x="80" y="175" font-size="9" fill="#64748b">t</text>
            <text x="180" y="175" font-size="9" fill="#64748b">t-10</text>
            <text x="280" y="175" font-size="9" fill="#64748b">t-20</text>
            <text x="380" y="175" font-size="9" fill="#64748b">t-30</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Mathematical Analysis'
        },
        {
          type: 'text',
          text: 'The gradient at time step k with respect to earlier step j involves a product of Jacobians:'
        },
        {
          type: 'formula',
          latex: '\\frac{\\partial h_k}{\\partial h_j} = \\prod_{t=j+1}^{k} \\frac{\\partial h_t}{\\partial h_{t-1}} = \\prod_{t=j+1}^{k} W_{hh}^T \\cdot \\text{diag}(1 - h_t^2)'
        },
        {
          type: 'text',
          text: 'Since tanh derivatives are bounded by 1 and typically much smaller, repeated multiplication causes exponential decay. For a sequence of length T, the gradient can shrink by a factor of 0.25^T or worse.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def demonstrate_vanishing_gradient():
    hidden_size = 100
    whh = np.random.randn(hidden_size, hidden_size) * 0.5

    gradient_norms = []
    gradient = np.ones((hidden_size, 1))

    for t in range(50):
        tanh_derivative = np.random.uniform(0.1, 1.0, (hidden_size, 1))
        gradient = whh.T @ (gradient * tanh_derivative)
        gradient_norms.append(np.linalg.norm(gradient))

    print("Gradient norm progression:")
    for i in [0, 10, 20, 30, 40, 49]:
        print(f"  Step {i}: {gradient_norms[i]:.2e}")

demonstrate_vanishing_gradient()

def check_weight_eigenvalues(whh):
    eigenvalues = np.linalg.eigvals(whh)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"Max eigenvalue magnitude: {max_eigenvalue:.4f}")
    if max_eigenvalue < 1:
        print("Gradients will vanish")
    elif max_eigenvalue > 1:
        print("Gradients may explode")
    else:
        print("Gradients stable")`
        },
        {
          type: 'subheading',
          text: 'Consequences for Learning'
        },
        {
          type: 'text',
          text: 'Vanishing gradients mean the network cannot learn long-term dependencies. If "The cat" appears at the start of a long sentence and "was" appears much later, the RNN cannot learn that "cat" determines "was" (not "were").'
        },
        {
          type: 'table',
          headers: ['Problem', 'Cause', 'Effect', 'Solution'],
          rows: [
            ['Vanishing', 'Eigenvalues < 1', 'Cannot learn long-term patterns', 'LSTM/GRU gates'],
            ['Exploding', 'Eigenvalues > 1', 'Numerical instability', 'Gradient clipping'],
            ['Short memory', 'Information decay', 'Forgets early inputs', 'Attention mechanism']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Why This Led to LSTMs',
          text: 'The vanishing gradient problem is why vanilla RNNs were largely replaced by LSTMs and GRUs. These architectures introduce gates that create "highways" for gradient flow, allowing learning over much longer sequences.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What causes vanishing gradients in RNNs?',
          options: [
            'Learning rate too high',
            'Repeated multiplication by weight matrices with eigenvalues < 1',
            'Too many layers',
            'Insufficient training data'
          ],
          correct: 1,
          explanation: 'Gradients flow backward through time by repeated multiplication with the recurrent weight matrix. When eigenvalues are less than 1, this causes exponential decay of gradients.'
        }
      ]
    },
    {
      id: 'lstm-architecture',
      title: 'LSTM Architecture',
      duration: '70 min',
      concepts: ['cell state', 'forget gate', 'input gate', 'output gate', 'gating mechanism'],
      content: [
        {
          type: 'heading',
          text: 'Long Short-Term Memory Networks'
        },
        {
          type: 'text',
          text: 'LSTMs solve the vanishing gradient problem with a clever innovation: the cell state. Think of it as a conveyor belt running through the entire sequence. Information can flow unchanged, or be modified by gates. This creates a highway for gradients to flow back through time.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowLSTM" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">LSTM Cell Architecture</text>
            <line x1="50" y1="60" x2="450" y2="60" stroke="#10b981" stroke-width="3"/>
            <text x="470" y="65" font-size="10" fill="#10b981">Cell State</text>
            <rect x="100" y="100" width="300" height="140" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" rx="8"/>
            <ellipse cx="140" cy="140" rx="25" ry="20" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
            <text x="140" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">×</text>
            <text x="140" y="175" text-anchor="middle" font-size="9" fill="#64748b">Forget</text>
            <ellipse cx="220" cy="140" rx="25" ry="20" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
            <text x="220" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#1d4ed8">×</text>
            <text x="220" y="175" text-anchor="middle" font-size="9" fill="#64748b">Input</text>
            <ellipse cx="300" cy="140" rx="25" ry="20" fill="#dcfce7" stroke="#22c55e" stroke-width="2"/>
            <text x="300" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">+</text>
            <ellipse cx="360" cy="140" rx="25" ry="20" fill="#f3e8ff" stroke="#a855f7" stroke-width="2"/>
            <text x="360" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#7c3aed">×</text>
            <text x="360" y="175" text-anchor="middle" font-size="9" fill="#64748b">Output</text>
            <rect x="140" y="200" width="40" height="25" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="160" y="217" text-anchor="middle" font-size="10" fill="#92400e">σ</text>
            <rect x="200" y="200" width="40" height="25" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="220" y="217" text-anchor="middle" font-size="10" fill="#1d4ed8">σ</text>
            <rect x="260" y="200" width="40" height="25" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="280" y="217" text-anchor="middle" font-size="10" fill="#4f46e5">tanh</text>
            <rect x="340" y="200" width="40" height="25" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="360" y="217" text-anchor="middle" font-size="10" fill="#7c3aed">σ</text>
            <line x1="140" y1="60" x2="140" y2="120" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="165" y1="140" x2="195" y2="140" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="245" y1="140" x2="275" y2="140" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="325" y1="140" x2="335" y2="140" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="300" y1="60" x2="300" y2="120" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="160" y1="200" x2="160" y2="165" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="140" y1="165" x2="140" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="220" y1="200" x2="220" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="280" y1="200" x2="280" y2="180" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="280" y1="180" x2="220" y2="180" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="220" y1="180" x2="220" y2="165" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="360" y1="200" x2="360" y2="160" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="360" y1="60" x2="360" y2="90" stroke="#6366f1" stroke-width="1.5"/>
            <rect x="350" y="90" width="20" height="15" fill="#e0e7ff" stroke="#6366f1" rx="2"/>
            <text x="360" y="101" text-anchor="middle" font-size="8" fill="#4f46e5">tanh</text>
            <line x1="360" y1="105" x2="360" y2="120" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="50" y1="250" x2="250" y2="250" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowLSTM)"/>
            <text x="150" y="265" text-anchor="middle" font-size="10" fill="#64748b">[h_{t-1}, x_t]</text>
            <line x1="385" y1="140" x2="450" y2="140" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowLSTM)"/>
            <text x="430" y="155" font-size="10" fill="#64748b">h_t</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'The Four Gates'
        },
        {
          type: 'text',
          text: 'An LSTM has four main components, each with a specific role:'
        },
        {
          type: 'text',
          text: '**Forget Gate**: Decides what information to discard from the cell state. "Should I forget the previous subject because a new one appeared?"'
        },
        {
          type: 'formula',
          latex: 'f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)'
        },
        {
          type: 'text',
          text: '**Input Gate**: Decides what new information to store. "Is this word important enough to remember?"'
        },
        {
          type: 'formula',
          latex: 'i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)'
        },
        {
          type: 'formula',
          latex: '\\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)'
        },
        {
          type: 'text',
          text: '**Cell State Update**: Combines forgetting old and adding new information.'
        },
        {
          type: 'formula',
          latex: 'C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t'
        },
        {
          type: 'text',
          text: '**Output Gate**: Decides what to output based on cell state. "What part of my memory is relevant to output now?"'
        },
        {
          type: 'formula',
          latex: 'o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)'
        },
        {
          type: 'formula',
          latex: 'h_t = o_t \\odot \\tanh(C_t)'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        scale = np.sqrt(2.0 / concat_size)

        self.wf = np.random.randn(hidden_size, concat_size) * scale
        self.wi = np.random.randn(hidden_size, concat_size) * scale
        self.wc = np.random.randn(hidden_size, concat_size) * scale
        self.wo = np.random.randn(hidden_size, concat_size) * scale

        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        concat = np.vstack([h_prev, x])

        f = sigmoid(self.wf @ concat + self.bf)
        i = sigmoid(self.wi @ concat + self.bi)
        c_tilde = np.tanh(self.wc @ concat + self.bc)
        o = sigmoid(self.wo @ concat + self.bo)

        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)

        return h, c, (f, i, c_tilde, o, concat)

lstm = LSTMCell(input_size=10, hidden_size=32)
h = np.zeros((32, 1))
c = np.zeros((32, 1))

for t in range(50):
    x = np.random.randn(10, 1)
    h, c, cache = lstm.forward(x, h, c)

print(f"Final hidden state norm: {np.linalg.norm(h):.4f}")
print(f"Final cell state norm: {np.linalg.norm(c):.4f}")`
        },
        {
          type: 'subheading',
          text: 'Why LSTMs Solve Vanishing Gradients'
        },
        {
          type: 'text',
          text: 'The cell state update C_t = f_t * C_{t-1} + ... creates an additive connection. Unlike multiplication in vanilla RNNs, addition allows gradients to flow unchanged when f_t ≈ 1. The forget gate learns when to preserve information across long distances.'
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Forget Gate Bias Initialization',
          text: 'The forget gate bias is typically initialized to 1 (not 0). This starts the LSTM in "remember everything" mode, allowing it to learn what to forget rather than what to remember.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the key innovation that allows LSTMs to capture long-term dependencies?',
          options: [
            'More hidden units',
            'The cell state with additive updates',
            'Deeper networks',
            'Faster training'
          ],
          correct: 1,
          explanation: 'The cell state acts as a highway for information flow. The additive update (C_t = f*C_{t-1} + i*C̃) allows gradients to flow unchanged through time when the forget gate is open, solving the vanishing gradient problem.'
        }
      ]
    },
    {
      id: 'gru-networks',
      title: 'GRU Networks',
      duration: '50 min',
      concepts: ['gated recurrent unit', 'update gate', 'reset gate', 'simplified architecture'],
      content: [
        {
          type: 'heading',
          text: 'Gated Recurrent Units'
        },
        {
          type: 'text',
          text: 'GRUs are a simplified version of LSTMs that combine the forget and input gates into a single "update gate" and merge the cell state and hidden state. Despite having fewer parameters, GRUs often perform comparably to LSTMs.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowGRU" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">GRU Cell Architecture</text>
            <rect x="100" y="50" width="300" height="130" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" rx="8"/>
            <rect x="130" y="140" width="50" height="25" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="155" y="157" text-anchor="middle" font-size="10" fill="#92400e">σ (z)</text>
            <text x="155" y="180" text-anchor="middle" font-size="9" fill="#64748b">Update</text>
            <rect x="220" y="140" width="50" height="25" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="245" y="157" text-anchor="middle" font-size="10" fill="#1d4ed8">σ (r)</text>
            <text x="245" y="180" text-anchor="middle" font-size="9" fill="#64748b">Reset</text>
            <rect x="310" y="140" width="50" height="25" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="335" y="157" text-anchor="middle" font-size="10" fill="#4f46e5">tanh</text>
            <text x="335" y="180" text-anchor="middle" font-size="9" fill="#64748b">Candidate</text>
            <ellipse cx="250" cy="90" rx="25" ry="18" fill="#dcfce7" stroke="#22c55e" stroke-width="2"/>
            <text x="250" y="95" text-anchor="middle" font-size="12" font-weight="bold" fill="#15803d">+</text>
            <ellipse cx="180" cy="90" rx="20" ry="15" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
            <text x="180" y="94" text-anchor="middle" font-size="10" fill="#92400e">×</text>
            <text x="180" y="75" text-anchor="middle" font-size="8" fill="#64748b">1-z</text>
            <ellipse cx="320" cy="90" rx="20" ry="15" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
            <text x="320" y="94" text-anchor="middle" font-size="10" fill="#92400e">×</text>
            <text x="320" y="75" text-anchor="middle" font-size="8" fill="#64748b">z</text>
            <line x1="50" y1="90" x2="160" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowGRU)"/>
            <text x="80" y="85" font-size="10" fill="#64748b">h_{t-1}</text>
            <line x1="200" y1="90" x2="225" y2="90" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="275" y1="90" x2="300" y2="90" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="340" y1="90" x2="450" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowGRU)"/>
            <text x="420" y="85" font-size="10" fill="#64748b">h_t</text>
            <line x1="155" y1="140" x2="155" y2="110" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="155" y1="110" x2="180" y2="105" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="155" y1="110" x2="320" y2="110" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="320" y1="110" x2="320" y2="105" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="335" y1="140" x2="335" y2="120" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="335" y1="120" x2="320" y2="105" stroke="#6366f1" stroke-width="1.5"/>
            <line x1="50" y1="200" x2="250" y2="200" stroke="#6366f1" stroke-width="2"/>
            <text x="150" y="215" text-anchor="middle" font-size="10" fill="#64748b">[h_{t-1}, x_t]</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'GRU Equations'
        },
        {
          type: 'text',
          text: '**Update Gate**: Controls how much of the previous state to keep vs how much to update with new information.'
        },
        {
          type: 'formula',
          latex: 'z_t = \\sigma(W_z \\cdot [h_{t-1}, x_t] + b_z)'
        },
        {
          type: 'text',
          text: '**Reset Gate**: Controls how much of the previous state to use when computing the candidate.'
        },
        {
          type: 'formula',
          latex: 'r_t = \\sigma(W_r \\cdot [h_{t-1}, x_t] + b_r)'
        },
        {
          type: 'text',
          text: '**Candidate Hidden State**: New potential state, computed with reset gate applied.'
        },
        {
          type: 'formula',
          latex: '\\tilde{h}_t = \\tanh(W_h \\cdot [r_t \\odot h_{t-1}, x_t] + b_h)'
        },
        {
          type: 'text',
          text: '**Final Hidden State**: Interpolation between old and candidate states.'
        },
        {
          type: 'formula',
          latex: 'h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        scale = np.sqrt(2.0 / concat_size)

        self.wz = np.random.randn(hidden_size, concat_size) * scale
        self.wr = np.random.randn(hidden_size, concat_size) * scale
        self.wh = np.random.randn(hidden_size, concat_size) * scale

        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        concat = np.vstack([h_prev, x])

        z = sigmoid(self.wz @ concat + self.bz)
        r = sigmoid(self.wr @ concat + self.br)

        concat_reset = np.vstack([r * h_prev, x])
        h_tilde = np.tanh(self.wh @ concat_reset + self.bh)

        h = (1 - z) * h_prev + z * h_tilde

        return h, (z, r, h_tilde)

gru = GRUCell(input_size=10, hidden_size=32)
h = np.zeros((32, 1))

for t in range(100):
    x = np.random.randn(10, 1)
    h, cache = gru.forward(x, h)

print(f"Hidden state norm after 100 steps: {np.linalg.norm(h):.4f}")`
        },
        {
          type: 'subheading',
          text: 'LSTM vs GRU Comparison'
        },
        {
          type: 'table',
          headers: ['Aspect', 'LSTM', 'GRU'],
          rows: [
            ['Gates', '3 (forget, input, output)', '2 (update, reset)'],
            ['States', '2 (cell, hidden)', '1 (hidden only)'],
            ['Parameters', 'More (4 weight matrices)', 'Fewer (3 weight matrices)'],
            ['Training', 'Slower', 'Faster'],
            ['Performance', 'Better on complex tasks', 'Comparable on many tasks']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'When to Use Which',
          text: 'Use GRUs when you need faster training and have limited data. Use LSTMs for complex tasks requiring fine-grained memory control. In practice, try both and see which works better for your specific task.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'How does a GRU simplify the LSTM architecture?',
          options: [
            'By removing all gates',
            'By combining forget and input gates into update gate, and merging cell/hidden states',
            'By using ReLU instead of tanh',
            'By adding more layers'
          ],
          correct: 1,
          explanation: 'GRUs combine the forget and input gates into a single update gate (z), and merge the cell state and hidden state into one. This reduces parameters while maintaining similar performance.'
        }
      ]
    },
    {
      id: 'bidirectional-rnns',
      title: 'Bidirectional RNNs',
      duration: '45 min',
      concepts: ['bidirectional processing', 'forward pass', 'backward pass', 'context aggregation'],
      content: [
        {
          type: 'heading',
          text: 'Processing Sequences in Both Directions'
        },
        {
          type: 'text',
          text: 'Standard RNNs only see past context when processing each element. But for many tasks, future context matters too. "I love this movie" vs "I love this movie... NOT!" The word "love" means different things depending on what comes after.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowBi" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
              <marker id="arrowBiRev" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Bidirectional RNN</text>
            <rect x="80" y="60" width="50" height="35" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="105" y="82" text-anchor="middle" font-size="10" fill="#4f46e5">→</text>
            <rect x="180" y="60" width="50" height="35" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="205" y="82" text-anchor="middle" font-size="10" fill="#4f46e5">→</text>
            <rect x="280" y="60" width="50" height="35" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="305" y="82" text-anchor="middle" font-size="10" fill="#4f46e5">→</text>
            <rect x="380" y="60" width="50" height="35" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="6"/>
            <text x="405" y="82" text-anchor="middle" font-size="10" fill="#4f46e5">→</text>
            <line x1="130" y1="78" x2="180" y2="78" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowBi)"/>
            <line x1="230" y1="78" x2="280" y2="78" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowBi)"/>
            <line x1="330" y1="78" x2="380" y2="78" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowBi)"/>
            <rect x="80" y="120" width="50" height="35" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="6"/>
            <text x="105" y="142" text-anchor="middle" font-size="10" fill="#92400e">←</text>
            <rect x="180" y="120" width="50" height="35" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="6"/>
            <text x="205" y="142" text-anchor="middle" font-size="10" fill="#92400e">←</text>
            <rect x="280" y="120" width="50" height="35" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="6"/>
            <text x="305" y="142" text-anchor="middle" font-size="10" fill="#92400e">←</text>
            <rect x="380" y="120" width="50" height="35" fill="#fef3c7" stroke="#f59e0b" stroke-width="2" rx="6"/>
            <text x="405" y="142" text-anchor="middle" font-size="10" fill="#92400e">←</text>
            <line x1="180" y1="138" x2="130" y2="138" stroke="#f59e0b" stroke-width="2" marker-end="url(#arrowBiRev)"/>
            <line x1="280" y1="138" x2="230" y2="138" stroke="#f59e0b" stroke-width="2" marker-end="url(#arrowBiRev)"/>
            <line x1="380" y1="138" x2="330" y2="138" stroke="#f59e0b" stroke-width="2" marker-end="url(#arrowBiRev)"/>
            <text x="105" y="180" text-anchor="middle" font-size="10" fill="#1e293b">x₁</text>
            <text x="205" y="180" text-anchor="middle" font-size="10" fill="#1e293b">x₂</text>
            <text x="305" y="180" text-anchor="middle" font-size="10" fill="#1e293b">x₃</text>
            <text x="405" y="180" text-anchor="middle" font-size="10" fill="#1e293b">x₄</text>
            <text x="40" y="82" font-size="10" fill="#6366f1">Forward</text>
            <text x="40" y="142" font-size="10" fill="#f59e0b">Backward</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'How Bidirectional RNNs Work'
        },
        {
          type: 'text',
          text: 'A bidirectional RNN runs two separate RNNs: one processing left-to-right, another right-to-left. Their outputs are combined (usually concatenated) to give each position access to both past and future context.'
        },
        {
          type: 'formula',
          latex: '\\vec{h}_t = \\text{RNN}_{forward}(x_t, \\vec{h}_{t-1})'
        },
        {
          type: 'formula',
          latex: '\\overleftarrow{h}_t = \\text{RNN}_{backward}(x_t, \\overleftarrow{h}_{t+1})'
        },
        {
          type: 'formula',
          latex: 'h_t = [\\vec{h}_t; \\overleftarrow{h}_t]'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        output = self.fc(combined)
        return output, lstm_out

model = BidirectionalLSTM(input_size=100, hidden_size=128)
batch = torch.randn(32, 50, 100)
output, all_states = model(batch)
print(f"Output shape: {output.shape}")
print(f"All states shape: {all_states.shape}")`
        },
        {
          type: 'subheading',
          text: 'When to Use Bidirectional RNNs'
        },
        {
          type: 'table',
          headers: ['Use Case', 'Bidirectional?', 'Reason'],
          rows: [
            ['Text classification', 'Yes', 'Full context available'],
            ['Named entity recognition', 'Yes', 'Context from both sides helps'],
            ['Language modeling', 'No', 'Cannot see future at generation time'],
            ['Real-time speech recognition', 'No', 'Future audio not available'],
            ['Machine translation (encoder)', 'Yes', 'Full source sentence available']
          ]
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'Computational Cost',
          text: 'Bidirectional RNNs double the parameters and computation. They also require the entire sequence upfront, making them unsuitable for streaming/real-time applications where inputs arrive one at a time.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why cant bidirectional RNNs be used for language generation?',
          options: [
            'They are too slow',
            'They require the full sequence including future tokens which dont exist during generation',
            'They have too many parameters',
            'They only work with short sequences'
          ],
          correct: 1,
          explanation: 'During text generation, we predict one token at a time. Future tokens dont exist yet, so the backward RNN has nothing to process. Bidirectional RNNs require the complete sequence upfront.'
        }
      ]
    }
  ]
}
