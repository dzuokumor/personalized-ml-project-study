export const glossary = [
  {
    term: 'Activation Function',
    definition: 'A function applied to the output of a neuron to introduce non-linearity. Common examples include ReLU, sigmoid, and tanh. Without activation functions, neural networks would only learn linear relationships.',
    category: 'fundamentals',
    usedin: ['LSTM Sentiment', 'CNN Classification']
  },
  {
    term: 'Attention Mechanism',
    definition: 'A technique that allows models to focus on relevant parts of the input when producing each part of the output. Computes weighted sums where weights indicate importance.',
    category: 'transformers',
    usedin: ['BART Summarization']
  },
  {
    term: 'Backpropagation',
    definition: 'Algorithm for computing gradients in neural networks by applying the chain rule from output to input. Enables efficient training by propagating error signals backwards through layers.',
    category: 'fundamentals',
    usedin: ['LSTM Sentiment', 'CNN Classification', 'BART Summarization']
  },
  {
    term: 'Batch Normalization',
    definition: 'Technique that normalizes layer inputs to have zero mean and unit variance. Stabilizes training, allows higher learning rates, and acts as regularization.',
    category: 'optimization',
    usedin: ['CNN Classification']
  },
  {
    term: 'Beam Search',
    definition: 'Decoding strategy that maintains k best candidates at each step. Balances exploration with computational cost, typically outperforming greedy decoding.',
    category: 'generation',
    usedin: ['BART Summarization']
  },
  {
    term: 'Binary Crossentropy',
    definition: 'Loss function for binary classification. Measures the difference between predicted probabilities and true binary labels. Heavily penalizes confident wrong predictions.',
    category: 'loss-functions',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'BPE (Byte Pair Encoding)',
    definition: 'Subword tokenization algorithm that iteratively merges frequent character pairs. Creates vocabulary that balances coverage and size.',
    category: 'nlp',
    usedin: ['BART Summarization']
  },
  {
    term: 'Cell State',
    definition: 'The memory component of an LSTM that runs through the entire sequence. Information flows along it unchanged unless modified by gates.',
    category: 'rnn',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Convolutional Layer',
    definition: 'Layer that applies learned filters across spatial positions. Exploits translation invariance and local connectivity, making it ideal for images.',
    category: 'cnn',
    usedin: ['CNN Classification']
  },
  {
    term: 'Dropout',
    definition: 'Regularization technique that randomly zeroes neuron outputs during training. Prevents co-adaptation and improves generalization.',
    category: 'regularization',
    usedin: ['LSTM Sentiment', 'CNN Classification']
  },
  {
    term: 'Embedding',
    definition: 'Dense vector representation of discrete items (words, categories). Learned during training to capture semantic relationships.',
    category: 'nlp',
    usedin: ['LSTM Sentiment', 'BART Summarization']
  },
  {
    term: 'Encoder-Decoder',
    definition: 'Architecture where an encoder processes input into representations and a decoder generates output from those representations. Used for sequence-to-sequence tasks.',
    category: 'architecture',
    usedin: ['BART Summarization']
  },
  {
    term: 'Feature Map',
    definition: 'Output of a convolutional layer. Each feature map shows where a particular pattern is detected in the input.',
    category: 'cnn',
    usedin: ['CNN Classification']
  },
  {
    term: 'Forget Gate',
    definition: 'LSTM component that decides what information to discard from the cell state. Outputs values between 0 (forget) and 1 (keep).',
    category: 'rnn',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Gradient Descent',
    definition: 'Optimization algorithm that updates parameters in the direction that reduces loss. The foundation of neural network training.',
    category: 'optimization',
    usedin: ['LSTM Sentiment', 'CNN Classification', 'BART Summarization']
  },
  {
    term: 'Hidden State',
    definition: 'The output of an RNN cell at each time step. Carries information about the sequence seen so far.',
    category: 'rnn',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Learning Rate',
    definition: 'Hyperparameter controlling the step size in gradient descent. Too high causes instability; too low causes slow training.',
    category: 'optimization',
    usedin: ['LSTM Sentiment', 'CNN Classification']
  },
  {
    term: 'LSTM (Long Short-Term Memory)',
    definition: 'RNN variant with gated memory cells that can learn long-term dependencies. Solves the vanishing gradient problem of standard RNNs.',
    category: 'rnn',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Max Pooling',
    definition: 'Downsampling operation that takes the maximum value in each region. Reduces spatial dimensions while preserving strongest features.',
    category: 'cnn',
    usedin: ['CNN Classification']
  },
  {
    term: 'Multi-Head Attention',
    definition: 'Parallel attention mechanisms that attend to different aspects of the input. Outputs are concatenated and projected.',
    category: 'transformers',
    usedin: ['BART Summarization']
  },
  {
    term: 'One-Hot Encoding',
    definition: 'Representing categorical variables as binary vectors with a single 1. Inefficient for large vocabularies, replaced by embeddings in NLP.',
    category: 'preprocessing',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Overfitting',
    definition: 'When a model learns training data too well, failing to generalize to new data. Addressed through regularization, dropout, and more data.',
    category: 'fundamentals',
    usedin: ['LSTM Sentiment', 'CNN Classification']
  },
  {
    term: 'Padding',
    definition: 'Adding zeros to sequences to make them equal length for batch processing. Position (pre/post) affects model behavior.',
    category: 'preprocessing',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'ReLU (Rectified Linear Unit)',
    definition: 'Activation function f(x) = max(0, x). Simple, fast, and avoids vanishing gradients for positive values.',
    category: 'fundamentals',
    usedin: ['CNN Classification', 'LSTM Sentiment']
  },
  {
    term: 'Self-Attention',
    definition: 'Attention where queries, keys, and values come from the same sequence. Allows each position to attend to all positions.',
    category: 'transformers',
    usedin: ['BART Summarization']
  },
  {
    term: 'Sigmoid',
    definition: 'Activation function that squashes values to (0, 1). Used for binary classification outputs and LSTM gates.',
    category: 'fundamentals',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Softmax',
    definition: 'Activation function that converts logits to probability distribution summing to 1. Used for multi-class classification.',
    category: 'fundamentals',
    usedin: ['CNN Classification', 'LSTM Sentiment', 'BART Summarization']
  },
  {
    term: 'Temperature Sampling',
    definition: 'Technique for controlling randomness in text generation. Lower temperature = more deterministic; higher = more random.',
    category: 'generation',
    usedin: ['LSTM Sentiment']
  },
  {
    term: 'Tokenization',
    definition: 'Breaking text into units (tokens) for processing. Can be word-level, subword, or character-level.',
    category: 'nlp',
    usedin: ['LSTM Sentiment', 'BART Summarization']
  },
  {
    term: 'Transformer',
    definition: 'Architecture based entirely on attention mechanisms, without recurrence. Enables parallel processing and captures long-range dependencies.',
    category: 'transformers',
    usedin: ['BART Summarization']
  },
  {
    term: 'Vanishing Gradient',
    definition: 'Problem where gradients become extremely small during backpropagation, preventing early layers from learning. Solved by LSTMs and residual connections.',
    category: 'fundamentals',
    usedin: ['LSTM Sentiment']
  }
]
