export const lstmcourse = {
  id: 'lstm-sentiment',
  title: 'LSTM Networks & Sentiment Analysis',
  icon: 'brain',
  description: 'Master recurrent neural networks through building a sentiment classifier and text generator.',
  difficulty: 'intermediate',
  sourceproject: 'sentiment_detection_lstm',
  lessons: [
    {
      id: 'rnn-history',
      title: 'The Evolution of Sequence Models',
      duration: '15 min read',
      concepts: ['RNN', 'History', 'Sequence Modeling'],
      content: [
        { type: 'heading', text: 'Why Sequences Matter' },
        { type: 'paragraph', text: 'Traditional neural networks treat each input independently. They have no memory of previous inputs, which makes them unsuitable for tasks where context matters—like understanding language, predicting stock prices, or analyzing time series data.' },
        { type: 'paragraph', text: 'Consider the sentence "The movie was not bad." A bag-of-words model might see "not" and "bad" as negative indicators, missing that together they express a positive sentiment. Sequence models solve this by processing inputs in order, maintaining a memory of what came before.' },

        { type: 'heading', text: 'A Brief History of Recurrent Networks' },
        { type: 'paragraph', text: 'The idea of recurrent neural networks dates back to the 1980s. John Hopfield introduced Hopfield Networks in 1982, which could store and retrieve patterns. However, these early networks struggled with learning long-term dependencies.' },
        { type: 'paragraph', text: 'In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published the backpropagation algorithm, enabling training of deeper networks. But standard RNNs faced a critical problem: the vanishing gradient problem, where gradients become extremely small during backpropagation, preventing the network from learning long-range dependencies.' },

        { type: 'subheading', text: 'The Vanishing Gradient Problem' },
        { type: 'paragraph', text: 'During backpropagation through time (BPTT), gradients are multiplied repeatedly. If these multiplications involve values less than 1, the gradient shrinks exponentially. After just 10-20 time steps, the gradient can become so small that early layers stop learning entirely.' },
        { type: 'formula', formula: '∂L/∂W = Σ (∂L/∂hₜ) × (∏ᵢ ∂hᵢ/∂hᵢ₋₁) × (∂h₀/∂W)' },
        { type: 'paragraph', text: 'The product term in the middle is where gradients vanish. If each ∂hᵢ/∂hᵢ₋₁ is less than 1, the product approaches zero.' },

        { type: 'heading', text: 'The LSTM Breakthrough' },
        { type: 'paragraph', text: 'In 1997, Sepp Hochreiter and Jürgen Schmidhuber introduced Long Short-Term Memory (LSTM) networks. The key innovation was the cell state—a highway that runs through the entire sequence, allowing information to flow unchanged. Gates control what information enters, leaves, or stays in this cell state.' },
        { type: 'paragraph', text: 'LSTMs didnt gain widespread adoption until 2014-2016 when GPU computing made training practical and researchers demonstrated remarkable results in speech recognition, machine translation, and text generation.' },

        { type: 'keypoints', points: [
          'Traditional neural networks lack memory of previous inputs',
          'The vanishing gradient problem prevented early RNNs from learning long sequences',
          'LSTM networks solved this with gated memory cells in 1997',
          'Modern deep learning hardware enabled practical LSTM applications'
        ]}
      ],
      quiz: [
        {
          question: 'What is the main limitation of traditional feedforward neural networks for sequence data?',
          options: ['They are too slow', 'They have no memory of previous inputs', 'They require too much data', 'They cannot handle numerical inputs'],
          correct: 1,
          explanation: 'Feedforward networks process each input independently without any notion of sequence or temporal context.'
        },
        {
          question: 'What problem did LSTM networks specifically solve?',
          options: ['Overfitting', 'Vanishing gradient problem', 'Lack of training data', 'Slow inference'],
          correct: 1,
          explanation: 'LSTMs introduced gated memory cells that allow gradients to flow unchanged, solving the vanishing gradient problem.'
        }
      ]
    },
    {
      id: 'lstm-architecture',
      title: 'LSTM Architecture Deep Dive',
      duration: '20 min read',
      concepts: ['LSTM', 'Gates', 'Cell State'],
      content: [
        { type: 'heading', text: 'The Four Components of an LSTM Cell' },
        { type: 'paragraph', text: 'An LSTM cell has four main components: the cell state, the forget gate, the input gate, and the output gate. Together, these components decide what information to keep, what to add, and what to output at each time step.' },

        { type: 'subheading', text: '1. The Cell State (Memory Highway)' },
        { type: 'paragraph', text: 'Think of the cell state as a conveyor belt running through the entire sequence. Information can flow along it unchanged, or be modified by the gates. This is the key to remembering information over long sequences.' },
        { type: 'formula', formula: 'Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ' },
        { type: 'paragraph', text: 'The cell state at time t equals the previous cell state (scaled by forget gate) plus new candidate values (scaled by input gate). The ⊙ symbol represents element-wise multiplication.' },

        { type: 'subheading', text: '2. The Forget Gate' },
        { type: 'paragraph', text: 'The forget gate decides what information to discard from the cell state. It looks at the previous hidden state and current input, outputting a value between 0 (completely forget) and 1 (completely keep) for each element in the cell state.' },
        { type: 'formula', formula: 'fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)' },
        { type: 'paragraph', text: 'For sentiment analysis, the forget gate might learn to discard information about neutral words like "the" or "a" while keeping sentiment-bearing words.' },

        { type: 'subheading', text: '3. The Input Gate' },
        { type: 'paragraph', text: 'The input gate controls what new information to add to the cell state. It has two parts: a sigmoid layer that decides which values to update, and a tanh layer that creates candidate values.' },
        { type: 'formula', formula: 'iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)' },
        { type: 'formula', formula: 'C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)' },

        { type: 'subheading', text: '4. The Output Gate' },
        { type: 'paragraph', text: 'Finally, the output gate decides what to output based on the cell state. The cell state is passed through tanh (scaling to -1 to 1) and multiplied by the output gate.' },
        { type: 'formula', formula: 'oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)' },
        { type: 'formula', formula: 'hₜ = oₜ ⊙ tanh(Cₜ)' },

        { type: 'heading', text: 'Your LSTM Implementation' },
        { type: 'paragraph', text: 'In your sentiment detection project, you used Keras to create an LSTM layer. Lets examine how the architecture maps to your code:' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'sentiment_detection_lstm',
          code: `def create_sentiment_model(vocab_size, embedding_dim=64, lstm_units=64, max_length=100):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        layers.LSTM(lstm_units),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model` },
        { type: 'paragraph', text: 'The lstm_units=64 parameter means your LSTM has 64 memory cells, each with its own forget, input, and output gates. This single line actually creates a complex network with thousands of parameters.' },

        { type: 'callout', variant: 'info', text: 'The mask_zero=True parameter tells the LSTM to ignore padding tokens (zeros), preventing the model from learning patterns from padding.' },

        { type: 'keypoints', points: [
          'LSTM cells have four components: cell state, forget gate, input gate, output gate',
          'The cell state acts as a memory highway allowing long-term information flow',
          'Gates use sigmoid activation to output values between 0 and 1',
          'Each LSTM unit contains all four gates operating in parallel'
        ]}
      ],
      quiz: [
        {
          question: 'What is the role of the forget gate in an LSTM?',
          options: ['Add new information', 'Decide what to output', 'Remove irrelevant information from cell state', 'Initialize the hidden state'],
          correct: 2,
          explanation: 'The forget gate outputs values between 0 and 1, determining how much of the previous cell state to keep or discard.'
        },
        {
          question: 'Why is mask_zero=True important in your Embedding layer?',
          options: ['It speeds up training', 'It reduces memory usage', 'It tells LSTM to ignore padding tokens', 'It normalizes the embeddings'],
          correct: 2,
          explanation: 'mask_zero=True ensures the LSTM doesnt learn from padding tokens, which would corrupt the learned representations.'
        },
        {
          question: 'What activation function do LSTM gates use?',
          options: ['ReLU', 'Sigmoid', 'Tanh', 'Softmax'],
          correct: 1,
          explanation: 'Gates use sigmoid to output values between 0 and 1, acting as "switches" that control information flow.'
        }
      ]
    },
    {
      id: 'text-preprocessing',
      title: 'Text Preprocessing for NLP',
      duration: '18 min read',
      concepts: ['Tokenization', 'Cleaning', 'Padding'],
      content: [
        { type: 'heading', text: 'Why Preprocessing Matters' },
        { type: 'paragraph', text: 'Neural networks operate on numbers, not text. Text preprocessing transforms raw text into numerical sequences that models can process. Poor preprocessing can severely limit model performance, while good preprocessing can make the difference between a mediocre and excellent model.' },

        { type: 'heading', text: 'Your Preprocessing Pipeline' },
        { type: 'paragraph', text: 'Lets walk through each step of your text preprocessing class and understand why each transformation matters.' },
        { type: 'code', language: 'python', filename: 'preprocessing.py', fromproject: 'sentiment_detection_lstm',
          code: `class textpreprocessor:
    def __init__(self, max_features=5000, max_length=100):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
        self.vocab_size = 0` },

        { type: 'subheading', text: 'Vocabulary Size (max_features)' },
        { type: 'paragraph', text: 'The max_features=5000 limits your vocabulary to the 5000 most common words. This is a trade-off between expressiveness and efficiency. Rare words often dont contribute much to sentiment and increase model complexity.' },
        { type: 'callout', variant: 'tip', text: 'For sentiment analysis, 5000-10000 words usually captures enough vocabulary. Increasing beyond this rarely improves accuracy but significantly increases training time.' },

        { type: 'subheading', text: 'OOV Token' },
        { type: 'paragraph', text: 'The oov_token="<OOV>" (Out Of Vocabulary) handles words not in your vocabulary. Without this, unknown words would be silently dropped, potentially losing important information.' },

        { type: 'heading', text: 'Text Cleaning' },
        { type: 'code', language: 'python', filename: 'preprocessing.py', fromproject: 'sentiment_detection_lstm',
          code: `def clean_text(self, text):
    text = text.lower()
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'[^a-z\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text` },

        { type: 'paragraph', text: 'Each regex pattern serves a specific purpose:' },
        { type: 'list', items: [
          'text.lower() - Normalizes case so "Good" and "good" are treated identically',
          '@\\w+ - Removes Twitter mentions (@username) which dont carry sentiment',
          'http\\S+|www\\S+ - Removes URLs which are just noise for sentiment',
          '[^a-z\\s] - Keeps only letters and spaces, removing punctuation and numbers',
          '\\s+ - Collapses multiple spaces into single spaces'
        ]},

        { type: 'heading', text: 'Tokenization and Padding' },
        { type: 'code', language: 'python', filename: 'preprocessing.py', fromproject: 'sentiment_detection_lstm',
          code: `def fit_transform(self, texts):
    cleaned = [self.clean_text(t) for t in texts]
    self.tokenizer.fit_on_texts(cleaned)
    self.vocab_size = min(self.max_features, len(self.tokenizer.word_index) + 1)
    sequences = self.tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
    return padded` },

        { type: 'subheading', text: 'Sequence Padding' },
        { type: 'paragraph', text: 'Neural networks require fixed-size inputs. Padding adds zeros to shorter sequences and truncates longer ones. The padding="post" argument adds padding at the end rather than the beginning, which works better with mask_zero=True.' },
        { type: 'callout', variant: 'warning', text: 'Using padding="pre" with mask_zero=True can cause issues because the LSTM would see zeros at the start, potentially affecting the hidden state initialization.' },

        { type: 'keypoints', points: [
          'Preprocessing transforms text to numerical sequences for neural networks',
          'Vocabulary limiting (max_features) balances expressiveness and efficiency',
          'Text cleaning removes noise like URLs, mentions, and punctuation',
          'Padding ensures all sequences have the same length for batch processing'
        ]}
      ],
      quiz: [
        {
          question: 'Why limit vocabulary to 5000 words instead of using all words?',
          options: ['Memory constraints only', 'Rare words add noise and increase complexity without improving accuracy', 'Python limitation', 'Keras requirement'],
          correct: 1,
          explanation: 'Rare words often dont contribute to sentiment and increase model parameters, leading to overfitting and slower training.'
        },
        {
          question: 'What does the OOV token handle?',
          options: ['Punctuation', 'Numbers', 'Words not in the vocabulary', 'Padding'],
          correct: 2,
          explanation: 'OOV (Out Of Vocabulary) token represents any word not in the top max_features words, preventing information loss.'
        }
      ]
    },
    {
      id: 'embeddings',
      title: 'Word Embeddings Explained',
      duration: '15 min read',
      concepts: ['Embeddings', 'Word2Vec', 'Semantic Space'],
      content: [
        { type: 'heading', text: 'From One-Hot to Dense Representations' },
        { type: 'paragraph', text: 'Early NLP represented words as one-hot vectors—sparse vectors with a 1 at the words index and 0s everywhere else. With a vocabulary of 5000 words, each word becomes a 5000-dimensional vector, mostly zeros. This is wasteful and captures no semantic relationships.' },
        { type: 'paragraph', text: 'Word embeddings compress these sparse vectors into dense, lower-dimensional representations where similar words have similar vectors. The famous example: king - man + woman ≈ queen shows how embeddings capture semantic relationships.' },

        { type: 'heading', text: 'How Embeddings Work' },
        { type: 'paragraph', text: 'An embedding layer is essentially a lookup table that maps each word index to a dense vector. In your model, with embedding_dim=64, each word is represented by 64 numbers learned during training.' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'sentiment_detection_lstm',
          code: `layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True)` },

        { type: 'paragraph', text: 'This creates a matrix of shape (vocab_size, embedding_dim). When a word with index 42 passes through, the layer returns the 42nd row of this matrix.' },

        { type: 'subheading', text: 'Learning vs Pre-trained Embeddings' },
        { type: 'paragraph', text: 'Your model learns embeddings from scratch during training. The alternative is using pre-trained embeddings like Word2Vec or GloVe, trained on billions of words. For domain-specific tasks like airline sentiment, learning embeddings often works better because the model learns representations specific to your domain.' },

        { type: 'heading', text: 'Embedding Dimensions' },
        { type: 'paragraph', text: 'The embedding_dim=64 is a hyperparameter. Higher dimensions can capture more nuanced relationships but require more data to train effectively. Common choices range from 50 to 300.' },
        { type: 'list', items: [
          '50-100 dimensions: Good for smaller datasets, faster training',
          '200-300 dimensions: Better for larger datasets, captures more nuance',
          'Your choice (64): Balanced for your dataset size'
        ]},

        { type: 'keypoints', points: [
          'Embeddings convert sparse one-hot vectors to dense representations',
          'Similar words have similar embedding vectors',
          'Embeddings are learned during training as model parameters',
          'Embedding dimension is a hyperparameter balancing expressiveness and data requirements'
        ]}
      ],
      quiz: [
        {
          question: 'What is the main advantage of embeddings over one-hot encoding?',
          options: ['Faster computation only', 'Dense representation capturing semantic relationships', 'Smaller file size', 'Easier to implement'],
          correct: 1,
          explanation: 'Embeddings capture semantic relationships (similar words have similar vectors) in a dense, efficient representation.'
        }
      ]
    },
    {
      id: 'binary-classification',
      title: 'Binary Classification & Loss Functions',
      duration: '12 min read',
      concepts: ['Sigmoid', 'Binary Crossentropy', 'Classification'],
      content: [
        { type: 'heading', text: 'The Binary Classification Task' },
        { type: 'paragraph', text: 'Your sentiment model performs binary classification: given a text, predict whether its positive (1) or negative (0). This is the simplest classification task but illustrates fundamental concepts that apply to all classification problems.' },

        { type: 'heading', text: 'Sigmoid Activation' },
        { type: 'paragraph', text: 'The final layer uses sigmoid activation, which squashes any input to a value between 0 and 1—perfect for representing probability.' },
        { type: 'formula', formula: 'σ(x) = 1 / (1 + e^(-x))' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'sentiment_detection_lstm',
          code: `layers.Dense(1, activation='sigmoid')` },
        { type: 'paragraph', text: 'The output represents P(positive|text)—the probability the text is positive. During inference, you typically use 0.5 as the decision threshold.' },

        { type: 'heading', text: 'Binary Crossentropy Loss' },
        { type: 'paragraph', text: 'Binary crossentropy measures how wrong the predictions are. It penalizes confident wrong predictions heavily while rewarding confident correct predictions.' },
        { type: 'formula', formula: 'L = -[y·log(p) + (1-y)·log(1-p)]' },
        { type: 'paragraph', text: 'Where y is the true label (0 or 1) and p is the predicted probability. If y=1 and p=0.9, loss is -log(0.9)=0.105. If y=1 and p=0.1, loss is -log(0.1)=2.3. Wrong predictions cost much more.' },

        { type: 'keypoints', points: [
          'Sigmoid maps outputs to [0,1] for probability interpretation',
          'Binary crossentropy penalizes confident wrong predictions heavily',
          'The output represents probability of the positive class',
          'Threshold of 0.5 is typically used for final predictions'
        ]}
      ],
      quiz: [
        {
          question: 'Why use sigmoid instead of softmax for binary classification?',
          options: ['Sigmoid is faster', 'Softmax only works with 3+ classes', 'Sigmoid outputs probability directly for binary case', 'No difference'],
          correct: 2,
          explanation: 'For binary classification, sigmoid directly outputs P(positive). Softmax would output two redundant probabilities that sum to 1.'
        }
      ]
    },
    {
      id: 'text-generation',
      title: 'Text Generation with LSTMs',
      duration: '20 min read',
      concepts: ['Language Model', 'Temperature Sampling', 'Sequence Generation'],
      content: [
        { type: 'heading', text: 'From Classification to Generation' },
        { type: 'paragraph', text: 'Your text generation model flips the LSTM architecture. Instead of many-to-one (sequence to label), it predicts the next word given previous words—the foundation of all language models, from your LSTM to GPT.' },

        { type: 'heading', text: 'Stacked LSTMs for Generation' },
        { type: 'code', language: 'python', filename: 'textgen/model.py', fromproject: 'sentiment_detection_lstm',
          code: `def create_textgen_model(vocab_size, sequence_length=50, embedding_dim=256, lstm_units=256):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=sequence_length),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.LSTM(lstm_units),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(vocab_size, activation='softmax')
    ])` },

        { type: 'paragraph', text: 'Key differences from sentiment model:' },
        { type: 'list', items: [
          'Two LSTM layers (stacked) for more complex pattern learning',
          'return_sequences=True on first LSTM outputs at every timestep',
          'Larger dimensions (256 vs 64) for richer representations',
          'Softmax output over entire vocabulary to predict next word',
          'Dropout for regularization to prevent overfitting'
        ]},

        { type: 'heading', text: 'Temperature Sampling' },
        { type: 'paragraph', text: 'The softmax output gives probability for each word. Simply picking the highest probability (argmax) produces repetitive, boring text. Temperature sampling adds controlled randomness.' },
        { type: 'code', language: 'python', filename: 'textgen/model.py', fromproject: 'sentiment_detection_lstm',
          code: `def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)` },

        { type: 'subheading', text: 'How Temperature Works' },
        { type: 'list', items: [
          'Temperature = 1.0: Original probability distribution',
          'Temperature < 1.0: Sharper distribution, more deterministic (safer choices)',
          'Temperature > 1.0: Flatter distribution, more random (creative but risky)'
        ]},
        { type: 'callout', variant: 'tip', text: 'For coherent text, use temperature 0.7-0.9. For creative exploration, try 1.2-1.5. Never go below 0.1 (too repetitive) or above 2.0 (nonsense).' },

        { type: 'keypoints', points: [
          'Text generation predicts next word given previous words',
          'Stacked LSTMs learn more complex patterns than single layers',
          'Temperature controls randomness in word selection',
          'Lower temperature = safer, higher temperature = more creative'
        ]}
      ],
      quiz: [
        {
          question: 'Why use return_sequences=True on the first LSTM but not the second?',
          options: ['Memory optimization', 'First LSTM feeds sequences to second LSTM, which outputs single vector', 'Keras requirement', 'Prevents overfitting'],
          correct: 1,
          explanation: 'The first LSTM outputs at each timestep for the second LSTM to process. The second LSTM outputs only the final state for classification/prediction.'
        },
        {
          question: 'What happens when temperature approaches 0?',
          options: ['Random output', 'Always picks highest probability word', 'Model crashes', 'Outputs empty strings'],
          correct: 1,
          explanation: 'Low temperature makes the distribution sharper, approaching argmax behavior where only the highest probability word is selected.'
        }
      ]
    }
  ]
}
