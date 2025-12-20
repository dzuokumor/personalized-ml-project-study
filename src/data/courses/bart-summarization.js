export const bartcourse = {
  id: 'bart-summarization',
  title: 'Transformers & Text Summarization',
  icon: 'document',
  description: 'Understand the transformer architecture through abstractive summarization with BART.',
  difficulty: 'advanced',
  sourceproject: 'BART-transformer-summary-gen',
  lessons: [
    {
      id: 'attention-revolution',
      title: 'The Attention Revolution',
      duration: '18 min read',
      concepts: ['Attention', 'Transformer', 'Self-Attention'],
      content: [
        { type: 'heading', text: 'From RNNs to Attention' },
        { type: 'paragraph', text: 'In 2017, the paper "Attention Is All You Need" by Vaswani et al. introduced the Transformer architecture, fundamentally changing NLP. While LSTMs process sequences one token at a time, Transformers process all tokens simultaneously using attention mechanisms.' },
        { type: 'paragraph', text: 'The key insight: instead of relying on recurrence to capture dependencies, let each token directly attend to every other token. This parallelization enables training on much larger datasets and captures long-range dependencies more effectively.' },

        { type: 'heading', text: 'The Attention Mechanism' },
        { type: 'paragraph', text: 'Attention computes a weighted sum of values, where weights are determined by how relevant each position is to the current position. Think of it as a soft lookup table where every entry contributes, but relevant entries contribute more.' },
        { type: 'formula', formula: 'Attention(Q, K, V) = softmax(QK^T / √dₖ)V' },
        { type: 'list', items: [
          'Q (Query): What am I looking for?',
          'K (Key): What do I contain?',
          'V (Value): What information do I provide?',
          '√dₖ: Scaling factor to prevent softmax saturation'
        ]},

        { type: 'subheading', text: 'Self-Attention' },
        { type: 'paragraph', text: 'In self-attention, Q, K, and V all come from the same sequence. Each token queries all other tokens (including itself) to build context-aware representations. This is how "bank" in "river bank" gets a different representation than "bank" in "bank account".' },

        { type: 'heading', text: 'Multi-Head Attention' },
        { type: 'paragraph', text: 'Instead of one attention function, Transformers use multiple "heads" that attend to different aspects. One head might focus on syntax, another on semantics, another on coreference. The outputs are concatenated and projected.' },
        { type: 'formula', formula: 'MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O' },

        { type: 'keypoints', points: [
          'Attention allows each token to directly interact with every other token',
          'Self-attention derives Q, K, V from the same sequence',
          'Multi-head attention captures different types of relationships',
          'Parallelization enables training on massive datasets'
        ]}
      ],
      quiz: [
        {
          question: 'What is the main advantage of attention over recurrence?',
          options: ['Smaller models', 'Direct access to all positions enables parallelization', 'Less training data needed', 'Simpler implementation'],
          correct: 1,
          explanation: 'Attention allows parallel processing of all positions and direct long-range dependencies without sequential processing.'
        },
        {
          question: 'Why divide by √dₖ in the attention formula?',
          options: ['Normalization', 'Prevent softmax from becoming too peaked', 'Speed optimization', 'Memory efficiency'],
          correct: 1,
          explanation: 'Without scaling, dot products grow large with dimension, causing softmax to have extremely small gradients.'
        }
      ]
    },
    {
      id: 'bart-architecture',
      title: 'BART: Denoising Autoencoder',
      duration: '15 min read',
      concepts: ['BART', 'Encoder-Decoder', 'Denoising'],
      content: [
        { type: 'heading', text: 'What Makes BART Special' },
        { type: 'paragraph', text: 'BART (Bidirectional and Auto-Regressive Transformers) combines ideas from BERT and GPT. Like BERT, it uses bidirectional encoding. Like GPT, it generates text autoregressively. This combination makes it excellent for generation tasks like summarization.' },

        { type: 'subheading', text: 'The Encoder-Decoder Architecture' },
        { type: 'paragraph', text: 'BART has two parts: an encoder that processes the input bidirectionally, and a decoder that generates output one token at a time. The decoder attends to both its own previous outputs and the encoder representations.' },

        { type: 'heading', text: 'Denoising Pre-training' },
        { type: 'paragraph', text: 'BART is pre-trained by corrupting text and learning to reconstruct the original. Corruptions include:' },
        { type: 'list', items: [
          'Token masking: Replace tokens with [MASK]',
          'Token deletion: Remove tokens entirely',
          'Sentence permutation: Shuffle sentence order',
          'Document rotation: Rotate document to start at random token',
          'Text infilling: Replace spans with single [MASK] token'
        ]},
        { type: 'paragraph', text: 'This diverse corruption teaches BART to understand and generate text at multiple levels—from individual tokens to document structure.' },

        { type: 'heading', text: 'Your BART Implementation' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'BART-transformer-summary-gen',
          code: `from transformers import BartTokenizer, BartForConditionalGeneration

class bartsummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)` },

        { type: 'paragraph', text: 'facebook/bart-large-cnn is BART fine-tuned on the CNN/DailyMail summarization dataset. The "large" variant has 400M parameters with 12 encoder and 12 decoder layers.' },

        { type: 'keypoints', points: [
          'BART combines bidirectional encoding with autoregressive decoding',
          'Pre-trained through diverse text corruption and reconstruction',
          'Encoder-decoder architecture ideal for sequence-to-sequence tasks',
          'Fine-tuned versions available for specific tasks like summarization'
        ]}
      ],
      quiz: [
        {
          question: 'How does BART differ from BERT?',
          options: ['BART is smaller', 'BART has a decoder for generation', 'BART uses different attention', 'BART requires less data'],
          correct: 1,
          explanation: 'BERT is encoder-only (good for understanding), while BART adds a decoder for generation tasks.'
        }
      ]
    },
    {
      id: 'beam-search',
      title: 'Beam Search Decoding',
      duration: '12 min read',
      concepts: ['Beam Search', 'Decoding', 'Generation'],
      content: [
        { type: 'heading', text: 'The Decoding Challenge' },
        { type: 'paragraph', text: 'Given a trained model, how do we generate the best output sequence? Greedy decoding picks the highest probability token at each step, but this often misses better sequences. Exhaustive search is computationally impossible.' },

        { type: 'heading', text: 'How Beam Search Works' },
        { type: 'paragraph', text: 'Beam search maintains k candidate sequences (beams) at each step. At each position, it expands all beams with all possible tokens, scores them, and keeps only the top k. This balances exploration with computational cost.' },
        { type: 'paragraph', text: 'With beam size 4, you explore 4 hypotheses in parallel, pruning less promising ones at each step.' },

        { type: 'heading', text: 'Your Generation Parameters' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'BART-transformer-summary-gen',
          code: `def summarize(self, text, max_length=150, min_length=40, num_beams=4):
    inputs = self.tokenizer(
        text,
        return_tensors='pt',
        max_length=1024,
        truncation=True
    )

    summary_ids = self.model.generate(
        inputs['input_ids'],
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=2.0,
        early_stopping=True
    )

    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary` },

        { type: 'subheading', text: 'Understanding the Parameters' },
        { type: 'list', items: [
          'num_beams=4: Keep 4 candidate sequences at each step',
          'length_penalty=2.0: Penalize shorter sequences (encourages complete summaries)',
          'early_stopping=True: Stop when all beams reach end token',
          'min_length=40: Force minimum output length',
          'max_length=150: Cap maximum output length'
        ]},

        { type: 'callout', variant: 'tip', text: 'Higher beam sizes improve quality but increase computation linearly. For production, 4-5 beams is usually the sweet spot.' },

        { type: 'keypoints', points: [
          'Greedy decoding often misses optimal sequences',
          'Beam search explores multiple hypotheses in parallel',
          'Length penalty prevents degenerate short outputs',
          'Early stopping optimizes inference when all beams complete'
        ]}
      ],
      quiz: [
        {
          question: 'Why use length_penalty > 1.0?',
          options: ['Speed up generation', 'Prevent very short outputs', 'Reduce memory usage', 'Improve token accuracy'],
          correct: 1,
          explanation: 'Length penalty > 1 penalizes shorter sequences, encouraging the model to produce complete, informative summaries.'
        }
      ]
    },
    {
      id: 'tokenization-bpe',
      title: 'Subword Tokenization',
      duration: '10 min read',
      concepts: ['BPE', 'Tokenization', 'Vocabulary'],
      content: [
        { type: 'heading', text: 'Beyond Word Tokenization' },
        { type: 'paragraph', text: 'Word-level tokenization struggles with rare words and morphological variations. "unhappiness", "unhappy", and "happiness" are treated as completely different tokens. Subword tokenization breaks words into meaningful pieces.' },

        { type: 'heading', text: 'Byte Pair Encoding (BPE)' },
        { type: 'paragraph', text: 'BPE starts with characters and iteratively merges the most frequent pairs. After training, common words remain whole while rare words split into subwords. "unhappiness" might become ["un", "happiness"] or ["un", "happy", "ness"].' },
        { type: 'paragraph', text: 'BART uses BPE with a vocabulary of 50,265 tokens—enough to represent any text while keeping embeddings manageable.' },

        { type: 'subheading', text: 'Advantages of Subword Tokenization' },
        { type: 'list', items: [
          'No out-of-vocabulary problem—any word can be represented',
          'Shares representations across morphological variants',
          'Balances vocabulary size with sequence length',
          'Captures meaningful subword patterns'
        ]},

        { type: 'keypoints', points: [
          'Subword tokenization balances vocabulary size and coverage',
          'BPE learns merges from training corpus frequency',
          'Rare words decompose into known subwords',
          'BART uses ~50K vocabulary covering all possible inputs'
        ]}
      ],
      quiz: [
        {
          question: 'What problem does subword tokenization solve?',
          options: ['Slow training', 'Out-of-vocabulary words', 'Memory limits', 'Overfitting'],
          correct: 1,
          explanation: 'Subword tokenization ensures any word can be represented by decomposing rare words into known subword units.'
        }
      ]
    }
  ]
}
