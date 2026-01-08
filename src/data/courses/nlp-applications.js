export const nlpapplications = {
  id: 'nlp-applications',
  title: 'NLP Applications',
  description: 'Build practical natural language processing systems',
  category: 'Applications',
  difficulty: 'Advanced',
  duration: '6 hours',
  lessons: [
    {
      id: 'text-preprocessing',
      title: 'Text Preprocessing & Tokenization',
      duration: '55 min',
      concepts: ['tokenization', 'normalization', 'stemming', 'lemmatization', 'subword tokenization'],
      content: [
        {
          type: 'heading',
          text: 'Preparing Text for ML Models'
        },
        {
          type: 'text',
          text: 'Raw text is messy—inconsistent casing, punctuation, typos, contractions. Preprocessing transforms text into a clean, consistent format that ML models can learn from effectively.'
        },
        {
          type: 'subheading',
          text: 'The Preprocessing Pipeline'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowPrep" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Text Preprocessing Pipeline</text>
            <rect x="30" y="50" width="80" height="40" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="70" y="75" text-anchor="middle" font-size="9" fill="#4f46e5">Raw Text</text>
            <rect x="130" y="50" width="70" height="40" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="165" y="75" text-anchor="middle" font-size="9" fill="#1d4ed8">Lowercase</text>
            <rect x="220" y="50" width="70" height="40" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="255" y="75" text-anchor="middle" font-size="9" fill="#15803d">Tokenize</text>
            <rect x="310" y="50" width="70" height="40" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="345" y="75" text-anchor="middle" font-size="9" fill="#92400e">Normalize</text>
            <rect x="400" y="50" width="70" height="40" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="435" y="75" text-anchor="middle" font-size="9" fill="#7c3aed">Encode</text>
            <line x1="110" y1="70" x2="130" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowPrep)"/>
            <line x1="200" y1="70" x2="220" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowPrep)"/>
            <line x1="290" y1="70" x2="310" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowPrep)"/>
            <line x1="380" y1="70" x2="400" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowPrep)"/>
            <text x="70" y="115" text-anchor="middle" font-size="8" fill="#64748b">"I'm LOVING it!!"</text>
            <text x="165" y="115" text-anchor="middle" font-size="8" fill="#64748b">"i'm loving it!!"</text>
            <text x="255" y="115" text-anchor="middle" font-size="8" fill="#64748b">["i'm", "loving", "it"]</text>
            <text x="345" y="115" text-anchor="middle" font-size="8" fill="#64748b">["i", "am", "love", "it"]</text>
            <text x="435" y="115" text-anchor="middle" font-size="8" fill="#64748b">[5, 23, 891, 12]</text>
          </svg>`
        },
        {
          type: 'code',
          language: 'python',
          code: `import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

text = "The cats were running quickly through the gardens!"
processed = preprocess_text(text)
print(f"Original: {text}")
print(f"Processed: {processed}")`
        },
        {
          type: 'subheading',
          text: 'Tokenization Strategies'
        },
        {
          type: 'table',
          headers: ['Method', 'Description', 'Example'],
          rows: [
            ['Word', 'Split on whitespace/punctuation', '"hello world" → ["hello", "world"]'],
            ['Character', 'Each character is a token', '"cat" → ["c", "a", "t"]'],
            ['Subword (BPE)', 'Merge frequent character pairs', '"unhappiness" → ["un", "happi", "ness"]'],
            ['SentencePiece', 'Language-agnostic subword', 'Works with any script']
          ]
        },
        {
          type: 'subheading',
          text: 'Modern Tokenization: BPE and Beyond'
        },
        {
          type: 'text',
          text: 'Modern models use subword tokenization (BPE, WordPiece, SentencePiece). This handles rare words by breaking them into known subwords, solving the out-of-vocabulary problem while keeping vocabulary size manageable.'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

text = "Transformers are revolutionizing NLP!"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"IDs: {ids}")

decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Preprocessing vs End-to-End',
          text: 'Modern Transformers often skip traditional preprocessing (stemming, stopwords) because they learn these patterns. However, basic cleaning (handling encoding, removing noise) is still valuable.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What advantage does subword tokenization have over word-level tokenization?',
          options: [
            'Faster processing',
            'Handles out-of-vocabulary words by breaking them into known subwords',
            'Smaller model size',
            'Better accuracy on all tasks'
          ],
          correct: 1,
          explanation: 'Subword tokenization can represent any word by combining known subword units. "Unhappiness" becomes ["un", "happi", "ness"], allowing the model to handle rare or novel words.'
        }
      ]
    },
    {
      id: 'word-embeddings',
      title: 'Word Embeddings',
      duration: '60 min',
      concepts: ['word2vec', 'GloVe', 'embedding space', 'semantic similarity'],
      content: [
        {
          type: 'heading',
          text: 'From Words to Vectors'
        },
        {
          type: 'text',
          text: 'Word embeddings represent words as dense vectors where semantically similar words are close together. "King" and "queen" are nearby, as are "Paris" and "France". This captures meaning in a way that sparse one-hot vectors cannot.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Word Embedding Space (2D Projection)</text>
            <line x1="50" y1="180" x2="450" y2="180" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="50" y1="180" x2="50" y2="40" stroke="#e2e8f0" stroke-width="1"/>
            <circle cx="120" cy="80" r="5" fill="#6366f1"/>
            <text x="120" y="70" text-anchor="middle" font-size="10" fill="#1e293b">king</text>
            <circle cx="180" cy="90" r="5" fill="#6366f1"/>
            <text x="180" y="80" text-anchor="middle" font-size="10" fill="#1e293b">queen</text>
            <circle cx="100" cy="140" r="5" fill="#6366f1"/>
            <text x="100" y="155" text-anchor="middle" font-size="10" fill="#1e293b">man</text>
            <circle cx="160" cy="150" r="5" fill="#6366f1"/>
            <text x="160" y="165" text-anchor="middle" font-size="10" fill="#1e293b">woman</text>
            <circle cx="320" cy="100" r="5" fill="#22c55e"/>
            <text x="320" y="90" text-anchor="middle" font-size="10" fill="#1e293b">Paris</text>
            <circle cx="380" cy="110" r="5" fill="#22c55e"/>
            <text x="380" y="100" text-anchor="middle" font-size="10" fill="#1e293b">France</text>
            <circle cx="300" cy="160" r="5" fill="#22c55e"/>
            <text x="300" y="175" text-anchor="middle" font-size="10" fill="#1e293b">London</text>
            <circle cx="360" cy="170" r="5" fill="#22c55e"/>
            <text x="360" y="185" text-anchor="middle" font-size="10" fill="#1e293b">England</text>
            <line x1="120" y1="80" x2="180" y2="90" stroke="#6366f1" stroke-width="1" stroke-dasharray="3,3"/>
            <line x1="100" y1="140" x2="160" y2="150" stroke="#6366f1" stroke-width="1" stroke-dasharray="3,3"/>
            <text x="250" y="210" text-anchor="middle" font-size="10" fill="#64748b">king - man + woman ≈ queen</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Word2Vec: Skip-gram and CBOW'
        },
        {
          type: 'text',
          text: 'Word2Vec learns embeddings by predicting context. **Skip-gram** predicts surrounding words from the center word. **CBOW** predicts the center word from surrounding words.'
        },
        {
          type: 'formula',
          latex: 'P(w_{context} | w_{center}) = \\frac{\\exp(v_{context} \\cdot v_{center})}{\\sum_w \\exp(v_w \\cdot v_{center})}'
        },
        {
          type: 'code',
          language: 'python',
          code: `from gensim.models import Word2Vec
import numpy as np

sentences = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'ran', 'in', 'the', 'park'],
    ['cats', 'and', 'dogs', 'are', 'pets'],
    ['the', 'king', 'wore', 'a', 'crown'],
    ['the', 'queen', 'sat', 'on', 'the', 'throne']
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, epochs=100)

if 'cat' in model.wv:
    print(f"Vector for 'cat': {model.wv['cat'][:5]}...")
    similar = model.wv.most_similar('cat', topn=3)
    print(f"Most similar to 'cat': {similar}")`
        },
        {
          type: 'subheading',
          text: 'GloVe: Global Vectors'
        },
        {
          type: 'text',
          text: 'GloVe combines global co-occurrence statistics with local context windows. It factorizes the log co-occurrence matrix, capturing both local and global patterns.'
        },
        {
          type: 'code',
          language: 'python',
          code: `import gensim.downloader as api

glove = api.load('glove-wiki-gigaword-100')

def analogy(a, b, c, model):
    result = model.most_similar(positive=[b, c], negative=[a], topn=1)
    return result[0][0]

print(f"king - man + woman = {analogy('man', 'king', 'woman', glove)}")
print(f"paris - france + italy = {analogy('france', 'paris', 'italy', glove)}")

similarity = glove.similarity('good', 'great')
print(f"Similarity(good, great): {similarity:.4f}")`
        },
        {
          type: 'table',
          headers: ['Embedding', 'Training', 'Strengths'],
          rows: [
            ['Word2Vec', 'Local context windows', 'Fast training, captures syntax'],
            ['GloVe', 'Global co-occurrence matrix', 'Captures global statistics'],
            ['FastText', 'Subword n-grams', 'Handles OOV, morphology'],
            ['ELMo', 'Bidirectional LSTM', 'Context-dependent embeddings']
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Static vs Contextual',
          text: 'Word2Vec and GloVe give each word ONE vector regardless of context. "Bank" has the same embedding whether referring to a river bank or a financial bank. Contextual embeddings (ELMo, BERT) give different vectors based on context.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does the famous "king - man + woman ≈ queen" example demonstrate?',
          options: [
            'Word embeddings can solve math problems',
            'Semantic relationships are encoded as vector directions',
            'Word2Vec always produces accurate results',
            'All words have equal distances'
          ],
          correct: 1,
          explanation: 'The king-queen, man-woman relationship shows that word embeddings capture semantic relationships as directions in vector space. The "gender" direction is similar in both pairs.'
        }
      ]
    },
    {
      id: 'sentiment-analysis',
      title: 'Sentiment Analysis',
      duration: '55 min',
      concepts: ['polarity', 'aspect-based', 'fine-grained sentiment', 'transfer learning'],
      content: [
        {
          type: 'heading',
          text: 'Understanding Opinion in Text'
        },
        {
          type: 'text',
          text: 'Sentiment analysis determines the emotional tone of text—positive, negative, or neutral. Applications range from product review analysis to social media monitoring and brand reputation management.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Sentiment Analysis Types</text>
            <rect x="30" y="50" width="130" height="80" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="95" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">Binary</text>
            <text x="95" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Positive / Negative</text>
            <text x="95" y="115" text-anchor="middle" font-size="8" fill="#64748b">"Great movie!" → +</text>
            <rect x="180" y="50" width="130" height="80" fill="#dbeafe" stroke="#3b82f6" rx="6"/>
            <text x="245" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1d4ed8">Fine-grained</text>
            <text x="245" y="95" text-anchor="middle" font-size="9" fill="#1e293b">1-5 stars / scale</text>
            <text x="245" y="115" text-anchor="middle" font-size="8" fill="#64748b">"Decent film" → 3/5</text>
            <rect x="330" y="50" width="140" height="80" fill="#f3e8ff" stroke="#a855f7" rx="6"/>
            <text x="400" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#7c3aed">Aspect-based</text>
            <text x="400" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Per feature/aspect</text>
            <text x="400" y="115" text-anchor="middle" font-size="8" fill="#64748b">Food: +, Service: -</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Traditional Approaches'
        },
        {
          type: 'code',
          language: 'python',
          code: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

texts = [
    "This movie was fantastic! I loved every minute.",
    "Terrible film. Complete waste of time.",
    "An amazing experience, highly recommended!",
    "Boring and predictable. Don't bother.",
    "Best movie I've seen this year!",
    "Awful acting and terrible plot."
]
labels = [1, 0, 1, 0, 1, 0]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(texts, labels)

test_texts = ["Great story and excellent acting!", "So boring I fell asleep"]
predictions = pipeline.predict(test_texts)
probs = pipeline.predict_proba(test_texts)

for text, pred, prob in zip(test_texts, predictions, probs):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = max(prob)
    print(f"'{text}' → {sentiment} ({confidence:.2%})")`
        },
        {
          type: 'subheading',
          text: 'Deep Learning for Sentiment'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I absolutely loved this product!",
    "Worst purchase I've ever made.",
    "It's okay, nothing special.",
    "The quality exceeded my expectations."
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"'{text}'")
    print(f"  → {result['label']}: {result['score']:.4f}\\n")`
        },
        {
          type: 'subheading',
          text: 'Aspect-Based Sentiment Analysis'
        },
        {
          type: 'text',
          text: 'ABSA extracts sentiments for specific aspects of a product or service. "The food was excellent but the service was slow" has positive sentiment for food, negative for service.'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline

classifier = pipeline("zero-shot-classification")

review = "The laptop has a beautiful screen and fast processor, but the battery life is disappointing."

aspects = ["screen quality", "processor speed", "battery life", "build quality"]

for aspect in aspects:
    result = classifier(
        review,
        candidate_labels=["positive", "negative", "neutral"],
        hypothesis_template=f"The sentiment about {aspect} is {{}}."
    )
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    print(f"{aspect}: {top_label} ({top_score:.2f})")`
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'Challenges',
          text: 'Sentiment analysis struggles with sarcasm ("Oh great, another meeting"), negation ("not bad" is positive), and domain-specific language. Always evaluate on domain-relevant test data.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does aspect-based sentiment analysis provide that binary sentiment does not?',
          options: [
            'Higher accuracy',
            'Faster processing',
            'Sentiment for specific features/aspects mentioned in text',
            'Language translation'
          ],
          correct: 2,
          explanation: 'ABSA identifies sentiments for individual aspects (food quality, service speed) rather than giving one overall sentiment. This provides more actionable insights for businesses.'
        }
      ]
    },
    {
      id: 'named-entity-recognition',
      title: 'Named Entity Recognition',
      duration: '55 min',
      concepts: ['NER', 'entity types', 'BIO tagging', 'sequence labeling'],
      content: [
        {
          type: 'heading',
          text: 'Identifying Entities in Text'
        },
        {
          type: 'text',
          text: 'Named Entity Recognition (NER) identifies and classifies named entities in text—people, organizations, locations, dates, etc. It is essential for information extraction, question answering, and knowledge graph construction.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 160" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">NER: Entity Identification</text>
            <rect x="30" y="50" width="60" height="30" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="60" y="70" text-anchor="middle" font-size="10" fill="#1e293b">Apple</text>
            <text x="60" y="95" text-anchor="middle" font-size="8" fill="#3b82f6">ORG</text>
            <text x="100" y="70" text-anchor="middle" font-size="10" fill="#64748b">CEO</text>
            <rect x="130" y="50" width="75" height="30" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="167" y="70" text-anchor="middle" font-size="10" fill="#1e293b">Tim Cook</text>
            <text x="167" y="95" text-anchor="middle" font-size="8" fill="#22c55e">PERSON</text>
            <text x="220" y="70" text-anchor="middle" font-size="10" fill="#64748b">visited</text>
            <rect x="255" y="50" width="50" height="30" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="280" y="70" text-anchor="middle" font-size="10" fill="#1e293b">Paris</text>
            <text x="280" y="95" text-anchor="middle" font-size="8" fill="#f59e0b">LOC</text>
            <text x="320" y="70" text-anchor="middle" font-size="10" fill="#64748b">on</text>
            <rect x="340" y="50" width="80" height="30" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="380" y="70" text-anchor="middle" font-size="10" fill="#1e293b">March 5th</text>
            <text x="380" y="95" text-anchor="middle" font-size="8" fill="#a855f7">DATE</text>
            <text x="430" y="70" text-anchor="middle" font-size="10" fill="#64748b">.</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'BIO Tagging Scheme'
        },
        {
          type: 'text',
          text: 'NER is framed as sequence labeling. Each token gets a tag: B-TYPE (beginning of entity), I-TYPE (inside entity), O (outside any entity).'
        },
        {
          type: 'table',
          headers: ['Token', 'Tag', 'Meaning'],
          rows: [
            ['New', 'B-LOC', 'Beginning of location'],
            ['York', 'I-LOC', 'Inside location (continuation)'],
            ['City', 'I-LOC', 'Inside location (continuation)'],
            ['is', 'O', 'Not an entity'],
            ['amazing', 'O', 'Not an entity']
          ]
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")

text = "Elon Musk founded SpaceX in 2002 and Tesla is headquartered in Austin, Texas."

entities = ner(text)

print(f"Text: {text}\\n")
print("Entities found:")
for entity in entities:
    print(f"  {entity['word']}: {entity['entity_group']} ({entity['score']:.2f})")`
        },
        {
          type: 'subheading',
          text: 'Training Custom NER Models'
        },
        {
          type: 'code',
          language: 'python',
          code: `import spacy
from spacy.tokens import DocBin
from spacy.training import Example

nlp = spacy.blank("en")

training_data = [
    ("iPhone 15 Pro was released by Apple", {"entities": [(0, 13, "PRODUCT"), (31, 36, "ORG")]}),
    ("Google announced Pixel 8 in October", {"entities": [(0, 6, "ORG"), (17, 24, "PRODUCT"), (28, 35, "DATE")]}),
]

ner = nlp.add_pipe("ner")
for _, annotations in training_data:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

optimizer = nlp.begin_training()
for epoch in range(30):
    losses = {}
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer, losses=losses)

doc = nlp("Samsung unveiled Galaxy S24 yesterday")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Common Entity Types',
          text: 'Standard types include PERSON, ORG, LOC, DATE, TIME, MONEY, PERCENT. Domain-specific NER adds types like DRUG, DISEASE, GENE for biomedical text or PRODUCT, FEATURE for e-commerce.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'In BIO tagging, what does the I- prefix indicate?',
          options: [
            'Important entity',
            'Initial token of an entity',
            'Inside/continuation of a multi-token entity',
            'Invalid entity'
          ],
          correct: 2,
          explanation: 'I- (Inside) marks tokens that continue an entity started by B- (Beginning). For "New York City" tagged as a location: New=B-LOC, York=I-LOC, City=I-LOC.'
        }
      ]
    },
    {
      id: 'text-classification',
      title: 'Text Classification',
      duration: '55 min',
      concepts: ['multiclass', 'multilabel', 'topic modeling', 'zero-shot classification'],
      content: [
        {
          type: 'heading',
          text: 'Categorizing Text at Scale'
        },
        {
          type: 'text',
          text: 'Text classification assigns predefined categories to text. Applications include spam detection, topic labeling, intent classification, and content moderation. It is one of the most common NLP tasks in production.'
        },
        {
          type: 'subheading',
          text: 'Classification Types'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Classification Types</text>
            <rect x="30" y="50" width="140" height="90" fill="#e0e7ff" stroke="#6366f1" rx="6"/>
            <text x="100" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#4f46e5">Multi-class</text>
            <text x="100" y="95" text-anchor="middle" font-size="9" fill="#1e293b">One label per document</text>
            <text x="100" y="115" text-anchor="middle" font-size="8" fill="#64748b">Sports | Politics | Tech</text>
            <text x="100" y="130" text-anchor="middle" font-size="8" fill="#22c55e">→ Sports ✓</text>
            <rect x="180" y="50" width="140" height="90" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="250" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">Multi-label</text>
            <text x="250" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Multiple labels per doc</text>
            <text x="250" y="115" text-anchor="middle" font-size="8" fill="#64748b">Tags: AI, Business, Tech</text>
            <text x="250" y="130" text-anchor="middle" font-size="8" fill="#22c55e">→ AI ✓, Tech ✓</text>
            <rect x="330" y="50" width="140" height="90" fill="#fef3c7" stroke="#f59e0b" rx="6"/>
            <text x="400" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">Hierarchical</text>
            <text x="400" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Nested categories</text>
            <text x="400" y="115" text-anchor="middle" font-size="8" fill="#64748b">Sports → Football → NFL</text>
            <text x="400" y="130" text-anchor="middle" font-size="8" fill="#22c55e">→ NFL ✓</text>
          </svg>`
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "Breaking: Major earthquake hits California",
    "Apple stock surges after earnings report",
    "New study reveals benefits of meditation",
    "Champions League final ends in dramatic penalty shootout"
]

for text in texts:
    result = classifier(text)[0]
    print(f"'{text[:50]}...'")
    print(f"  Label: {result['label']}, Score: {result['score']:.4f}\\n")`
        },
        {
          type: 'subheading',
          text: 'Zero-Shot Classification'
        },
        {
          type: 'text',
          text: 'Zero-shot classification uses language understanding to classify into categories never seen during training. It works by framing classification as natural language inference.'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline

zero_shot = pipeline("zero-shot-classification")

text = "The new electric vehicle has a range of 400 miles and charges in 20 minutes."

categories = ["automotive", "technology", "environment", "finance", "sports"]

result = zero_shot(text, categories)

print(f"Text: '{text}'\\n")
print("Category scores:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.4f}")`
        },
        {
          type: 'subheading',
          text: 'Fine-tuning for Custom Classification'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

data = {
    "text": ["Great service!", "Terrible experience", "It was okay", "Amazing!", "Awful"],
    "label": [2, 0, 1, 2, 0]
}
dataset = Dataset.from_dict(data)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Label Imbalance',
          text: 'Real-world classification often has imbalanced classes (99% non-spam, 1% spam). Address this with oversampling, class weights, or focal loss. Always report per-class metrics, not just accuracy.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'How does zero-shot classification work without task-specific training?',
          options: [
            'It uses random guessing',
            'It frames classification as natural language inference using language understanding',
            'It requires a massive labeled dataset',
            'It only works for binary classification'
          ],
          correct: 1,
          explanation: 'Zero-shot classification uses pre-trained language understanding to determine if a text entails a label hypothesis (e.g., "This text is about sports"). No task-specific training needed.'
        }
      ]
    },
    {
      id: 'seq2seq-translation',
      title: 'Sequence-to-Sequence & Translation',
      duration: '60 min',
      concepts: ['encoder-decoder', 'machine translation', 'beam search', 'BLEU score'],
      content: [
        {
          type: 'heading',
          text: 'Transforming Sequences'
        },
        {
          type: 'text',
          text: 'Sequence-to-sequence models transform one sequence into another: translation (English→French), summarization (article→summary), or question answering (context+question→answer). The encoder processes input, the decoder generates output.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowSeq2Seq" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Seq2Seq for Translation</text>
            <rect x="30" y="80" width="180" height="60" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="8"/>
            <text x="120" y="115" text-anchor="middle" font-size="12" font-weight="bold" fill="#4f46e5">Encoder</text>
            <rect x="290" y="80" width="180" height="60" fill="#dcfce7" stroke="#22c55e" stroke-width="2" rx="8"/>
            <text x="380" y="115" text-anchor="middle" font-size="12" font-weight="bold" fill="#15803d">Decoder</text>
            <line x1="210" y1="110" x2="290" y2="110" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowSeq2Seq)"/>
            <text x="250" y="100" text-anchor="middle" font-size="9" fill="#64748b">context</text>
            <text x="120" y="55" text-anchor="middle" font-size="10" fill="#1e293b">"Hello, how are you?"</text>
            <line x1="120" y1="60" x2="120" y2="80" stroke="#6366f1" stroke-width="1.5"/>
            <text x="380" y="165" text-anchor="middle" font-size="10" fill="#1e293b">"Bonjour, comment allez-vous?"</text>
            <line x1="380" y1="140" x2="380" y2="155" stroke="#22c55e" stroke-width="1.5"/>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Modern Translation with Transformers'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

texts = [
    "Hello, how are you?",
    "Machine learning is transforming the world.",
    "The weather is beautiful today."
]

for text in texts:
    result = translator(text)[0]
    print(f"EN: {text}")
    print(f"FR: {result['translation_text']}\\n")`
        },
        {
          type: 'subheading',
          text: 'Decoding Strategies'
        },
        {
          type: 'text',
          text: 'The decoder generates output one token at a time. How we select each token affects quality:'
        },
        {
          type: 'table',
          headers: ['Strategy', 'Method', 'Trade-off'],
          rows: [
            ['Greedy', 'Pick highest probability token', 'Fast but may miss better sequences'],
            ['Beam Search', 'Keep top-k candidates at each step', 'Better quality, more computation'],
            ['Sampling', 'Sample from probability distribution', 'More diverse, less deterministic'],
            ['Top-k/Top-p', 'Sample from top candidates only', 'Balance quality and diversity']
          ]
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "translate English to German: The house is beautiful."
inputs = tokenizer(text, return_tensors="pt")

greedy_output = model.generate(**inputs, max_length=50)
beam_output = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
sample_output = model.generate(**inputs, max_length=50, do_sample=True, top_k=50, temperature=0.7)

print(f"Input: {text}")
print(f"Greedy: {tokenizer.decode(greedy_output[0], skip_special_tokens=True)}")
print(f"Beam: {tokenizer.decode(beam_output[0], skip_special_tokens=True)}")
print(f"Sample: {tokenizer.decode(sample_output[0], skip_special_tokens=True)}")`
        },
        {
          type: 'subheading',
          text: 'Evaluation: BLEU Score'
        },
        {
          type: 'text',
          text: 'BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between generated and reference translations. Higher is better, but it has limitations—it does not capture meaning, only surface similarity.'
        },
        {
          type: 'code',
          language: 'python',
          code: `from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['the', 'cat', 'sat', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

score = sentence_bleu(reference, candidate)
print(f"BLEU score: {score:.4f}")

references = [[['the', 'cat', 'sat']], [['hello', 'world']]]
candidates = [['the', 'cat', 'sat'], ['hi', 'world']]
corpus_score = corpus_bleu(references, candidates)
print(f"Corpus BLEU: {corpus_score:.4f}")`
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'BLEU Limitations',
          text: 'BLEU does not measure fluency, meaning, or grammaticality—only n-gram overlap. "The cat sat" and "Sat the cat" have similar BLEU but different quality. Consider METEOR, BERTScore, or human evaluation for comprehensive assessment.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the advantage of beam search over greedy decoding?',
          options: [
            'Faster generation',
            'Less memory usage',
            'Considers multiple candidate sequences to find better overall output',
            'Produces more random outputs'
          ],
          correct: 2,
          explanation: 'Beam search keeps track of the top-k most likely sequences at each step, avoiding getting stuck in locally optimal but globally suboptimal paths that greedy decoding might find.'
        }
      ]
    }
  ]
}
