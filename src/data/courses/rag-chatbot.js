export const ragcourse = {
  id: 'rag-chatbot',
  title: 'RAG & Vector Databases',
  icon: 'chat',
  description: 'Build AI chatbots that answer questions from your own documents using Retrieval-Augmented Generation.',
  difficulty: 'advanced',
  sourceproject: 'RAG-chatbot',
  lessons: [
    {
      id: 'what-is-rag',
      title: 'What is RAG?',
      duration: '12 min read',
      concepts: ['RAG', 'LLM Limitations', 'Knowledge Grounding'],
      content: [
        { type: 'heading', text: 'The LLM Knowledge Problem' },
        { type: 'paragraph', text: 'Large Language Models like GPT and Mistral are trained on massive datasets, but they have limitations: their knowledge is frozen at training time, they can hallucinate facts, and they know nothing about your private data.' },
        { type: 'paragraph', text: 'Imagine asking an LLM about your company policies or personal projects. It simply cannot answer accurately because that information was never in its training data.' },

        { type: 'heading', text: 'Retrieval-Augmented Generation' },
        { type: 'paragraph', text: 'RAG solves this by combining retrieval with generation. Instead of relying solely on the model\'s internal knowledge, we first retrieve relevant documents from a knowledge base, then provide them as context for the LLM to generate an answer.' },
        { type: 'list', items: [
          'User asks a question',
          'System retrieves relevant documents from knowledge base',
          'Retrieved context + question sent to LLM',
          'LLM generates answer grounded in the retrieved facts'
        ]},

        { type: 'subheading', text: 'Why RAG Works' },
        { type: 'paragraph', text: 'RAG leverages the best of both worlds: the reasoning and language capabilities of LLMs, combined with accurate, up-to-date information from your documents. The LLM acts as a reasoning engine over retrieved facts rather than a knowledge store.' },

        { type: 'heading', text: 'The RAG Architecture' },
        { type: 'code', language: 'python', filename: 'rag.py', fromproject: 'RAG-chatbot',
          code: `class ragchatbot:
    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        self.documents_path = Path(__file__).parent / "documents"
        self.vectorstore_path = Path(__file__).parent / "vectorstore"

        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1024
        )` },

        { type: 'paragraph', text: 'The architecture has three main components: a document store, a vector database for retrieval, and an LLM for generation. We use Mistral 7B through OpenRouter as our LLM.' },

        { type: 'keypoints', points: [
          'RAG grounds LLM responses in retrieved documents',
          'Solves knowledge cutoff and hallucination problems',
          'Enables LLMs to answer questions about private data',
          'Combines retrieval accuracy with generative fluency'
        ]}
      ],
      quiz: [
        {
          question: 'What problem does RAG primarily solve?',
          options: ['Slow inference', 'LLMs lacking knowledge of private/current data', 'High API costs', 'Model size'],
          correct: 1,
          explanation: 'RAG addresses the limitation that LLMs only know their training data by retrieving relevant documents at query time.'
        },
        {
          question: 'In RAG, what role does the LLM play?',
          options: ['Document storage', 'Similarity search', 'Reasoning over retrieved context', 'Embedding generation'],
          correct: 2,
          explanation: 'The LLM acts as a reasoning engine that generates answers based on the retrieved context, not as a knowledge store.'
        }
      ]
    },
    {
      id: 'vector-embeddings',
      title: 'Vector Embeddings & Semantic Search',
      duration: '15 min read',
      concepts: ['Embeddings', 'Semantic Search', 'Sentence Transformers'],
      content: [
        { type: 'heading', text: 'From Text to Vectors' },
        { type: 'paragraph', text: 'How do we find documents relevant to a query? Keyword matching fails when the query uses different words than the documents. We need semantic search that understands meaning.' },
        { type: 'paragraph', text: 'Embeddings convert text into dense vectors where semantically similar texts have similar vectors. "What are David\'s skills?" and "Tell me about his technical abilities" should retrieve the same documents.' },

        { type: 'heading', text: 'Sentence Transformers' },
        { type: 'paragraph', text: 'We use sentence-transformers/all-MiniLM-L6-v2, a lightweight model that produces 384-dimensional embeddings. It\'s trained to place semantically similar sentences close together in vector space.' },
        { type: 'code', language: 'python', filename: 'rag.py', fromproject: 'RAG-chatbot',
          code: `from langchain_community.embeddings import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)` },

        { type: 'subheading', text: 'How Embeddings Work' },
        { type: 'paragraph', text: 'The embedding model was trained on millions of sentence pairs, learning to produce similar vectors for paraphrases and different vectors for unrelated sentences. At inference, it encodes any text into this learned semantic space.' },

        { type: 'heading', text: 'Similarity Search' },
        { type: 'paragraph', text: 'With both documents and queries as vectors, finding relevant documents becomes a nearest-neighbor search. We use cosine similarity to measure how aligned two vectors are.' },
        { type: 'formula', formula: 'similarity(A, B) = (A · B) / (||A|| × ||B||)' },
        { type: 'paragraph', text: 'A similarity of 1 means identical direction (same meaning), 0 means orthogonal (unrelated), and -1 means opposite.' },

        { type: 'keypoints', points: [
          'Embeddings encode semantic meaning into dense vectors',
          'Similar meanings result in similar vectors',
          'Enables finding relevant documents regardless of exact wording',
          'Sentence Transformers are optimized for sentence-level similarity'
        ]}
      ],
      quiz: [
        {
          question: 'Why use embeddings instead of keyword matching?',
          options: ['Faster search', 'Smaller storage', 'Captures semantic meaning', 'Simpler implementation'],
          correct: 2,
          explanation: 'Embeddings capture semantic meaning, so "skills" and "abilities" are recognized as similar concepts.'
        },
        {
          question: 'What does all-MiniLM-L6-v2 produce?',
          options: ['Text summaries', '384-dimensional vectors', 'Classification labels', 'Token probabilities'],
          correct: 1,
          explanation: 'This sentence transformer model outputs 384-dimensional dense vectors representing the semantic content of text.'
        }
      ]
    },
    {
      id: 'faiss-vectorstore',
      title: 'FAISS Vector Database',
      duration: '12 min read',
      concepts: ['FAISS', 'Vector Database', 'Approximate Search'],
      content: [
        { type: 'heading', text: 'The Search Problem' },
        { type: 'paragraph', text: 'With thousands of document chunks, each as a 384-dimensional vector, how do we quickly find the most similar ones to a query? Brute-force comparison is O(n) for each query—too slow for production.' },

        { type: 'heading', text: 'FAISS: Facebook AI Similarity Search' },
        { type: 'paragraph', text: 'FAISS is a library for efficient similarity search. It uses various indexing strategies to enable sub-linear search times. For our use case, it makes retrieval nearly instantaneous even with large document collections.' },
        { type: 'code', language: 'python', filename: 'rag.py', fromproject: 'RAG-chatbot',
          code: `from langchain_community.vectorstores import FAISS

def _create_vectorstore(self):
    loader = DirectoryLoader(
        str(self.documents_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(documents)

    self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
    self.vectorstore.save_local(str(self.vectorstore_path))` },

        { type: 'heading', text: 'Document Chunking' },
        { type: 'paragraph', text: 'Documents are split into chunks of ~1000 characters with 200-character overlap. Why? Embeddings work best on focused passages, and overlap ensures we don\'t split important context between chunks.' },

        { type: 'subheading', text: 'RecursiveCharacterTextSplitter' },
        { type: 'paragraph', text: 'This splitter tries to keep semantically related text together. It first tries to split on paragraphs, then sentences, then words, only resorting to character-level splits when necessary.' },

        { type: 'heading', text: 'Persistence' },
        { type: 'paragraph', text: 'FAISS indexes can be saved to disk and loaded later, avoiding re-embedding documents on every startup. This is critical for production where embedding thousands of chunks takes time.' },
        { type: 'code', language: 'python', filename: 'rag.py', fromproject: 'RAG-chatbot',
          code: `def _load_vectorstore(self):
    self.vectorstore = FAISS.load_local(
        str(self.vectorstore_path),
        self.embeddings,
        allow_dangerous_deserialization=True
    )` },

        { type: 'keypoints', points: [
          'FAISS enables fast similarity search over large vector collections',
          'Documents are chunked into ~1000 character passages',
          'Chunk overlap preserves context across boundaries',
          'Vectorstore persists to disk for fast startup'
        ]}
      ],
      quiz: [
        {
          question: 'Why chunk documents before embedding?',
          options: ['Reduce storage', 'Embeddings work better on focused passages', 'API limits', 'Faster embedding'],
          correct: 1,
          explanation: 'Embedding models are optimized for sentence/paragraph level text, not entire documents. Chunks ensure focused, relevant retrieval.'
        },
        {
          question: 'What is the purpose of chunk overlap?',
          options: ['Increase storage', 'Preserve context across chunk boundaries', 'Speed up search', 'Reduce duplicates'],
          correct: 1,
          explanation: 'Overlap ensures that important context split between chunks is still captured in at least one chunk.'
        }
      ]
    },
    {
      id: 'langchain-pipeline',
      title: 'LangChain Retrieval Pipeline',
      duration: '14 min read',
      concepts: ['LangChain', 'ConversationalRetrievalChain', 'Prompt Engineering'],
      content: [
        { type: 'heading', text: 'What is LangChain?' },
        { type: 'paragraph', text: 'LangChain is a framework for building LLM applications. It provides abstractions for common patterns like retrieval, chains, and agents. Instead of manually orchestrating retrieval and generation, LangChain handles the plumbing.' },

        { type: 'heading', text: 'ConversationalRetrievalChain' },
        { type: 'paragraph', text: 'This chain handles the full RAG pipeline: it takes a question, retrieves relevant documents, formats them into a prompt, sends to the LLM, and returns the response. It also maintains chat history for multi-turn conversations.' },
        { type: 'code', language: 'python', filename: 'rag.py', fromproject: 'RAG-chatbot',
          code: `from langchain.chains import ConversationalRetrievalChain

def _create_chain(self):
    self.chain = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )` },

        { type: 'subheading', text: 'The Retriever' },
        { type: 'paragraph', text: 'search_kwargs={"k": 4} means we retrieve the 4 most relevant chunks for each query. More chunks provide more context but risk including irrelevant information and hitting token limits.' },

        { type: 'heading', text: 'Prompt Engineering' },
        { type: 'paragraph', text: 'The prompt template shapes how the LLM uses retrieved context. We instruct it to be a portfolio assistant, use the provided information, and maintain conversational style.' },
        { type: 'code', language: 'python', filename: 'rag.py', fromproject: 'RAG-chatbot',
          code: `system_prompt = """You are David Zuokumor's portfolio assistant. Use the information provided to answer questions about David.

Rules:
- Be concise, friendly, and natural
- Never say "based on the context" or mention documents
- Only greet on the first message of a conversation
- When ranking projects by complexity, follow the tier system in the data
- Always complete your responses fully

{context}

Chat history: {chat_history}

Question: {question}

Response:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=system_prompt
)` },

        { type: 'paragraph', text: 'Notice the prompt includes {context} where retrieved documents are inserted, {chat_history} for conversation continuity, and {question} for the current query.' },

        { type: 'keypoints', points: [
          'LangChain abstracts the RAG pipeline into reusable components',
          'ConversationalRetrievalChain handles retrieval, prompting, and generation',
          'Prompt engineering shapes how the LLM uses retrieved context',
          'Chat history enables multi-turn conversations'
        ]}
      ],
      quiz: [
        {
          question: 'What does k=4 in the retriever mean?',
          options: ['4 tokens per chunk', 'Retrieve 4 most relevant chunks', '4 conversation turns', '4 second timeout'],
          correct: 1,
          explanation: 'The k parameter controls how many document chunks are retrieved for each query.'
        },
        {
          question: 'Why include chat_history in the prompt?',
          options: ['Debugging', 'Enable multi-turn conversations', 'Reduce latency', 'Smaller context'],
          correct: 1,
          explanation: 'Chat history allows the model to understand references to previous messages and maintain conversational context.'
        }
      ]
    },
    {
      id: 'deployment-architecture',
      title: 'Split Deployment Architecture',
      duration: '10 min read',
      concepts: ['FastAPI', 'Docker', 'HuggingFace Spaces', 'Vercel'],
      content: [
        { type: 'heading', text: 'Why Split Backend and Frontend?' },
        { type: 'paragraph', text: 'The RAG chatbot uses a split deployment: Python backend on HuggingFace Spaces, React frontend on Vercel. This leverages each platform\'s strengths—HuggingFace for ML workloads, Vercel for static/React apps.' },

        { type: 'heading', text: 'FastAPI Backend' },
        { type: 'paragraph', text: 'The backend exposes the RAG pipeline through REST endpoints. FastAPI provides automatic OpenAPI docs, async support, and Pydantic validation.' },
        { type: 'code', language: 'python', filename: 'app.py', fromproject: 'RAG-chatbot',
          code: `from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ragchatbot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class chatrequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: chatrequest):
    response = chatbot.chat(request.message)
    return {"response": response}` },

        { type: 'heading', text: 'Docker Deployment' },
        { type: 'paragraph', text: 'HuggingFace Spaces runs Docker containers. The Dockerfile installs dependencies, copies code, sets permissions, and runs uvicorn.' },
        { type: 'code', language: 'dockerfile', filename: 'Dockerfile', fromproject: 'RAG-chatbot',
          code: `FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

RUN useradd -m -u 1000 user
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R user:user /app

USER user
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]` },

        { type: 'callout', variant: 'tip', text: 'HuggingFace Spaces requires the app to run on port 7860 and needs proper file permissions for non-root users.' },

        { type: 'keypoints', points: [
          'Split architecture leverages platform-specific strengths',
          'FastAPI backend handles RAG logic and LLM calls',
          'Docker containerizes the backend for HuggingFace Spaces',
          'CORS middleware enables cross-origin frontend requests'
        ]}
      ],
      quiz: [
        {
          question: 'Why use Docker for the backend?',
          options: ['Faster inference', 'Consistent environment across deployment', 'Smaller model size', 'Better prompts'],
          correct: 1,
          explanation: 'Docker ensures the same environment locally and on HuggingFace Spaces, avoiding "works on my machine" issues.'
        }
      ]
    }
  ]
}
