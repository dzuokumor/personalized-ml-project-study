export const mlopscurriculum = {
  id: 'mlops-curriculum',
  title: 'MLOps & Deployment',
  description: 'Deploy and maintain ML systems in production',
  category: 'Production',
  difficulty: 'Intermediate',
  duration: '5 hours',
  lessons: [
    {
      id: 'model-serialization',
      title: 'Model Serialization',
      duration: '50 min',
      concepts: ['pickle', 'joblib', 'ONNX', 'model formats', 'versioning'],
      content: [
        {
          type: 'heading',
          text: 'Saving and Loading Models'
        },
        {
          type: 'text',
          text: 'Model serialization saves trained models to disk for later use—deployment, sharing, or continued training. Different formats have different trade-offs in terms of portability, size, and framework compatibility.'
        },
        {
          type: 'subheading',
          text: 'Common Serialization Formats'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Model Serialization Formats</text>
            <rect x="30" y="50" width="100" height="90" fill="#e0e7ff" stroke="#6366f1" rx="6"/>
            <text x="80" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#4f46e5">Pickle/Joblib</text>
            <text x="80" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Python native</text>
            <text x="80" y="110" text-anchor="middle" font-size="8" fill="#64748b">Easy, but</text>
            <text x="80" y="125" text-anchor="middle" font-size="8" fill="#64748b">Python-only</text>
            <rect x="150" y="50" width="100" height="90" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="200" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">ONNX</text>
            <text x="200" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Cross-platform</text>
            <text x="200" y="110" text-anchor="middle" font-size="8" fill="#64748b">Portable,</text>
            <text x="200" y="125" text-anchor="middle" font-size="8" fill="#64748b">optimizable</text>
            <rect x="270" y="50" width="100" height="90" fill="#fef3c7" stroke="#f59e0b" rx="6"/>
            <text x="320" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">SavedModel</text>
            <text x="320" y="95" text-anchor="middle" font-size="9" fill="#1e293b">TensorFlow</text>
            <text x="320" y="110" text-anchor="middle" font-size="8" fill="#64748b">Full graph,</text>
            <text x="320" y="125" text-anchor="middle" font-size="8" fill="#64748b">TF Serving</text>
            <rect x="390" y="50" width="100" height="90" fill="#dbeafe" stroke="#3b82f6" rx="6"/>
            <text x="440" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#1d4ed8">TorchScript</text>
            <text x="440" y="95" text-anchor="middle" font-size="9" fill="#1e293b">PyTorch</text>
            <text x="440" y="110" text-anchor="middle" font-size="8" fill="#64748b">JIT compiled,</text>
            <text x="440" y="125" text-anchor="middle" font-size="8" fill="#64748b">C++ deployable</text>
          </svg>`
        },
        {
          type: 'code',
          language: 'python',
          code: `import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print(f"Predictions match: {(model.predict(X[:5]) == loaded_model.predict(X[:5])).all()}")`
        },
        {
          type: 'subheading',
          text: 'PyTorch Model Saving'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()

torch.save(model.state_dict(), 'model_weights.pth')
model_loaded = SimpleNet()
model_loaded.load_state_dict(torch.load('model_weights.pth'))

torch.save(model, 'model_full.pth')
model_full = torch.load('model_full.pth')

scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')
loaded_scripted = torch.jit.load('model_scripted.pt')`
        },
        {
          type: 'subheading',
          text: 'ONNX for Cross-Platform Deployment'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
import onnx
import onnxruntime as ort
import numpy as np

model = SimpleNet()
model.eval()

dummy_input = torch.randn(1, 784)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

session = ort.InferenceSession("model.onnx")
input_data = np.random.randn(1, 784).astype(np.float32)
outputs = session.run(None, {'input': input_data})
print(f"ONNX output shape: {outputs[0].shape}")`
        },
        {
          type: 'table',
          headers: ['Format', 'Framework', 'Use Case', 'Pros'],
          rows: [
            ['pickle/joblib', 'Scikit-learn', 'Quick experiments', 'Simple, preserves all attributes'],
            ['state_dict', 'PyTorch', 'Training checkpoints', 'Flexible, architecture-independent'],
            ['TorchScript', 'PyTorch', 'Production deployment', 'No Python needed, optimized'],
            ['SavedModel', 'TensorFlow', 'TF Serving', 'Full graph, signatures'],
            ['ONNX', 'Cross-platform', 'Multi-framework deploy', 'Portable, optimization tools']
          ]
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'Security Warning',
          text: 'Pickle can execute arbitrary code when loading. Never load pickled files from untrusted sources. For sharing models, prefer safer formats like ONNX or SavedModel.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why is ONNX useful for model deployment?',
          options: [
            'It is the fastest format',
            'It enables cross-platform deployment regardless of training framework',
            'It automatically improves model accuracy',
            'It only works with Python'
          ],
          correct: 1,
          explanation: 'ONNX (Open Neural Network Exchange) provides a framework-agnostic format. A model trained in PyTorch can be converted to ONNX and deployed using TensorFlow, ONNX Runtime, or other runtimes.'
        }
      ]
    },
    {
      id: 'ml-apis',
      title: 'Building ML APIs',
      duration: '60 min',
      concepts: ['REST API', 'FastAPI', 'Flask', 'request handling', 'input validation'],
      content: [
        {
          type: 'heading',
          text: 'Serving Models via APIs'
        },
        {
          type: 'text',
          text: 'APIs allow other applications to use your ML model over HTTP. FastAPI and Flask are popular Python frameworks. FastAPI is newer, faster, and has better async support and automatic documentation.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowAPI" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">ML API Architecture</text>
            <rect x="30" y="60" width="80" height="60" fill="#e0e7ff" stroke="#6366f1" rx="6"/>
            <text x="70" y="95" text-anchor="middle" font-size="10" fill="#4f46e5">Client</text>
            <rect x="170" y="60" width="160" height="60" fill="#dcfce7" stroke="#22c55e" stroke-width="2" rx="6"/>
            <text x="250" y="85" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">FastAPI Server</text>
            <text x="250" y="105" text-anchor="middle" font-size="9" fill="#1e293b">/predict endpoint</text>
            <rect x="390" y="60" width="80" height="60" fill="#fef3c7" stroke="#f59e0b" rx="6"/>
            <text x="430" y="95" text-anchor="middle" font-size="10" fill="#92400e">ML Model</text>
            <line x1="110" y1="80" x2="170" y2="80" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowAPI)"/>
            <text x="140" y="72" font-size="8" fill="#64748b">JSON</text>
            <line x1="330" y1="90" x2="390" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowAPI)"/>
            <line x1="170" y1="100" x2="110" y2="100" stroke="#22c55e" stroke-width="2" marker-end="url(#arrowAPI)"/>
            <text x="140" y="112" font-size="8" fill="#64748b">Result</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'FastAPI ML Service'
        },
        {
          type: 'code',
          language: 'python',
          code: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List

app = FastAPI(title="ML Prediction API", version="1.0")

model = joblib.load("model.joblib")

class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=20, max_items=20)

    class Config:
        schema_extra = {
            "example": {
                "features": [0.1] * 20
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max()
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}`
        },
        {
          type: 'subheading',
          text: 'Batch Predictions'
        },
        {
          type: 'code',
          language: 'python',
          code: `from fastapi import FastAPI, BackgroundTasks
from typing import List
import asyncio

class BatchInput(BaseModel):
    samples: List[List[float]]

class BatchOutput(BaseModel):
    predictions: List[int]
    job_id: str = None

predictions_cache = {}

@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(input_data: BatchInput):
    features = np.array(input_data.samples)
    predictions = model.predict(features).tolist()
    return BatchOutput(predictions=predictions)

async def process_batch(job_id: str, features: np.ndarray):
    await asyncio.sleep(0.1)
    predictions = model.predict(features).tolist()
    predictions_cache[job_id] = predictions

@app.post("/predict/async")
async def predict_async(input_data: BatchInput, background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())
    features = np.array(input_data.samples)
    background_tasks.add_task(process_batch, job_id, features)
    return {"job_id": job_id, "status": "processing"}

@app.get("/predict/result/{job_id}")
async def get_result(job_id: str):
    if job_id in predictions_cache:
        return {"predictions": predictions_cache[job_id], "status": "complete"}
    return {"status": "processing"}`
        },
        {
          type: 'subheading',
          text: 'Testing Your API'
        },
        {
          type: 'code',
          language: 'python',
          code: `import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [0.5] * 20}
)
print(response.json())

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"samples": [[0.5] * 20, [0.3] * 20, [0.7] * 20]}
)
print(response.json())`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'API Best Practices',
          text: 'Always include input validation, health endpoints, proper error handling, request logging, and rate limiting. Document your API using OpenAPI/Swagger (automatic with FastAPI at /docs).'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What advantage does FastAPI have over Flask for ML APIs?',
          options: [
            'Flask is no longer maintained',
            'FastAPI has automatic validation, async support, and auto-generated docs',
            'FastAPI is the only Python web framework',
            'Flask cannot serve ML models'
          ],
          correct: 1,
          explanation: 'FastAPI provides automatic request validation via Pydantic, native async/await support for better concurrency, and automatic OpenAPI documentation generation at /docs.'
        }
      ]
    },
    {
      id: 'containerization',
      title: 'Containerization with Docker',
      duration: '60 min',
      concepts: ['Docker', 'Dockerfile', 'images', 'containers', 'multi-stage builds'],
      content: [
        {
          type: 'heading',
          text: 'Packaging Models in Containers'
        },
        {
          type: 'text',
          text: 'Docker containers package your model, code, and dependencies into a portable unit. This ensures your model runs the same everywhere—your laptop, a server, or cloud. No more "works on my machine" problems.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Container vs Traditional Deployment</text>
            <rect x="30" y="50" width="180" height="110" fill="#fecaca" stroke="#ef4444" rx="6"/>
            <text x="120" y="75" text-anchor="middle" font-size="10" font-weight="bold" fill="#dc2626">Traditional</text>
            <rect x="45" y="90" width="70" height="25" fill="#fee2e2" stroke="#fca5a5" rx="3"/>
            <text x="80" y="107" text-anchor="middle" font-size="8" fill="#1e293b">App</text>
            <rect x="125" y="90" width="70" height="25" fill="#fee2e2" stroke="#fca5a5" rx="3"/>
            <text x="160" y="107" text-anchor="middle" font-size="8" fill="#1e293b">Deps</text>
            <rect x="45" y="125" width="150" height="25" fill="#fee2e2" stroke="#fca5a5" rx="3"/>
            <text x="120" y="142" text-anchor="middle" font-size="8" fill="#1e293b">Host OS + Runtime</text>
            <rect x="290" y="50" width="180" height="110" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="380" y="75" text-anchor="middle" font-size="10" font-weight="bold" fill="#15803d">Containerized</text>
            <rect x="305" y="90" width="70" height="55" fill="#bbf7d0" stroke="#86efac" rx="3"/>
            <text x="340" y="110" text-anchor="middle" font-size="8" fill="#1e293b">Container 1</text>
            <text x="340" y="125" text-anchor="middle" font-size="7" fill="#64748b">App+Deps</text>
            <rect x="385" y="90" width="70" height="55" fill="#bbf7d0" stroke="#86efac" rx="3"/>
            <text x="420" y="110" text-anchor="middle" font-size="8" fill="#1e293b">Container 2</text>
            <text x="420" y="125" text-anchor="middle" font-size="7" fill="#64748b">App+Deps</text>
            <rect x="305" y="150" width="150" height="15" fill="#86efac" stroke="#22c55e" rx="2"/>
            <text x="380" y="161" text-anchor="middle" font-size="7" fill="#1e293b">Docker Engine</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Dockerfile for ML API'
        },
        {
          type: 'code',
          language: 'dockerfile',
          code: `FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.joblib .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`
        },
        {
          type: 'subheading',
          text: 'Multi-Stage Build for Smaller Images'
        },
        {
          type: 'code',
          language: 'dockerfile',
          code: `FROM python:3.10 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY model.joblib .
COPY app.py .

RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`
        },
        {
          type: 'subheading',
          text: 'Building and Running'
        },
        {
          type: 'code',
          language: 'bash',
          code: `# Build the image
docker build -t ml-api:v1 .

# Run the container
docker run -d -p 8000:8000 --name ml-service ml-api:v1

# Check logs
docker logs ml-service

# Test the API
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.5, 0.3, 0.2, ...]}'

# Stop and remove
docker stop ml-service
docker rm ml-service`
        },
        {
          type: 'subheading',
          text: 'Docker Compose for Multi-Service'
        },
        {
          type: 'code',
          language: 'yaml',
          code: `version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.joblib
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Image Size Tips',
          text: 'Use slim/alpine base images, multi-stage builds, and .dockerignore. Remove cache after pip install. A typical ML image might be 1-2GB; with optimization, you can often get it under 500MB.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the main benefit of containerizing an ML model?',
          options: [
            'Faster inference',
            'Consistent environment across development, testing, and production',
            'Smaller model size',
            'Better model accuracy'
          ],
          correct: 1,
          explanation: 'Containers package the model with all dependencies, ensuring identical behavior everywhere. This eliminates "works on my machine" issues caused by different Python versions, library versions, or system configurations.'
        }
      ]
    },
    {
      id: 'cloud-deployment',
      title: 'Cloud Deployment',
      duration: '55 min',
      concepts: ['AWS', 'GCP', 'Azure', 'serverless', 'Kubernetes'],
      content: [
        {
          type: 'heading',
          text: 'Deploying to the Cloud'
        },
        {
          type: 'text',
          text: 'Cloud platforms provide scalable infrastructure for ML models. Options range from simple serverless functions to managed ML services to full Kubernetes clusters. Choose based on scale, cost, and operational complexity.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Cloud Deployment Options</text>
            <rect x="30" y="50" width="130" height="120" fill="#e0e7ff" stroke="#6366f1" rx="6"/>
            <text x="95" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#4f46e5">Serverless</text>
            <text x="95" y="95" text-anchor="middle" font-size="9" fill="#1e293b">Lambda, Cloud Run</text>
            <text x="95" y="115" text-anchor="middle" font-size="8" fill="#22c55e">✓ Simple</text>
            <text x="95" y="130" text-anchor="middle" font-size="8" fill="#22c55e">✓ Auto-scale</text>
            <text x="95" y="145" text-anchor="middle" font-size="8" fill="#ef4444">✗ Cold starts</text>
            <rect x="180" y="50" width="130" height="120" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="245" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">Managed ML</text>
            <text x="245" y="95" text-anchor="middle" font-size="9" fill="#1e293b">SageMaker, Vertex AI</text>
            <text x="245" y="115" text-anchor="middle" font-size="8" fill="#22c55e">✓ ML-optimized</text>
            <text x="245" y="130" text-anchor="middle" font-size="8" fill="#22c55e">✓ Monitoring</text>
            <text x="245" y="145" text-anchor="middle" font-size="8" fill="#ef4444">✗ Vendor lock-in</text>
            <rect x="330" y="50" width="140" height="120" fill="#fef3c7" stroke="#f59e0b" rx="6"/>
            <text x="400" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">Kubernetes</text>
            <text x="400" y="95" text-anchor="middle" font-size="9" fill="#1e293b">EKS, GKE, AKS</text>
            <text x="400" y="115" text-anchor="middle" font-size="8" fill="#22c55e">✓ Full control</text>
            <text x="400" y="130" text-anchor="middle" font-size="8" fill="#22c55e">✓ Portable</text>
            <text x="400" y="145" text-anchor="middle" font-size="8" fill="#ef4444">✗ Complex</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'AWS Lambda Deployment'
        },
        {
          type: 'code',
          language: 'python',
          code: `import json
import boto3
import joblib
import numpy as np

model = None

def load_model():
    global model
    if model is None:
        s3 = boto3.client('s3')
        s3.download_file('my-bucket', 'model.joblib', '/tmp/model.joblib')
        model = joblib.load('/tmp/model.joblib')
    return model

def lambda_handler(event, context):
    model = load_model()

    body = json.loads(event['body'])
    features = np.array(body['features']).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': int(prediction),
            'probability': float(probability)
        })
    }`
        },
        {
          type: 'subheading',
          text: 'Google Cloud Run Deployment'
        },
        {
          type: 'code',
          language: 'bash',
          code: `# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-api

# Deploy to Cloud Run
gcloud run deploy ml-api \\
  --image gcr.io/PROJECT_ID/ml-api \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated \\
  --memory 2Gi \\
  --cpu 2 \\
  --min-instances 1 \\
  --max-instances 10`
        },
        {
          type: 'subheading',
          text: 'Kubernetes Deployment'
        },
        {
          type: 'code',
          language: 'yaml',
          code: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer`
        },
        {
          type: 'table',
          headers: ['Platform', 'Best For', 'Scaling', 'Cost Model'],
          rows: [
            ['Lambda/Cloud Functions', 'Low traffic, sporadic', 'Auto (0 to 1000s)', 'Per request'],
            ['Cloud Run/App Runner', 'Medium traffic, containers', 'Auto (0 to 100s)', 'Per request + time'],
            ['EC2/Compute Engine', 'Steady traffic', 'Manual/Auto', 'Per hour'],
            ['Kubernetes', 'High traffic, complex', 'Auto (HPA)', 'Per node-hour']
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is a key advantage of serverless deployment for ML models?',
          options: [
            'Lowest latency',
            'Automatic scaling to zero when not in use, reducing costs',
            'Best for GPU inference',
            'Most control over infrastructure'
          ],
          correct: 1,
          explanation: 'Serverless platforms automatically scale from zero to handle any load, and you only pay when requests are processed. This is cost-effective for variable or low-traffic workloads.'
        }
      ]
    },
    {
      id: 'monitoring',
      title: 'Monitoring and Maintenance',
      duration: '55 min',
      concepts: ['model drift', 'data drift', 'metrics', 'alerting', 'retraining'],
      content: [
        {
          type: 'heading',
          text: 'Keeping Models Healthy in Production'
        },
        {
          type: 'text',
          text: 'Deployment is not the end—it is the beginning of model maintenance. Models degrade over time as data distributions shift. Monitoring detects problems before they impact users; good alerting enables rapid response.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Model Monitoring Loop</text>
            <ellipse cx="100" cy="100" rx="60" ry="40" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
            <text x="100" y="105" text-anchor="middle" font-size="10" fill="#4f46e5">Production</text>
            <ellipse cx="250" cy="100" rx="60" ry="40" fill="#dcfce7" stroke="#22c55e" stroke-width="2"/>
            <text x="250" y="105" text-anchor="middle" font-size="10" fill="#15803d">Monitor</text>
            <ellipse cx="400" cy="100" rx="60" ry="40" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
            <text x="400" y="105" text-anchor="middle" font-size="10" fill="#92400e">Retrain</text>
            <path d="M 160 100 L 190 100" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowAPI)"/>
            <path d="M 310 100 L 340 100" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowAPI)"/>
            <path d="M 400 60 Q 400 30 250 30 Q 100 30 100 60" stroke="#6366f1" stroke-width="2" fill="none" marker-end="url(#arrowAPI)"/>
            <text x="250" y="160" text-anchor="middle" font-size="9" fill="#64748b">Continuous feedback loop</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Types of Drift'
        },
        {
          type: 'table',
          headers: ['Type', 'Definition', 'Detection'],
          rows: [
            ['Data Drift', 'Input distribution changes', 'Statistical tests on features'],
            ['Concept Drift', 'Relationship between X and y changes', 'Monitor prediction accuracy'],
            ['Model Decay', 'Performance degrades over time', 'Track metrics over time'],
            ['Label Drift', 'Target distribution changes', 'Monitor class frequencies']
          ]
        },
        {
          type: 'subheading',
          text: 'Implementing Monitoring'
        },
        {
          type: 'code',
          language: 'python',
          code: `import numpy as np
from scipy import stats
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self, reference_data):
        self.reference_mean = reference_data.mean(axis=0)
        self.reference_std = reference_data.std(axis=0)
        self.predictions = []
        self.actuals = []
        self.timestamps = []

    def log_prediction(self, features, prediction, actual=None):
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)
        self.timestamps.append(datetime.now())

        drift_score = self.detect_data_drift(features)
        if drift_score > 0.05:
            logging.warning(f"Data drift detected: {drift_score:.4f}")

    def detect_data_drift(self, features):
        z_scores = (features - self.reference_mean) / (self.reference_std + 1e-8)
        return np.abs(z_scores).mean()

    def calculate_accuracy(self, window=1000):
        if len(self.actuals) < window:
            return None
        recent_preds = self.predictions[-window:]
        recent_actuals = self.actuals[-window:]
        return np.mean(np.array(recent_preds) == np.array(recent_actuals))

    def get_prediction_distribution(self):
        if not self.predictions:
            return {}
        unique, counts = np.unique(self.predictions, return_counts=True)
        return dict(zip(unique.tolist(), (counts / len(self.predictions)).tolist()))`
        },
        {
          type: 'subheading',
          text: 'Prometheus Metrics'
        },
        {
          type: 'code',
          language: 'python',
          code: `from prometheus_client import Counter, Histogram, Gauge, start_http_server

prediction_counter = Counter(
    'ml_predictions_total',
    'Total predictions made',
    ['model_version', 'class']
)

latency_histogram = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

accuracy_gauge = Gauge(
    'ml_model_accuracy',
    'Rolling accuracy of the model'
)

drift_gauge = Gauge(
    'ml_data_drift_score',
    'Current data drift score'
)

import time

@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()

    prediction = model.predict([input_data.features])[0]

    latency = time.time() - start_time
    latency_histogram.observe(latency)
    prediction_counter.labels(model_version='v1', class_=str(prediction)).inc()

    return {"prediction": int(prediction)}`
        },
        {
          type: 'subheading',
          text: 'Alerting Strategy'
        },
        {
          type: 'code',
          language: 'yaml',
          code: `# Prometheus alerting rules
groups:
- name: ml-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(ml_prediction_errors_total[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High prediction error rate"

  - alert: ModelDrift
    expr: ml_data_drift_score > 0.1
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Data drift detected"

  - alert: HighLatency
    expr: histogram_quantile(0.95, ml_prediction_latency_seconds_bucket) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "95th percentile latency above 1 second"`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Retraining Triggers',
          text: 'Consider retraining when: accuracy drops below threshold, drift score exceeds threshold, on a schedule (weekly/monthly), or when significant new data is available. Automate the retraining pipeline for faster response.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is data drift in the context of ML monitoring?',
          options: [
            'Model weights changing over time',
            'Changes in the input data distribution compared to training data',
            'Slow inference speed',
            'Memory leaks in the application'
          ],
          correct: 1,
          explanation: 'Data drift occurs when the statistical properties of input data change over time. A model trained on one distribution may perform poorly when inputs shift to a different distribution.'
        }
      ]
    }
  ]
}
