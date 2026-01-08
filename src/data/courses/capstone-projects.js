export const capstoneprojects = {
  id: 'capstone-projects',
  title: 'Capstone Projects',
  description: 'Build portfolio-ready end-to-end ML projects',
  category: 'Projects',
  difficulty: 'Advanced',
  duration: '4 hours',
  lessons: [
    {
      id: 'classification-project',
      title: 'End-to-End Classification Project',
      duration: '60 min',
      concepts: ['project structure', 'data pipeline', 'model selection', 'deployment'],
      content: [
        {
          type: 'heading',
          text: 'Building a Complete Classification System'
        },
        {
          type: 'text',
          text: 'This capstone guides you through building a production-ready classification system from scratch. You will handle real data, make architectural decisions, and deploy a working API. This is the kind of project that demonstrates real ML engineering skills.'
        },
        {
          type: 'subheading',
          text: 'Project: Customer Churn Prediction'
        },
        {
          type: 'text',
          text: 'We will build a system that predicts which customers are likely to leave a service. This is a common business problem with real impact—preventing churn directly increases revenue.'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowCap" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1"/>
              </marker>
            </defs>
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">End-to-End ML Pipeline</text>
            <rect x="20" y="60" width="70" height="50" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="55" y="90" text-anchor="middle" font-size="9" fill="#4f46e5">Raw Data</text>
            <rect x="110" y="60" width="70" height="50" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="145" y="85" text-anchor="middle" font-size="9" fill="#1d4ed8">Clean &</text>
            <text x="145" y="97" text-anchor="middle" font-size="9" fill="#1d4ed8">Process</text>
            <rect x="200" y="60" width="70" height="50" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="235" y="85" text-anchor="middle" font-size="9" fill="#15803d">Feature</text>
            <text x="235" y="97" text-anchor="middle" font-size="9" fill="#15803d">Engineer</text>
            <rect x="290" y="60" width="70" height="50" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="325" y="85" text-anchor="middle" font-size="9" fill="#92400e">Train &</text>
            <text x="325" y="97" text-anchor="middle" font-size="9" fill="#92400e">Evaluate</text>
            <rect x="380" y="60" width="70" height="50" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="415" y="90" text-anchor="middle" font-size="9" fill="#7c3aed">Deploy</text>
            <line x1="90" y1="85" x2="110" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="180" y1="85" x2="200" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="270" y1="85" x2="290" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="360" y1="85" x2="380" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <rect x="110" y="130" width="250" height="40" fill="#f8fafc" stroke="#e2e8f0" rx="4"/>
            <text x="235" y="155" text-anchor="middle" font-size="10" fill="#64748b">Version control + Experiment tracking + CI/CD</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Step 1: Project Setup'
        },
        {
          type: 'code',
          language: 'bash',
          code: `# Project structure
churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── api/
│       ├── __init__.py
│       └── app.py
├── tests/
├── config/
│   └── config.yaml
├── requirements.txt
├── Dockerfile
└── README.md`
        },
        {
          type: 'subheading',
          text: 'Step 2: Data Pipeline'
        },
        {
          type: 'code',
          language: 'python',
          code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    def preprocess(self, df, fit=True):
        df = df.copy()
        df = df.dropna()

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col == 'churn':
                continue
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        return df

    def split_features_target(self, df, target='churn'):
        X = df.drop(columns=[target])
        y = df[target].map({'Yes': 1, 'No': 0})
        self.feature_names = X.columns.tolist()
        return X, y

    def scale_features(self, X, fit=True):
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

pipeline = DataPipeline()
df = pipeline.load_data('data/raw/customer_churn.csv')
df = pipeline.preprocess(df, fit=True)
X, y = pipeline.split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train_scaled = pipeline.scale_features(X_train, fit=True)
X_test_scaled = pipeline.scale_features(X_test, fit=False)`
        },
        {
          type: 'subheading',
          text: 'Step 3: Model Training and Selection'
        },
        {
          type: 'code',
          language: 'python',
          code: `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model': model,
            'auc': roc_auc_score(y_test, y_prob),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"{name}: AUC = {results[name]['auc']:.4f}")

    best_model_name = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_model_name]['model']

    joblib.dump(best_model, 'models/best_model.joblib')
    joblib.dump(pipeline, 'models/pipeline.joblib')

    return best_model, results

best_model, results = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)`
        },
        {
          type: 'subheading',
          text: 'Step 4: API Deployment'
        },
        {
          type: 'code',
          language: 'python',
          code: `from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

model = joblib.load('models/best_model.joblib')
pipeline = joblib.load('models/pipeline.joblib')

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    payment_method: str

class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])
    df = pipeline.preprocess(df, fit=False)
    X = pipeline.scale_features(df, fit=False)

    prob = model.predict_proba(X)[0, 1]
    will_churn = prob > 0.5
    risk_level = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"

    return PredictionResponse(
        churn_probability=float(prob),
        will_churn=bool(will_churn),
        risk_level=risk_level
    )`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Portfolio Tip',
          text: 'Document your decisions: why you chose certain models, how you handled imbalanced data, what tradeoffs you made. This shows ML engineering thinking, not just code execution.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why is it important to save the preprocessing pipeline along with the model?',
          options: [
            'To reduce file size',
            'Because inference data must be transformed the same way as training data',
            'To speed up predictions',
            'It is not important'
          ],
          correct: 1,
          explanation: 'The model expects input in the same format it was trained on. The preprocessing pipeline (scalers, encoders) must be saved and applied identically during inference to ensure consistent predictions.'
        }
      ]
    },
    {
      id: 'nlp-project',
      title: 'End-to-End NLP Project',
      duration: '60 min',
      concepts: ['text classification', 'transformers', 'fine-tuning', 'evaluation'],
      content: [
        {
          type: 'heading',
          text: 'Building a Text Classification System'
        },
        {
          type: 'text',
          text: 'This project takes you through building a sentiment analysis system using modern NLP techniques. You will fine-tune a transformer model and deploy it as an API.'
        },
        {
          type: 'subheading',
          text: 'Project: Product Review Sentiment Analyzer'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">NLP Pipeline Architecture</text>
            <rect x="30" y="50" width="90" height="50" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="75" y="80" text-anchor="middle" font-size="9" fill="#4f46e5">Raw Reviews</text>
            <rect x="140" y="50" width="90" height="50" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="185" y="75" text-anchor="middle" font-size="9" fill="#1d4ed8">Tokenizer</text>
            <text x="185" y="88" text-anchor="middle" font-size="8" fill="#64748b">(BERT)</text>
            <rect x="250" y="50" width="90" height="50" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="295" y="75" text-anchor="middle" font-size="9" fill="#15803d">Transformer</text>
            <text x="295" y="88" text-anchor="middle" font-size="8" fill="#64748b">(DistilBERT)</text>
            <rect x="360" y="50" width="90" height="50" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="405" y="75" text-anchor="middle" font-size="9" fill="#92400e">Classifier</text>
            <text x="405" y="88" text-anchor="middle" font-size="8" fill="#64748b">Head</text>
            <line x1="120" y1="75" x2="140" y2="75" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="230" y1="75" x2="250" y2="75" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="340" y1="75" x2="360" y2="75" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <rect x="140" y="120" width="210" height="35" fill="#f3e8ff" stroke="#a855f7" rx="4"/>
            <text x="245" y="142" text-anchor="middle" font-size="10" fill="#7c3aed">Positive | Negative | Neutral</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Step 1: Data Preparation'
        },
        {
          type: 'code',
          language: 'python',
          code: `from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd

dataset = load_dataset("amazon_polarity", split="train[:50000]")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

print(f"Training samples: {len(tokenized_dataset['train'])}")
print(f"Test samples: {len(tokenized_dataset['test'])}")`
        },
        {
          type: 'subheading',
          text: 'Step 2: Model Fine-tuning'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")`
        },
        {
          type: 'subheading',
          text: 'Step 3: Inference Pipeline'
        },
        {
          type: 'code',
          language: 'python',
          code: `from transformers import pipeline
import torch

class SentimentAnalyzer:
    def __init__(self, model_path="./final_model"):
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        self.label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "positive"
        }

    def analyze(self, text):
        result = self.classifier(text)[0]
        return {
            "sentiment": self.label_map.get(result["label"], result["label"]),
            "confidence": result["score"]
        }

    def analyze_batch(self, texts):
        results = self.classifier(texts)
        return [
            {
                "text": text,
                "sentiment": self.label_map.get(r["label"], r["label"]),
                "confidence": r["score"]
            }
            for text, r in zip(texts, results)
        ]

analyzer = SentimentAnalyzer()

reviews = [
    "This product is amazing! Best purchase ever.",
    "Terrible quality. Complete waste of money.",
    "It's okay, nothing special but does the job."
]

for review in reviews:
    result = analyzer.analyze(review)
    print(f"'{review[:50]}...'")
    print(f"  → {result['sentiment']} ({result['confidence']:.2%})\\n")`
        },
        {
          type: 'subheading',
          text: 'Step 4: Deploy as API'
        },
        {
          type: 'code',
          language: 'python',
          code: `from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Sentiment Analysis API")

analyzer = SentimentAnalyzer()

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float

@app.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(input_data: TextInput):
    result = analyzer.analyze(input_data.text)
    return SentimentResult(**result)

@app.post("/analyze/batch")
async def analyze_batch(input_data: BatchInput):
    results = analyzer.analyze_batch(input_data.texts)
    return results`
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'GPU Optimization',
          text: 'For production, consider ONNX export or TorchScript for faster inference. Batch requests together when possible. Use model quantization (INT8) for CPU deployment.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why use a pre-trained transformer like DistilBERT instead of training from scratch?',
          options: [
            'Pre-trained models are smaller',
            'Transfer learning from massive text corpora gives better results with less data',
            'Pre-trained models do not need fine-tuning',
            'They are faster to train from scratch'
          ],
          correct: 1,
          explanation: 'Pre-trained transformers have learned rich language representations from billions of words. Fine-tuning adapts this knowledge to your specific task with much less data than training from scratch.'
        }
      ]
    },
    {
      id: 'cv-project',
      title: 'End-to-End Computer Vision Project',
      duration: '60 min',
      concepts: ['image classification', 'transfer learning', 'data augmentation', 'model optimization'],
      content: [
        {
          type: 'heading',
          text: 'Building an Image Classification System'
        },
        {
          type: 'text',
          text: 'This project walks through building a production image classifier using transfer learning. You will handle real images, augment data, fine-tune a pre-trained model, and deploy with optimized inference.'
        },
        {
          type: 'subheading',
          text: 'Project: Medical Image Classification'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Transfer Learning Pipeline</text>
            <rect x="30" y="60" width="80" height="70" fill="#e0e7ff" stroke="#6366f1" rx="4"/>
            <text x="70" y="90" text-anchor="middle" font-size="9" fill="#4f46e5">Input</text>
            <text x="70" y="105" text-anchor="middle" font-size="9" fill="#4f46e5">Image</text>
            <text x="70" y="120" text-anchor="middle" font-size="8" fill="#64748b">224×224</text>
            <rect x="130" y="60" width="100" height="70" fill="#dbeafe" stroke="#3b82f6" rx="4"/>
            <text x="180" y="85" text-anchor="middle" font-size="9" fill="#1d4ed8">Pre-trained</text>
            <text x="180" y="100" text-anchor="middle" font-size="9" fill="#1d4ed8">ResNet50</text>
            <text x="180" y="115" text-anchor="middle" font-size="8" fill="#64748b">(frozen)</text>
            <rect x="250" y="60" width="100" height="70" fill="#dcfce7" stroke="#22c55e" rx="4"/>
            <text x="300" y="85" text-anchor="middle" font-size="9" fill="#15803d">Custom</text>
            <text x="300" y="100" text-anchor="middle" font-size="9" fill="#15803d">Classifier</text>
            <text x="300" y="115" text-anchor="middle" font-size="8" fill="#64748b">(trainable)</text>
            <rect x="370" y="60" width="100" height="70" fill="#fef3c7" stroke="#f59e0b" rx="4"/>
            <text x="420" y="95" text-anchor="middle" font-size="9" fill="#92400e">Prediction</text>
            <line x1="110" y1="95" x2="130" y2="95" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="230" y1="95" x2="250" y2="95" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
            <line x1="350" y1="95" x2="370" y2="95" stroke="#6366f1" stroke-width="2" marker-end="url(#arrowCap)"/>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'Step 1: Data Loading and Augmentation'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
val_dataset = datasets.ImageFolder('data/val', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

class_names = train_dataset.classes
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}")`
        },
        {
          type: 'subheading',
          text: 'Step 2: Transfer Learning Model'
        },
        {
          type: 'code',
          language: 'python',
          code: `import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers=2):
        layers = list(self.backbone.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier(num_classes=len(class_names)).to(device)`
        },
        {
          type: 'subheading',
          text: 'Step 3: Training Loop'
        },
        {
          type: 'code',
          language: 'python',
          code: `from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return model

model = train_model(model, train_loader, val_loader, epochs=10)`
        },
        {
          type: 'subheading',
          text: 'Step 4: Inference API'
        },
        {
          type: 'code',
          language: 'python',
          code: `from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI(title="Image Classification API")

model = ImageClassifier(num_classes=3)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

class_names = ['normal', 'pneumonia', 'covid']

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    return {
        "prediction": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {
            name: float(probs[i]) for i, name in enumerate(class_names)
        }
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    return result`
        },
        {
          type: 'callout',
          variant: 'warning',
          title: 'Medical AI Disclaimer',
          text: 'Medical image classification requires rigorous validation, regulatory approval, and should assist (not replace) medical professionals. Always include appropriate disclaimers and validation with domain experts.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why freeze the backbone when starting transfer learning?',
          options: [
            'To make the model smaller',
            'To preserve learned features and only train the new classifier head initially',
            'To speed up inference',
            'Because pre-trained weights cannot be changed'
          ],
          correct: 1,
          explanation: 'Freezing the backbone preserves the valuable feature representations learned from ImageNet. We first train only the new classification head, then optionally fine-tune later layers if needed.'
        }
      ]
    },
    {
      id: 'portfolio-building',
      title: 'Building Your Portfolio',
      duration: '60 min',
      concepts: ['documentation', 'GitHub', 'presentation', 'interviews'],
      content: [
        {
          type: 'heading',
          text: 'Presenting Your ML Work'
        },
        {
          type: 'text',
          text: 'Technical skills alone do not land jobs—you need to communicate your work effectively. A strong portfolio demonstrates not just what you built, but how you think about problems. This lesson covers how to document, present, and discuss your projects.'
        },
        {
          type: 'subheading',
          text: 'Project Documentation Structure'
        },
        {
          type: 'visualization',
          svg: `<svg viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
            <text x="250" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">Portfolio Project Anatomy</text>
            <rect x="30" y="50" width="200" height="130" fill="#e0e7ff" stroke="#6366f1" rx="6"/>
            <text x="130" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#4f46e5">README.md</text>
            <text x="45" y="95" font-size="9" fill="#1e293b">• Problem statement</text>
            <text x="45" y="110" font-size="9" fill="#1e293b">• Solution approach</text>
            <text x="45" y="125" font-size="9" fill="#1e293b">• Results & metrics</text>
            <text x="45" y="140" font-size="9" fill="#1e293b">• How to run</text>
            <text x="45" y="155" font-size="9" fill="#1e293b">• Architecture diagram</text>
            <text x="45" y="170" font-size="9" fill="#1e293b">• Demo/screenshots</text>
            <rect x="260" y="50" width="200" height="130" fill="#dcfce7" stroke="#22c55e" rx="6"/>
            <text x="360" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">Supporting Docs</text>
            <text x="275" y="95" font-size="9" fill="#1e293b">• notebooks/ - exploration</text>
            <text x="275" y="110" font-size="9" fill="#1e293b">• docs/ - deep dives</text>
            <text x="275" y="125" font-size="9" fill="#1e293b">• CONTRIBUTING.md</text>
            <text x="275" y="140" font-size="9" fill="#1e293b">• requirements.txt</text>
            <text x="275" y="155" font-size="9" fill="#1e293b">• Dockerfile</text>
            <text x="275" y="170" font-size="9" fill="#1e293b">• tests/</text>
          </svg>`
        },
        {
          type: 'subheading',
          text: 'README Template'
        },
        {
          type: 'code',
          language: 'markdown',
          code: `# Project Name

## Overview
One paragraph explaining what this project does and why it matters.

## Demo
![Demo GIF](demo.gif)

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| F1 Score | 0.93 |
| Inference Time | 45ms |

## Architecture
\`\`\`
[Input] → [Preprocessing] → [Model] → [Postprocessing] → [Output]
\`\`\`

## Key Decisions
1. **Model Choice**: Used DistilBERT because [reason]
2. **Data Handling**: Addressed class imbalance using [technique]
3. **Deployment**: Chose FastAPI for [reason]

## Quick Start
\`\`\`bash
git clone https://github.com/user/project
cd project
pip install -r requirements.txt
python src/api/app.py
\`\`\`

## Project Structure
\`\`\`
├── data/           # Data processing
├── models/         # Model definitions
├── notebooks/      # Exploration
├── src/            # Source code
└── tests/          # Unit tests
\`\`\`

## What I Learned
- Handling imbalanced datasets with SMOTE
- Deploying models with Docker
- Setting up CI/CD for ML projects`
        },
        {
          type: 'subheading',
          text: 'GitHub Best Practices'
        },
        {
          type: 'table',
          headers: ['Element', 'Why It Matters', 'Tips'],
          rows: [
            ['Commit history', 'Shows your process', 'Clear, atomic commits'],
            ['Branch strategy', 'Shows organization', 'feature/, fix/ branches'],
            ['Issues/PRs', 'Shows planning', 'Document decisions'],
            ['CI/CD', 'Shows professionalism', 'GitHub Actions for tests'],
            ['License', 'Shows awareness', 'MIT for portfolio projects']
          ]
        },
        {
          type: 'subheading',
          text: 'Talking About Your Projects'
        },
        {
          type: 'text',
          text: 'In interviews, use the STAR method adapted for ML projects:'
        },
        {
          type: 'text',
          text: '**Situation**: "The company needed to reduce customer churn, which was costing $2M annually."'
        },
        {
          type: 'text',
          text: '**Task**: "I was responsible for building a predictive model to identify at-risk customers."'
        },
        {
          type: 'text',
          text: '**Approach**: "I compared logistic regression, random forests, and gradient boosting. I chose XGBoost because it handled our imbalanced data well and provided feature importance for business insights."'
        },
        {
          type: 'text',
          text: '**Result**: "The model achieved 85% precision on high-risk customers, enabling targeted retention campaigns that reduced churn by 15%."'
        },
        {
          type: 'subheading',
          text: 'Common Interview Questions'
        },
        {
          type: 'code',
          language: 'text',
          code: `Questions about your projects:

1. "Walk me through your most challenging project."
   → Focus on technical challenges and how you solved them

2. "Why did you choose [model/approach]?"
   → Show you considered alternatives and made informed decisions

3. "What would you do differently?"
   → Shows self-reflection and learning mindset

4. "How did you validate your model?"
   → Demonstrate understanding of evaluation beyond accuracy

5. "How would you scale this to production?"
   → Shows systems thinking beyond just model building

Prepare concrete answers with metrics and specific examples.`
        },
        {
          type: 'subheading',
          text: 'Portfolio Checklist'
        },
        {
          type: 'text',
          text: 'Before sharing a project, ensure it has:'
        },
        {
          type: 'keypoints',
          points: [
            'Clear README with problem statement and results',
            'Working code that others can run',
            'Clean, documented codebase (no secrets, no notebooks with output)',
            'At least one visual (diagram, chart, or demo)',
            'Explanation of your decisions and tradeoffs',
            'Tests or validation of key components',
            'Professional commit history (no "fix stuff" commits)'
          ]
        },
        {
          type: 'callout',
          variant: 'info',
          title: 'Quality Over Quantity',
          text: '3 excellent projects beat 10 mediocre ones. Each project should demonstrate a different skill: one showing data engineering, one showing deep learning, one showing deployment. Depth matters more than breadth.'
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the most important element in a portfolio project README?',
          options: [
            'Code complexity',
            'Length of documentation',
            'Clear problem statement, approach, and measurable results',
            'Number of libraries used'
          ],
          correct: 2,
          explanation: 'Recruiters and hiring managers want to quickly understand: what problem you solved, how you approached it, and what impact you achieved. Clear results with metrics are more impressive than complex code.'
        }
      ]
    }
  ]
}
