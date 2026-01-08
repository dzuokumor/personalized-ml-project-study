export const coremlconcepts = {
  id: 'core-ml-concepts',
  title: 'Core ML Concepts',
  description: 'Understand the fundamental concepts that underpin all machine learning: types of learning, the ML pipeline, feature engineering, and the critical balance between underfitting and overfitting.',
  difficulty: 'beginner',
  estimatedhours: 12,
  lessons: [
    {
      id: 'what-is-ml',
      title: 'What is Machine Learning?',
      duration: '35 min',
      content: [
        {
          type: 'text',
          content: `Let's cut through the hype and understand what machine learning actually is, at its core.`
        },
        {
          type: 'heading',
          content: 'The Traditional Programming Paradigm'
        },
        {
          type: 'text',
          content: `In traditional programming, you write explicit rules:

**Input**: Email text
**Rules**: If email contains "winner" and "lottery" → spam
**Output**: Spam or not spam

The programmer explicitly codes every rule. This works when:
- Rules are simple and well-defined
- Edge cases are manageable
- The problem doesn't change over time

But what about recognizing faces? Writing rules for "this pixel pattern = human face" is nearly impossible.`
        },
        {
          type: 'heading',
          content: 'The Machine Learning Paradigm'
        },
        {
          type: 'text',
          content: `ML flips the paradigm:

**Input**: Email text + Labels (spam/not spam)
**Algorithm**: Learns patterns from data
**Output**: Rules (learned model)

Instead of coding rules, we provide examples and let the algorithm discover patterns. The "program" writes itself from data.

**Arthur Samuel's Definition (1959):**
"Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."

**Tom Mitchell's Definition (1997, more precise):**
"A computer program learns from experience E with respect to task T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."`
        },
        {
          type: 'heading',
          content: 'When to Use Machine Learning'
        },
        {
          type: 'text',
          content: `ML is the right choice when:

1. **Patterns exist** in the data but are complex/unknown
2. **Rules are hard to specify** explicitly (face recognition, speech)
3. **Data is available** in sufficient quantity
4. **The environment changes** (spam evolves, user preferences shift)

ML is NOT the right choice when:
- Simple rules work (e.g., "if age < 18, deny alcohol purchase")
- No data is available
- Explainability is critical and required by law
- The cost of errors is catastrophic and unacceptable`
        },
        {
          type: 'heading',
          content: 'The Core Components of ML'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Every ML system has these components:

# 1. DATA - The raw material
X = [[5.1, 3.5, 1.4, 0.2],   # Features: sepal length, width, petal length, width
     [4.9, 3.0, 1.4, 0.2],
     [7.0, 3.2, 4.7, 1.4]]
y = [0, 0, 1]                 # Labels: 0=setosa, 1=versicolor

# 2. MODEL - The learnable function
# f(X) = y_predicted
# Could be: linear model, decision tree, neural network, etc.

# 3. LOSS FUNCTION - Measures how wrong we are
# loss = sum((y_predicted - y_actual)^2) / n

# 4. OPTIMIZATION - Adjusts model to minimize loss
# Gradient descent, Adam, etc.

# In code:
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()      # Model architecture
model.fit(X, y)                   # Learning (minimize loss via optimization)
predictions = model.predict(X_new) # Inference`
        },
        {
          type: 'heading',
          content: 'A Taste of How Learning Works'
        },
        {
          type: 'text',
          content: `Let's see the simplest possible example: fitting a line to data.

**The Problem:** Given points (x, y), find a line y = mx + b that fits them.

**The Model:** y_pred = m * x + b (parameters: m, b)

**The Loss:** Mean Squared Error = average of (y_pred - y_actual)²

**The Learning:** Adjust m and b to minimize the loss`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt

# Our data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.2, 5.8, 8.1, 9.9])

# The "learning" - finding best m and b
# (In reality, we use calculus or gradient descent)
# For a line, there's a closed-form solution:

n = len(X)
m = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / (n * np.sum(X**2) - np.sum(X)**2)
b = (np.sum(y) - m * np.sum(X)) / n

print(f"Learned: y = {m:.2f}x + {b:.2f}")
# Output: y = 1.97x + 0.10

# Make predictions
y_pred = m * X + b

# How good is our fit?
mse = np.mean((y - y_pred)**2)
print(f"Mean Squared Error: {mse:.4f}")

# This is machine learning in its purest form:
# Data → Model → Loss → Optimization → Learned Parameters`
        },
        {
          type: 'heading',
          content: 'ML vs Traditional Software'
        },
        {
          type: 'text',
          content: `| Aspect | Traditional Software | Machine Learning |
|--------|---------------------|------------------|
| Logic | Explicitly coded | Learned from data |
| Updates | Manual code changes | Retrain on new data |
| Edge cases | Must anticipate all | Learns from examples |
| Debugging | Step through code | Analyze model behavior |
| Testing | Unit tests | Validation metrics |
| Behavior | Deterministic | Probabilistic |

**Key Insight:** ML models are functions approximated from data. They're powerful but come with uncertainty - they can be wrong, and understanding *why* they're wrong is often challenging.`
        },
        {
          type: 'text',
          content: `**Core Takeaways:**
- ML learns patterns from data instead of following explicit rules
- Every ML system has: data, model, loss function, optimization
- ML excels when patterns exist but rules are hard to specify
- ML outputs are probabilistic, not deterministic`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What fundamentally distinguishes ML from traditional programming?',
          options: [
            'ML is faster',
            'ML learns rules from data rather than having rules explicitly programmed',
            'ML only works with numbers',
            'ML requires more computing power'
          ],
          correct: 1,
          explanation: 'The key distinction is that ML derives rules/patterns from data, while traditional programming requires explicit rule coding by programmers.'
        },
        {
          type: 'multiple-choice',
          question: 'Which of these is NOT a core component of ML systems?',
          options: ['Data', 'Loss function', 'Database', 'Model'],
          correct: 2,
          explanation: 'The core components of ML are: data (examples), model (learnable function), loss function (measures error), and optimization (adjusts parameters). A database is a storage system, not a learning component.'
        }
      ]
    },
    {
      id: 'types-of-learning',
      title: 'Types of Learning',
      duration: '45 min',
      content: [
        {
          type: 'text',
          content: `Machine learning is divided into categories based on the type of feedback the learning algorithm receives. Understanding these categories helps you choose the right approach for your problem.`
        },
        {
          type: 'heading',
          content: 'Supervised Learning'
        },
        {
          type: 'text',
          content: `In supervised learning, we have input-output pairs. The algorithm learns to map inputs to outputs.

**Analogy:** Learning with a teacher who provides correct answers.

**Data Format:** (input, label) pairs
- Input: Email text
- Label: Spam or not spam

**Key Property:** We know the "ground truth" for training examples.`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Supervised Learning Examples

# ========== CLASSIFICATION ==========
# Predict discrete categories

# Binary classification: two classes
# - Email: spam vs not spam
# - Medical: disease vs healthy
# - Finance: fraud vs legitimate

# Multi-class classification: multiple classes
# - Image: cat vs dog vs bird
# - Text: positive vs negative vs neutral sentiment
# - Digit recognition: 0-9

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = LogisticRegression()
clf.fit(X, y)                    # Learn mapping
predictions = clf.predict(X)      # Predict class
probabilities = clf.predict_proba(X)  # Class probabilities

# ========== REGRESSION ==========
# Predict continuous values

# Examples:
# - House price prediction
# - Temperature forecasting
# - Stock price prediction
# - Age estimation from photo

from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1500], [2000], [2500], [3000]])  # Square footage
y = np.array([200000, 250000, 320000, 380000])   # Price

reg = LinearRegression()
reg.fit(X, y)
predicted_price = reg.predict([[2200]])  # Predict price for 2200 sqft`
        },
        {
          type: 'heading',
          content: 'Unsupervised Learning'
        },
        {
          type: 'text',
          content: `In unsupervised learning, we only have inputs - no labels. The algorithm must find structure in the data on its own.

**Analogy:** Learning without a teacher, discovering patterns yourself.

**Data Format:** Inputs only
- Input: Customer purchase history
- No labels: We don't know which customers are similar

**Key Property:** We don't know the "right answer" - we're exploring.`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Unsupervised Learning Examples

# ========== CLUSTERING ==========
# Group similar items together

# Examples:
# - Customer segmentation
# - Document grouping
# - Image compression
# - Anomaly detection

from sklearn.cluster import KMeans
import numpy as np

# Customer data: age, income, spending
X = np.array([[25, 50000, 2000],
              [35, 80000, 3500],
              [45, 120000, 5000],
              [22, 45000, 1800],
              [55, 150000, 6000]])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
cluster_labels = kmeans.labels_  # Which cluster each customer belongs to
# No "correct" labels - algorithm found groupings

# ========== DIMENSIONALITY REDUCTION ==========
# Reduce features while preserving information

# Examples:
# - Visualization (reduce to 2D/3D)
# - Noise reduction
# - Feature extraction
# - Data compression

from sklearn.decomposition import PCA

# High-dimensional data (e.g., 100 features)
X_high_dim = np.random.randn(1000, 100)

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)
# Can now visualize on 2D plot

# ========== ASSOCIATION ==========
# Find relationships between items

# Examples:
# - Market basket analysis (people who buy X also buy Y)
# - Recommendation systems
# - DNA sequence analysis`
        },
        {
          type: 'heading',
          content: 'Semi-Supervised Learning'
        },
        {
          type: 'text',
          content: `Semi-supervised learning uses a small amount of labeled data with a large amount of unlabeled data.

**Why it matters:** Labeling data is expensive and time-consuming. If you have 1 million images but can only afford to label 1000, semi-supervised learning can help.

**Intuition:** Use labeled examples to understand structure, then propagate labels to similar unlabeled examples.`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Semi-supervised: Use both labeled and unlabeled data

# Scenario:
# - 100 labeled examples (expensive to obtain)
# - 10,000 unlabeled examples (cheap/free)

# Approach 1: Self-training
# 1. Train on labeled data
# 2. Predict on unlabeled data
# 3. Add high-confidence predictions to training set
# 4. Repeat

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

# -1 indicates unlabeled
y_partial = np.array([0, 1, 0, -1, -1, -1, -1, -1, 1, -1])

model = SelfTrainingClassifier(SVC(probability=True))
model.fit(X, y_partial)

# Approach 2: Label Propagation
# Spread labels through similar examples
from sklearn.semi_supervised import LabelPropagation

label_prop = LabelPropagation()
label_prop.fit(X, y_partial)
all_labels = label_prop.transduction_  # Labels for all points`
        },
        {
          type: 'heading',
          content: 'Reinforcement Learning'
        },
        {
          type: 'text',
          content: `In reinforcement learning (RL), an agent learns by interacting with an environment and receiving rewards or penalties.

**Analogy:** Learning through trial and error, like training a dog with treats.

**Key Components:**
- **Agent**: The learner/decision maker
- **Environment**: What the agent interacts with
- **State**: Current situation
- **Action**: What agent can do
- **Reward**: Feedback signal

**Examples:**
- Game playing (AlphaGo, Atari games)
- Robotics (walking, manipulation)
- Autonomous vehicles
- Trading strategies`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Reinforcement Learning: Agent-Environment Loop

# Conceptual example (not complete implementation)
class SimpleAgent:
    def __init__(self):
        self.q_table = {}  # State -> Action values

    def choose_action(self, state):
        # Exploration vs exploitation
        if random.random() < 0.1:  # Explore
            return random.choice(actions)
        else:  # Exploit
            return max(self.q_table[state], key=lambda a: self.q_table[state][a])

    def learn(self, state, action, reward, next_state):
        # Q-learning update
        old_value = self.q_table[state][action]
        future_value = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_value + 0.1 * (reward + 0.9 * future_value - old_value)

# Training loop
agent = SimpleAgent()
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state`
        },
        {
          type: 'heading',
          content: 'Self-Supervised Learning'
        },
        {
          type: 'text',
          content: `Self-supervised learning generates labels from the data itself. This is how modern large language models like GPT are trained.

**Key Idea:** Create a prediction task where labels can be automatically derived from the input.

**Examples:**
- **Language Models:** Given "The cat sat on the ___", predict "mat"
- **Image Models:** Predict rotation angle of an image
- **Contrastive Learning:** Learn that two augmented versions of same image are similar

**Why it matters:** Can learn from massive amounts of unlabeled data, then fine-tune with small labeled datasets.`
        },
        {
          type: 'text',
          content: `**Summary of Learning Types:**

| Type | Labels | Example |
|------|--------|---------|
| Supervised | Full labels | Classify emails as spam |
| Unsupervised | No labels | Cluster customers |
| Semi-supervised | Some labels | Image classification with few labels |
| Reinforcement | Rewards | Game playing |
| Self-supervised | Generated from data | Language modeling |`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Customer segmentation (grouping similar customers) is an example of:',
          options: ['Supervised learning', 'Unsupervised learning', 'Reinforcement learning', 'Semi-supervised learning'],
          correct: 1,
          explanation: 'Customer segmentation is clustering, which is unsupervised learning. We don\'t have predefined labels for customer groups - the algorithm discovers the groupings from the data.'
        },
        {
          type: 'multiple-choice',
          question: 'Predicting house prices from features like square footage is:',
          options: ['Classification', 'Regression', 'Clustering', 'Reinforcement learning'],
          correct: 1,
          explanation: 'House price prediction outputs a continuous value (the price), making it a regression problem. Classification would output discrete categories.'
        },
        {
          type: 'multiple-choice',
          question: 'What makes reinforcement learning different from supervised learning?',
          options: [
            'RL uses more data',
            'RL learns from rewards through interaction, not from labeled examples',
            'RL is faster',
            'RL only works with images'
          ],
          correct: 1,
          explanation: 'RL learns through trial-and-error interaction with an environment, receiving reward signals. Supervised learning requires explicit input-output pairs provided upfront.'
        }
      ]
    },
    {
      id: 'ml-pipeline',
      title: 'The ML Pipeline',
      duration: '50 min',
      content: [
        {
          type: 'text',
          content: `Building an ML system is much more than training a model. It's a systematic process from raw data to deployed solution. Understanding this pipeline is crucial for success.`
        },
        {
          type: 'heading',
          content: 'The Complete ML Pipeline'
        },
        {
          type: 'text',
          content: `**1. Problem Definition** → What are we trying to solve?
**2. Data Collection** → Gather relevant data
**3. Data Exploration** → Understand your data
**4. Data Preprocessing** → Clean and prepare data
**5. Feature Engineering** → Create informative features
**6. Model Selection** → Choose appropriate algorithms
**7. Training** → Fit model to data
**8. Evaluation** → Measure performance
**9. Hyperparameter Tuning** → Optimize model
**10. Deployment** → Put model into production
**11. Monitoring** → Track performance over time

In practice, this is iterative - you'll loop back frequently.`
        },
        {
          type: 'heading',
          content: 'Step 1: Problem Definition'
        },
        {
          type: 'text',
          content: `Before writing any code, answer these questions:

**Business Questions:**
- What problem are we solving?
- What does success look like?
- What's the impact of wrong predictions?
- What decisions will this model inform?

**Technical Questions:**
- What type of problem is this? (classification, regression, clustering)
- What data is available?
- What's our target variable?
- What baseline should we beat?

**Constraints:**
- Latency requirements (real-time? batch?)
- Interpretability needs
- Computational resources
- Legal/ethical considerations`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Example: Loan Default Prediction

# Problem Definition:
# - Predict if a loan applicant will default
# - Binary classification (default: yes/no)
# - False negative costly (approve bad loan) - emphasis on recall
# - False positive also costly (deny good customer) - but less so
# - Model must be explainable (regulatory requirement)

# Success metrics:
# - Beat baseline (approve everyone who applied last year)
# - AUC-ROC > 0.85
# - Precision > 0.70 at recall > 0.80

# Constraints:
# - Prediction must be < 1 second (real-time decisions)
# - Cannot use protected attributes (race, gender)
# - Model must be interpretable for regulatory compliance`
        },
        {
          type: 'heading',
          content: 'Step 2-4: Data Collection, Exploration, Preprocessing'
        },
        {
          type: 'code',
          language: 'python',
          content: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== DATA COLLECTION ==========
df = pd.read_csv('loans.csv')

# ========== DATA EXPLORATION ==========
print(df.shape)
print(df.info())
print(df.describe())

# Check target distribution
print(df['default'].value_counts(normalize=True))

# Visualize distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(['income', 'debt_ratio', 'credit_score']):
    df[col].hist(ax=axes[0, i], bins=50)
    axes[0, i].set_title(col)

# Correlation with target
for col in df.select_dtypes(include=[np.number]).columns:
    corr = df[col].corr(df['default'])
    print(f"{col}: {corr:.3f}")

# ========== DATA PREPROCESSING ==========

# Handle missing values
print(df.isnull().sum())
df['income'].fillna(df['income'].median(), inplace=True)
df.dropna(subset=['credit_score'], inplace=True)  # Can't impute credit score

# Handle outliers
q99 = df['income'].quantile(0.99)
df['income'] = df['income'].clip(upper=q99)

# Encode categorical variables
df = pd.get_dummies(df, columns=['employment_status', 'loan_purpose'])

# Create train/test split BEFORE any fitting
from sklearn.model_selection import train_test_split
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features (fit on train only!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`
        },
        {
          type: 'heading',
          content: 'Step 5-8: Feature Engineering to Evaluation'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== FEATURE ENGINEERING ==========

# Create new features from domain knowledge
df['debt_to_income'] = df['total_debt'] / (df['income'] + 1)
df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1)
df['income_per_dependent'] = df['income'] / (df['dependents'] + 1)

# Interaction features
df['income_x_credit'] = df['income'] * df['credit_score']

# Binning
df['credit_tier'] = pd.cut(df['credit_score'],
                           bins=[0, 580, 670, 740, 850],
                           labels=['poor', 'fair', 'good', 'excellent'])

# ========== MODEL SELECTION & TRAINING ==========
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Try multiple models
models = {
    'logistic': LogisticRegression(),
    'random_forest': RandomForestClassifier(n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        'auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred)
    }
    print(f"\\n{name}:")
    print(f"AUC: {results[name]['auc']:.4f}")
    print(results[name]['report'])

# ========== EVALUATION ==========
from sklearn.metrics import confusion_matrix, precision_recall_curve

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall curve (important for imbalanced data)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()`
        },
        {
          type: 'heading',
          content: 'Step 9-11: Tuning, Deployment, Monitoring'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== HYPERPARAMETER TUNING ==========
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ========== SAVE FOR DEPLOYMENT ==========
import joblib
from sklearn.pipeline import Pipeline

# Create a complete pipeline
full_pipeline = Pipeline([
    ('scaler', scaler),
    ('model', best_model)
])

joblib.dump(full_pipeline, 'loan_model.joblib')

# ========== DEPLOYMENT (conceptual) ==========
# In production:
# pipeline = joblib.load('loan_model.joblib')
# prediction = pipeline.predict(new_applicant_data)

# ========== MONITORING ==========
# Track these metrics over time:
# - Model performance (accuracy, AUC on new data)
# - Data drift (feature distributions changing)
# - Prediction drift (output distribution changing)
# - Latency and throughput

# Example: Check for data drift
def check_data_drift(train_data, new_data, threshold=0.1):
    drift_detected = []
    for col in train_data.columns:
        train_mean = train_data[col].mean()
        new_mean = new_data[col].mean()
        drift = abs(train_mean - new_mean) / (train_mean + 1e-10)
        if drift > threshold:
            drift_detected.append((col, drift))
    return drift_detected`
        },
        {
          type: 'text',
          content: `**Pipeline Best Practices:**
1. Version control everything: code, data, models
2. Automate the pipeline where possible
3. Document each step and decision
4. Always have a baseline to compare against
5. Monitor deployed models continuously
6. Plan for model retraining from the start`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'When should you split data into train/test sets?',
          options: [
            'After feature scaling',
            'Before any preprocessing that learns from data',
            'After model training',
            'It doesn\'t matter when'
          ],
          correct: 1,
          explanation: 'You must split before any preprocessing that learns from data (like scaling, imputation with mean, etc.) to prevent data leakage from test set into training.'
        },
        {
          type: 'multiple-choice',
          question: 'What is data drift?',
          options: [
            'When data is lost during transfer',
            'When the distribution of input data changes over time compared to training data',
            'When the model weights drift',
            'When labels are incorrect'
          ],
          correct: 1,
          explanation: 'Data drift occurs when the statistical properties of the input data change over time, potentially degrading model performance. For example, customer behavior changing during a pandemic.'
        }
      ]
    },
    {
      id: 'features-engineering',
      title: 'Features and Feature Engineering',
      duration: '55 min',
      content: [
        {
          type: 'text',
          content: `Features are the language your model uses to understand the world. Good features can make a simple model outperform a complex one with poor features.

**Quote:** "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering." - Andrew Ng`
        },
        {
          type: 'heading',
          content: 'What Are Features?'
        },
        {
          type: 'text',
          content: `Features are the input variables your model uses to make predictions.

**Raw Data → Features → Model → Prediction**

For a house price prediction:
- **Raw data:** 2500 sqft, 3 bedrooms, built 1995, 123 Main St
- **Features (numerical):** [2500, 3, 28, 35.2, -82.4]
  - Square footage
  - Number of bedrooms
  - Age of house
  - Latitude
  - Longitude

The quality of your features directly impacts what patterns your model can learn.`
        },
        {
          type: 'heading',
          content: 'Types of Features'
        },
        {
          type: 'code',
          language: 'python',
          content: `import pandas as pd
import numpy as np

# ========== NUMERICAL FEATURES ==========
# Continuous: height, weight, temperature
# Discrete: count of items, number of bedrooms

# Scale matters! Distance-based algorithms need scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler: scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ========== CATEGORICAL FEATURES ==========
# Nominal: colors, countries (no order)
# Ordinal: ratings (bad, ok, good), education level (has order)

# One-hot encoding for nominal
df = pd.get_dummies(df, columns=['color'])
# color_red, color_blue, color_green columns

# Ordinal encoding for ordinal
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['bad', 'ok', 'good']])
df['rating_encoded'] = enc.fit_transform(df[['rating']])

# Label encoding (for tree-based models, can use any encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])

# ========== TEXT FEATURES ==========
# Bag of words, TF-IDF, embeddings

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df['description'])

# ========== DATETIME FEATURES ==========
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days`
        },
        {
          type: 'heading',
          content: 'Feature Engineering Techniques'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== MATHEMATICAL TRANSFORMATIONS ==========

# Log transform (for right-skewed data)
df['income_log'] = np.log1p(df['income'])  # log(1+x) handles zeros

# Square root (mild compression)
df['income_sqrt'] = np.sqrt(df['income'])

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Creates: x1, x2, x1^2, x2^2, x1*x2

# ========== INTERACTION FEATURES ==========
# Combine features that might have joint effects

df['age_x_income'] = df['age'] * df['income']
df['rooms_per_sqft'] = df['rooms'] / df['sqft']
df['price_per_sqft'] = df['price'] / df['sqft']

# ========== BINNING / DISCRETIZATION ==========
# Convert continuous to categorical

# Equal-width bins
df['age_bin'] = pd.cut(df['age'], bins=5)

# Custom bins based on domain knowledge
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 18, 35, 50, 65, 100],
                         labels=['child', 'young_adult', 'middle', 'senior', 'elderly'])

# Quantile bins (equal frequency)
df['income_quantile'] = pd.qcut(df['income'], q=5)

# ========== AGGREGATION FEATURES ==========
# When you have groups/categories

# Group statistics
df['avg_spend_by_category'] = df.groupby('category')['spend'].transform('mean')
df['spend_vs_category_avg'] = df['spend'] - df['avg_spend_by_category']

# Ranking within groups
df['spend_rank_in_category'] = df.groupby('category')['spend'].rank(pct=True)

# Count encodings
category_counts = df['category'].value_counts()
df['category_frequency'] = df['category'].map(category_counts)`
        },
        {
          type: 'heading',
          content: 'Feature Selection'
        },
        {
          type: 'text',
          content: `More features isn't always better. Too many features can:
- Cause overfitting
- Slow down training
- Make interpretation harder
- Introduce noise

Feature selection helps identify the most informative features.`
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== FILTER METHODS ==========
# Based on statistical properties, independent of model

# Correlation with target
correlations = df.corr()['target'].abs().sort_values(ascending=False)
print(correlations)

# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)

# Statistical tests
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# ========== WRAPPER METHODS ==========
# Use model performance to select features

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]

# ========== EMBEDDED METHODS ==========
# Feature selection built into model training

# L1 regularization (Lasso) - drives coefficients to zero
from sklearn.linear_model import LassoCV
lasso = LassoCV()
lasso.fit(X, y)
important_features = X.columns[lasso.coef_ != 0]

# Tree-based feature importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)

# ========== DIMENSIONALITY REDUCTION ==========
# Create new features that capture variance

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")`
        },
        {
          type: 'heading',
          content: 'Common Feature Engineering Mistakes'
        },
        {
          type: 'text',
          content: `**1. Data Leakage**
Using information that wouldn't be available at prediction time.
- Don't use future data to predict the past
- Don't use the target variable (even indirectly)

**2. Not Handling Categorical Features Properly**
- Tree-based models can handle label encoding
- Distance-based models (SVM, KNN) need one-hot encoding

**3. Ignoring Domain Knowledge**
- Ratios often matter more than absolute values
- Time-based features should reflect cyclical patterns
- Domain experts know which combinations matter

**4. Over-Engineering**
- Creating too many features causes overfitting
- Complex features are harder to maintain
- Start simple, add complexity if needed

**5. Not Scaling Features**
- Many algorithms (gradient descent, distance-based) require scaling
- Tree-based models don't need scaling`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why might log-transforming a feature be useful?',
          options: [
            'It always improves model performance',
            'It can make right-skewed distributions more normal and handle outliers',
            'It makes the feature smaller',
            'It\'s required for all machine learning models'
          ],
          correct: 1,
          explanation: 'Log transformation compresses the range of values, making right-skewed distributions more symmetric and reducing the impact of extreme outliers. This can help many algorithms learn better.'
        },
        {
          type: 'multiple-choice',
          question: 'Which encoding should you use for categorical variables with distance-based models like KNN?',
          options: ['Label encoding', 'One-hot encoding', 'Target encoding', 'No encoding needed'],
          correct: 1,
          explanation: 'Distance-based models calculate distances between feature values. Label encoding (0, 1, 2) implies "2 is closer to 1 than to 0" which is wrong for nominal categories. One-hot encoding treats each category equally.'
        }
      ]
    },
    {
      id: 'train-val-test',
      title: 'Training, Validation, and Testing',
      duration: '45 min',
      content: [
        {
          type: 'text',
          content: `How you split your data determines whether you can trust your results. This lesson explains why we need multiple data splits and how to do it correctly.`
        },
        {
          type: 'heading',
          content: 'Why Split Data?'
        },
        {
          type: 'text',
          content: `**The Problem:**
If you test your model on data it trained on, you're asking: "Can you remember what you saw?" Not: "Can you generalize to new data?"

A student who memorizes test answers gets 100% on that test but might fail a different one. We want models that understand concepts, not just memorize examples.

**The Goal:**
Estimate how the model will perform on **unseen** data - data it has never encountered during training or tuning.`
        },
        {
          type: 'heading',
          content: 'The Train-Test Split'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import train_test_split

# Basic 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42  # For reproducibility
)

# Training set: Used to fit the model
# Test set: Used ONLY for final evaluation (never for training or tuning)

model.fit(X_train, y_train)
final_score = model.score(X_test, y_test)

# CRITICAL: The test set should be touched exactly once - at the end
# Looking at test set multiple times and adjusting = data leakage`
        },
        {
          type: 'heading',
          content: 'Why We Need a Validation Set'
        },
        {
          type: 'text',
          content: `**The Problem with Just Train-Test:**
What if you need to:
- Try different models (logistic regression vs random forest)
- Tune hyperparameters (learning rate, max depth)
- Select features

You can't use the test set for these decisions - that would contaminate it. You need a third set: **validation**.`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Three-way split: 60% train, 20% validation, 20% test

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Purpose of each set:
# - Training: Fit model parameters (weights)
# - Validation: Tune hyperparameters, select model, monitor for overfitting
# - Test: Final unbiased evaluation (touch only once!)

# Example workflow:
models = [
    LogisticRegression(),
    RandomForestClassifier(n_estimators=100),
    RandomForestClassifier(n_estimators=200)
]

best_score = 0
best_model = None

for model in models:
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)  # Use validation for selection
    print(f"{model}: {val_score:.4f}")
    if val_score > best_score:
        best_score = val_score
        best_model = model

# Only now, with the final model chosen, evaluate on test set
final_score = best_model.score(X_test, y_test)
print(f"Final test score: {final_score:.4f}")`
        },
        {
          type: 'heading',
          content: 'Cross-Validation'
        },
        {
          type: 'text',
          content: `**The Problem with a Single Validation Set:**
With limited data, a single split might be unrepresentative. Your results depend on which examples happened to be in validation.

**Cross-Validation Solution:**
Split data into K folds, use each fold as validation once, average results.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import cross_val_score, KFold

# K-Fold Cross-Validation
# Data is split into K parts
# Model is trained K times, each time using a different part as validation

scores = cross_val_score(
    RandomForestClassifier(),
    X, y,
    cv=5,  # 5-fold CV
    scoring='accuracy'
)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# What happens in 5-fold CV:
# Fold 1: Train on folds 2,3,4,5 - Validate on fold 1
# Fold 2: Train on folds 1,3,4,5 - Validate on fold 2
# Fold 3: Train on folds 1,2,4,5 - Validate on fold 3
# Fold 4: Train on folds 1,2,3,5 - Validate on fold 4
# Fold 5: Train on folds 1,2,3,4 - Validate on fold 5
# Average all 5 validation scores

# More control with KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    # Train and evaluate...`
        },
        {
          type: 'heading',
          content: 'Stratified Splitting'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import StratifiedKFold, train_test_split

# Problem: Random splits might unbalance classes
# Example: 90% class A, 10% class B
# A random split might give a fold with 95% A, 5% B - not representative

# Solution: Stratified splitting preserves class proportions

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Preserves class proportions
    random_state=42
)

# Verify proportions are preserved
print(f"Original: {pd.Series(y).value_counts(normalize=True)}")
print(f"Train: {pd.Series(y_train).value_counts(normalize=True)}")
print(f"Test: {pd.Series(y_test).value_counts(normalize=True)}")

# Stratified cross-validation
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # Each fold has same class proportions as original data
    pass

# Use stratified CV by default for classification!
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5))`
        },
        {
          type: 'heading',
          content: 'Time Series Splitting'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import TimeSeriesSplit

# Problem: For time series, random splits cause data leakage
# Training on 2023 data to predict 2022 events = cheating!

# Solution: Time-based splits where train always precedes test

tscv = TimeSeriesSplit(n_splits=5)

# Visualize the splits
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: Train {train_idx[0]}-{train_idx[-1]}, Val {val_idx[0]}-{val_idx[-1]}")

# Output:
# Fold 0: Train 0-19, Val 20-29
# Fold 1: Train 0-29, Val 30-39
# Fold 2: Train 0-39, Val 40-49
# ...

# Notice: Training set grows, validation moves forward in time

# For simple train-test split with time series:
# Sort by date first, then split by index (not randomly!)
df = df.sort_values('date')
split_point = int(len(df) * 0.8)
train = df.iloc[:split_point]
test = df.iloc[split_point:]`
        },
        {
          type: 'heading',
          content: 'Nested Cross-Validation'
        },
        {
          type: 'text',
          content: `**For rigorous model selection and evaluation:**

When you tune hyperparameters using CV, you're optimizing for that particular CV split. The CV score might be optimistic.

**Nested CV:** Use an outer loop for evaluation, inner loop for tuning.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import cross_val_score, GridSearchCV

# Nested CV: Outer loop for evaluation, inner loop for tuning

# Inner loop: Find best hyperparameters
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
inner_cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=inner_cv,
    scoring='accuracy'
)

# Outer loop: Evaluate the grid search procedure
outer_cv = StratifiedKFold(n_splits=5)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)

print(f"Nested CV scores: {nested_scores}")
print(f"Mean: {nested_scores.mean():.4f} (+/- {nested_scores.std() * 2:.4f}")

# This gives unbiased estimate of how well your tuning procedure works`
        },
        {
          type: 'text',
          content: `**Key Takeaways:**
1. **Test set:** Touch ONLY once, for final evaluation
2. **Validation set:** Use for model selection and hyperparameter tuning
3. **Cross-validation:** More reliable than single validation split
4. **Stratified splits:** Essential for imbalanced classification
5. **Time series:** Always respect temporal ordering
6. **Nested CV:** Most rigorous, but computationally expensive`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why should you only evaluate on the test set once?',
          options: [
            'To save computation time',
            'Because multiple evaluations would cause the score to change',
            'Because using test results to make decisions contaminates the estimate',
            'Because sklearn only allows one evaluation'
          ],
          correct: 2,
          explanation: 'If you use test set results to make decisions (like "test score is low, let me try a different model"), you\'re indirectly training on the test set. This makes your test score optimistic and unreliable.'
        },
        {
          type: 'multiple-choice',
          question: 'What does stratified splitting ensure?',
          options: [
            'Equal sized splits',
            'Random sampling',
            'Class proportions are preserved in each split',
            'No missing values in splits'
          ],
          correct: 2,
          explanation: 'Stratified splitting ensures that each split has approximately the same percentage of samples from each class as the original dataset. This is crucial for imbalanced datasets.'
        }
      ]
    },
    {
      id: 'overfitting-underfitting',
      title: 'Overfitting and Underfitting',
      duration: '50 min',
      content: [
        {
          type: 'text',
          content: `Every ML practitioner's constant battle: finding the sweet spot between a model that's too simple (underfitting) and one that's too complex (overfitting).`
        },
        {
          type: 'heading',
          content: 'The Core Problem'
        },
        {
          type: 'text',
          content: `**We want models that generalize** - perform well on new, unseen data.

Two ways to fail:
1. **Underfitting:** Model is too simple to capture the pattern
2. **Overfitting:** Model memorizes training data instead of learning the pattern

**Analogy:**
- Underfitting: A student who only learned "most questions are answered C"
- Overfitting: A student who memorized all practice test answers but can't solve new problems`
        },
        {
          type: 'heading',
          content: 'Visualizing the Problem'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# True function: y = sin(x) + noise
np.random.seed(42)
X = np.linspace(0, 6, 30).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.randn(30) * 0.1

# Test data
X_test = np.linspace(0, 6, 100).reshape(-1, 1)
y_test_true = np.sin(X_test).ravel()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

degrees = [1, 4, 15]
titles = ['Underfitting (degree=1)', 'Good Fit (degree=4)', 'Overfitting (degree=15)']

for ax, degree, title in zip(axes, degrees, titles):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    y_pred = model.predict(X_test)

    ax.scatter(X, y, label='Training data')
    ax.plot(X_test, y_pred, 'r-', label='Model')
    ax.plot(X_test, y_test_true, 'g--', label='True function')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Key observations:
# degree=1: Can't capture the curve (underfitting)
# degree=4: Captures pattern without noise (good)
# degree=15: Fits every training point but wiggles wildly (overfitting)`
        },
        {
          type: 'heading',
          content: 'Diagnosing the Problem'
        },
        {
          type: 'code',
          language: 'python',
          content: `# The key diagnostic: compare training vs validation performance

def diagnose(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    print(f"Training score: {train_score:.4f}")
    print(f"Validation score: {val_score:.4f}")
    print(f"Gap: {train_score - val_score:.4f}")

    if train_score < 0.7:
        print("Diagnosis: UNDERFITTING")
        print("- Model too simple")
        print("- Try: more complex model, more features, less regularization")
    elif train_score - val_score > 0.1:
        print("Diagnosis: OVERFITTING")
        print("- Model too complex")
        print("- Try: more data, simpler model, more regularization")
    else:
        print("Diagnosis: GOOD FIT")

# Example outputs:
# Underfitting: Train=0.55, Val=0.52, Gap=0.03
# Overfitting: Train=0.99, Val=0.75, Gap=0.24
# Good fit: Train=0.90, Val=0.88, Gap=0.02`
        },
        {
          type: 'heading',
          content: 'Learning Curves'
        },
        {
          type: 'text',
          content: `Learning curves plot performance vs training set size. They reveal whether more data will help and diagnose fitting issues.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy'
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Interpreting learning curves:

# UNDERFITTING pattern:
# - Both curves plateau at low score
# - Gap between curves is small
# - More data won't help - need more complex model

# OVERFITTING pattern:
# - Training score much higher than validation
# - Large gap between curves
# - Gap decreases with more data - more data helps!

# GOOD FIT pattern:
# - Both curves converge to high score
# - Small gap at end
# - More data might help slightly`
        },
        {
          type: 'heading',
          content: 'Combating Overfitting'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== 1. GET MORE DATA ==========
# The best solution when possible
# More data = harder to memorize = must learn patterns

# ========== 2. REGULARIZATION ==========
# Penalize model complexity

# L1 Regularization (Lasso): drives weights to zero
# Good for feature selection
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)  # Higher alpha = more regularization

# L2 Regularization (Ridge): shrinks weights toward zero
# Good when all features are relevant
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# Elastic Net: combines L1 and L2
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# For neural networks: weight decay = L2 regularization

# ========== 3. SIMPLIFY THE MODEL ==========
# Reduce model capacity

# Fewer parameters
model = RandomForestClassifier(n_estimators=10)  # Instead of 1000

# Limit tree depth
model = DecisionTreeClassifier(max_depth=5)

# Fewer features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)

# ========== 4. DROPOUT (Neural Networks) ==========
# Randomly disable neurons during training
# Forces network to learn redundant representations

# ========== 5. EARLY STOPPING ==========
# Stop training when validation performance stops improving

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 rounds
    tol=1e-4
)

# ========== 6. DATA AUGMENTATION ==========
# Artificially increase training data
# Common in computer vision: rotations, flips, crops`
        },
        {
          type: 'heading',
          content: 'Combating Underfitting'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== 1. USE A MORE COMPLEX MODEL ==========
# Move from linear to non-linear

# From linear regression
from sklearn.linear_model import LinearRegression

# To random forest (can capture non-linear relationships)
from sklearn.ensemble import RandomForestRegressor

# Or neural networks for very complex patterns

# ========== 2. ADD MORE FEATURES ==========
# Feature engineering can unlock patterns

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# Domain-specific features
# e.g., for text: add word embeddings, n-grams

# ========== 3. REDUCE REGULARIZATION ==========
# Too much regularization can cause underfitting

# Decrease alpha
model = Ridge(alpha=0.01)  # Instead of 1.0

# ========== 4. TRAIN LONGER (Neural Networks) ==========
# More epochs = more time to learn

# ========== 5. DECREASE DROPOUT (Neural Networks) ==========
# Too much dropout can prevent learning`
        },
        {
          type: 'heading',
          content: 'The Model Complexity Sweet Spot'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Validation curves: Performance vs model complexity
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 10, 20, 50, 100]

train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', label='Training score')
plt.plot(param_range, val_mean, 'o-', label='Validation score')
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Validation Curve')
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.show()

# Typical pattern:
# - Training score increases with complexity
# - Validation score increases, then decreases
# - Sweet spot: where validation score is highest
# - Beyond sweet spot: overfitting`
        },
        {
          type: 'text',
          content: `**Summary:**

| Issue | Train Score | Val Score | Gap | Solution |
|-------|-------------|-----------|-----|----------|
| Underfitting | Low | Low | Small | More complex model, more features |
| Overfitting | High | Low | Large | More data, regularization, simpler model |
| Good Fit | High | High | Small | You're done! |

**Remember:** It's usually better to start simple and add complexity than to start complex and try to regularize.`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'High training accuracy but low validation accuracy indicates:',
          options: ['Underfitting', 'Overfitting', 'Good generalization', 'Data leakage'],
          correct: 1,
          explanation: 'A large gap between training and validation performance, with high training score, is the classic sign of overfitting - the model memorized training data but doesn\'t generalize.'
        },
        {
          type: 'multiple-choice',
          question: 'Which technique helps combat overfitting?',
          options: ['Adding more features', 'Increasing model complexity', 'Regularization', 'Training longer'],
          correct: 2,
          explanation: 'Regularization penalizes complexity, forcing the model to learn simpler patterns that generalize better. Adding features/complexity or training longer could make overfitting worse.'
        }
      ]
    },
    {
      id: 'bias-variance-tradeoff',
      title: 'The Bias-Variance Tradeoff',
      duration: '45 min',
      content: [
        {
          type: 'text',
          content: `The bias-variance tradeoff is the mathematical framework behind underfitting and overfitting. Understanding it helps you make principled decisions about model selection.`
        },
        {
          type: 'heading',
          content: 'What Are Bias and Variance?'
        },
        {
          type: 'text',
          content: `Imagine training many models on different random samples from the same underlying distribution.

**Bias:** How far off is the average prediction from the true value?
- High bias: Model consistently misses the target (systematic error)
- Think: A broken scale that always reads 5 pounds too high

**Variance:** How much do predictions vary across different training sets?
- High variance: Model is very sensitive to training data
- Think: A shaky hand - hits different spots each time

**The Goal:** Low bias AND low variance. But there's a tradeoff.`
        },
        {
          type: 'heading',
          content: 'The Mathematical Decomposition'
        },
        {
          type: 'text',
          content: `For any model, the expected prediction error can be decomposed:

**Total Error = Bias² + Variance + Irreducible Noise**

- **Bias²:** Error from wrong assumptions (model too simple)
- **Variance:** Error from sensitivity to training data (model too complex)
- **Irreducible Noise:** Random error in the data itself (can't be reduced)

This is why you can't minimize both simultaneously:
- Simple models → Low variance, high bias (underfitting)
- Complex models → Low bias, high variance (overfitting)`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt

# Demonstration: Train multiple models on different data samples

np.random.seed(0)

# True function
def true_function(x):
    return np.sin(x)

# Generate multiple training sets
n_samples = 20
n_datasets = 200
X_range = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)

# Models with different complexity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

models = {
    'High Bias (degree=1)': make_pipeline(PolynomialFeatures(1), LinearRegression()),
    'Balanced (degree=4)': make_pipeline(PolynomialFeatures(4), LinearRegression()),
    'High Variance (degree=15)': make_pipeline(PolynomialFeatures(15), LinearRegression())
}

# Train on multiple datasets and collect predictions
for name, model in models.items():
    predictions = []

    for _ in range(n_datasets):
        # New random sample
        X_train = np.random.uniform(0, 2*np.pi, n_samples).reshape(-1, 1)
        y_train = true_function(X_train).ravel() + np.random.randn(n_samples) * 0.3

        model.fit(X_train, y_train)
        y_pred = model.predict(X_range)
        predictions.append(y_pred)

    predictions = np.array(predictions)

    # Calculate bias and variance
    mean_prediction = predictions.mean(axis=0)
    bias_squared = (mean_prediction - true_function(X_range).ravel())**2
    variance = predictions.var(axis=0)

    print(f"{name}:")
    print(f"  Mean Bias²: {bias_squared.mean():.4f}")
    print(f"  Mean Variance: {variance.mean():.4f}")
    print(f"  Total: {(bias_squared + variance).mean():.4f}")
    print()`
        },
        {
          type: 'heading',
          content: 'Intuition with Examples'
        },
        {
          type: 'text',
          content: `**High Bias Models (Underfitting):**

Examples:
- Linear regression on curved data
- Shallow decision tree on complex patterns
- Logistic regression on non-linear boundaries

Characteristics:
- Consistent errors across different training sets
- Can't capture the true pattern
- Training and test error both high

**High Variance Models (Overfitting):**

Examples:
- Very deep decision tree
- High-degree polynomial regression
- KNN with k=1
- Unregularized neural network

Characteristics:
- Wildly different predictions on different training sets
- Captures noise as if it were signal
- Training error low, test error high`
        },
        {
          type: 'heading',
          content: 'Model-Specific Bias-Variance'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Different models have different bias-variance profiles

# ========== LINEAR MODELS ==========
# Generally: Low variance, potentially high bias
from sklearn.linear_model import LinearRegression, Ridge

# Ridge regression: add regularization to reduce variance
# Higher alpha = lower variance, higher bias
ridge_low_alpha = Ridge(alpha=0.01)   # Lower bias, higher variance
ridge_high_alpha = Ridge(alpha=100)   # Higher bias, lower variance

# ========== TREE-BASED MODELS ==========
# Single tree: Low bias, high variance
from sklearn.tree import DecisionTreeClassifier

# Deep tree: can fit any training data (low bias) but overfits (high variance)
deep_tree = DecisionTreeClassifier(max_depth=None)

# Shallow tree: can't fit complex data (high bias) but more stable (low variance)
shallow_tree = DecisionTreeClassifier(max_depth=3)

# ========== ENSEMBLE METHODS ==========
# Random Forest: reduces variance through averaging
# Boosting: reduces bias through iterative correction

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest: Average many high-variance trees
# Result: lower variance while maintaining low bias
rf = RandomForestClassifier(n_estimators=100)

# Gradient Boosting: Sequentially fix errors
# Each tree corrects previous errors (reduces bias)
# But can overfit if too many trees (increases variance)
gb = GradientBoostingClassifier(n_estimators=100)

# ========== K-NEAREST NEIGHBORS ==========
from sklearn.neighbors import KNeighborsClassifier

# k=1: lowest bias (can perfectly fit training data)
#      highest variance (sensitive to noise)
knn_1 = KNeighborsClassifier(n_neighbors=1)

# k=large: higher bias (averages too much)
#          lower variance (more stable)
knn_100 = KNeighborsClassifier(n_neighbors=100)`
        },
        {
          type: 'heading',
          content: 'Managing the Tradeoff'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Strategies for optimal bias-variance balance

# ========== 1. REGULARIZATION ==========
# Add penalty for complexity

from sklearn.linear_model import RidgeCV

# Cross-validation finds optimal regularization
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge.fit(X_train, y_train)
print(f"Best alpha: {ridge.alpha_}")  # Optimal bias-variance tradeoff

# ========== 2. ENSEMBLE METHODS ==========
# Combine multiple models to reduce variance

# Bagging: Train multiple models on bootstrap samples, average predictions
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(n_estimators=50)

# Random Forest: Bagging + random feature selection
rf = RandomForestClassifier(n_estimators=100)

# ========== 3. CROSS-VALIDATION ==========
# Reliable estimate of generalization error
from sklearn.model_selection import cross_val_score

# Select model with best CV score
for model in [shallow_tree, deep_tree, rf]:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{model.__class__.__name__}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ========== 4. MORE DATA ==========
# More data always helps reduce variance
# (Doesn't help with bias - need different model for that)

# ========== 5. FEATURE ENGINEERING ==========
# Good features can reduce bias without adding variance
# Better than adding model complexity`
        },
        {
          type: 'heading',
          content: 'The Modern Perspective: Double Descent'
        },
        {
          type: 'text',
          content: `**Classical view:** Error decreases then increases with complexity (U-shaped curve)

**Modern discovery:** Very complex models (like deep neural networks) can have a "double descent" - error decreases, increases, then decreases again with extreme overparameterization.

**Why?**
- With massive overparameterization, there are many solutions that fit training data
- Optimization (gradient descent) finds "simple" solutions among these
- This implicit regularization prevents overfitting

**Practical implication:** Very large models can work well, but this doesn't negate the bias-variance tradeoff - it's just that these models have implicit regularization that balances the tradeoff automatically.`
        },
        {
          type: 'text',
          content: `**Key Takeaways:**

1. **Bias:** Systematic error from model assumptions being wrong
2. **Variance:** Sensitivity to training data randomness
3. **Tradeoff:** Reducing one typically increases the other
4. **Goal:** Find the complexity that minimizes total error
5. **Tools:** Regularization, ensembles, cross-validation, more data
6. **Rule of thumb:** Start simple, add complexity only if underfitting`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'A model that gives very different predictions when trained on different random samples of the same data has:',
          options: ['High bias', 'High variance', 'Low bias', 'Good generalization'],
          correct: 1,
          explanation: 'High variance means the model\'s predictions are very sensitive to the particular training data used. Different samples lead to very different models.'
        },
        {
          type: 'multiple-choice',
          question: 'How does increasing regularization affect bias and variance?',
          options: [
            'Increases both bias and variance',
            'Decreases both bias and variance',
            'Increases bias, decreases variance',
            'Decreases bias, increases variance'
          ],
          correct: 2,
          explanation: 'Regularization constrains the model, making it simpler. This increases bias (might miss patterns) but decreases variance (more stable predictions across datasets).'
        },
        {
          type: 'multiple-choice',
          question: 'Random Forests reduce variance compared to a single decision tree by:',
          options: [
            'Using deeper trees',
            'Training multiple trees and averaging their predictions',
            'Using fewer features',
            'Using more regularization'
          ],
          correct: 1,
          explanation: 'Random Forests train many trees on different bootstrap samples and random feature subsets, then average predictions. Averaging reduces variance while maintaining the low bias of individual trees.'
        }
      ]
    }
  ]
}
