export const randomforestcourse = {
  id: 'random-forest-regression',
  title: 'Random Forest & Feature Engineering',
  icon: 'tree',
  description: 'Master ensemble methods and feature engineering through travel time prediction.',
  difficulty: 'intermediate',
  sourceproject: 'machine-learning-summative',
  lessons: [
    {
      id: 'ensemble-intro',
      title: 'Ensemble Learning Fundamentals',
      duration: '15 min read',
      concepts: ['Ensemble', 'Bagging', 'Wisdom of Crowds'],
      content: [
        { type: 'heading', text: 'The Power of Many' },
        { type: 'paragraph', text: 'Ensemble learning combines multiple models to produce better predictions than any single model. The intuition is simple: a committee of diverse experts often makes better decisions than any individual expert.' },
        { type: 'paragraph', text: 'This is called the "wisdom of crowds" effect. If each model makes independent errors, averaging their predictions cancels out individual mistakes while preserving the correct signal.' },

        { type: 'heading', text: 'Types of Ensembles' },
        { type: 'subheading', text: 'Bagging (Bootstrap Aggregating)' },
        { type: 'paragraph', text: 'Train multiple models on random subsets of the training data (with replacement). Each model sees a different view of the data. Random Forest uses bagging with decision trees.' },

        { type: 'subheading', text: 'Boosting' },
        { type: 'paragraph', text: 'Train models sequentially, with each new model focusing on examples the previous models got wrong. XGBoost and AdaBoost are popular boosting algorithms.' },

        { type: 'subheading', text: 'Stacking' },
        { type: 'paragraph', text: 'Train a meta-model to combine predictions from multiple base models. The meta-model learns optimal weights for each base model.' },

        { type: 'heading', text: 'Why Ensembles Work' },
        { type: 'paragraph', text: 'Consider three classifiers with 70% individual accuracy. If they make independent errors, the majority vote has:' },
        { type: 'formula', formula: 'P(majority correct) = P(3 correct) + P(2 correct) = 0.7³ + 3(0.7²)(0.3) ≈ 78.4%' },
        { type: 'paragraph', text: 'Adding more uncorrelated classifiers further improves accuracy. This is why diversity among ensemble members is crucial.' },

        { type: 'keypoints', points: [
          'Ensembles combine multiple models for better predictions',
          'Bagging trains on random data subsets',
          'Boosting focuses on hard examples sequentially',
          'Diversity among models is key to ensemble success'
        ]}
      ],
      quiz: [
        {
          question: 'What is the key difference between bagging and boosting?',
          options: ['Speed of training', 'Bagging trains in parallel, boosting trains sequentially', 'Number of trees', 'Type of base model'],
          correct: 1,
          explanation: 'Bagging trains models independently on random subsets, while boosting trains sequentially with each model learning from previous errors.'
        }
      ]
    },
    {
      id: 'random-forest-deep',
      title: 'Random Forest Architecture',
      duration: '18 min read',
      concepts: ['Decision Tree', 'Random Forest', 'Feature Subsampling'],
      content: [
        { type: 'heading', text: 'Decision Trees: The Building Block' },
        { type: 'paragraph', text: 'A decision tree splits data by asking yes/no questions. At each node, it finds the split that best separates the target variable. For regression, this typically means minimizing variance within each resulting group.' },
        { type: 'paragraph', text: 'Trees are interpretable but prone to overfitting—they can memorize training data perfectly while failing on new data.' },

        { type: 'heading', text: 'Random Forest: Trees + Randomness' },
        { type: 'paragraph', text: 'Random Forest adds two layers of randomness to decision trees:' },
        { type: 'list', items: [
          'Bootstrap sampling: Each tree trains on a random subset of rows (with replacement)',
          'Feature subsampling: At each split, only a random subset of features is considered'
        ]},
        { type: 'paragraph', text: 'This double randomness decorrelates the trees. Even if one feature dominates, different trees will focus on different features, creating diversity.' },

        { type: 'heading', text: 'The Travel Time Prediction Model' },
        { type: 'paragraph', text: 'This model predicts inter-city travel time in Nigeria using road and environmental features. The FastAPI endpoint shows the full prediction flow:' },
        { type: 'code', language: 'python', filename: 'prediction.py', fromproject: 'machine-learning-summative',
          code: `class TravelInput(BaseModel):
    road_length_km: float = Field(..., gt=0, description="Length of the road segment in kilometers")
    weather: Literal[0, 1, 2, 3] = Field(..., description="0=Clear, 1=Cloudy, 2=Rainy, 3=Foggy")
    direction: Literal[0, 1] = Field(..., description="1 if route is FROM Lagos, 0 if TO Lagos")
    congestion_level: Literal[1, 2, 3] = Field(..., description="1=Low, 2=Medium, 3=High")` },

        { type: 'callout', variant: 'info', text: 'Pydantic Field validators ensure valid inputs before prediction. This is crucial for production ML systems.' },

        { type: 'keypoints', points: [
          'Decision trees split data to minimize variance in target',
          'Random Forest adds bootstrap sampling and feature subsampling',
          'Double randomness decorrelates trees for better ensemble performance',
          'Typically use sqrt(n_features) features per split for classification, n/3 for regression'
        ]}
      ],
      quiz: [
        {
          question: 'Why does Random Forest subsample features at each split?',
          options: ['Speed optimization', 'Decorrelates trees by preventing dominant features from appearing everywhere', 'Reduces memory', 'Required by scikit-learn'],
          correct: 1,
          explanation: 'Feature subsampling ensures different trees focus on different features, creating the diversity needed for effective ensembling.'
        }
      ]
    },
    {
      id: 'feature-engineering',
      title: 'Feature Engineering for ML',
      duration: '20 min read',
      concepts: ['Feature Engineering', 'Transformation', 'Interaction'],
      content: [
        { type: 'heading', text: 'Why Feature Engineering Matters' },
        { type: 'paragraph', text: 'Raw data rarely captures the patterns models need. Feature engineering creates new features that make relationships easier to learn. A skilled data scientist can often improve model performance more through feature engineering than by tuning algorithms.' },

        { type: 'heading', text: 'The Feature Transformations' },
        { type: 'code', language: 'python', filename: 'prediction.py', fromproject: 'machine-learning-summative',
          code: `@app.post("/predict-time")
def predict_travel_time(data: TravelInput):
    input_df = pd.DataFrame([[
        data.road_length_km,
        data.weather,
        data.direction,
        data.congestion_level
    ]], columns=['Road Length (km)', 'Weather', 'direction', 'Congestion Level'])

    input_df['sqrt_distance'] = input_df['Road Length (km)'] ** 0.5
    input_df['distance_weather'] = input_df['Road Length (km)'] * input_df['Weather'] / 4

    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)[0]

    return {
        "predicted_travel_time_min": round(prediction, 2)
    }` },

        { type: 'subheading', text: 'Square Root Transformation' },
        { type: 'paragraph', text: 'sqrt_distance = road_length ** 0.5 captures diminishing returns. Travel time doesnt increase linearly with distance—longer trips involve highways with higher average speeds. The square root compresses large distances.' },

        { type: 'subheading', text: 'Feature Interaction' },
        { type: 'paragraph', text: 'distance_weather = road_length * weather / 4 captures how weather effects scale with distance. Rain on a short trip matters less than rain on a long trip. This interaction feature encodes that relationship explicitly.' },

        { type: 'heading', text: 'Common Feature Engineering Techniques' },
        { type: 'list', items: [
          'Log/sqrt transforms: Compress skewed distributions',
          'Polynomial features: Capture non-linear relationships',
          'Interactions: Multiply features to capture combined effects',
          'Binning: Convert continuous to categorical for non-linear patterns',
          'Date features: Extract day of week, month, hour from timestamps'
        ]},

        { type: 'keypoints', points: [
          'Feature engineering often matters more than algorithm choice',
          'Transformations can linearize non-linear relationships',
          'Interaction features capture combined effects of multiple variables',
          'Domain knowledge guides effective feature creation'
        ]}
      ],
      quiz: [
        {
          question: 'Why create a distance * weather interaction feature?',
          options: ['Reduces dimensionality', 'Captures how weather impact scales with trip length', 'Speeds up training', 'Required for Random Forest'],
          correct: 1,
          explanation: 'The interaction captures that weather matters more on longer trips—rain affects a 200km trip more than a 10km trip.'
        }
      ]
    },
    {
      id: 'scaling-deployment',
      title: 'Model Scaling and Deployment',
      duration: '12 min read',
      concepts: ['Scaling', 'Joblib', 'FastAPI'],
      content: [
        { type: 'heading', text: 'Why Scale Features?' },
        { type: 'paragraph', text: 'Many algorithms (KNN, SVM, neural networks) are sensitive to feature scales. A feature ranging 0-1000 would dominate one ranging 0-1. Standard scaling (z-score normalization) puts all features on comparable scales.' },
        { type: 'formula', formula: 'z = (x - μ) / σ' },
        { type: 'paragraph', text: 'Random Forests are actually scale-invariant (trees split on value thresholds, not distances), but scaling was included in the pipeline, possibly for comparison with other algorithms during development.' },

        { type: 'heading', text: 'Persisting Models with Joblib' },
        { type: 'code', language: 'python', filename: 'prediction.py', fromproject: 'machine-learning-summative',
          code: `model = joblib.load("../models/best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")` },

        { type: 'paragraph', text: 'Joblib efficiently serializes scikit-learn models and numpy arrays. Critical point: you must save both the model AND the scaler. The scaler contains the mean and standard deviation from training data—without it, you cant scale new inputs correctly.' },

        { type: 'heading', text: 'FastAPI for ML Deployment' },
        { type: 'paragraph', text: 'FastAPI provides automatic request validation, OpenAPI documentation, and async support. The prediction endpoint validates inputs with Pydantic before they reach the model, catching errors early.' },

        { type: 'callout', variant: 'warning', text: 'Always validate inputs before prediction. Invalid inputs can cause cryptic model errors or worse—silent incorrect predictions.' },

        { type: 'keypoints', points: [
          'Scaling normalizes features to comparable ranges',
          'Save both model and preprocessing objects (scaler)',
          'FastAPI provides validation and documentation automatically',
          'Input validation prevents silent prediction errors'
        ]}
      ],
      quiz: [
        {
          question: 'Why must you save the scaler alongside the model?',
          options: ['Performance optimization', 'Contains training set statistics needed to scale new inputs', 'FastAPI requirement', 'Reduces file size'],
          correct: 1,
          explanation: 'The scaler stores training set mean and std. Without it, you cannot properly scale new inputs to match what the model expects.'
        }
      ]
    }
  ]
}
