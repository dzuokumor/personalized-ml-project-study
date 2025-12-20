export const mlopscourse = {
  id: 'mlops-deployment',
  title: 'MLOps & Production Systems',
  icon: 'rocket',
  description: 'Learn production ML deployment, monitoring, and scaling with Docker and FastAPI.',
  difficulty: 'advanced',
  sourceproject: 'ml-deployment',
  lessons: [
    {
      id: 'mlops-intro',
      title: 'Introduction to MLOps',
      duration: '12 min read',
      concepts: ['MLOps', 'DevOps', 'ML Lifecycle'],
      content: [
        { type: 'heading', text: 'What is MLOps?' },
        { type: 'paragraph', text: 'MLOps (Machine Learning Operations) applies DevOps principles to machine learning systems. While a Jupyter notebook can train a model, deploying and maintaining it in production requires infrastructure, monitoring, and continuous improvement processes.' },
        { type: 'paragraph', text: 'Only ~10% of a production ML system is the actual model code. The rest is data pipelines, serving infrastructure, monitoring, and retraining systems.' },

        { type: 'heading', text: 'The ML Lifecycle' },
        { type: 'list', items: [
          'Data Collection: Gathering and storing training data',
          'Data Preparation: Cleaning, labeling, feature engineering',
          'Model Training: Experimentation, hyperparameter tuning',
          'Model Evaluation: Validation, A/B testing',
          'Deployment: Serving predictions at scale',
          'Monitoring: Tracking performance, detecting drift',
          'Retraining: Updating models as data changes'
        ]},

        { type: 'heading', text: 'The Production System' },
        { type: 'paragraph', text: 'My ml-deployment project implements a complete MLOps pipeline: model serving with FastAPI, containerization with Docker, monitoring, and retraining capabilities. This is a realistic production architecture.' },

        { type: 'keypoints', points: [
          'MLOps applies DevOps practices to ML systems',
          'Model code is only ~10% of production ML systems',
          'The ML lifecycle includes deployment, monitoring, and retraining',
          'Production ML requires infrastructure beyond notebooks'
        ]}
      ],
      quiz: [
        {
          question: 'Why is MLOps important?',
          options: ['Makes training faster', 'Bridges the gap between experimentation and production', 'Reduces model size', 'Improves accuracy'],
          correct: 1,
          explanation: 'MLOps provides the infrastructure and processes to deploy, monitor, and maintain ML models in production environments.'
        }
      ]
    },
    {
      id: 'model-serving',
      title: 'Model Serving with FastAPI',
      duration: '18 min read',
      concepts: ['Serving', 'API', 'Inference'],
      content: [
        { type: 'heading', text: 'From Notebook to API' },
        { type: 'paragraph', text: 'A trained model in a notebook is useful for analysis. A model behind an API can serve thousands of predictions per second. Model serving turns ML models into reliable, scalable services.' },

        { type: 'heading', text: 'The Model Class' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'ml-deployment',
          code: `class LandCoverModel:
    def __init__(self):
        self.model = None
        self.reverse_mapping = None
        self.code_to_name = {
            10: 'trees',
            20: 'shrubland',
            30: 'grassland',
            40: 'cropland',
            50: 'built-up',
            60: 'bare_sparse',
            80: 'water',
            90: 'wetland',
            95: 'mangroves'
        }
        self.load_model()` },

        { type: 'paragraph', text: 'The model is loaded once at startup and reused for all predictions. This avoids the overhead of loading weights for every request. The class encapsulates all model-related logic.' },

        { type: 'heading', text: 'Prediction with Confidence' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'ml-deployment',
          code: `def predict(self, image_array):
    predictions = self.model.predict(image_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])

    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3 = []
    for idx in top_3_indices:
        top_3.append({
            'class': self.code_to_name.get(self.reverse_mapping.get(int(idx))),
            'confidence': float(predictions[0][idx])
        })

    return {
        'predicted_class': class_code,
        'predicted_class_name': class_name,
        'confidence': confidence,
        'top_3': top_3_with_names
    }` },

        { type: 'paragraph', text: 'Returning top-3 predictions with confidence scores is production best practice. It helps users understand model uncertainty and catch edge cases.' },

        { type: 'keypoints', points: [
          'Load models once at startup to minimize latency',
          'Encapsulate model logic in dedicated classes',
          'Return confidence scores alongside predictions',
          'Top-k predictions help identify uncertain cases'
        ]}
      ],
      quiz: [
        {
          question: 'Why load the model at startup rather than per-request?',
          options: ['Saves disk space', 'Avoids repeated loading overhead for low-latency serving', 'Required by FastAPI', 'Improves accuracy'],
          correct: 1,
          explanation: 'Loading model weights takes hundreds of milliseconds. Loading once and reusing enables sub-millisecond prediction times.'
        }
      ]
    },
    {
      id: 'docker-containers',
      title: 'Containerization with Docker',
      duration: '15 min read',
      concepts: ['Docker', 'Containers', 'Orchestration'],
      content: [
        { type: 'heading', text: 'Why Containers?' },
        { type: 'paragraph', text: 'ML models depend on specific Python versions, library versions, and system configurations. "Works on my machine" is not acceptable for production. Containers package the entire environment, ensuring consistency from development to production.' },

        { type: 'heading', text: 'Docker Fundamentals' },
        { type: 'list', items: [
          'Image: Read-only template containing OS, dependencies, and code',
          'Container: Running instance of an image',
          'Dockerfile: Instructions to build an image',
          'Docker Compose: Orchestrate multiple containers'
        ]},

        { type: 'heading', text: 'Container Best Practices for ML' },
        { type: 'list', items: [
          'Use slim base images (python:3.11-slim) to reduce size',
          'Install dependencies before copying code (layer caching)',
          'Dont include training data in images',
          'Use multi-stage builds for smaller production images',
          'Set resource limits (memory, CPU) for predictable performance'
        ]},

        { type: 'heading', text: 'Horizontal Scaling' },
        { type: 'paragraph', text: 'This project tested scaling from 1 to 2 containers, achieving 40% throughput improvement. Horizontal scaling (more containers) is often easier than vertical scaling (bigger machines) for ML serving.' },

        { type: 'callout', variant: 'tip', text: 'Stateless services scale best. Keep model weights in the container or shared storage, not in memory across requests.' },

        { type: 'keypoints', points: [
          'Containers ensure environment consistency across deployments',
          'Use slim base images and layer caching for efficient builds',
          'Horizontal scaling adds more container instances',
          'Stateless design enables elastic scaling'
        ]}
      ],
      quiz: [
        {
          question: 'What problem do containers solve for ML deployment?',
          options: ['Speed up training', 'Environment consistency across development and production', 'Reduce model size', 'Improve accuracy'],
          correct: 1,
          explanation: 'Containers package the exact environment (OS, libraries, configs) ensuring the model runs identically everywhere.'
        }
      ]
    },
    {
      id: 'model-retraining',
      title: 'Model Retraining Pipeline',
      duration: '18 min read',
      concepts: ['Retraining', 'Data Drift', 'Continuous Learning'],
      content: [
        { type: 'heading', text: 'Why Retrain?' },
        { type: 'paragraph', text: 'Models degrade over time. Data drift occurs when the real-world data distribution shifts from training data. A fraud detection model trained on 2020 data may fail on 2024 patterns. Continuous retraining keeps models relevant.' },

        { type: 'heading', text: 'The Retraining Implementation' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'ml-deployment',
          code: `def retrain(self, train_data_path, epochs=10, batch_size=8, log_callback=None, progress_callback=None):
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    log(f"Starting retraining...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )` },

        { type: 'paragraph', text: 'The retraining function includes logging callbacks for real-time progress updates. This is essential for production systems where training might take hours—users need visibility into progress.' },

        { type: 'heading', text: 'Retraining Best Practices' },
        { type: 'code', language: 'python', filename: 'model.py', fromproject: 'ml-deployment',
          code: `backup_path = f"models/model_rgb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
if os.path.exists(MODEL_PATH):
    os.rename(MODEL_PATH, backup_path)
    log(f"Previous model backed up to {backup_path}")

self.model.save(MODEL_PATH)
log(f"Retrained model saved successfully!")

log("Reloading model into memory...")
self.model = keras.models.load_model(MODEL_PATH)` },

        { type: 'list', items: [
          'Backup previous model before overwriting',
          'Validate new model performance before deployment',
          'Log all training metrics for debugging',
          'Reload model into memory after saving',
          'Consider shadow deployment for A/B testing'
        ]},

        { type: 'keypoints', points: [
          'Data drift causes model performance degradation',
          'Backup models before retraining',
          'Progress callbacks enable monitoring long training jobs',
          'Validate new models before replacing production versions'
        ]}
      ],
      quiz: [
        {
          question: 'Why backup the model before retraining?',
          options: ['Saves disk space', 'Enables rollback if new model performs worse', 'Required by TensorFlow', 'Speeds up training'],
          correct: 1,
          explanation: 'If the retrained model performs worse, you can quickly restore the previous version without re-training from scratch.'
        }
      ]
    },
    {
      id: 'load-testing',
      title: 'Load Testing & Performance',
      duration: '12 min read',
      concepts: ['Load Testing', 'Latency', 'Throughput'],
      content: [
        { type: 'heading', text: 'Why Load Test?' },
        { type: 'paragraph', text: 'A model that works for one user might crash under 100 concurrent users. Load testing simulates production traffic to identify bottlenecks, measure latency, and determine scaling requirements before real users are affected.' },

        { type: 'heading', text: 'Key Metrics' },
        { type: 'list', items: [
          'Latency (p50, p95, p99): Time to respond. p99 matters more than average.',
          'Throughput: Requests per second the system can handle',
          'Error rate: Percentage of failed requests under load',
          'Resource utilization: CPU, memory, GPU usage'
        ]},

        { type: 'heading', text: 'The Load Test Results' },
        { type: 'paragraph', text: 'This project tested with Locust, comparing 1 vs 2 containers under 50-100 concurrent users. The 40% throughput improvement with 2 containers demonstrates near-linear horizontal scaling—an excellent result.' },

        { type: 'callout', variant: 'info', text: 'Not all systems scale linearly. Shared resources (databases, model weights) can become bottlenecks. Test to find the scaling limits.' },

        { type: 'heading', text: 'Load Testing Best Practices' },
        { type: 'list', items: [
          'Test with realistic request patterns and payloads',
          'Gradually increase load to find breaking points',
          'Monitor system resources during tests',
          'Test failure scenarios (what happens when DB is slow?)',
          'Run tests in production-like environments'
        ]},

        { type: 'keypoints', points: [
          'Load testing reveals bottlenecks before production',
          'p99 latency matters more than average for user experience',
          'Horizontal scaling should show near-linear throughput gains',
          'Test with realistic traffic patterns'
        ]}
      ],
      quiz: [
        {
          question: 'Why focus on p99 latency rather than average latency?',
          options: ['Its easier to measure', 'It shows the worst-case experience users actually encounter', 'Its always lower', 'Industry standard'],
          correct: 1,
          explanation: 'Average hides outliers. p99 ensures 99% of users experience acceptable latency—the slow 1% often matters most.'
        }
      ]
    }
  ]
}
