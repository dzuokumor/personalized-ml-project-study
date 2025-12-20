export const healthcarecourse = {
  id: 'healthcare-ml-pipeline',
  title: 'Healthcare ML & Data Pipelines',
  icon: 'health',
  description: 'Build end-to-end ML pipelines with dual database architecture for healthcare predictions.',
  difficulty: 'advanced',
  sourceproject: 'database-prediction-pipeline-group-11',
  lessons: [
    {
      id: 'healthcare-ml-intro',
      title: 'ML in Healthcare Context',
      duration: '12 min read',
      concepts: ['Healthcare AI', 'Ethics', 'Regulations'],
      content: [
        { type: 'heading', text: 'The Promise and Responsibility' },
        { type: 'paragraph', text: 'Healthcare ML can save lives—detecting diseases early, predicting complications, optimizing treatments. But it also carries unique responsibilities. Errors affect real patients. Bias can harm vulnerable populations. Regulations like HIPAA govern data handling.' },

        { type: 'heading', text: 'The Cardiovascular Prediction System' },
        { type: 'paragraph', text: 'This project predicts cardiovascular disease risk from patient data. This is a common healthcare ML application with real clinical value—early identification enables preventive interventions.' },

        { type: 'heading', text: 'Key Considerations for Healthcare ML' },
        { type: 'list', items: [
          'Data Privacy: Patient data requires strict access controls and encryption',
          'Explainability: Clinicians need to understand why predictions are made',
          'Bias: Models must be fair across demographic groups',
          'Validation: Clinical validation differs from standard ML metrics',
          'Regulation: FDA approval may be required for clinical deployment'
        ]},

        { type: 'callout', variant: 'warning', text: 'Healthcare ML systems should augment, not replace, clinical judgment. Always include human oversight in the decision loop.' },

        { type: 'keypoints', points: [
          'Healthcare ML has high impact but requires extra responsibility',
          'Privacy, explainability, and fairness are critical',
          'Clinical validation extends beyond accuracy metrics',
          'Regulatory compliance may be required'
        ]}
      ],
      quiz: [
        {
          question: 'Why is explainability especially important in healthcare ML?',
          options: ['Regulatory requirement only', 'Clinicians need to understand and trust predictions to act on them', 'Improves accuracy', 'Reduces compute cost'],
          correct: 1,
          explanation: 'Clinicians must understand model reasoning to appropriately incorporate predictions into patient care decisions.'
        }
      ]
    },
    {
      id: 'etl-pipelines',
      title: 'ETL Pipelines for ML',
      duration: '15 min read',
      concepts: ['ETL', 'Data Pipeline', 'Preprocessing'],
      content: [
        { type: 'heading', text: 'Extract, Transform, Load' },
        { type: 'paragraph', text: 'ETL pipelines move data from sources (medical records, devices, labs) through transformations (cleaning, feature engineering) into storage (databases, feature stores) for model training and inference.' },

        { type: 'heading', text: 'The Pipeline Architecture' },
        { type: 'paragraph', text: 'This project uses a dual-database architecture: SQL (PostgreSQL) for structured patient data with ACID guarantees, and MongoDB for flexible prediction logging and unstructured data.' },

        { type: 'subheading', text: 'SQL for Structured Data' },
        { type: 'paragraph', text: 'Patient demographics, medical measurements, and diagnoses have fixed schemas that benefit from relational databases. Foreign keys ensure data integrity—every diagnosis links to a valid patient.' },

        { type: 'subheading', text: 'MongoDB for Flexibility' },
        { type: 'paragraph', text: 'Prediction logs may evolve as models change. MongoDB documents can have varying fields without schema migrations. This flexibility suits rapidly evolving ML outputs.' },

        { type: 'heading', text: 'Data Flow' },
        { type: 'list', items: [
          '1. Patient data entered via API → stored in SQL',
          '2. Medical measurements recorded → SQL with foreign key to patient',
          '3. Prediction request received → fetch patient data from SQL',
          '4. Model generates prediction → log result to MongoDB',
          '5. Historical predictions queryable from MongoDB'
        ]},

        { type: 'keypoints', points: [
          'ETL pipelines automate data movement and transformation',
          'SQL provides ACID guarantees for critical patient data',
          'NoSQL offers flexibility for evolving ML outputs',
          'Choose database type based on data characteristics'
        ]}
      ],
      quiz: [
        {
          question: 'Why use both SQL and MongoDB in the same system?',
          options: ['Cost optimization', 'SQL for structured data with integrity, MongoDB for flexible evolving data', 'Performance only', 'Industry requirement'],
          correct: 1,
          explanation: 'Different data types benefit from different database paradigms. Patient records need ACID; prediction logs need flexibility.'
        }
      ]
    },
    {
      id: 'api-design',
      title: 'RESTful API Design for ML',
      duration: '15 min read',
      concepts: ['REST', 'API Design', 'Pydantic'],
      content: [
        { type: 'heading', text: 'Designing ML APIs' },
        { type: 'paragraph', text: 'Good API design makes ML systems usable. Clear endpoints, validated inputs, informative errors, and consistent responses enable integration with clinical workflows and other systems.' },

        { type: 'heading', text: 'The API Structure' },
        { type: 'code', language: 'python', filename: 'routers/prediction.py', fromproject: 'database-prediction-pipeline-group-11',
          code: `router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
    responses={500: {"description": "Internal server error"}}
)

@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def log_prediction(prediction: PredictionCreate):
    try:
        new_prediction = crud.create_prediction(prediction)
        return new_prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log prediction: {str(e)}"
        )` },

        { type: 'subheading', text: 'API Design Principles' },
        { type: 'list', items: [
          'Resource-oriented: /patients, /predictions, /diagnoses',
          'HTTP verbs: GET (read), POST (create), PUT (update), DELETE',
          'Status codes: 201 Created, 404 Not Found, 500 Server Error',
          'Validation: Pydantic models catch invalid inputs before processing',
          'Documentation: FastAPI auto-generates OpenAPI/Swagger docs'
        ]},

        { type: 'heading', text: 'Input Validation with Pydantic' },
        { type: 'paragraph', text: 'Pydantic models define expected input structure with types and constraints. Invalid requests are rejected with clear error messages before reaching the business logic.' },

        { type: 'keypoints', points: [
          'REST APIs use resources and HTTP verbs consistently',
          'Input validation prevents invalid data from reaching models',
          'Proper status codes communicate success/failure clearly',
          'Auto-generated documentation enables easy integration'
        ]}
      ],
      quiz: [
        {
          question: 'Why use Pydantic for request validation?',
          options: ['Faster processing', 'Catches invalid inputs with clear errors before business logic', 'Required by FastAPI', 'Reduces code size'],
          correct: 1,
          explanation: 'Pydantic validates request structure and types, returning informative errors for invalid inputs automatically.'
        }
      ]
    },
    {
      id: 'crud-operations',
      title: 'CRUD Operations & Data Access',
      duration: '12 min read',
      concepts: ['CRUD', 'Repository Pattern', 'ORM'],
      content: [
        { type: 'heading', text: 'Create, Read, Update, Delete' },
        { type: 'paragraph', text: 'CRUD operations are the fundamental data access patterns. Separating data access logic into dedicated functions (repository pattern) keeps code organized and testable.' },

        { type: 'heading', text: 'The CRUD Layer' },
        { type: 'paragraph', text: 'This project has separate CRUD modules for SQL (using SQLAlchemy) and MongoDB (using PyMongo). This separation allows each to use idiomatic patterns for their database type.' },

        { type: 'subheading', text: 'SQL with SQLAlchemy ORM' },
        { type: 'paragraph', text: 'SQLAlchemy maps Python classes to database tables. You write Python code; it generates SQL. This abstraction improves portability and prevents SQL injection.' },

        { type: 'subheading', text: 'MongoDB with PyMongo' },
        { type: 'paragraph', text: 'PyMongo provides direct document operations. Insert, find, update, and delete map naturally to MongoDB operations. Documents are Python dictionaries.' },

        { type: 'heading', text: 'Best Practices' },
        { type: 'list', items: [
          'Separate data access from business logic',
          'Use ORMs for type safety and SQL injection prevention',
          'Handle database errors gracefully',
          'Log database operations for debugging',
          'Use transactions for multi-step operations'
        ]},

        { type: 'keypoints', points: [
          'CRUD operations form the foundation of data access',
          'Repository pattern separates data logic from business logic',
          'ORMs provide type safety and SQL injection prevention',
          'Different databases may need different access patterns'
        ]}
      ],
      quiz: [
        {
          question: 'Why separate CRUD operations into dedicated modules?',
          options: ['Faster execution', 'Keeps code organized, testable, and separates concerns', 'Database requirement', 'Reduces memory usage'],
          correct: 1,
          explanation: 'The repository pattern separates data access from business logic, making code easier to test, maintain, and modify.'
        }
      ]
    },
    {
      id: 'prediction-logging',
      title: 'ML Prediction Logging',
      duration: '10 min read',
      concepts: ['Logging', 'Audit Trail', 'Reproducibility'],
      content: [
        { type: 'heading', text: 'Why Log Predictions?' },
        { type: 'paragraph', text: 'Prediction logging creates an audit trail essential for healthcare. You need to know what predictions were made, when, with what inputs, and which model version. This enables debugging, compliance, and model improvement.' },

        { type: 'heading', text: 'What to Log' },
        { type: 'list', items: [
          'Input features: What data the model received',
          'Output prediction: The model result and confidence',
          'Model version: Which model made the prediction',
          'Timestamp: When the prediction was made',
          'Patient reference: Link to patient record (if applicable)',
          'Request metadata: User, session, system info'
        ]},

        { type: 'heading', text: 'MongoDB for Prediction Logs' },
        { type: 'paragraph', text: 'This project logs predictions to MongoDB. This is a common pattern—predictions are append-heavy (many writes, few updates) and may have varying structure as models evolve.' },

        { type: 'callout', variant: 'info', text: 'Prediction logs are valuable training data. Comparing predictions to actual outcomes enables monitoring and retraining.' },

        { type: 'keypoints', points: [
          'Prediction logging is essential for healthcare compliance',
          'Log inputs, outputs, model version, and timestamps',
          'Append-heavy workloads suit document databases',
          'Prediction logs enable monitoring and improvement'
        ]}
      ],
      quiz: [
        {
          question: 'Why is logging model version with predictions important?',
          options: ['Saves disk space', 'Enables tracing predictions back to specific model for debugging', 'Improves accuracy', 'Required by FastAPI'],
          correct: 1,
          explanation: 'When investigating a prediction, you need to know which model version made it to reproduce and debug the behavior.'
        }
      ]
    }
  ]
}
