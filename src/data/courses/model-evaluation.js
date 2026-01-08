export const modelevaluation = {
  id: 'model-evaluation',
  title: 'Model Evaluation & Tuning',
  description: 'Master the art of measuring model performance, cross-validation techniques, and hyperparameter optimization',
  category: 'Classical ML',
  difficulty: 'Intermediate',
  duration: '5 hours',
  lessons: [
    {
      id: 'classification-metrics',
      title: 'Classification Metrics',
      duration: '55 min',
      concepts: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
      content: [
        {
          type: 'heading',
          content: 'Beyond Accuracy'
        },
        {
          type: 'text',
          content: `"My model has 99% accuracy!" Sounds great, right? But what if you're detecting fraud and only 1% of transactions are fraudulent? A model that predicts "not fraud" every time would be 99% accurate - and completely useless.

Accuracy alone is often misleading. We need metrics that capture what we actually care about.`
        },
        {
          type: 'heading',
          content: 'The Confusion Matrix'
        },
        {
          type: 'text',
          content: `The confusion matrix is the foundation of classification metrics. It shows how predictions match reality:`
        },
        {
          type: 'visualization',
          title: 'Confusion Matrix Structure',
          svg: `<svg viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="280" fill="#f8fafc"/>

            <!-- Title -->
            <text x="200" y="25" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">Confusion Matrix</text>

            <!-- Headers -->
            <text x="200" y="55" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">Predicted</text>
            <text x="240" y="75" text-anchor="middle" font-size="10" fill="#64748b">Positive</text>
            <text x="320" y="75" text-anchor="middle" font-size="10" fill="#64748b">Negative</text>

            <text x="130" y="115" text-anchor="middle" font-size="11" fill="#475569" font-weight="500" transform="rotate(-90, 130, 140)">Actual</text>
            <text x="160" y="120" text-anchor="middle" font-size="10" fill="#64748b">Positive</text>
            <text x="160" y="190" text-anchor="middle" font-size="10" fill="#64748b">Negative</text>

            <!-- Matrix cells -->
            <rect x="190" y="85" width="80" height="60" fill="#10b981" rx="4"/>
            <text x="230" y="110" text-anchor="middle" font-size="11" fill="white" font-weight="600">TP</text>
            <text x="230" y="125" text-anchor="middle" font-size="9" fill="#d1fae5">True Positive</text>

            <rect x="280" y="85" width="80" height="60" fill="#ef4444" rx="4"/>
            <text x="320" y="110" text-anchor="middle" font-size="11" fill="white" font-weight="600">FN</text>
            <text x="320" y="125" text-anchor="middle" font-size="9" fill="#fecaca">False Negative</text>

            <rect x="190" y="155" width="80" height="60" fill="#f59e0b" rx="4"/>
            <text x="230" y="180" text-anchor="middle" font-size="11" fill="white" font-weight="600">FP</text>
            <text x="230" y="195" text-anchor="middle" font-size="9" fill="#fef3c7">False Positive</text>

            <rect x="280" y="155" width="80" height="60" fill="#10b981" rx="4"/>
            <text x="320" y="180" text-anchor="middle" font-size="11" fill="white" font-weight="600">TN</text>
            <text x="320" y="195" text-anchor="middle" font-size="9" fill="#d1fae5">True Negative</text>

            <!-- Descriptions -->
            <text x="200" y="240" text-anchor="middle" font-size="9" fill="#10b981">Green = Correct predictions</text>
            <text x="200" y="255" text-anchor="middle" font-size="9" fill="#ef4444">Red/Orange = Errors</text>
          </svg>`,
          caption: 'The confusion matrix shows all possible prediction outcomes'
        },
        {
          type: 'text',
          content: `**True Positive (TP)**: Predicted positive, actually positive (correct!)
**True Negative (TN)**: Predicted negative, actually negative (correct!)
**False Positive (FP)**: Predicted positive, actually negative (Type I error)
**False Negative (FN)**: Predicted negative, actually positive (Type II error)

Think of it like a medical test:
- **FP**: Healthy person told they're sick (unnecessary worry)
- **FN**: Sick person told they're healthy (dangerous miss)`
        },
        {
          type: 'heading',
          content: 'Key Metrics'
        },
        {
          type: 'subheading',
          content: 'Accuracy'
        },
        {
          type: 'formula',
          content: 'Accuracy = (TP + TN) / (TP + TN + FP + FN)'
        },
        {
          type: 'text',
          content: `What fraction of all predictions were correct? Simple, but misleading for imbalanced classes.`
        },
        {
          type: 'subheading',
          content: 'Precision (Positive Predictive Value)'
        },
        {
          type: 'formula',
          content: 'Precision = TP / (TP + FP)'
        },
        {
          type: 'text',
          content: `When the model predicts positive, how often is it right?

**High precision** means few false positives. Important when false positives are costly (e.g., spam filter - you don't want legitimate emails marked as spam).`
        },
        {
          type: 'subheading',
          content: 'Recall (Sensitivity, True Positive Rate)'
        },
        {
          type: 'formula',
          content: 'Recall = TP / (TP + FN)'
        },
        {
          type: 'text',
          content: `Of all actual positives, how many did the model catch?

**High recall** means few false negatives. Important when false negatives are costly (e.g., cancer screening - you don't want to miss a case).`
        },
        {
          type: 'visualization',
          title: 'Precision vs Recall',
          svg: `<svg viewBox="0 0 450 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="200" fill="#f8fafc"/>

            <!-- Precision diagram -->
            <g transform="translate(30,30)">
              <text x="75" y="0" text-anchor="middle" font-size="11" fill="#3b82f6" font-weight="600">Precision</text>
              <text x="75" y="15" text-anchor="middle" font-size="9" fill="#64748b">"Of predicted positives..."</text>

              <!-- Predicted positive box -->
              <rect x="20" y="30" width="110" height="80" fill="#dbeafe" stroke="#3b82f6" stroke-width="2" rx="4"/>
              <text x="75" y="125" text-anchor="middle" font-size="9" fill="#3b82f6">Predicted Positive</text>

              <!-- TP and FP inside -->
              <rect x="30" y="40" width="40" height="60" fill="#10b981" rx="2"/>
              <text x="50" y="75" text-anchor="middle" font-size="10" fill="white" font-weight="500">TP</text>

              <rect x="80" y="40" width="40" height="60" fill="#f59e0b" rx="2"/>
              <text x="100" y="75" text-anchor="middle" font-size="10" fill="white" font-weight="500">FP</text>

              <text x="75" y="145" text-anchor="middle" font-size="9" fill="#475569">TP / (TP + FP)</text>
            </g>

            <!-- Recall diagram -->
            <g transform="translate(250,30)">
              <text x="75" y="0" text-anchor="middle" font-size="11" fill="#10b981" font-weight="600">Recall</text>
              <text x="75" y="15" text-anchor="middle" font-size="9" fill="#64748b">"Of actual positives..."</text>

              <!-- Actual positive box -->
              <rect x="20" y="30" width="110" height="80" fill="#d1fae5" stroke="#10b981" stroke-width="2" rx="4"/>
              <text x="75" y="125" text-anchor="middle" font-size="9" fill="#10b981">Actual Positive</text>

              <!-- TP and FN inside -->
              <rect x="30" y="40" width="40" height="60" fill="#10b981" rx="2"/>
              <text x="50" y="75" text-anchor="middle" font-size="10" fill="white" font-weight="500">TP</text>

              <rect x="80" y="40" width="40" height="60" fill="#ef4444" rx="2"/>
              <text x="100" y="75" text-anchor="middle" font-size="10" fill="white" font-weight="500">FN</text>

              <text x="75" y="145" text-anchor="middle" font-size="9" fill="#475569">TP / (TP + FN)</text>
            </g>
          </svg>`,
          caption: 'Precision looks at predicted positives, recall looks at actual positives'
        },
        {
          type: 'subheading',
          content: 'F1 Score'
        },
        {
          type: 'text',
          content: `The harmonic mean of precision and recall - balances both concerns:`
        },
        {
          type: 'formula',
          content: 'F1 = 2 × (Precision × Recall) / (Precision + Recall)'
        },
        {
          type: 'text',
          content: `F1 ranges from 0 to 1. A model needs both good precision AND good recall to achieve high F1.

**Why harmonic mean?** It penalizes extreme imbalance. If precision is 1.0 but recall is 0.0, F1 = 0 (not 0.5 like arithmetic mean would give).`
        },
        {
          type: 'heading',
          content: 'The Precision-Recall Trade-off'
        },
        {
          type: 'text',
          content: `You can't maximize both precision and recall. There's always a trade-off:

- **Increase threshold** → Fewer positive predictions → Higher precision, lower recall
- **Decrease threshold** → More positive predictions → Lower precision, higher recall

The right balance depends on the business problem.`
        },
        {
          type: 'visualization',
          title: 'Precision-Recall Trade-off',
          svg: `<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="220" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="60" y1="180" x2="360" y2="180" stroke="#475569" stroke-width="2"/>
            <line x1="60" y1="180" x2="60" y2="30" stroke="#475569" stroke-width="2"/>

            <text x="210" y="210" text-anchor="middle" font-size="10" fill="#475569">Threshold</text>
            <text x="30" y="105" font-size="10" fill="#475569" transform="rotate(-90,30,105)">Score</text>

            <!-- Precision curve -->
            <path d="M80,150 Q150,100 200,70 Q250,50 330,40" fill="none" stroke="#3b82f6" stroke-width="2.5"/>

            <!-- Recall curve -->
            <path d="M80,40 Q150,50 200,80 Q250,120 330,160" fill="none" stroke="#10b981" stroke-width="2.5"/>

            <!-- Legend -->
            <line x1="100" y1="20" x2="130" y2="20" stroke="#3b82f6" stroke-width="2.5"/>
            <text x="135" y="24" font-size="10" fill="#3b82f6">Precision</text>

            <line x1="220" y1="20" x2="250" y2="20" stroke="#10b981" stroke-width="2.5"/>
            <text x="255" y="24" font-size="10" fill="#10b981">Recall</text>

            <!-- Annotations -->
            <line x1="200" y1="30" x2="200" y2="180" stroke="#ef4444" stroke-width="1" stroke-dasharray="4"/>
            <text x="200" y="195" text-anchor="middle" font-size="9" fill="#ef4444">Balance point</text>
          </svg>`,
          caption: 'Raising the threshold increases precision but decreases recall'
        },
        {
          type: 'heading',
          content: 'ROC Curve and AUC'
        },
        {
          type: 'text',
          content: `The **ROC curve** (Receiver Operating Characteristic) plots True Positive Rate vs False Positive Rate at various thresholds.

**AUC** (Area Under the Curve) summarizes the ROC curve into a single number:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing (diagonal line)
- AUC < 0.5: Worse than random (predictions are backwards)`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report
)
import numpy as np

y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0])
y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.15, 0.6, 0.25, 0.05])

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\\nAccuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")

print("\\nClassification Report:")
print(classification_report(y_true, y_pred))`
        },
        {
          type: 'heading',
          content: 'Which Metric to Use?'
        },
        {
          type: 'table',
          headers: ['Scenario', 'Priority', 'Metric'],
          rows: [
            ['Balanced classes', 'General performance', 'Accuracy, F1'],
            ['Imbalanced classes', 'Overall discrimination', 'ROC-AUC'],
            ['False positives costly', 'Minimize FP', 'Precision'],
            ['False negatives costly', 'Minimize FN', 'Recall'],
            ['Need balance', 'Both FP and FN matter', 'F1 Score']
          ],
          caption: 'Choosing the right metric for your problem'
        },
        {
          type: 'keypoints',
          points: [
            'Accuracy can be misleading for imbalanced classes',
            'Precision measures "of predicted positives, how many are correct"',
            'Recall measures "of actual positives, how many did we catch"',
            'F1 Score balances precision and recall',
            'ROC-AUC measures overall discrimination ability'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'In a cancer screening test, which metric should be prioritized?',
          options: [
            'Precision - avoid false alarms',
            'Recall - catch all cancer cases',
            'Accuracy - overall correctness',
            'Specificity - identify healthy patients'
          ],
          correct: 1,
          explanation: 'In cancer screening, missing a cancer case (false negative) is much worse than a false alarm. High recall ensures we catch all cases, even if some healthy people need follow-up tests.'
        },
        {
          type: 'multiple-choice',
          question: 'What does an ROC-AUC of 0.5 indicate?',
          options: [
            'Perfect classifier',
            'Good classifier',
            'Random guessing',
            'Inverse predictions'
          ],
          correct: 2,
          explanation: 'AUC = 0.5 means the classifier is no better than random guessing. The ROC curve follows the diagonal line from (0,0) to (1,1).'
        }
      ]
    },
    {
      id: 'regression-metrics',
      title: 'Regression Metrics',
      duration: '45 min',
      concepts: ['MSE', 'RMSE', 'MAE', 'R-squared'],
      content: [
        {
          type: 'heading',
          content: 'Measuring Prediction Error'
        },
        {
          type: 'text',
          content: `In regression, we predict continuous values. How close are our predictions to the actual values? That's what regression metrics measure.

Each metric captures different aspects of error - understanding their differences helps you choose the right one.`
        },
        {
          type: 'heading',
          content: 'Mean Squared Error (MSE)'
        },
        {
          type: 'formula',
          content: 'MSE = (1/n) Σ(yᵢ - ŷᵢ)²'
        },
        {
          type: 'text',
          content: `MSE squares each error, then averages them. Squaring has important effects:

**Advantages:**
- Differentiable everywhere (good for optimization)
- Penalizes large errors more than small ones

**Disadvantages:**
- Units are squared (hard to interpret)
- Very sensitive to outliers
- Large errors dominate the metric`
        },
        {
          type: 'visualization',
          title: 'MSE Penalizes Large Errors',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="50" y1="140" x2="350" y2="140" stroke="#475569" stroke-width="1.5"/>

            <!-- Prediction line -->
            <line x1="50" y1="90" x2="350" y2="90" stroke="#3b82f6" stroke-width="2"/>
            <text x="360" y="95" font-size="9" fill="#3b82f6">Predicted</text>

            <!-- Small error -->
            <circle cx="100" cy="80" r="5" fill="#10b981"/>
            <line x1="100" y1="80" x2="100" y2="90" stroke="#10b981" stroke-width="2"/>
            <text x="100" y="155" text-anchor="middle" font-size="9" fill="#64748b">Error: 10</text>
            <text x="100" y="168" text-anchor="middle" font-size="9" fill="#10b981">Squared: 100</text>

            <!-- Medium error -->
            <circle cx="200" cy="60" r="5" fill="#f59e0b"/>
            <line x1="200" y1="60" x2="200" y2="90" stroke="#f59e0b" stroke-width="2"/>
            <text x="200" y="155" text-anchor="middle" font-size="9" fill="#64748b">Error: 30</text>
            <text x="200" y="168" text-anchor="middle" font-size="9" fill="#f59e0b">Squared: 900</text>

            <!-- Large error -->
            <circle cx="300" cy="30" r="5" fill="#ef4444"/>
            <line x1="300" y1="30" x2="300" y2="90" stroke="#ef4444" stroke-width="2"/>
            <text x="300" y="155" text-anchor="middle" font-size="9" fill="#64748b">Error: 60</text>
            <text x="300" y="168" text-anchor="middle" font-size="9" fill="#ef4444">Squared: 3600</text>

            <!-- Title -->
            <text x="200" y="20" text-anchor="middle" font-size="10" fill="#475569">Doubling error quadruples MSE contribution</text>
          </svg>`,
          caption: 'Large errors contribute disproportionately to MSE'
        },
        {
          type: 'heading',
          content: 'Root Mean Squared Error (RMSE)'
        },
        {
          type: 'formula',
          content: 'RMSE = √MSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]'
        },
        {
          type: 'text',
          content: `RMSE is simply the square root of MSE. This brings the error back to the original units.

If you're predicting house prices in dollars, RMSE is also in dollars - much more interpretable than MSE (which would be in dollars²).

**RMSE = 50,000** means "on average, predictions are about $50,000 off"`
        },
        {
          type: 'heading',
          content: 'Mean Absolute Error (MAE)'
        },
        {
          type: 'formula',
          content: 'MAE = (1/n) Σ|yᵢ - ŷᵢ|'
        },
        {
          type: 'text',
          content: `MAE uses absolute values instead of squares. Each error contributes proportionally to its size.

**Advantages:**
- Same units as target variable
- More robust to outliers than MSE/RMSE
- Easy to interpret

**Disadvantages:**
- Not differentiable at zero (less convenient for optimization)
- Treats all errors equally (sometimes you want to penalize large errors more)`
        },
        {
          type: 'subheading',
          content: 'MAE vs RMSE: When Does It Matter?'
        },
        {
          type: 'text',
          content: `If you have outliers or extreme errors:
- **RMSE will be much larger than MAE** (because squaring amplifies large errors)
- **MAE is more representative** of typical error

If all errors are similar in size:
- **RMSE ≈ MAE** (they converge)

**Rule of thumb**: If RMSE >> MAE, you have some large errors. Investigate!`
        },
        {
          type: 'heading',
          content: 'R-squared (Coefficient of Determination)'
        },
        {
          type: 'formula',
          content: 'R² = 1 - (SS_res / SS_tot) = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]'
        },
        {
          type: 'text',
          content: `R² measures how much of the variance in y your model explains.

- **R² = 1**: Model explains all variance (perfect predictions)
- **R² = 0**: Model explains no variance (no better than predicting mean)
- **R² < 0**: Model is worse than predicting mean (very bad!)

**Interpretation**: "X% of the variance in y is explained by the model"`
        },
        {
          type: 'visualization',
          title: 'R² Visual Interpretation',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- R² = 0.9 -->
            <g transform="translate(30,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">R² = 0.9</text>
              <rect x="10" y="10" width="120" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Regression line -->
              <line x1="25" y1="90" x2="115" y2="25" stroke="#3b82f6" stroke-width="2"/>

              <!-- Points close to line -->
              <circle cx="30" cy="85" r="4" fill="#10b981"/>
              <circle cx="45" cy="72" r="4" fill="#10b981"/>
              <circle cx="60" cy="62" r="4" fill="#10b981"/>
              <circle cx="75" cy="50" r="4" fill="#10b981"/>
              <circle cx="90" cy="40" r="4" fill="#10b981"/>
              <circle cx="105" cy="30" r="4" fill="#10b981"/>

              <text x="70" y="125" text-anchor="middle" font-size="9" fill="#64748b">Tight fit</text>
            </g>

            <!-- R² = 0.5 -->
            <g transform="translate(170,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#f59e0b" font-weight="600">R² = 0.5</text>
              <rect x="10" y="10" width="120" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Regression line -->
              <line x1="25" y1="90" x2="115" y2="25" stroke="#3b82f6" stroke-width="2"/>

              <!-- Points scattered -->
              <circle cx="30" cy="75" r="4" fill="#f59e0b"/>
              <circle cx="45" cy="85" r="4" fill="#f59e0b"/>
              <circle cx="60" cy="45" r="4" fill="#f59e0b"/>
              <circle cx="75" cy="65" r="4" fill="#f59e0b"/>
              <circle cx="90" cy="30" r="4" fill="#f59e0b"/>
              <circle cx="105" cy="50" r="4" fill="#f59e0b"/>

              <text x="70" y="125" text-anchor="middle" font-size="9" fill="#64748b">Moderate fit</text>
            </g>

            <!-- R² = 0.1 -->
            <g transform="translate(310,30)">
              <text x="45" y="0" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="600">R² = 0.1</text>
              <rect x="5" y="10" width="80" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Regression line (nearly flat) -->
              <line x1="15" y1="60" x2="75" y2="55" stroke="#3b82f6" stroke-width="2"/>

              <!-- Points very scattered -->
              <circle cx="20" cy="25" r="3" fill="#ef4444"/>
              <circle cx="30" cy="85" r="3" fill="#ef4444"/>
              <circle cx="40" cy="40" r="3" fill="#ef4444"/>
              <circle cx="50" cy="90" r="3" fill="#ef4444"/>
              <circle cx="60" cy="35" r="3" fill="#ef4444"/>
              <circle cx="70" cy="75" r="3" fill="#ef4444"/>

              <text x="45" y="125" text-anchor="middle" font-size="9" fill="#64748b">Poor fit</text>
            </g>
          </svg>`,
          caption: 'Higher R² means the model explains more variance in the data'
        },
        {
          type: 'heading',
          content: 'Adjusted R²'
        },
        {
          type: 'text',
          content: `Regular R² always increases when you add more features - even useless ones! **Adjusted R²** penalizes model complexity:`
        },
        {
          type: 'formula',
          content: 'Adjusted R² = 1 - [(1-R²)(n-1)/(n-p-1)]'
        },
        {
          type: 'text',
          content: `Where n = number of samples, p = number of features.

Adjusted R² only increases if a new feature improves the model more than expected by chance. Use this when comparing models with different numbers of features.`
        },
        {
          type: 'heading',
          content: 'Metrics in Practice'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 210, 240, 350])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

if rmse > mae * 1.2:
    print("\\nNote: RMSE >> MAE suggests outliers or large errors")`
        },
        {
          type: 'heading',
          content: 'Choosing the Right Metric'
        },
        {
          type: 'table',
          headers: ['Metric', 'Use When', 'Interpretation'],
          rows: [
            ['MAE', 'Outliers exist, want typical error', 'Average absolute error in original units'],
            ['RMSE', 'Large errors should be penalized more', 'Similar to MAE but penalizes outliers'],
            ['R²', 'Comparing models, explaining variance', 'Fraction of variance explained'],
            ['MAPE', 'Want percentage error', 'Average % deviation (avoid with zeros)']
          ],
          caption: 'Metric selection guide'
        },
        {
          type: 'keypoints',
          points: [
            'MSE squares errors, penalizing large errors heavily',
            'RMSE brings MSE back to original units',
            'MAE is more robust to outliers than RMSE',
            'R² measures the proportion of variance explained',
            'If RMSE >> MAE, investigate outliers'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Which metric is most robust to outliers?',
          options: [
            'MSE',
            'RMSE',
            'MAE',
            'R²'
          ],
          correct: 2,
          explanation: 'MAE (Mean Absolute Error) uses absolute values rather than squares, so outliers contribute proportionally to their size rather than quadratically. This makes MAE more robust to extreme values.'
        },
        {
          type: 'multiple-choice',
          question: 'What does R² = 0.75 mean?',
          options: [
            '75% of predictions are correct',
            '75% of the variance in y is explained by the model',
            'The model has 75% accuracy',
            'The error is 75% of the mean'
          ],
          correct: 1,
          explanation: 'R² = 0.75 means the model explains 75% of the variance in the target variable. The remaining 25% is unexplained variance (noise or missing features).'
        }
      ]
    },
    {
      id: 'cross-validation',
      title: 'Cross-Validation Techniques',
      duration: '55 min',
      concepts: ['K-Fold', 'Stratified', 'Leave-One-Out', 'Time Series CV'],
      content: [
        {
          type: 'heading',
          content: 'Why Cross-Validation?'
        },
        {
          type: 'text',
          content: `A single train-test split has a problem: your results depend heavily on which data ended up in which set. Lucky split? Great results. Unlucky split? Poor results.

**Cross-validation** uses multiple splits to get a more reliable estimate of model performance. Instead of one number, you get a distribution of scores.`
        },
        {
          type: 'heading',
          content: 'K-Fold Cross-Validation'
        },
        {
          type: 'text',
          content: `The most common technique. Split data into K equal parts (folds):

1. Train on K-1 folds, validate on 1 fold
2. Repeat K times, each fold serving as validation once
3. Average the K scores

Typical values: K = 5 or K = 10`
        },
        {
          type: 'visualization',
          title: '5-Fold Cross-Validation',
          svg: `<svg viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="220" fill="#f8fafc"/>

            <!-- Title -->
            <text x="225" y="20" text-anchor="middle" font-size="11" fill="#475569" font-weight="600">5-Fold Cross-Validation</text>

            <!-- Legend -->
            <rect x="120" y="30" width="12" height="12" fill="#3b82f6" rx="2"/>
            <text x="137" y="40" font-size="9" fill="#475569">Training</text>
            <rect x="200" y="30" width="12" height="12" fill="#10b981" rx="2"/>
            <text x="217" y="40" font-size="9" fill="#475569">Validation</text>

            <!-- Fold 1 -->
            <g transform="translate(50,55)">
              <text x="-10" y="12" font-size="9" fill="#64748b" text-anchor="end">Fold 1</text>
              <rect x="0" y="0" width="60" height="18" fill="#10b981" rx="2"/>
              <rect x="65" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="130" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="195" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="260" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <text x="340" y="12" font-size="9" fill="#64748b">→ Score₁</text>
            </g>

            <!-- Fold 2 -->
            <g transform="translate(50,80)">
              <text x="-10" y="12" font-size="9" fill="#64748b" text-anchor="end">Fold 2</text>
              <rect x="0" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="65" y="0" width="60" height="18" fill="#10b981" rx="2"/>
              <rect x="130" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="195" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="260" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <text x="340" y="12" font-size="9" fill="#64748b">→ Score₂</text>
            </g>

            <!-- Fold 3 -->
            <g transform="translate(50,105)">
              <text x="-10" y="12" font-size="9" fill="#64748b" text-anchor="end">Fold 3</text>
              <rect x="0" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="65" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="130" y="0" width="60" height="18" fill="#10b981" rx="2"/>
              <rect x="195" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="260" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <text x="340" y="12" font-size="9" fill="#64748b">→ Score₃</text>
            </g>

            <!-- Fold 4 -->
            <g transform="translate(50,130)">
              <text x="-10" y="12" font-size="9" fill="#64748b" text-anchor="end">Fold 4</text>
              <rect x="0" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="65" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="130" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="195" y="0" width="60" height="18" fill="#10b981" rx="2"/>
              <rect x="260" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <text x="340" y="12" font-size="9" fill="#64748b">→ Score₄</text>
            </g>

            <!-- Fold 5 -->
            <g transform="translate(50,155)">
              <text x="-10" y="12" font-size="9" fill="#64748b" text-anchor="end">Fold 5</text>
              <rect x="0" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="65" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="130" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="195" y="0" width="60" height="18" fill="#3b82f6" rx="2"/>
              <rect x="260" y="0" width="60" height="18" fill="#10b981" rx="2"/>
              <text x="340" y="12" font-size="9" fill="#64748b">→ Score₅</text>
            </g>

            <!-- Final score -->
            <text x="225" y="200" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">
              Final Score = Average(Score₁, Score₂, Score₃, Score₄, Score₅)
            </text>
          </svg>`,
          caption: 'Each fold serves as validation once while others train the model'
        },
        {
          type: 'heading',
          content: 'Stratified K-Fold'
        },
        {
          type: 'text',
          content: `For classification with imbalanced classes, regular K-Fold might create folds with different class distributions. **Stratified K-Fold** ensures each fold has the same class proportions as the full dataset.

**Example**: If 10% of samples are positive, each fold will have ~10% positive samples.

Always use stratified K-Fold for classification!`
        },
        {
          type: 'heading',
          content: 'Leave-One-Out Cross-Validation (LOOCV)'
        },
        {
          type: 'text',
          content: `The extreme case: K = n (number of samples). Each sample serves as the validation set once.

**Pros**:
- Uses almost all data for training each time
- No randomness - results are deterministic

**Cons**:
- Computationally expensive (n model trainings)
- High variance in the estimate

Use for small datasets where every sample matters.`
        },
        {
          type: 'heading',
          content: 'Time Series Cross-Validation'
        },
        {
          type: 'text',
          content: `Regular K-Fold doesn't work for time series - you can't train on future data to predict the past!

**Time Series Split**: Always train on past, validate on future. The training set grows over time.`
        },
        {
          type: 'visualization',
          title: 'Time Series Cross-Validation',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Time arrow -->
            <line x1="50" y1="160" x2="350" y2="160" stroke="#475569" stroke-width="1.5" marker-end="url(#timearrow)"/>
            <text x="200" y="175" text-anchor="middle" font-size="9" fill="#64748b">Time →</text>

            <!-- Split 1 -->
            <g transform="translate(50,30)">
              <text x="-10" y="10" font-size="9" fill="#64748b" text-anchor="end">Split 1</text>
              <rect x="0" y="0" width="60" height="16" fill="#3b82f6" rx="2"/>
              <rect x="65" y="0" width="30" height="16" fill="#10b981" rx="2"/>
              <rect x="100" y="0" width="200" height="16" fill="#e2e8f0" rx="2"/>
            </g>

            <!-- Split 2 -->
            <g transform="translate(50,55)">
              <text x="-10" y="10" font-size="9" fill="#64748b" text-anchor="end">Split 2</text>
              <rect x="0" y="0" width="90" height="16" fill="#3b82f6" rx="2"/>
              <rect x="95" y="0" width="30" height="16" fill="#10b981" rx="2"/>
              <rect x="130" y="0" width="170" height="16" fill="#e2e8f0" rx="2"/>
            </g>

            <!-- Split 3 -->
            <g transform="translate(50,80)">
              <text x="-10" y="10" font-size="9" fill="#64748b" text-anchor="end">Split 3</text>
              <rect x="0" y="0" width="120" height="16" fill="#3b82f6" rx="2"/>
              <rect x="125" y="0" width="30" height="16" fill="#10b981" rx="2"/>
              <rect x="160" y="0" width="140" height="16" fill="#e2e8f0" rx="2"/>
            </g>

            <!-- Split 4 -->
            <g transform="translate(50,105)">
              <text x="-10" y="10" font-size="9" fill="#64748b" text-anchor="end">Split 4</text>
              <rect x="0" y="0" width="150" height="16" fill="#3b82f6" rx="2"/>
              <rect x="155" y="0" width="30" height="16" fill="#10b981" rx="2"/>
              <rect x="190" y="0" width="110" height="16" fill="#e2e8f0" rx="2"/>
            </g>

            <!-- Legend -->
            <rect x="100" y="135" width="12" height="12" fill="#3b82f6" rx="2"/>
            <text x="117" y="145" font-size="9" fill="#475569">Train (past)</text>
            <rect x="190" y="135" width="12" height="12" fill="#10b981" rx="2"/>
            <text x="207" y="145" font-size="9" fill="#475569">Validate (future)</text>

            <defs>
              <marker id="timearrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Time series CV: training set expands, always predicting future from past'
        },
        {
          type: 'heading',
          content: 'Implementation'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold,
    LeaveOneOut, TimeSeriesSplit
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=1000, random_state=42)
model = RandomForestClassifier(random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(model, X, y, cv=kfold)
print(f"K-Fold: {scores_kfold.mean():.3f} (+/- {scores_kfold.std()*2:.3f})")

stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_strat = cross_val_score(model, X, y, cv=stratified)
print(f"Stratified: {scores_strat.mean():.3f} (+/- {scores_strat.std()*2:.3f})")

tscv = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(model, X, y, cv=tscv)
print(f"Time Series: {scores_ts.mean():.3f} (+/- {scores_ts.std()*2:.3f})")`
        },
        {
          type: 'heading',
          content: 'Choosing K'
        },
        {
          type: 'table',
          headers: ['K Value', 'Training Size', 'Variance', 'Computation'],
          rows: [
            ['K = 2', '50% data', 'High', 'Fast'],
            ['K = 5', '80% data', 'Moderate', 'Moderate'],
            ['K = 10', '90% data', 'Low', 'Slow'],
            ['K = n (LOOCV)', '99%+ data', 'Very High', 'Very Slow']
          ],
          caption: 'Trade-offs when selecting K'
        },
        {
          type: 'callout',
          variant: 'tip',
          content: 'K=5 or K=10 are good defaults. Use K=5 for large datasets (faster) and K=10 for smaller datasets (more reliable estimate).'
        },
        {
          type: 'keypoints',
          points: [
            'Cross-validation gives more reliable performance estimates than single splits',
            'K-Fold divides data into K parts, using each as validation once',
            'Always use Stratified K-Fold for classification',
            'Use Time Series Split when data has temporal ordering',
            'Report mean and standard deviation of cross-validation scores'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why should you use Stratified K-Fold for classification?',
          options: [
            'It is faster than regular K-Fold',
            'It ensures each fold has the same class distribution',
            'It uses more data for training',
            'It reduces model variance'
          ],
          correct: 1,
          explanation: 'Stratified K-Fold maintains the same class proportions in each fold as in the full dataset. This prevents folds with very different class distributions, which could lead to unreliable performance estimates.'
        },
        {
          type: 'multiple-choice',
          question: 'What is wrong with using regular K-Fold for time series data?',
          options: [
            'It is too slow',
            'It does not shuffle the data',
            'It allows training on future data to predict the past',
            'It uses too many folds'
          ],
          correct: 2,
          explanation: 'Regular K-Fold can put future data in the training set, which is unrealistic for time series prediction. Time Series Split ensures you always train on the past and validate on the future.'
        }
      ]
    },
    {
      id: 'hyperparameter-tuning',
      title: 'Hyperparameter Tuning',
      duration: '50 min',
      concepts: ['Grid Search', 'Random Search', 'Bayesian Optimization'],
      content: [
        {
          type: 'heading',
          content: 'Parameters vs Hyperparameters'
        },
        {
          type: 'text',
          content: `**Parameters**: Learned from data during training (weights, coefficients)
**Hyperparameters**: Set before training, control the learning process

Examples of hyperparameters:
- Learning rate
- Number of trees in a forest
- Maximum depth of trees
- Regularization strength
- Number of hidden layers

Finding good hyperparameters is crucial - the same model with different hyperparameters can perform very differently.`
        },
        {
          type: 'heading',
          content: 'Grid Search'
        },
        {
          type: 'text',
          content: `The simplest approach: try all combinations of hyperparameter values.

Define a grid of values for each hyperparameter, then exhaustively evaluate every combination using cross-validation.`
        },
        {
          type: 'visualization',
          title: 'Grid Search Visualization',
          svg: `<svg viewBox="0 0 350 250" xmlns="http://www.w3.org/2000/svg">
            <rect width="350" height="250" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="80" y1="200" x2="300" y2="200" stroke="#475569" stroke-width="1.5"/>
            <line x1="80" y1="200" x2="80" y2="50" stroke="#475569" stroke-width="1.5"/>

            <text x="190" y="235" text-anchor="middle" font-size="10" fill="#475569">max_depth</text>
            <text x="35" y="125" font-size="10" fill="#475569" transform="rotate(-90,35,125)">n_estimators</text>

            <!-- X-axis labels -->
            <text x="120" y="215" text-anchor="middle" font-size="9" fill="#64748b">3</text>
            <text x="180" y="215" text-anchor="middle" font-size="9" fill="#64748b">5</text>
            <text x="240" y="215" text-anchor="middle" font-size="9" fill="#64748b">7</text>

            <!-- Y-axis labels -->
            <text x="70" y="175" text-anchor="end" font-size="9" fill="#64748b">50</text>
            <text x="70" y="125" text-anchor="end" font-size="9" fill="#64748b">100</text>
            <text x="70" y="75" text-anchor="end" font-size="9" fill="#64748b">200</text>

            <!-- Grid points with scores -->
            <circle cx="120" cy="170" r="15" fill="#fecaca"/>
            <text x="120" y="174" text-anchor="middle" font-size="8" fill="#475569">.82</text>

            <circle cx="180" cy="170" r="15" fill="#fde68a"/>
            <text x="180" y="174" text-anchor="middle" font-size="8" fill="#475569">.85</text>

            <circle cx="240" cy="170" r="15" fill="#fde68a"/>
            <text x="240" y="174" text-anchor="middle" font-size="8" fill="#475569">.84</text>

            <circle cx="120" cy="120" r="15" fill="#fde68a"/>
            <text x="120" y="124" text-anchor="middle" font-size="8" fill="#475569">.86</text>

            <circle cx="180" cy="120" r="15" fill="#86efac"/>
            <text x="180" y="124" text-anchor="middle" font-size="8" fill="#475569">.89</text>

            <circle cx="240" cy="120" r="15" fill="#fde68a"/>
            <text x="240" y="124" text-anchor="middle" font-size="8" fill="#475569">.87</text>

            <circle cx="120" cy="70" r="15" fill="#fde68a"/>
            <text x="120" y="74" text-anchor="middle" font-size="8" fill="#475569">.85</text>

            <circle cx="180" cy="70" r="15" fill="#86efac" stroke="#10b981" stroke-width="3"/>
            <text x="180" y="74" text-anchor="middle" font-size="8" fill="#475569">.91</text>

            <circle cx="240" cy="70" r="15" fill="#86efac"/>
            <text x="240" y="74" text-anchor="middle" font-size="8" fill="#475569">.90</text>

            <!-- Best label -->
            <text x="180" y="45" text-anchor="middle" font-size="9" fill="#10b981" font-weight="600">Best!</text>
          </svg>`,
          caption: 'Grid search evaluates all 9 combinations, finds best at max_depth=5, n_estimators=200'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")`
        },
        {
          type: 'text',
          content: `**Problem with Grid Search**: Exhaustive search is expensive!

3 values × 4 values × 3 values = 36 combinations
× 5 folds = 180 model trainings

Adding more hyperparameters or values explodes exponentially.`
        },
        {
          type: 'heading',
          content: 'Random Search'
        },
        {
          type: 'text',
          content: `Instead of trying all combinations, sample random points from the hyperparameter space.

**Surprising fact**: Random search often finds solutions as good as grid search with far fewer iterations. Why?

Grid search wastes evaluations when some hyperparameters don't matter much. Random search explores more unique values.`
        },
        {
          type: 'visualization',
          title: 'Grid vs Random Search',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Grid Search -->
            <g transform="translate(30,20)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">Grid Search (9 points)</text>
              <rect x="10" y="10" width="120" height="120" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Grid points -->
              <circle cx="35" cy="35" r="5" fill="#3b82f6"/>
              <circle cx="70" cy="35" r="5" fill="#3b82f6"/>
              <circle cx="105" cy="35" r="5" fill="#3b82f6"/>
              <circle cx="35" cy="70" r="5" fill="#3b82f6"/>
              <circle cx="70" cy="70" r="5" fill="#3b82f6"/>
              <circle cx="105" cy="70" r="5" fill="#3b82f6"/>
              <circle cx="35" cy="105" r="5" fill="#3b82f6"/>
              <circle cx="70" cy="105" r="5" fill="#3b82f6"/>
              <circle cx="105" cy="105" r="5" fill="#3b82f6"/>

              <text x="70" y="150" text-anchor="middle" font-size="9" fill="#64748b">Only 3 unique values per axis</text>
            </g>

            <!-- Random Search -->
            <g transform="translate(220,20)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">Random Search (9 points)</text>
              <rect x="10" y="10" width="120" height="120" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Random points -->
              <circle cx="25" cy="45" r="5" fill="#10b981"/>
              <circle cx="55" cy="25" r="5" fill="#10b981"/>
              <circle cx="90" cy="60" r="5" fill="#10b981"/>
              <circle cx="40" cy="85" r="5" fill="#10b981"/>
              <circle cx="110" cy="35" r="5" fill="#10b981"/>
              <circle cx="75" cy="95" r="5" fill="#10b981"/>
              <circle cx="100" cy="110" r="5" fill="#10b981"/>
              <circle cx="30" cy="115" r="5" fill="#10b981"/>
              <circle cx="65" cy="55" r="5" fill="#10b981"/>

              <text x="70" y="150" text-anchor="middle" font-size="9" fill="#64748b">9 unique values per axis</text>
            </g>
          </svg>`,
          caption: 'Random search explores more unique values with the same number of evaluations'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")`
        },
        {
          type: 'heading',
          content: 'Bayesian Optimization'
        },
        {
          type: 'text',
          content: `Both grid and random search ignore results from previous evaluations. **Bayesian optimization** learns from previous evaluations to focus on promising regions.

It builds a probabilistic model of the objective function and uses it to choose the next point to evaluate. This is especially useful when:
- Evaluations are expensive (deep learning)
- The hyperparameter space is large
- You have a limited budget`
        },
        {
          type: 'code',
          language: 'python',
          content: `from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 20),
    'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0)
}

bayes_search = BayesSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)`
        },
        {
          type: 'heading',
          content: 'Best Practices'
        },
        {
          type: 'callout',
          variant: 'warning',
          content: 'Always use a separate test set that was never used during hyperparameter tuning. Cross-validation scores during tuning are optimistically biased!'
        },
        {
          type: 'table',
          headers: ['Method', 'When to Use', 'Budget'],
          rows: [
            ['Grid Search', 'Few hyperparameters, small search space', 'High'],
            ['Random Search', 'Many hyperparameters, uncertain ranges', 'Medium'],
            ['Bayesian', 'Expensive evaluations, complex spaces', 'Low']
          ],
          caption: 'Choosing a hyperparameter tuning strategy'
        },
        {
          type: 'keypoints',
          points: [
            'Hyperparameters control the learning process and must be set before training',
            'Grid search is exhaustive but expensive',
            'Random search is often more efficient than grid search',
            'Bayesian optimization uses previous results to guide the search',
            'Always evaluate final performance on a held-out test set'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why might random search outperform grid search with the same number of evaluations?',
          options: [
            'Random search is faster to compute',
            'Random search explores more unique values for each hyperparameter',
            'Random search uses cross-validation',
            'Random search finds the global optimum'
          ],
          correct: 1,
          explanation: 'Random search samples more unique values for each hyperparameter. If some hyperparameters matter more than others, this gives more chances to find good values for the important ones.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the main advantage of Bayesian optimization?',
          options: [
            'It tries all possible combinations',
            'It learns from previous evaluations to focus on promising regions',
            'It is the fastest method',
            'It requires no hyperparameters'
          ],
          correct: 1,
          explanation: 'Bayesian optimization builds a model of the objective function from previous evaluations. This allows it to intelligently choose which hyperparameters to try next, making it more sample-efficient.'
        }
      ]
    },
    {
      id: 'learning-curves',
      title: 'Learning Curves & Diagnostics',
      duration: '45 min',
      concepts: ['Learning Curves', 'Validation Curves', 'Diagnosing Problems'],
      content: [
        {
          type: 'heading',
          content: 'What Learning Curves Tell Us'
        },
        {
          type: 'text',
          content: `Learning curves plot model performance against training set size. They reveal fundamental issues with your model:

- Is the model underfitting?
- Is the model overfitting?
- Would more data help?
- Is there too much noise?

Reading learning curves is one of the most valuable diagnostic skills in ML.`
        },
        {
          type: 'heading',
          content: 'The Ideal Learning Curve'
        },
        {
          type: 'visualization',
          title: 'Learning Curve Patterns',
          svg: `<svg viewBox="0 0 450 400" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="400" fill="#f8fafc"/>

            <!-- Good Fit -->
            <g transform="translate(20,20)">
              <text x="90" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">Good Fit</text>
              <rect x="10" y="10" width="160" height="110" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="25" y1="100" x2="155" y2="100" stroke="#94a3b8" stroke-width="1"/>
              <line x1="25" y1="100" x2="25" y2="25" stroke="#94a3b8" stroke-width="1"/>

              <!-- Training curve -->
              <path d="M30,30 Q60,45 90,55 Q120,62 150,65" fill="none" stroke="#3b82f6" stroke-width="2"/>

              <!-- Validation curve -->
              <path d="M30,90 Q60,70 90,60 Q120,55 150,52" fill="none" stroke="#10b981" stroke-width="2"/>

              <!-- Convergence zone -->
              <rect x="130" y="50" width="25" height="18" fill="#d1fae5" opacity="0.5" rx="2"/>

              <text x="90" y="115" text-anchor="middle" font-size="8" fill="#64748b">Curves converge at high score</text>
            </g>

            <!-- High Bias (Underfitting) -->
            <g transform="translate(230,20)">
              <text x="90" y="0" text-anchor="middle" font-size="10" fill="#f59e0b" font-weight="600">Underfitting (High Bias)</text>
              <rect x="10" y="10" width="160" height="110" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="25" y1="100" x2="155" y2="100" stroke="#94a3b8" stroke-width="1"/>
              <line x1="25" y1="100" x2="25" y2="25" stroke="#94a3b8" stroke-width="1"/>

              <!-- Both curves plateau low -->
              <path d="M30,60 Q60,65 90,68 Q120,70 150,70" fill="none" stroke="#3b82f6" stroke-width="2"/>
              <path d="M30,85 Q60,75 90,72 Q120,71 150,71" fill="none" stroke="#10b981" stroke-width="2"/>

              <!-- Low zone -->
              <line x1="25" y1="45" x2="155" y2="45" stroke="#f59e0b" stroke-width="1" stroke-dasharray="3"/>
              <text x="160" y="48" font-size="7" fill="#f59e0b">Target</text>

              <text x="90" y="115" text-anchor="middle" font-size="8" fill="#64748b">Both curves plateau low</text>
            </g>

            <!-- High Variance (Overfitting) -->
            <g transform="translate(20,170)">
              <text x="90" y="0" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="600">Overfitting (High Variance)</text>
              <rect x="10" y="10" width="160" height="110" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="25" y1="100" x2="155" y2="100" stroke="#94a3b8" stroke-width="1"/>
              <line x1="25" y1="100" x2="25" y2="25" stroke="#94a3b8" stroke-width="1"/>

              <!-- Training very high -->
              <path d="M30,30 Q60,32 90,33 Q120,34 150,35" fill="none" stroke="#3b82f6" stroke-width="2"/>

              <!-- Validation much lower -->
              <path d="M30,90 Q60,75 90,65 Q120,60 150,55" fill="none" stroke="#10b981" stroke-width="2"/>

              <!-- Gap indicator -->
              <line x1="140" y1="35" x2="140" y2="55" stroke="#ef4444" stroke-width="1"/>
              <text x="145" y="47" font-size="7" fill="#ef4444">Gap!</text>

              <text x="90" y="115" text-anchor="middle" font-size="8" fill="#64748b">Large gap between curves</text>
            </g>

            <!-- More Data Needed -->
            <g transform="translate(230,170)">
              <text x="90" y="0" text-anchor="middle" font-size="10" fill="#8b5cf6" font-weight="600">Need More Data</text>
              <rect x="10" y="10" width="160" height="110" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Axes -->
              <line x1="25" y1="100" x2="155" y2="100" stroke="#94a3b8" stroke-width="1"/>
              <line x1="25" y1="100" x2="25" y2="25" stroke="#94a3b8" stroke-width="1"/>

              <!-- Curves still converging -->
              <path d="M30,30 Q60,38 90,44 Q120,48 150,50" fill="none" stroke="#3b82f6" stroke-width="2"/>
              <path d="M30,95 Q60,80 90,68 Q120,62 150,58" fill="none" stroke="#10b981" stroke-width="2"/>

              <!-- Trend arrows -->
              <path d="M150,50 L165,47" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="2"/>
              <path d="M150,58 L165,55" stroke="#10b981" stroke-width="1.5" stroke-dasharray="2"/>

              <text x="90" y="115" text-anchor="middle" font-size="8" fill="#64748b">Curves still converging</text>
            </g>

            <!-- Legend -->
            <g transform="translate(100,340)">
              <line x1="0" y1="0" x2="20" y2="0" stroke="#3b82f6" stroke-width="2"/>
              <text x="25" y="4" font-size="9" fill="#475569">Training Score</text>

              <line x1="120" y1="0" x2="140" y2="0" stroke="#10b981" stroke-width="2"/>
              <text x="145" y="4" font-size="9" fill="#475569">Validation Score</text>
            </g>
          </svg>`,
          caption: 'Different learning curve patterns reveal different model issues'
        },
        {
          type: 'heading',
          content: 'Interpreting the Patterns'
        },
        {
          type: 'text',
          content: `**Good Fit**: Both curves converge at a high score with a small gap. The model has learned the true patterns.

**Underfitting (High Bias)**: Both curves plateau at a low score. The model is too simple.
*Fix*: Use a more complex model, add features, reduce regularization

**Overfitting (High Variance)**: Training score is high, but validation score is much lower.
*Fix*: Get more data, use regularization, simplify the model, use dropout

**Need More Data**: Curves are still converging as data increases.
*Fix*: Collect more training data`
        },
        {
          type: 'heading',
          content: 'Validation Curves'
        },
        {
          type: 'text',
          content: `Validation curves plot performance against a hyperparameter value. They help you find the sweet spot - too little complexity underfits, too much overfits.`
        },
        {
          type: 'visualization',
          title: 'Validation Curve Example',
          svg: `<svg viewBox="0 0 380 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="380" height="220" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="60" y1="180" x2="340" y2="180" stroke="#475569" stroke-width="1.5"/>
            <line x1="60" y1="180" x2="60" y2="30" stroke="#475569" stroke-width="1.5"/>

            <text x="200" y="210" text-anchor="middle" font-size="10" fill="#475569">Model Complexity (e.g., max_depth)</text>
            <text x="25" y="105" font-size="10" fill="#475569" transform="rotate(-90,25,105)">Score</text>

            <!-- Regions -->
            <rect x="60" y="30" width="80" height="150" fill="#fef3c7" opacity="0.3"/>
            <rect x="140" y="30" width="100" height="150" fill="#d1fae5" opacity="0.3"/>
            <rect x="240" y="30" width="100" height="150" fill="#fee2e2" opacity="0.3"/>

            <text x="100" y="45" text-anchor="middle" font-size="8" fill="#f59e0b">Underfit</text>
            <text x="190" y="45" text-anchor="middle" font-size="8" fill="#10b981">Good</text>
            <text x="290" y="45" text-anchor="middle" font-size="8" fill="#ef4444">Overfit</text>

            <!-- Training curve -->
            <path d="M80,140 Q120,100 160,70 Q200,50 240,40 Q280,35 320,32"
                  fill="none" stroke="#3b82f6" stroke-width="2.5"/>

            <!-- Validation curve -->
            <path d="M80,150 Q120,110 160,80 Q200,70 240,85 Q280,110 320,140"
                  fill="none" stroke="#10b981" stroke-width="2.5"/>

            <!-- Optimal point -->
            <circle cx="190" cy="75" r="6" fill="#10b981" stroke="white" stroke-width="2"/>
            <text x="195" y="65" font-size="9" fill="#10b981">Optimal</text>

            <!-- Legend -->
            <line x1="100" y1="195" x2="120" y2="195" stroke="#3b82f6" stroke-width="2.5"/>
            <text x="125" y="198" font-size="9" fill="#475569">Train</text>
            <line x1="180" y1="195" x2="200" y2="195" stroke="#10b981" stroke-width="2.5"/>
            <text x="205" y="198" font-size="9" fill="#475569">Validation</text>
          </svg>`,
          caption: 'Find the complexity level where validation score peaks'
        },
        {
          type: 'heading',
          content: 'Creating Diagnostic Plots'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X, y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')`
        },
        {
          type: 'heading',
          content: 'Diagnostic Checklist'
        },
        {
          type: 'table',
          headers: ['Symptom', 'Diagnosis', 'Solutions'],
          rows: [
            ['Train high, val low', 'Overfitting', 'More data, regularization, simpler model'],
            ['Both low, converged', 'Underfitting', 'More complex model, more features'],
            ['Both improving', 'Need more data', 'Collect more training samples'],
            ['High variance in scores', 'Noisy data or small dataset', 'More data, robust model'],
            ['Val curve drops at high complexity', 'Overfitting', 'Tune regularization']
          ],
          caption: 'Common patterns and their solutions'
        },
        {
          type: 'keypoints',
          points: [
            'Learning curves plot performance vs training set size',
            'Large train-validation gap indicates overfitting',
            'Both curves low indicates underfitting',
            'Validation curves help tune hyperparameters',
            'Use these diagnostics before collecting more data or changing models'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'If both training and validation scores are low and converged, what should you do?',
          options: [
            'Collect more data',
            'Use a more complex model',
            'Increase regularization',
            'Use a smaller learning rate'
          ],
          correct: 1,
          explanation: 'When both curves plateau at a low score, the model is underfitting (high bias). It needs more capacity to learn the patterns, so you should try a more complex model or add more features.'
        },
        {
          type: 'multiple-choice',
          question: 'What does a large gap between training and validation curves indicate?',
          options: [
            'The model is underfitting',
            'The model is overfitting',
            'More data is needed',
            'The learning rate is too high'
          ],
          correct: 1,
          explanation: 'A large gap means the model performs much better on training data than validation data - classic overfitting. The model has memorized the training data but does not generalize well.'
        }
      ]
    },
    {
      id: 'model-selection',
      title: 'Model Selection Strategies',
      duration: '40 min',
      concepts: ['Train-Val-Test', 'Nested CV', 'Model Comparison'],
      content: [
        {
          type: 'heading',
          content: 'The Three-Way Split'
        },
        {
          type: 'text',
          content: `A common mistake: using the test set during model development. This leads to overly optimistic estimates because you've indirectly optimized for the test set.

The proper approach uses three separate sets:

**Training Set**: Fit model parameters
**Validation Set**: Tune hyperparameters, select model architecture
**Test Set**: Final, unbiased performance estimate (touch once at the very end!)`
        },
        {
          type: 'visualization',
          title: 'Proper Data Splitting',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Full dataset bar -->
            <rect x="30" y="30" width="340" height="30" fill="#e2e8f0" rx="4"/>
            <text x="200" y="50" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">Full Dataset</text>

            <!-- Split arrow -->
            <path d="M200,65 L200,80" stroke="#64748b" stroke-width="1.5"/>

            <!-- Three sets -->
            <rect x="30" y="85" width="180" height="30" fill="#3b82f6" rx="4"/>
            <text x="120" y="105" text-anchor="middle" font-size="10" fill="white" font-weight="500">Training (60%)</text>

            <rect x="215" y="85" width="80" height="30" fill="#10b981" rx="4"/>
            <text x="255" y="105" text-anchor="middle" font-size="10" fill="white" font-weight="500">Val (20%)</text>

            <rect x="300" y="85" width="70" height="30" fill="#ef4444" rx="4"/>
            <text x="335" y="105" text-anchor="middle" font-size="10" fill="white" font-weight="500">Test (20%)</text>

            <!-- Usage labels -->
            <text x="120" y="135" text-anchor="middle" font-size="8" fill="#3b82f6">Fit model</text>
            <text x="255" y="135" text-anchor="middle" font-size="8" fill="#10b981">Tune & select</text>
            <text x="335" y="135" text-anchor="middle" font-size="8" fill="#ef4444">Final eval</text>

            <!-- Warning -->
            <text x="335" y="155" text-anchor="middle" font-size="7" fill="#ef4444">⚠ Use ONCE</text>
          </svg>`,
          caption: 'Each set has a distinct purpose in the model development workflow'
        },
        {
          type: 'heading',
          content: 'Nested Cross-Validation'
        },
        {
          type: 'text',
          content: `When data is limited, a single validation set may give unreliable estimates. **Nested CV** provides an unbiased estimate even when doing hyperparameter tuning:

**Outer loop**: K-fold CV for performance estimation
**Inner loop**: K-fold CV for hyperparameter tuning (inside each outer fold)

This separates the data used for tuning from the data used for evaluation.`
        },
        {
          type: 'visualization',
          title: 'Nested Cross-Validation',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Outer loop label -->
            <text x="200" y="20" text-anchor="middle" font-size="11" fill="#475569" font-weight="600">Nested CV (5×3)</text>

            <!-- Outer fold visualization -->
            <g transform="translate(30,35)">
              <text x="-5" y="12" font-size="9" fill="#475569" text-anchor="end">Outer</text>

              <!-- 5 outer folds -->
              <rect x="0" y="0" width="60" height="16" fill="#ef4444" rx="2"/>
              <rect x="65" y="0" width="60" height="16" fill="#3b82f6" rx="2"/>
              <rect x="130" y="0" width="60" height="16" fill="#3b82f6" rx="2"/>
              <rect x="195" y="0" width="60" height="16" fill="#3b82f6" rx="2"/>
              <rect x="260" y="0" width="60" height="16" fill="#3b82f6" rx="2"/>

              <text x="30" y="30" font-size="7" fill="#ef4444" text-anchor="middle">Test</text>
              <text x="175" y="30" font-size="7" fill="#3b82f6" text-anchor="middle">Train (for inner CV)</text>
            </g>

            <!-- Inner loop -->
            <g transform="translate(95,75)">
              <text x="-65" y="30" font-size="9" fill="#475569" text-anchor="end">Inner</text>

              <!-- Inner folds (within training) -->
              <rect x="0" y="0" width="55" height="14" fill="#10b981" rx="2"/>
              <rect x="58" y="0" width="55" height="14" fill="#3b82f6" rx="2"/>
              <rect x="116" y="0" width="55" height="14" fill="#3b82f6" rx="2"/>

              <rect x="0" y="18" width="55" height="14" fill="#3b82f6" rx="2"/>
              <rect x="58" y="18" width="55" height="14" fill="#10b981" rx="2"/>
              <rect x="116" y="18" width="55" height="14" fill="#3b82f6" rx="2"/>

              <rect x="0" y="36" width="55" height="14" fill="#3b82f6" rx="2"/>
              <rect x="58" y="36" width="55" height="14" fill="#3b82f6" rx="2"/>
              <rect x="116" y="36" width="55" height="14" fill="#10b981" rx="2"/>

              <text x="200" y="28" font-size="8" fill="#64748b">→ Best hyperparams</text>
            </g>

            <!-- Flow -->
            <g transform="translate(30,150)">
              <rect x="0" y="0" width="100" height="20" fill="#e2e8f0" rx="4"/>
              <text x="50" y="14" text-anchor="middle" font-size="8" fill="#475569">Inner CV tunes</text>

              <path d="M105,10 L130,10" stroke="#64748b" stroke-width="1" marker-end="url(#flowarrow)"/>

              <rect x="135" y="0" width="100" height="20" fill="#e2e8f0" rx="4"/>
              <text x="185" y="14" text-anchor="middle" font-size="8" fill="#475569">Retrain on all inner</text>

              <path d="M240,10 L265,10" stroke="#64748b" stroke-width="1" marker-end="url(#flowarrow)"/>

              <rect x="270" y="0" width="100" height="20" fill="#e2e8f0" rx="4"/>
              <text x="320" y="14" text-anchor="middle" font-size="8" fill="#475569">Eval on outer test</text>
            </g>

            <defs>
              <marker id="flowarrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#64748b"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Inner CV tunes hyperparameters, outer CV estimates generalization'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100]}

nested_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    score = grid_search.score(X_test, y_test)
    nested_scores.append(score)

print(f"Nested CV Score: {np.mean(nested_scores):.3f} (+/- {np.std(nested_scores)*2:.3f})")`
        },
        {
          type: 'heading',
          content: 'Comparing Models Properly'
        },
        {
          type: 'text',
          content: `When comparing models, statistical significance matters. Two models with means 0.85 and 0.87 might not be meaningfully different if they have high variance.

**Paired t-test**: Compare fold-by-fold scores
**McNemar's test**: Compare predictions directly
**Confidence intervals**: If they overlap, difference may not be significant`
        },
        {
          type: 'code',
          language: 'python',
          content: `from scipy.stats import ttest_rel
import numpy as np

scores_model_a = cross_val_score(model_a, X, y, cv=10)
scores_model_b = cross_val_score(model_b, X, y, cv=10)

t_stat, p_value = ttest_rel(scores_model_a, scores_model_b)

print(f"Model A: {scores_model_a.mean():.3f} (+/- {scores_model_a.std()*2:.3f})")
print(f"Model B: {scores_model_b.mean():.3f} (+/- {scores_model_b.std()*2:.3f})")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
else:
    print("No significant difference")`
        },
        {
          type: 'heading',
          content: 'Model Selection Workflow'
        },
        {
          type: 'table',
          headers: ['Step', 'Action', 'Data Used'],
          rows: [
            ['1', 'Split data into train and test', 'All data'],
            ['2', 'Choose candidate models', 'Domain knowledge'],
            ['3', 'Tune each model with CV', 'Training only'],
            ['4', 'Compare best version of each', 'Training only'],
            ['5', 'Select final model', 'CV results'],
            ['6', 'Evaluate on test set ONCE', 'Test set']
          ],
          caption: 'Systematic model selection process'
        },
        {
          type: 'callout',
          variant: 'warning',
          content: 'The test set should only be used ONCE at the very end. If you use it multiple times, you risk overfitting to the test set and getting an overly optimistic estimate.'
        },
        {
          type: 'keypoints',
          points: [
            'Use train/validation/test split or nested CV',
            'Never touch the test set during model development',
            'Nested CV gives unbiased estimates when tuning hyperparameters',
            'Use statistical tests to compare models properly',
            'Report confidence intervals, not just point estimates'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the purpose of nested cross-validation?',
          options: [
            'To speed up training',
            'To get unbiased performance estimates while also tuning hyperparameters',
            'To reduce the need for a test set',
            'To increase model accuracy'
          ],
          correct: 1,
          explanation: 'Nested CV separates the data used for hyperparameter tuning (inner CV) from the data used for performance estimation (outer CV). This prevents optimistic bias from tuning on the same data you evaluate on.'
        },
        {
          type: 'multiple-choice',
          question: 'When should you evaluate on the test set?',
          options: [
            'After each hyperparameter change',
            'After each cross-validation fold',
            'Once, at the very end of model development',
            'Before starting model selection'
          ],
          correct: 2,
          explanation: 'The test set should only be used once, at the very end, to get an unbiased estimate of final model performance. Using it multiple times risks overfitting to the test set.'
        }
      ]
    }
  ]
}
