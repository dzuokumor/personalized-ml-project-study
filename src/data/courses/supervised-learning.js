export const supervisedlearning = {
  id: 'supervised-learning',
  title: 'Supervised Learning',
  description: 'Master all major supervised learning algorithms: from linear and logistic regression to decision trees, random forests, SVMs, and ensemble methods. Build models from scratch and understand how they work.',
  difficulty: 'intermediate',
  estimatedhours: 18,
  lessons: [
    {
      id: 'linear-regression-scratch',
      title: 'Linear Regression from Scratch',
      duration: '55 min',
      content: [
        {
          type: 'text',
          content: `Linear regression is the "hello world" of machine learning. It's simple enough to implement from scratch yet introduces all the core concepts: models, loss functions, and optimization.`
        },
        {
          type: 'heading',
          content: 'The Problem and Intuition'
        },
        {
          type: 'text',
          content: `**Given:** Data points (x, y) showing a relationship
**Goal:** Find the best-fit line y = mx + b

"Best fit" means the line that minimizes the total prediction error across all points.

**Why linear?** Many relationships are approximately linear, and even when they're not, linear models provide a useful baseline and are interpretable.`
        },
        {
          type: 'heading',
          content: 'The Model'
        },
        {
          type: 'text',
          content: `**Simple Linear Regression (one feature):**
ŷ = w₀ + w₁x

Where:
- ŷ is the predicted value
- w₀ is the bias (y-intercept)
- w₁ is the weight (slope)
- x is the input feature

**Multiple Linear Regression (multiple features):**
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

In vector form: ŷ = Xw where X includes a column of 1s for the bias.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # True: y = 4 + 3x + noise

# Visualize
plt.scatter(X, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Our Data')
plt.show()

# The goal: find w₀ and w₁ such that
# y ≈ w₀ + w₁*X
# True values: w₀ = 4, w₁ = 3`
        },
        {
          type: 'heading',
          content: 'The Loss Function'
        },
        {
          type: 'text',
          content: `How do we measure "best fit"? We need a **loss function** that quantifies error.

**Mean Squared Error (MSE):**

MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Why squared?
1. Makes all errors positive (can't cancel out)
2. Penalizes large errors more than small ones
3. Is differentiable everywhere (needed for optimization)
4. Has a closed-form solution for linear regression`
        },
        {
          type: 'code',
          language: 'python',
          content: `def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

# Example: measure loss for different lines
def predict(X, w0, w1):
    return w0 + w1 * X

# Try different parameters
for w0, w1 in [(0, 0), (4, 0), (4, 3), (3.5, 3.2)]:
    y_pred = predict(X, w0, w1)
    loss = mse_loss(y, y_pred)
    print(f"w0={w0}, w1={w1}: MSE = {loss:.4f}")

# Output:
# w0=0, w1=0: MSE = 29.42
# w0=4, w1=0: MSE = 10.23
# w0=4, w1=3: MSE = 0.97  # Close to true values!
# w0=3.5, w1=3.2: MSE = 1.04`
        },
        {
          type: 'heading',
          content: 'The Closed-Form Solution (Normal Equation)'
        },
        {
          type: 'text',
          content: `For linear regression, we can solve for the optimal weights directly using calculus.

**Derivation (simplified):**
1. We want to minimize MSE = ||y - Xw||²
2. Take derivative with respect to w, set to zero
3. Solve for w

**Result: w = (XᵀX)⁻¹Xᵀy**

This is called the **Normal Equation**.`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Add column of 1s for bias term
X_b = np.c_[np.ones((100, 1)), X]  # X_b = [1, X]

# Normal equation: w = (X^T X)^(-1) X^T y
w_optimal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"Optimal weights: w0 = {w_optimal[0,0]:.4f}, w1 = {w_optimal[1,0]:.4f}")
# Output: w0 = 4.2150, w1 = 2.7701
# Close to true values of 4 and 3!

# Predictions with optimal weights
y_pred = X_b @ w_optimal

# Visualize
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Best fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title(f'Linear Regression: y = {w_optimal[0,0]:.2f} + {w_optimal[1,0]:.2f}x')
plt.show()

print(f"Final MSE: {mse_loss(y, y_pred):.4f}")`
        },
        {
          type: 'heading',
          content: 'Implementing from Scratch'
        },
        {
          type: 'code',
          language: 'python',
          content: `class linearregression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """Fit using the normal equation."""
        # Add bias column
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Normal equation
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        """Make predictions."""
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]
        return X_b @ self.weights

    def score(self, X, y):
        """R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)

# Test our implementation
model = linearregression()
model.fit(X, y)

print(f"Weights: {model.weights.flatten()}")
print(f"R² Score: {model.score(X, y):.4f}")

# Compare with sklearn
from sklearn.linear_model import LinearRegression as SklearnLR
sklearn_model = SklearnLR()
sklearn_model.fit(X, y.ravel())
print(f"Sklearn: intercept={sklearn_model.intercept_:.4f}, coef={sklearn_model.coef_[0]:.4f}")`
        },
        {
          type: 'heading',
          content: 'Multiple Features'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Multiple linear regression: y = w0 + w1*x1 + w2*x2 + ...

# Generate data with multiple features
np.random.seed(42)
n_samples = 200
X_multi = np.random.randn(n_samples, 3)  # 3 features
true_weights = np.array([5, 2, -3, 1.5])  # [bias, w1, w2, w3]
y_multi = X_multi @ true_weights[1:] + true_weights[0] + np.random.randn(n_samples) * 0.5

# Fit
model = linearregression()
model.fit(X_multi, y_multi.reshape(-1, 1))

print("True weights:", true_weights)
print("Learned weights:", model.weights.flatten())

# Output:
# True weights: [ 5.   2.  -3.   1.5]
# Learned weights: [5.01 2.02 -2.98 1.49]  # Very close!`
        },
        {
          type: 'heading',
          content: 'Assumptions and Limitations'
        },
        {
          type: 'text',
          content: `**Linear regression assumes:**

1. **Linearity:** Relationship between X and y is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of errors
4. **Normality:** Errors are normally distributed (for inference)
5. **No multicollinearity:** Features are not highly correlated with each other

**Limitations:**
- Can't capture non-linear relationships (unless you add polynomial features)
- Sensitive to outliers (squared error)
- Assumes additive effects (no interactions unless explicitly added)

**When to use:**
- Baseline model (always try first)
- When interpretability matters (coefficients = feature importance)
- When relationship is approximately linear`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why do we use squared error in linear regression?',
          options: [
            'It makes errors smaller',
            'It penalizes large errors more, is differentiable, and has a closed-form solution',
            'It\'s faster to compute',
            'It\'s the only option'
          ],
          correct: 1,
          explanation: 'Squared error penalizes large errors more than small ones, is differentiable everywhere (needed for optimization), ensures positive errors, and allows for a closed-form solution via the normal equation.'
        },
        {
          type: 'multiple-choice',
          question: 'What does the normal equation w = (XᵀX)⁻¹Xᵀy compute?',
          options: [
            'The average of the data',
            'The optimal weights that minimize MSE',
            'The maximum likelihood estimate',
            'The ridge regression solution'
          ],
          correct: 1,
          explanation: 'The normal equation directly computes the weights that minimize the Mean Squared Error - it\'s the closed-form solution to the linear regression optimization problem.'
        }
      ]
    },
    {
      id: 'gradient-descent',
      title: 'Gradient Descent Deep Dive',
      duration: '60 min',
      content: [
        {
          type: 'text',
          content: `The normal equation works for linear regression, but what about more complex models? **Gradient descent** is the universal optimization algorithm that powers all of deep learning.`
        },
        {
          type: 'heading',
          content: 'The Intuition'
        },
        {
          type: 'text',
          content: `Imagine you're blindfolded on a hilly landscape and want to find the lowest point.

**Strategy:** Feel the slope under your feet, take a step downhill. Repeat.

This is gradient descent:
1. Compute the gradient (slope) of the loss function
2. Take a step in the direction that decreases the loss
3. Repeat until convergence

The **gradient** tells us the direction of steepest ascent, so we go the opposite direction (steepest descent).`
        },
        {
          type: 'heading',
          content: 'The Math'
        },
        {
          type: 'text',
          content: `**Update rule:**
w_new = w_old - η * ∇L(w)

Where:
- w is the weight vector
- η (eta) is the **learning rate** - size of each step
- ∇L(w) is the **gradient** of the loss with respect to w

**For MSE loss:**
L = (1/n) Σ(yᵢ - (w₀ + w₁xᵢ))²

**Gradient:**
∂L/∂w₀ = (-2/n) Σ(yᵢ - ŷᵢ)
∂L/∂w₁ = (-2/n) Σ(yᵢ - ŷᵢ) * xᵢ`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias column
X_b = np.c_[np.ones((100, 1)), X]

def compute_gradient(X_b, y, weights):
    """Compute gradient of MSE loss."""
    n = len(y)
    predictions = X_b @ weights
    errors = predictions - y
    gradient = (2/n) * X_b.T @ errors
    return gradient

def gradient_descent(X_b, y, learning_rate=0.1, n_iterations=1000):
    """Batch gradient descent."""
    n_features = X_b.shape[1]
    weights = np.random.randn(n_features, 1)  # Random initialization

    history = {'weights': [], 'loss': []}

    for i in range(n_iterations):
        # Compute gradient
        gradient = compute_gradient(X_b, y, weights)

        # Update weights
        weights = weights - learning_rate * gradient

        # Track progress
        loss = np.mean((X_b @ weights - y) ** 2)
        history['weights'].append(weights.copy())
        history['loss'].append(loss)

    return weights, history

# Run gradient descent
final_weights, history = gradient_descent(X_b, y, learning_rate=0.1, n_iterations=100)

print(f"Final weights: {final_weights.flatten()}")
print(f"Final loss: {history['loss'][-1]:.4f}")

# Plot loss over time
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Loss Over Training')

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.6)
y_pred = X_b @ final_weights
plt.plot(X, y_pred, 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Final Fit')
plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'Learning Rate: The Crucial Hyperparameter'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Learning rate affects convergence dramatically

learning_rates = [0.001, 0.01, 0.1, 0.5, 1.5]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, lr in zip(axes, learning_rates):
    _, history = gradient_descent(X_b, y, learning_rate=lr, n_iterations=50)
    ax.plot(history['loss'])
    ax.set_title(f'LR = {lr}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_ylim([0, max(history['loss'][:10])])

plt.tight_layout()
plt.show()

# Observations:
# lr=0.001: Very slow convergence
# lr=0.01:  Slow but steady
# lr=0.1:   Good convergence
# lr=0.5:   Fast but might oscillate
# lr=1.5:   Diverges! Loss explodes

# Rule of thumb: Start with 0.01 or 0.001, adjust based on behavior`
        },
        {
          type: 'heading',
          content: 'Types of Gradient Descent'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== BATCH GRADIENT DESCENT ==========
# Uses ALL data points to compute gradient
# Pros: Stable gradient, guaranteed convergence direction
# Cons: Slow for large datasets

def batch_gd(X, y, lr=0.01, epochs=100):
    weights = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
        gradient = (2/len(y)) * X.T @ (X @ weights - y)
        weights -= lr * gradient
    return weights

# ========== STOCHASTIC GRADIENT DESCENT (SGD) ==========
# Uses ONE random data point per update
# Pros: Fast updates, can escape local minima
# Cons: Noisy gradient, might not converge exactly

def sgd(X, y, lr=0.01, epochs=100):
    weights = np.zeros((X.shape[1], 1))
    n = len(y)
    for _ in range(epochs):
        for i in np.random.permutation(n):  # Random order
            xi = X[i:i+1]  # Single sample
            yi = y[i:i+1]
            gradient = 2 * xi.T @ (xi @ weights - yi)
            weights -= lr * gradient
    return weights

# ========== MINI-BATCH GRADIENT DESCENT ==========
# Uses small batches (e.g., 32, 64, 128 samples)
# Best of both worlds: stable enough, fast enough
# This is what everyone actually uses!

def minibatch_gd(X, y, lr=0.01, epochs=100, batch_size=32):
    weights = np.zeros((X.shape[1], 1))
    n = len(y)
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            gradient = (2/len(y_batch)) * X_batch.T @ (X_batch @ weights - y_batch)
            weights -= lr * gradient
    return weights

# Compare
batch_weights = batch_gd(X_b, y)
sgd_weights = sgd(X_b, y)
minibatch_weights = minibatch_gd(X_b, y)

print(f"Batch GD: {batch_weights.flatten()}")
print(f"SGD: {sgd_weights.flatten()}")
print(f"Mini-batch: {minibatch_weights.flatten()}")`
        },
        {
          type: 'heading',
          content: 'Advanced Optimizers'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ========== MOMENTUM ==========
# Accelerate in consistent direction, dampen oscillations
# Like a ball rolling downhill - builds up speed

def sgd_momentum(X, y, lr=0.01, momentum=0.9, epochs=100):
    weights = np.zeros((X.shape[1], 1))
    velocity = np.zeros_like(weights)

    for _ in range(epochs):
        gradient = (2/len(y)) * X.T @ (X @ weights - y)
        velocity = momentum * velocity - lr * gradient
        weights += velocity

    return weights

# ========== ADAGRAD ==========
# Adapts learning rate per parameter
# Parameters with large gradients get smaller updates

def adagrad(X, y, lr=0.1, epochs=100, epsilon=1e-8):
    weights = np.zeros((X.shape[1], 1))
    grad_squared_sum = np.zeros_like(weights)

    for _ in range(epochs):
        gradient = (2/len(y)) * X.T @ (X @ weights - y)
        grad_squared_sum += gradient ** 2
        weights -= (lr / (np.sqrt(grad_squared_sum) + epsilon)) * gradient

    return weights

# ========== ADAM ==========
# Combines momentum + adaptive learning rates
# The go-to optimizer for deep learning

def adam(X, y, lr=0.001, beta1=0.9, beta2=0.999, epochs=100, epsilon=1e-8):
    weights = np.zeros((X.shape[1], 1))
    m = np.zeros_like(weights)  # First moment (momentum)
    v = np.zeros_like(weights)  # Second moment (adaptive LR)

    for t in range(1, epochs + 1):
        gradient = (2/len(y)) * X.T @ (X @ weights - y)

        # Update moments
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update weights
        weights -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return weights

# Compare optimizers
print("Momentum:", sgd_momentum(X_b, y).flatten())
print("AdaGrad:", adagrad(X_b, y).flatten())
print("Adam:", adam(X_b, y, epochs=1000).flatten())`
        },
        {
          type: 'heading',
          content: 'Feature Scaling is Critical'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Why scaling matters for gradient descent

# Create unscaled data
X_unscaled = np.c_[
    np.random.randn(100) * 1000,   # Feature 1: large scale
    np.random.randn(100) * 0.01    # Feature 2: small scale
]
y_demo = 3 * X_unscaled[:, 0] + 500 * X_unscaled[:, 1] + np.random.randn(100)

# Without scaling: gradient descent struggles
# The loss landscape is elongated (different curvature in different directions)
# Large features dominate the gradient

# With scaling: all features on same scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

# Compare convergence
X_unscaled_b = np.c_[np.ones((100, 1)), X_unscaled]
X_scaled_b = np.c_[np.ones((100, 1)), X_scaled]

_, hist_unscaled = gradient_descent(X_unscaled_b, y_demo.reshape(-1,1), lr=0.0000001, n_iterations=100)
_, hist_scaled = gradient_descent(X_scaled_b, y_demo.reshape(-1,1), lr=0.1, n_iterations=100)

# Scaled version converges MUCH faster
print(f"Unscaled final loss: {hist_unscaled['loss'][-1]:.4f}")
print(f"Scaled final loss: {hist_scaled['loss'][-1]:.4f}")

# ALWAYS scale features before gradient descent!`
        },
        {
          type: 'text',
          content: `**Key Takeaways:**
1. Gradient descent is an iterative optimization algorithm
2. Learning rate is critical - too small is slow, too large diverges
3. Mini-batch gradient descent is the practical choice
4. Adam optimizer is the modern default
5. Always scale your features!`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What happens if the learning rate is too high?',
          options: [
            'Training is too slow',
            'The model underfits',
            'The loss oscillates or diverges (explodes)',
            'The model overfits'
          ],
          correct: 2,
          explanation: 'With a too-high learning rate, updates overshoot the minimum, causing the loss to oscillate wildly or even increase (diverge). The algorithm fails to converge.'
        },
        {
          type: 'multiple-choice',
          question: 'Why is mini-batch gradient descent preferred over batch or pure SGD?',
          options: [
            'It uses less memory',
            'It balances stable gradient estimates with computational efficiency',
            'It always finds the global minimum',
            'It doesn\'t require a learning rate'
          ],
          correct: 1,
          explanation: 'Mini-batch GD provides more stable gradients than single-sample SGD while being much faster than full-batch GD. It\'s the practical sweet spot used in modern deep learning.'
        }
      ]
    },
    {
      id: 'logistic-regression',
      title: 'Logistic Regression & Classification',
      duration: '55 min',
      content: [
        {
          type: 'text',
          content: `Despite its name, logistic regression is a **classification** algorithm. It's the linear model for predicting probabilities and class membership.`
        },
        {
          type: 'heading',
          content: 'From Regression to Classification'
        },
        {
          type: 'text',
          content: `**Problem:** Linear regression outputs any real number (-∞ to +∞)
**Need:** Probabilities (0 to 1)

**Solution:** Pass the linear output through the **sigmoid function**:

σ(z) = 1 / (1 + e^(-z))

This "squashes" any input to the range (0, 1).`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot sigmoid
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z))
plt.xlabel('z (linear output)')
plt.ylabel('σ(z) (probability)')
plt.title('Sigmoid Function')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.grid(True)

# Key properties:
# σ(0) = 0.5 (decision boundary)
# σ(large positive) → 1
# σ(large negative) → 0
# Smooth and differentiable everywhere

# Derivative (useful for gradient descent)
# dσ/dz = σ(z) * (1 - σ(z))
plt.subplot(1, 2, 2)
plt.plot(z, sigmoid(z) * (1 - sigmoid(z)))
plt.xlabel('z')
plt.ylabel("σ'(z)")
plt.title('Sigmoid Derivative')
plt.grid(True)
plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'The Logistic Regression Model'
        },
        {
          type: 'text',
          content: `**Model:**
P(y=1|x) = σ(w₀ + w₁x₁ + ... + wₙxₙ) = σ(wᵀx)

**Prediction:**
- If P(y=1|x) > 0.5: predict class 1
- If P(y=1|x) ≤ 0.5: predict class 0

The decision boundary is where wᵀx = 0 (linear in feature space).`
        },
        {
          type: 'heading',
          content: 'The Loss Function: Cross-Entropy'
        },
        {
          type: 'text',
          content: `We can't use MSE for classification - it doesn't work well with probabilities.

**Binary Cross-Entropy Loss:**
L = -1/n Σ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]

**Intuition:**
- When y=1: Loss = -log(p̂). Low loss when p̂ is high.
- When y=0: Loss = -log(1-p̂). Low loss when p̂ is low.
- Heavily penalizes confident wrong predictions.`
        },
        {
          type: 'code',
          language: 'python',
          content: `def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Binary cross-entropy loss."""
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Visualize why cross-entropy works better
p_pred = np.linspace(0.01, 0.99, 100)

plt.figure(figsize=(10, 4))

# When true label is 1
plt.subplot(1, 2, 1)
plt.plot(p_pred, -np.log(p_pred), label='Cross-entropy')
plt.plot(p_pred, (1 - p_pred)**2, label='Squared error')
plt.xlabel('Predicted probability')
plt.ylabel('Loss')
plt.title('True Label = 1')
plt.legend()

# When true label is 0
plt.subplot(1, 2, 2)
plt.plot(p_pred, -np.log(1 - p_pred), label='Cross-entropy')
plt.plot(p_pred, p_pred**2, label='Squared error')
plt.xlabel('Predicted probability')
plt.ylabel('Loss')
plt.title('True Label = 0')
plt.legend()

plt.tight_layout()
plt.show()

# Cross-entropy penalizes confident mistakes MORE severely
# This helps the model learn more effectively`
        },
        {
          type: 'heading',
          content: 'Implementing from Scratch'
        },
        {
          type: 'code',
          language: 'python',
          content: `class logisticregression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iter):
            # Forward pass
            linear = X @ self.weights + self.bias
            predictions = sigmoid(linear)

            # Compute gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = X @ self.weights + self.bias
        return sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Test on binary classification data
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42, n_clusters_per_class=1)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train our model
model = logisticregression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
print(f"Our accuracy: {model.score(X_test, y_test):.4f}")

# Compare with sklearn
from sklearn.linear_model import LogisticRegression as SklearnLR
sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)
print(f"Sklearn accuracy: {sklearn_model.score(X_test, y_test):.4f}")`
        },
        {
          type: 'heading',
          content: 'Visualizing the Decision Boundary'
        },
        {
          type: 'code',
          language: 'python',
          content: `def plot_decision_boundary(model, X, y):
    """Plot decision boundary for 2D data."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

plot_decision_boundary(model, X_test, y_test)

# The decision boundary is LINEAR
# This is logistic regression's main limitation`
        },
        {
          type: 'heading',
          content: 'Multi-Class Classification'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Logistic regression can be extended to multiple classes

# ========== ONE-VS-REST (OvR) ==========
# Train K binary classifiers, one per class
# For class k: "class k" vs "everything else"
# Predict class with highest probability

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target  # 3 classes

model = LogisticRegression(multi_class='ovr')  # One vs Rest
model.fit(X, y)
print(f"OvR Accuracy: {model.score(X, y):.4f}")
print(f"Probabilities shape: {model.predict_proba(X[:1]).shape}")  # (1, 3)

# ========== SOFTMAX (Multinomial) ==========
# Generalize sigmoid to multiple classes
# P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
# All probabilities sum to 1

model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_softmax.fit(X, y)
print(f"Softmax Accuracy: {model_softmax.score(X, y):.4f}")

# Softmax is generally preferred for >2 classes
# Probabilities are better calibrated`
        },
        {
          type: 'heading',
          content: 'Regularization'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Logistic regression can overfit, especially with many features
# Use regularization to constrain weights

# L2 regularization (Ridge): penalizes sum of squared weights
# L1 regularization (Lasso): penalizes sum of absolute weights (can zero out features)

# In sklearn, C = 1/λ (inverse of regularization strength)
# Smaller C = stronger regularization

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                          n_redundant=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compare different regularization strengths
for C in [0.001, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"C={C}: Train={train_acc:.3f}, Test={test_acc:.3f}")

# Low C (high regularization): might underfit
# High C (low regularization): might overfit`
        },
        {
          type: 'text',
          content: `**When to Use Logistic Regression:**
- Binary or multi-class classification
- When you need probabilities, not just predictions
- When interpretability matters (coefficients indicate feature importance)
- As a baseline before trying complex models
- When decision boundary is approximately linear`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why do we use cross-entropy loss instead of MSE for logistic regression?',
          options: [
            'Cross-entropy is faster to compute',
            'MSE doesn\'t work at all',
            'Cross-entropy penalizes confident wrong predictions more severely and has better gradient properties',
            'There\'s no difference'
          ],
          correct: 2,
          explanation: 'Cross-entropy loss heavily penalizes confident wrong predictions (predicting 0.99 when truth is 0), while MSE doesn\'t distinguish as effectively. This leads to better gradient flow and faster learning.'
        },
        {
          type: 'multiple-choice',
          question: 'What type of decision boundary does logistic regression create?',
          options: ['Curved', 'Linear', 'Circular', 'Random'],
          correct: 1,
          explanation: 'Logistic regression creates a linear decision boundary. The sigmoid function outputs probabilities, but the boundary where P=0.5 is a hyperplane (linear in feature space).'
        }
      ]
    },
    {
      id: 'decision-trees',
      title: 'Decision Trees',
      duration: '50 min',
      content: [
        {
          type: 'text',
          content: `Decision trees make predictions by learning a sequence of if-then rules. They're intuitive, interpretable, and the foundation of powerful ensemble methods.`
        },
        {
          type: 'heading',
          content: 'The Intuition'
        },
        {
          type: 'text',
          content: `**How humans make decisions:**
"Should I play tennis today?"
- Is it raining? → If yes, don't play
- Is it too hot? → If yes, don't play
- Otherwise → Play!

**Decision trees learn these rules from data automatically.**

Each internal node tests a feature, each branch represents an outcome, and each leaf gives a prediction.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train a decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree for Iris Classification')
plt.show()

# How to read the tree:
# - Each box shows: split condition, impurity, samples, class distribution
# - Left branch: condition is True
# - Right branch: condition is False
# - Color indicates majority class`
        },
        {
          type: 'heading',
          content: 'How Splits Are Chosen'
        },
        {
          type: 'text',
          content: `The algorithm recursively splits data to maximize "purity" - making each subset as homogeneous as possible.

**Splitting Criteria:**

**Gini Impurity:**
Gini = 1 - Σ pᵢ²

Measures probability of misclassifying a randomly chosen sample.
- Gini = 0: pure node (all same class)
- Gini = 0.5: maximum impurity for binary (50-50 split)

**Entropy (Information Gain):**
Entropy = -Σ pᵢ log₂(pᵢ)

Measures uncertainty/disorder.
- Entropy = 0: pure node
- Entropy = 1: maximum uncertainty for binary`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

def gini_impurity(y):
    """Calculate Gini impurity."""
    if len(y) == 0:
        return 0
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum(proportions ** 2)

def entropy(y):
    """Calculate entropy."""
    if len(y) == 0:
        return 0
    proportions = np.bincount(y) / len(y)
    proportions = proportions[proportions > 0]  # Avoid log(0)
    return -np.sum(proportions * np.log2(proportions))

def information_gain(y, left_y, right_y, criterion='gini'):
    """Calculate information gain from a split."""
    if criterion == 'gini':
        func = gini_impurity
    else:
        func = entropy

    parent_impurity = func(y)
    n = len(y)
    n_left, n_right = len(left_y), len(right_y)

    child_impurity = (n_left/n) * func(left_y) + (n_right/n) * func(right_y)
    return parent_impurity - child_impurity

# Example: Which split is better?
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 50-50 split

# Split A: 4 samples each side
split_a_left = np.array([0, 0, 0, 0])
split_a_right = np.array([1, 1, 1, 1])
print(f"Split A gain: {information_gain(y, split_a_left, split_a_right):.3f}")

# Split B: Imperfect split
split_b_left = np.array([0, 0, 0, 1])
split_b_right = np.array([0, 1, 1, 1])
print(f"Split B gain: {information_gain(y, split_b_left, split_b_right):.3f}")

# Split A is better (more information gain)`
        },
        {
          type: 'heading',
          content: 'The Splitting Algorithm'
        },
        {
          type: 'code',
          language: 'python',
          content: `def find_best_split(X, y):
    """Find the best feature and threshold to split on."""
    best_gain = 0
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        thresholds = np.unique(feature_values)

        for threshold in thresholds:
            # Split data
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            # Calculate gain
            gain = information_gain(y, y[left_mask], y[right_mask])

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

# The full algorithm:
# 1. Find best split for current node
# 2. If gain > 0 and stopping criteria not met:
#    - Split data
#    - Recursively build left and right subtrees
# 3. Otherwise, create leaf node with majority class`
        },
        {
          type: 'heading',
          content: 'Controlling Tree Complexity'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Decision trees easily overfit - they can perfectly memorize training data
# Control complexity with hyperparameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Unrestricted tree - likely overfitting
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)
print(f"Full tree - Train: {tree_full.score(X_train, y_train):.3f}, Test: {tree_full.score(X_test, y_test):.3f}")
print(f"  Depth: {tree_full.get_depth()}, Leaves: {tree_full.get_n_leaves()}")

# Hyperparameters for control:

# max_depth: Maximum depth of tree
tree_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_depth.fit(X_train, y_train)
print(f"max_depth=3 - Train: {tree_depth.score(X_train, y_train):.3f}, Test: {tree_depth.score(X_test, y_test):.3f}")

# min_samples_split: Minimum samples required to split a node
tree_split = DecisionTreeClassifier(min_samples_split=10, random_state=42)
tree_split.fit(X_train, y_train)
print(f"min_samples_split=10 - Train: {tree_split.score(X_train, y_train):.3f}, Test: {tree_split.score(X_test, y_test):.3f}")

# min_samples_leaf: Minimum samples required at a leaf
tree_leaf = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_leaf.fit(X_train, y_train)
print(f"min_samples_leaf=5 - Train: {tree_leaf.score(X_train, y_train):.3f}, Test: {tree_leaf.score(X_test, y_test):.3f}")

# max_features: Number of features to consider for each split
tree_features = DecisionTreeClassifier(max_features='sqrt', random_state=42)
tree_features.fit(X_train, y_train)
print(f"max_features='sqrt' - Train: {tree_features.score(X_train, y_train):.3f}, Test: {tree_features.score(X_test, y_test):.3f}")`
        },
        {
          type: 'heading',
          content: 'Decision Trees for Regression'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.1

# Fit regression trees with different depths
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
X_test = np.linspace(0, 5, 500).reshape(-1, 1)

for ax, depth in zip(axes, [2, 5, 10]):
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X, y)
    y_pred = tree.predict(X_test)

    ax.scatter(X, y, s=20, edgecolor='black', label='Data')
    ax.plot(X_test, y_pred, 'r-', label='Prediction')
    ax.set_title(f'Depth = {depth}')
    ax.legend()

plt.tight_layout()
plt.show()

# For regression:
# - Split criterion: minimize variance (MSE)
# - Leaf prediction: mean of samples in leaf
# - Still uses recursive splitting`
        },
        {
          type: 'heading',
          content: 'Feature Importance'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Trees provide natural feature importance

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X, y)

# Feature importance = total reduction in impurity from splits on that feature
importances = tree.feature_importances_

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances)
plt.yticks(range(len(importances)), iris.feature_names)
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importances')
plt.show()

# Interpretation:
# Higher = more important for making decisions
# Can use for feature selection`
        },
        {
          type: 'text',
          content: `**Pros of Decision Trees:**
- Highly interpretable (visualize the rules)
- No feature scaling needed
- Handle non-linear relationships
- Handle mixed data types
- Fast prediction (just follow the path)

**Cons:**
- Prone to overfitting
- High variance (small data changes can create very different trees)
- Greedy algorithm (locally optimal, not globally)
- Can create biased trees if classes are imbalanced`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does a Gini impurity of 0 mean?',
          options: [
            'Maximum impurity',
            'The node is completely pure (all samples same class)',
            'The node is useless',
            '50-50 class split'
          ],
          correct: 1,
          explanation: 'Gini = 0 means all samples in the node belong to the same class - the node is perfectly pure. This is what we want at leaf nodes.'
        },
        {
          type: 'multiple-choice',
          question: 'Which hyperparameter limits tree depth to prevent overfitting?',
          options: ['min_samples_leaf', 'criterion', 'max_depth', 'splitter'],
          correct: 2,
          explanation: 'max_depth directly limits how deep the tree can grow. Shallower trees are simpler and less prone to overfitting.'
        }
      ]
    },
    {
      id: 'random-forests',
      title: 'Random Forests',
      duration: '50 min',
      content: [
        {
          type: 'text',
          content: `Random Forests combine many decision trees to create a model that's more accurate and less prone to overfitting. It's one of the most successful and widely-used ML algorithms.`
        },
        {
          type: 'heading',
          content: 'The Ensemble Idea'
        },
        {
          type: 'text',
          content: `**"Wisdom of crowds"** - Many weak opinions combined often outperform a single expert.

**For ML:** Train multiple models, combine their predictions.
- Each model makes different errors
- Errors tend to cancel out when averaged
- Result: better accuracy and lower variance

**Random Forest = Many Decision Trees + Randomness**`
        },
        {
          type: 'heading',
          content: 'How Random Forests Work'
        },
        {
          type: 'text',
          content: `**Two sources of randomness:**

1. **Bootstrap Sampling (Bagging):**
   - Each tree is trained on a random sample of the data (with replacement)
   - ~63% of data is used; rest is "out-of-bag" (OOB)

2. **Random Feature Selection:**
   - At each split, only consider a random subset of features
   - Typically sqrt(n_features) for classification
   - This decorrelates the trees

**Final Prediction:**
- Classification: Majority vote
- Regression: Average of predictions`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single decision tree (high variance)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print(f"Single tree - Train: {tree.score(X_train, y_train):.3f}, Test: {tree.score(X_test, y_test):.3f}")

# Random Forest (reduced variance)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest - Train: {rf.score(X_train, y_train):.3f}, Test: {rf.score(X_test, y_test):.3f}")

# The gap between train and test is much smaller for RF!`
        },
        {
          type: 'heading',
          content: 'Why Random Forests Are Better'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Visualize variance reduction

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

n_trees_range = [1, 5, 10, 25, 50, 100, 200]
test_scores = []
test_stds = []

for n_trees in n_trees_range:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5)
    test_scores.append(scores.mean())
    test_stds.append(scores.std())

plt.figure(figsize=(10, 6))
plt.errorbar(n_trees_range, test_scores, yerr=test_stds, marker='o', capsize=5)
plt.xlabel('Number of Trees')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Random Forest: Accuracy vs Number of Trees')
plt.xscale('log')
plt.grid(True)
plt.show()

# Observations:
# - Accuracy increases with more trees (up to a point)
# - Variance (error bars) decreases with more trees
# - After ~100 trees, diminishing returns`
        },
        {
          type: 'heading',
          content: 'Hyperparameters'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import GridSearchCV

# Key hyperparameters:
param_grid = {
    # Number of trees - more is usually better, but slower
    'n_estimators': [100, 200, 300],

    # Max depth - controls tree complexity
    'max_depth': [None, 10, 20, 30],

    # Features to consider at each split
    # 'sqrt', 'log2', or fraction
    'max_features': ['sqrt', 'log2', 0.5],

    # Minimum samples to split
    'min_samples_split': [2, 5, 10],

    # Minimum samples at leaf
    'min_samples_leaf': [1, 2, 4]
}

# For demonstration, use smaller grid
small_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                          small_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")`
        },
        {
          type: 'heading',
          content: 'Out-of-Bag (OOB) Score'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Each tree is trained on ~63% of data (bootstrap sample)
# The other ~37% can be used for validation - "out-of-bag"
# Free validation without needing a separate validation set!

rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.4f}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.4f}")

# OOB score is usually a good estimate of test performance
# Useful for hyperparameter tuning without cross-validation`
        },
        {
          type: 'heading',
          content: 'Feature Importance'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Random Forests aggregate feature importance across all trees
# More stable than single tree importance

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

# Sort by importance
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.show()

# Print top 5 features
print("Top 5 features:")
for i in range(5):
    print(f"  Feature {indices[i]}: {importances[indices[i]]:.4f} (+/- {std[indices[i]]:.4f})")`
        },
        {
          type: 'heading',
          content: 'Random Forest vs Single Tree'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Compare decision boundaries
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=200, noise=0.3, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, model, title in [
    (axes[0], DecisionTreeClassifier(random_state=42), 'Single Decision Tree'),
    (axes[1], RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest')
]:
    model.fit(X_moons, y_moons)

    # Plot decision boundary
    x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'{title}\\nAccuracy: {model.score(X_moons, y_moons):.3f}')

plt.tight_layout()
plt.show()

# Random Forest has smoother boundaries (averaging effect)`
        },
        {
          type: 'text',
          content: `**When to Use Random Forests:**
- Default choice for tabular data
- When you need good accuracy without much tuning
- When feature importance matters
- When interpretability is secondary to accuracy

**Pros:**
- Excellent accuracy out-of-the-box
- Robust to overfitting
- Handles high-dimensional data well
- No feature scaling needed
- Built-in feature importance

**Cons:**
- Slower than single trees
- Less interpretable than single tree
- Can be memory-intensive for large datasets`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What are the two sources of randomness in Random Forests?',
          options: [
            'Random weights and random features',
            'Bootstrap sampling and random feature selection at splits',
            'Random learning rate and random depth',
            'Random labels and random samples'
          ],
          correct: 1,
          explanation: 'Random Forests use (1) bootstrap sampling to train each tree on different data subsets, and (2) random feature selection at each split to decorrelate the trees.'
        },
        {
          type: 'multiple-choice',
          question: 'Why do Random Forests have lower variance than single decision trees?',
          options: [
            'They use simpler trees',
            'Averaging many different trees cancels out individual errors',
            'They use regularization',
            'They train faster'
          ],
          correct: 1,
          explanation: 'Each tree makes different errors due to randomness. When we average (or vote on) many trees, these errors tend to cancel out, reducing overall variance.'
        }
      ]
    },
    {
      id: 'support-vector-machines',
      title: 'Support Vector Machines',
      duration: '55 min',
      content: [
        {
          type: 'text',
          content: `Support Vector Machines (SVMs) find the optimal boundary between classes by maximizing the margin. They're powerful for both linear and non-linear classification.`
        },
        {
          type: 'heading',
          content: 'The Intuition: Maximum Margin'
        },
        {
          type: 'text',
          content: `**Many hyperplanes can separate two classes. Which is best?**

SVM chooses the hyperplane that maximizes the **margin** - the distance to the nearest points from each class.

**Why maximum margin?**
- More robust to new data
- Better generalization
- Points far from boundary are classified more confidently

**Support Vectors:** The closest points to the boundary that define the margin.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Create linearly separable data
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Train SVM
svm = SVC(kernel='linear', C=1000)  # High C for hard margin
svm.fit(X, y)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))

# Plot data
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)

# Plot decision boundary and margins
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                     np.linspace(ylim[0], ylim[1], 200))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Decision boundary, margins, and decision function
ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'],
           colors=['blue', 'black', 'red'])

# Highlight support vectors
ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
           s=200, linewidth=2, facecolors='none', edgecolors='green',
           label='Support Vectors')

ax.legend()
ax.set_title('SVM: Maximum Margin Classifier')
plt.show()

print(f"Number of support vectors: {len(svm.support_vectors_)}")`
        },
        {
          type: 'heading',
          content: 'The Math (Simplified)'
        },
        {
          type: 'text',
          content: `**Decision boundary:** wᵀx + b = 0

**Margins:** wᵀx + b = +1 and wᵀx + b = -1

**Margin width:** 2/||w||

**Optimization problem:**
Minimize ||w||² (equivalently, maximize margin)
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i

This is a convex optimization problem with a unique solution.`
        },
        {
          type: 'heading',
          content: 'Soft Margin: Handling Non-Separable Data'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Real data is often not perfectly separable
# Soft margin SVM allows some misclassifications

# C parameter controls the tradeoff:
# - Small C: Allow more margin violations (soft margin)
# - Large C: Fewer violations allowed (approaches hard margin)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
C_values = [0.01, 1, 100]

for ax, C in zip(axes, C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)

    # Decision boundary
    xlim = ax.get_xlim() if ax.get_xlim()[0] < ax.get_xlim()[1] else X[:, 0].min() - 1, X[:, 0].max() + 1
    ylim = ax.get_ylim() if ax.get_ylim()[0] < ax.get_ylim()[1] else X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='green', linewidths=2)
    ax.set_title(f'C = {C}\\n{len(svm.support_)} support vectors')

plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'The Kernel Trick'
        },
        {
          type: 'text',
          content: `**Problem:** Linear SVM can only create linear boundaries.
**Solution:** Map data to higher dimensions where it becomes linearly separable.

**The Kernel Trick:** Instead of explicitly computing the transformation, use a kernel function that computes inner products in the transformed space directly.

**Common Kernels:**
- Linear: K(x, x') = xᵀx'
- Polynomial: K(x, x') = (γxᵀx' + r)^d
- RBF (Gaussian): K(x, x') = exp(-γ||x - x'||²)
- Sigmoid: K(x, x') = tanh(γxᵀx' + r)`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.datasets import make_circles

# Create non-linearly separable data
X_circles, y_circles = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for ax, kernel in zip(axes, kernels):
    svm = SVC(kernel=kernel, C=1.0)
    svm.fit(X_circles, y_circles)

    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200),
                         np.linspace(-1.5, 1.5, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'{kernel.upper()} kernel\\nAccuracy: {svm.score(X_circles, y_circles):.3f}')

plt.tight_layout()
plt.show()

# Linear kernel fails (data not linearly separable)
# RBF handles it perfectly!`
        },
        {
          type: 'heading',
          content: 'RBF Kernel: The Most Popular'
        },
        {
          type: 'code',
          language: 'python',
          content: `# RBF kernel: K(x, x') = exp(-gamma * ||x - x'||^2)
# gamma controls the "reach" of each point

# High gamma: Each point has small influence (complex, wiggly boundary)
# Low gamma: Each point has large influence (smooth boundary)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
gamma_values = [0.1, 1, 10, 100]

for ax, gamma in zip(axes, gamma_values):
    svm = SVC(kernel='rbf', gamma=gamma, C=1.0)
    svm.fit(X_circles, y_circles)

    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200),
                         np.linspace(-1.5, 1.5, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'gamma = {gamma}')

plt.tight_layout()
plt.show()

# Typically tune C and gamma together via grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_circles, y_circles)
print(f"Best params: {grid.best_params_}")`
        },
        {
          type: 'heading',
          content: 'Scaling is Essential'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SVMs are sensitive to feature scales
# The distance calculations are affected by scale

# BAD: Without scaling
svm_unscaled = SVC(kernel='rbf')
svm_unscaled.fit(X, y)
print(f"Without scaling: {svm_unscaled.score(X, y):.3f}")

# GOOD: With scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
pipeline.fit(X, y)
print(f"With scaling: {pipeline.score(X, y):.3f}")

# ALWAYS scale features for SVM!`
        },
        {
          type: 'text',
          content: `**When to Use SVM:**
- Medium-sized datasets (slow for very large data)
- High-dimensional data (handles well)
- When clear margin of separation exists
- Both linear and non-linear problems

**Pros:**
- Effective in high dimensions
- Memory efficient (only stores support vectors)
- Versatile kernel options
- Robust to overfitting (especially in high dimensions)

**Cons:**
- Slow for large datasets (O(n²) to O(n³))
- Requires feature scaling
- No probability estimates (by default)
- Sensitive to C and gamma choices`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What do support vectors represent?',
          options: [
            'All training points',
            'The points closest to the decision boundary that define the margin',
            'Outliers in the data',
            'The centroids of each class'
          ],
          correct: 1,
          explanation: 'Support vectors are the training points that lie on the margin boundaries. They are the most critical points that define the decision boundary - removing them would change the solution.'
        },
        {
          type: 'multiple-choice',
          question: 'What does the RBF kernel allow SVMs to do?',
          options: [
            'Train faster',
            'Handle more data',
            'Create non-linear decision boundaries',
            'Avoid overfitting'
          ],
          correct: 2,
          explanation: 'The RBF kernel implicitly maps data to a higher-dimensional space where it may become linearly separable, allowing SVMs to create non-linear boundaries in the original space.'
        }
      ]
    },
    {
      id: 'knn-naive-bayes',
      title: 'KNN and Naive Bayes',
      duration: '45 min',
      content: [
        {
          type: 'text',
          content: `Two fundamentally different approaches to classification: K-Nearest Neighbors (instance-based) and Naive Bayes (probabilistic). Both are simple yet surprisingly effective.`
        },
        {
          type: 'heading',
          content: 'K-Nearest Neighbors (KNN)'
        },
        {
          type: 'text',
          content: `**The simplest ML algorithm:** To classify a new point, find the K closest training points and vote.

**No training phase!** KNN is a "lazy learner" - it stores all training data and does computation at prediction time.

**Key idea:** Similar points have similar labels.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # Use first 2 features for visualization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Effect of K
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
k_values = [1, 5, 15, 50]

for ax, k in zip(axes, k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'K = {k}\\nAccuracy: {knn.score(X_test, y_test):.3f}')

plt.tight_layout()
plt.show()

# K=1: Very complex boundary (overfitting)
# K=50: Very smooth boundary (underfitting)
# K=5 or 15: Good balance`
        },
        {
          type: 'heading',
          content: 'KNN Distance Metrics'
        },
        {
          type: 'code',
          language: 'python',
          content: `# KNN depends on "distance" between points
# Common distance metrics:

# Euclidean (L2): sqrt(sum((x-y)^2))
# Manhattan (L1): sum(|x-y|)
# Minkowski: generalization of L1 and L2

from sklearn.neighbors import KNeighborsClassifier

# Compare metrics
for metric in ['euclidean', 'manhattan', 'minkowski']:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    print(f"{metric}: {knn.score(X_test, y_test):.3f}")

# Weighted KNN: Closer neighbors have more influence
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train, y_train)
print(f"Weighted: {knn_weighted.score(X_test, y_test):.3f}")

# IMPORTANT: Scale features for KNN!
# Distance is affected by scale
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipeline.fit(X_train, y_train)
print(f"With scaling: {pipeline.score(X_test, y_test):.3f}")`
        },
        {
          type: 'heading',
          content: 'Naive Bayes'
        },
        {
          type: 'text',
          content: `**A probabilistic classifier using Bayes' theorem:**

P(y|X) = P(X|y) * P(y) / P(X)

**The "Naive" assumption:** Features are conditionally independent given the class.

P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)

This assumption is often wrong but Naive Bayes still works well in practice!`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Different Naive Bayes variants for different data types

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# ========== GAUSSIAN NAIVE BAYES ==========
# For continuous features
# Assumes features follow Gaussian distribution within each class

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Gaussian NB: {gnb.score(X_test, y_test):.3f}")

# ========== MULTINOMIAL NAIVE BAYES ==========
# For count data (e.g., word counts in text)
# Great for text classification!

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Text classification example
categories = ['sci.space', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
X_text = newsgroups.data
y_text = newsgroups.target

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X_text)

mnb = MultinomialNB()
mnb.fit(X_counts, y_text)
print(f"Multinomial NB for text: {mnb.score(X_counts, y_text):.3f}")

# ========== BERNOULLI NAIVE BAYES ==========
# For binary features (presence/absence)
bnb = BernoulliNB()
# Convert to binary (presence of word)
X_binary = (X_counts > 0).astype(int)
bnb.fit(X_binary, y_text)
print(f"Bernoulli NB: {bnb.score(X_binary, y_text):.3f}")`
        },
        {
          type: 'heading',
          content: 'Why Naive Bayes Works'
        },
        {
          type: 'text',
          content: `Despite the "naive" independence assumption being almost always violated:

1. **We only need ranking:** For classification, we don't need exact probabilities - just which class has highest probability

2. **Errors can cancel out:** Dependency violations in opposite directions may balance out

3. **Simple model = low variance:** The simple model is less prone to overfitting

4. **Works well with high dimensions:** When features >> samples, simpler models are better

**Best use cases:**
- Text classification (spam detection, sentiment analysis)
- Real-time prediction (very fast)
- When training data is limited
- As a baseline model`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Spam classification example (classic Naive Bayes application)

# Simulated email data
emails = [
    ("Win free lottery money now!", 1),  # spam
    ("Meeting at 3pm tomorrow", 0),       # not spam
    ("You've won a prize!", 1),
    ("Project deadline extended", 0),
    ("Free viagra pills", 1),
    ("Lunch plans for today?", 0),
    ("Congratulations winner!", 1),
    ("Quarterly report attached", 0)
]

X_emails = [e[0] for e in emails]
y_emails = [e[1] for e in emails]

# Simple pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

spam_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

spam_classifier.fit(X_emails, y_emails)

# Test on new emails
test_emails = [
    "Free money waiting for you!",
    "Can we schedule a call tomorrow?"
]

predictions = spam_classifier.predict(test_emails)
probas = spam_classifier.predict_proba(test_emails)

for email, pred, proba in zip(test_emails, predictions, probas):
    print(f"'{email[:30]}...'")
    print(f"  Prediction: {'SPAM' if pred else 'NOT SPAM'}")
    print(f"  Confidence: {max(proba):.2%}")`
        },
        {
          type: 'heading',
          content: 'KNN vs Naive Bayes Comparison'
        },
        {
          type: 'text',
          content: `| Aspect | KNN | Naive Bayes |
|--------|-----|-------------|
| Training | None (lazy) | Fast (learn distributions) |
| Prediction | Slow (compute distances) | Very fast |
| Memory | Stores all data | Stores statistics |
| Assumptions | None | Feature independence |
| Interpretability | Low | Medium (feature probabilities) |
| High dimensions | Struggles (curse of dimensionality) | Works well |
| Scaling needed | Yes | No |
| Best for | Small-medium data, low dimensions | Text, high dimensions, fast prediction |`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the "naive" assumption in Naive Bayes?',
          options: [
            'The data is normally distributed',
            'Features are independent given the class',
            'All classes have equal probability',
            'The model is simple'
          ],
          correct: 1,
          explanation: 'Naive Bayes assumes that features are conditionally independent given the class label. This means P(X|y) = P(x1|y) * P(x2|y) * ... - the probability of seeing all features is the product of individual feature probabilities.'
        },
        {
          type: 'multiple-choice',
          question: 'Why does KNN struggle in high dimensions?',
          options: [
            'It needs more memory',
            'Distances become less meaningful as dimensions increase (curse of dimensionality)',
            'It runs slower',
            'It can\'t handle many features'
          ],
          correct: 1,
          explanation: 'In high dimensions, all points become roughly equidistant (curse of dimensionality). The concept of "nearest neighbor" becomes less meaningful when everything is equally far away.'
        }
      ]
    },
    {
      id: 'ensemble-methods',
      title: 'Ensemble Methods: Boosting & Bagging',
      duration: '55 min',
      content: [
        {
          type: 'text',
          content: `Ensemble methods combine multiple models to create a stronger predictor. We've seen Random Forests (bagging); now let's understand the full landscape including boosting.`
        },
        {
          type: 'heading',
          content: 'The Big Picture'
        },
        {
          type: 'text',
          content: `**Two main ensemble strategies:**

1. **Bagging (Bootstrap Aggregating):**
   - Train models in parallel on bootstrap samples
   - Combine by averaging (regression) or voting (classification)
   - Reduces variance
   - Example: Random Forest

2. **Boosting:**
   - Train models sequentially, each fixing previous errors
   - Each model focuses on hard examples
   - Reduces bias
   - Examples: AdaBoost, Gradient Boosting, XGBoost`
        },
        {
          type: 'heading',
          content: 'Bagging'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Single tree
tree = DecisionTreeClassifier(random_state=42)
scores_tree = cross_val_score(tree, X, y, cv=5)
print(f"Single tree: {scores_tree.mean():.3f} (+/- {scores_tree.std():.3f})")

# Bagged trees
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,        # Number of trees
    max_samples=0.8,        # Fraction of samples per tree
    max_features=1.0,       # Fraction of features per tree
    bootstrap=True,         # Sample with replacement
    random_state=42
)
scores_bag = cross_val_score(bagging, X, y, cv=5)
print(f"Bagged trees: {scores_bag.mean():.3f} (+/- {scores_bag.std():.3f})")

# Key insight: Bagging reduces variance without increasing bias
# Standard deviation is much lower for bagging!`
        },
        {
          type: 'heading',
          content: 'AdaBoost: Adaptive Boosting'
        },
        {
          type: 'text',
          content: `**AdaBoost algorithm:**
1. Train a weak learner on the data
2. Increase weights of misclassified examples
3. Train next learner on reweighted data
4. Repeat, then combine all learners

**Each model focuses on examples the previous models got wrong!**`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.ensemble import AdaBoostClassifier

# AdaBoost with decision stumps (depth=1 trees)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
scores_ada = cross_val_score(ada, X, y, cv=5)
print(f"AdaBoost: {scores_ada.mean():.3f} (+/- {scores_ada.std():.3f})")

# Visualize how AdaBoost combines weak learners
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
n_estimators_list = [1, 5, 20, 100]

for ax, n_est in zip(axes, n_estimators_list):
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        random_state=42
    )
    ada.fit(X_moon, y_moon)

    xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
    Z = ada.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'{n_est} weak learners\\nAccuracy: {ada.score(X_moon, y_moon):.3f}')

plt.tight_layout()
plt.show()

# Many weak learners → complex boundary`
        },
        {
          type: 'heading',
          content: 'Gradient Boosting'
        },
        {
          type: 'text',
          content: `**Gradient Boosting trains models on residuals (errors) of previous models.**

**Algorithm:**
1. Fit initial model (usually mean for regression)
2. Compute residuals: r = y - prediction
3. Fit new model to predict residuals
4. Update: prediction = prediction + learning_rate * new_model
5. Repeat

**Key insight:** Each tree corrects the errors of the ensemble so far.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np

# Gradient Boosting Classifier
gb = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Shrinkage - smaller = more robust
    max_depth=3,           # Depth of each tree
    subsample=0.8,         # Fraction of samples per tree (stochastic GB)
    random_state=42
)
scores_gb = cross_val_score(gb, X, y, cv=5)
print(f"Gradient Boosting: {scores_gb.mean():.3f} (+/- {scores_gb.std():.3f})")

# Learning rate tradeoff
# Lower learning rate = need more trees, but better generalization
# Higher learning rate = faster training, risk of overfitting

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, lr in zip(axes, [0.01, 0.1, 1.0]):
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3, random_state=42)
    gb.fit(X, y)

    # Track training deviance
    train_scores = [gb.loss_(y, y_pred) for y_pred in gb.staged_decision_function(X)]
    ax.plot(train_scores)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'LR = {lr}')

plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'XGBoost: Extreme Gradient Boosting'
        },
        {
          type: 'code',
          language: 'python',
          content: `# XGBoost is an optimized implementation of gradient boosting
# Features: regularization, parallel processing, handling missing values

# pip install xgboost
import xgboost as xgb

# XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,  # Random feature selection (like RF)
    reg_alpha=0,           # L1 regularization
    reg_lambda=1,          # L2 regularization
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

scores_xgb = cross_val_score(xgb_clf, X, y, cv=5)
print(f"XGBoost: {scores_xgb.mean():.3f} (+/- {scores_xgb.std():.3f})")

# Early stopping to prevent overfitting
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_early = xgb.XGBClassifier(
    n_estimators=1000,  # High number, will stop early
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    early_stopping_rounds=10
)

xgb_early.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {xgb_early.best_iteration}")
print(f"Best score: {xgb_early.best_score:.4f}")`
        },
        {
          type: 'heading',
          content: 'LightGBM and CatBoost'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Other popular gradient boosting libraries

# LightGBM: Fast, efficient, handles large datasets
# pip install lightgbm
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=31,       # LightGBM uses leaf-wise growth
    random_state=42
)
scores_lgb = cross_val_score(lgb_clf, X, y, cv=5)
print(f"LightGBM: {scores_lgb.mean():.3f} (+/- {scores_lgb.std():.3f})")

# CatBoost: Handles categorical features natively
# pip install catboost
from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    depth=3,
    random_state=42,
    verbose=False
)
scores_cat = cross_val_score(cat_clf, X, y, cv=5)
print(f"CatBoost: {scores_cat.mean():.3f} (+/- {scores_cat.std():.3f}")`
        },
        {
          type: 'heading',
          content: 'Bagging vs Boosting Summary'
        },
        {
          type: 'text',
          content: `| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Base learners | Often full trees | Often shallow trees (stumps) |
| Weights | Equal | Based on performance |
| Overfitting risk | Low | Higher (but regularizable) |
| Example | Random Forest | XGBoost, LightGBM |
| Best when | High variance models | High bias models |

**Practical recommendations:**
1. **Start with Random Forest** - good baseline, hard to mess up
2. **Try Gradient Boosting** - often beats RF with tuning
3. **Use XGBoost/LightGBM** for competitions and production
4. **Use early stopping** to prevent overfitting in boosting`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the main difference between bagging and boosting?',
          options: [
            'Bagging uses trees, boosting uses linear models',
            'Bagging trains models in parallel; boosting trains sequentially with each model focusing on previous errors',
            'Bagging is faster',
            'Boosting can only do classification'
          ],
          correct: 1,
          explanation: 'Bagging trains independent models in parallel on bootstrap samples. Boosting trains models sequentially, with each new model focusing on correcting the errors of the previous ensemble.'
        },
        {
          type: 'multiple-choice',
          question: 'What does the learning rate in gradient boosting control?',
          options: [
            'How fast the training runs',
            'The contribution of each tree to the final prediction (lower = more robust)',
            'The depth of trees',
            'The number of features'
          ],
          correct: 1,
          explanation: 'The learning rate (shrinkage) scales the contribution of each tree. Lower learning rates require more trees but typically generalize better. It\'s a regularization parameter.'
        }
      ]
    }
  ]
}
