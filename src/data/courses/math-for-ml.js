export const mathforml = {
  id: 'math-for-ml',
  title: 'Mathematics for ML',
  icon: 'calculator',
  description: 'Master the essential mathematics that powers machine learning: linear algebra, calculus, probability, and statistics.',
  difficulty: 'beginner',
  estimatedhours: 12,
  path: 'foundations',
  lessons: [
    {
      id: 'vectors',
      title: 'Vectors and Vector Operations',
      duration: '25 min',
      concepts: ['vectors', 'dot product', 'magnitude', 'unit vectors'],
      content: [
        { type: 'heading', text: 'What is a Vector?' },
        { type: 'paragraph', text: 'Before we dive into complex neural networks, we need to understand the fundamental building block of all machine learning: vectors. A vector is simply an ordered list of numbers. That\'s it. But this simple concept is the foundation of everything in ML.' },
        { type: 'callout', variant: 'tip', text: 'Think of a vector as a point in space or a direction with magnitude. In ML, we use vectors to represent data points, features, weights, and gradients.' },
        { type: 'subheading', text: 'Why Vectors Matter in ML' },
        { type: 'paragraph', text: 'Every data point in machine learning is represented as a vector. An image? A vector of pixel values. A sentence? A vector of word embeddings. A customer? A vector of features like age, income, and purchase history.' },
        { type: 'paragraph', text: 'Let\'s say we have a house with 3 bedrooms, 2 bathrooms, and 1500 square feet. We represent this as:' },
        { type: 'formula', formula: 'x = [3, 2, 1500]' },
        { type: 'paragraph', text: 'This vector x has 3 components (or dimensions). In ML, we often work with vectors that have hundreds or even millions of dimensions.' },
        { type: 'subheading', text: 'Vector Notation' },
        { type: 'paragraph', text: 'Vectors are typically written as lowercase bold letters (x) or with an arrow (→x). We\'ll use lowercase letters. A vector with n components is written as:' },
        { type: 'formula', formula: 'x = [x₁, x₂, x₃, ..., xₙ]' },
        { type: 'paragraph', text: 'The subscript indicates the position (index) of each element. In programming, we usually start counting from 0, but in math notation, we often start from 1.' },
        { type: 'code', language: 'python', filename: 'vectors.py', code: `import numpy as np

# Creating a vector
house_features = np.array([3, 2, 1500])
print(f"House vector: {house_features}")
print(f"Number of dimensions: {len(house_features)}")

# Accessing elements
bedrooms = house_features[0]  # First element (index 0)
print(f"Bedrooms: {bedrooms}")` },
        { type: 'heading', text: 'Vector Operations' },
        { type: 'subheading', text: '1. Vector Addition' },
        { type: 'paragraph', text: 'Adding two vectors means adding their corresponding elements. Both vectors must have the same number of dimensions.' },
        { type: 'formula', formula: 'a + b = [a₁ + b₁, a₂ + b₂, ..., aₙ + bₙ]' },
        { type: 'paragraph', text: 'Intuition: If you walk 3 steps east then 4 steps north, vector addition tells you where you end up relative to where you started.' },
        { type: 'code', language: 'python', filename: 'vector_addition.py', code: `import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = a + b
print(f"a + b = {result}")  # [5, 7, 9]` },
        { type: 'diagram', caption: 'Vector Addition: The resulting vector (purple) is the diagonal of the parallelogram formed by vectors a and b', svg: `<svg width="320" height="240" viewBox="0 0 320 240">
          <defs>
            <marker id="arrowhead-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6"/>
            </marker>
            <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#22c55e"/>
            </marker>
            <marker id="arrowhead-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#8b5cf6"/>
            </marker>
          </defs>
          <rect width="320" height="240" fill="#f8fafc"/>
          <g transform="translate(40, 200)">
            <line x1="0" y1="0" x2="260" y2="0" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="0" y1="0" x2="0" y2="-180" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="0" y1="0" x2="100" y2="-60" stroke="#3b82f6" stroke-width="3" marker-end="url(#arrowhead-blue)"/>
            <text x="55" y="-25" font-size="14" fill="#3b82f6" font-weight="bold">a</text>
            <line x1="0" y1="0" x2="60" y2="-100" stroke="#22c55e" stroke-width="3" marker-end="url(#arrowhead-green)"/>
            <text x="20" y="-55" font-size="14" fill="#22c55e" font-weight="bold">b</text>
            <line x1="0" y1="0" x2="160" y2="-160" stroke="#8b5cf6" stroke-width="3" marker-end="url(#arrowhead-purple)"/>
            <text x="85" y="-95" font-size="14" fill="#8b5cf6" font-weight="bold">a + b</text>
            <line x1="100" y1="-60" x2="160" y2="-160" stroke="#22c55e" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>
            <line x1="60" y1="-100" x2="160" y2="-160" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>
            <circle cx="0" cy="0" r="4" fill="#1e293b"/>
            <text x="-15" y="20" font-size="12" fill="#64748b">Origin</text>
          </g>
        </svg>` },
        { type: 'subheading', text: '2. Scalar Multiplication' },
        { type: 'paragraph', text: 'Multiplying a vector by a scalar (a single number) scales each element by that number.' },
        { type: 'formula', formula: 'c · x = [c·x₁, c·x₂, ..., c·xₙ]' },
        { type: 'paragraph', text: 'Intuition: Scaling a vector by 2 makes it twice as long (doubles its magnitude) but keeps it pointing in the same direction.' },
        { type: 'code', language: 'python', filename: 'scalar_mult.py', code: `import numpy as np

x = np.array([1, 2, 3])
scaled = 3 * x
print(f"3 * x = {scaled}")  # [3, 6, 9]

# Negative scalar flips direction
flipped = -1 * x
print(f"-1 * x = {flipped}")  # [-1, -2, -3]` },
        { type: 'diagram', caption: 'Scalar Multiplication: The vector keeps its direction but changes length. Negative scalars flip the direction.', svg: `<svg width="400" height="180" viewBox="0 0 400 180">
          <defs>
            <marker id="arrow-orig" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
            </marker>
            <marker id="arrow-scaled" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6"/>
            </marker>
            <marker id="arrow-neg" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444"/>
            </marker>
          </defs>
          <rect width="400" height="180" fill="#f8fafc"/>
          <g transform="translate(200, 90)">
            <line x1="-180" y1="0" x2="180" y2="0" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="0" y1="-70" x2="0" y2="70" stroke="#e2e8f0" stroke-width="1"/>
            <line x1="0" y1="0" x2="50" y2="-30" stroke="#64748b" stroke-width="3" marker-end="url(#arrow-orig)"/>
            <text x="30" y="-40" font-size="12" fill="#64748b" font-weight="bold">x</text>
            <line x1="0" y1="0" x2="120" y2="-72" stroke="#3b82f6" stroke-width="3" marker-end="url(#arrow-scaled)"/>
            <text x="80" y="-55" font-size="12" fill="#3b82f6" font-weight="bold">2x</text>
            <line x1="0" y1="0" x2="-50" y2="30" stroke="#ef4444" stroke-width="3" marker-end="url(#arrow-neg)"/>
            <text x="-70" y="25" font-size="12" fill="#ef4444" font-weight="bold">-x</text>
            <circle cx="0" cy="0" r="3" fill="#1e293b"/>
          </g>
          <text x="200" y="170" text-anchor="middle" font-size="11" fill="#64748b">Original (gray), Doubled (blue), Negated (red)</text>
        </svg>` },
        { type: 'heading', text: 'The Dot Product' },
        { type: 'paragraph', text: 'The dot product is one of the most important operations in machine learning. It takes two vectors and returns a single number (scalar).' },
        { type: 'formula', formula: 'a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σᵢ aᵢbᵢ' },
        { type: 'callout', variant: 'info', text: 'The dot product is everywhere in ML: in linear regression predictions, neural network forward passes, attention mechanisms, and similarity calculations.' },
        { type: 'paragraph', text: 'Let\'s break down what the dot product actually computes:' },
        { type: 'list', items: [
          'Multiply corresponding elements together',
          'Sum up all the products',
          'Result is a single number'
        ]},
        { type: 'code', language: 'python', filename: 'dot_product.py', code: `import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Manual calculation
manual = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(f"Manual: {manual}")  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# Using numpy
dot = np.dot(a, b)
print(f"np.dot: {dot}")  # 32

# Alternative syntax
dot2 = a @ b
print(f"a @ b: {dot2}")  # 32` },
        { type: 'subheading', text: 'Geometric Interpretation' },
        { type: 'paragraph', text: 'The dot product has a beautiful geometric meaning. It measures how much two vectors point in the same direction.' },
        { type: 'formula', formula: 'a · b = |a| × |b| × cos(θ)' },
        { type: 'paragraph', text: 'Where |a| and |b| are the magnitudes (lengths) of the vectors, and θ is the angle between them.' },
        { type: 'list', items: [
          'If vectors point in the same direction (θ = 0°): dot product is maximum (positive)',
          'If vectors are perpendicular (θ = 90°): dot product is 0',
          'If vectors point in opposite directions (θ = 180°): dot product is minimum (negative)'
        ]},
        { type: 'diagram', caption: 'Dot Product Geometric Meaning: The dot product measures how aligned two vectors are', svg: `<svg width="500" height="160" viewBox="0 0 500 160">
          <rect width="500" height="160" fill="#f8fafc"/>
          <g transform="translate(80, 80)">
            <text x="0" y="-55" text-anchor="middle" font-size="11" fill="#64748b">Same direction</text>
            <text x="0" y="-42" text-anchor="middle" font-size="10" fill="#22c55e" font-weight="bold">a·b > 0</text>
            <line x1="-40" y1="0" x2="40" y2="0" stroke="#3b82f6" stroke-width="2.5"/>
            <polygon points="40,0 32,-5 32,5" fill="#3b82f6"/>
            <line x1="-40" y1="8" x2="30" y2="8" stroke="#22c55e" stroke-width="2.5"/>
            <polygon points="30,8 22,3 22,13" fill="#22c55e"/>
            <text x="0" y="35" text-anchor="middle" font-size="10" fill="#64748b">θ ≈ 0°</text>
          </g>
          <g transform="translate(250, 80)">
            <text x="0" y="-55" text-anchor="middle" font-size="11" fill="#64748b">Perpendicular</text>
            <text x="0" y="-42" text-anchor="middle" font-size="10" fill="#f59e0b" font-weight="bold">a·b = 0</text>
            <line x1="-40" y1="0" x2="40" y2="0" stroke="#3b82f6" stroke-width="2.5"/>
            <polygon points="40,0 32,-5 32,5" fill="#3b82f6"/>
            <line x1="0" y1="30" x2="0" y2="-25" stroke="#22c55e" stroke-width="2.5"/>
            <polygon points="0,-25 -5,-17 5,-17" fill="#22c55e"/>
            <path d="M 12 0 A 12 12 0 0 0 0 -12" fill="none" stroke="#94a3b8" stroke-width="1"/>
            <text x="0" y="50" text-anchor="middle" font-size="10" fill="#64748b">θ = 90°</text>
          </g>
          <g transform="translate(420, 80)">
            <text x="0" y="-55" text-anchor="middle" font-size="11" fill="#64748b">Opposite</text>
            <text x="0" y="-42" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="bold">a·b < 0</text>
            <line x1="-40" y1="0" x2="40" y2="0" stroke="#3b82f6" stroke-width="2.5"/>
            <polygon points="40,0 32,-5 32,5" fill="#3b82f6"/>
            <line x1="30" y1="0" x2="-35" y2="0" stroke="#22c55e" stroke-width="2.5"/>
            <polygon points="-35,0 -27,-5 -27,5" fill="#22c55e"/>
            <text x="0" y="35" text-anchor="middle" font-size="10" fill="#64748b">θ = 180°</text>
          </g>
        </svg>` },
        { type: 'callout', variant: 'tip', text: 'This is why dot products are used for similarity! The cosine similarity between two vectors is just their dot product divided by the product of their magnitudes.' },
        { type: 'heading', text: 'Vector Magnitude (Length)' },
        { type: 'paragraph', text: 'The magnitude of a vector is its length. For a 2D vector, you might recognize this as the Pythagorean theorem.' },
        { type: 'formula', formula: '|x| = √(x₁² + x₂² + ... + xₙ²) = √(x · x)' },
        { type: 'paragraph', text: 'Notice that the magnitude is the square root of the dot product of a vector with itself!' },
        { type: 'code', language: 'python', filename: 'magnitude.py', code: `import numpy as np

x = np.array([3, 4])

# Manual calculation (Pythagorean theorem)
manual = np.sqrt(3**2 + 4**2)
print(f"Manual: {manual}")  # 5.0

# Using numpy
magnitude = np.linalg.norm(x)
print(f"np.linalg.norm: {magnitude}")  # 5.0

# Works for any dimension
high_dim = np.array([1, 2, 3, 4, 5])
print(f"5D magnitude: {np.linalg.norm(high_dim)}")` },
        { type: 'heading', text: 'Unit Vectors' },
        { type: 'paragraph', text: 'A unit vector is a vector with magnitude 1. It represents pure direction without any magnitude. Any vector can be converted to a unit vector by dividing by its magnitude.' },
        { type: 'formula', formula: 'û = x / |x|' },
        { type: 'paragraph', text: 'Unit vectors are crucial in ML for normalization. When we normalize data or use L2 normalization in embeddings, we\'re often creating unit vectors.' },
        { type: 'code', language: 'python', filename: 'unit_vectors.py', code: `import numpy as np

x = np.array([3, 4])

# Create unit vector
magnitude = np.linalg.norm(x)
unit_vector = x / magnitude

print(f"Original: {x}")
print(f"Unit vector: {unit_vector}")  # [0.6, 0.8]
print(f"Magnitude of unit vector: {np.linalg.norm(unit_vector)}")  # 1.0` },
        { type: 'keypoints', points: [
          'Vectors are ordered lists of numbers that represent data in ML',
          'The dot product measures similarity between vectors',
          'Magnitude is the length of a vector (Euclidean distance from origin)',
          'Unit vectors have magnitude 1 and represent pure direction'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the dot product of vectors [1, 2, 3] and [4, 5, 6]?',
          options: ['15', '32', '21', '12'],
          correct: 1,
          explanation: 'Dot product = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32'
        },
        {
          type: 'multiple-choice',
          question: 'If two vectors are perpendicular, what is their dot product?',
          options: ['1', '-1', '0', 'Undefined'],
          correct: 2,
          explanation: 'Perpendicular vectors form a 90° angle. Since cos(90°) = 0, and dot product = |a||b|cos(θ), the result is 0.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the magnitude of the vector [3, 4]?',
          options: ['7', '5', '12', '25'],
          correct: 1,
          explanation: 'Magnitude = √(3² + 4²) = √(9 + 16) = √25 = 5. This is the classic 3-4-5 Pythagorean triple!'
        },
        {
          type: 'multiple-choice',
          question: 'Why are unit vectors important in ML?',
          options: ['They are faster to compute', 'They represent pure direction without magnitude', 'They use less memory', 'They only work in 3D'],
          correct: 1,
          explanation: 'Unit vectors have magnitude 1, so they represent only direction. This is useful for normalization and comparing directions without magnitude affecting the comparison.'
        }
      ]
    },
    {
      id: 'matrices-basics',
      title: 'Matrices and Linear Transformations',
      duration: '30 min',
      concepts: ['matrices', 'linear transformations', 'matrix-vector multiplication'],
      content: [
        { type: 'heading', text: 'What is a Matrix?' },
        { type: 'paragraph', text: 'A matrix is a 2D grid of numbers. While a vector is a list, a matrix is a table. We describe a matrix by its shape: rows × columns.' },
        { type: 'formula', formula: 'A = [a₁₁  a₁₂  a₁₃]\n    [a₂₁  a₂₂  a₂₃]   ← This is a 2×3 matrix (2 rows, 3 columns)' },
        { type: 'paragraph', text: 'The element aᵢⱼ is in row i and column j. In NumPy, we access it as A[i, j] (with 0-based indexing).' },
        { type: 'callout', variant: 'info', text: 'In deep learning, weight matrices connect layers. A matrix of shape (100, 50) can transform a 50-dimensional input into a 100-dimensional output.' },
        { type: 'code', language: 'python', filename: 'matrices.py', code: `import numpy as np

# Creating a matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(f"Shape: {A.shape}")  # (2, 3) - 2 rows, 3 columns
print(f"Element at row 0, col 1: {A[0, 1]}")  # 2
print(f"First row: {A[0]}")  # [1, 2, 3]
print(f"Second column: {A[:, 1]}")  # [2, 5]` },
        { type: 'heading', text: 'Matrix-Vector Multiplication' },
        { type: 'paragraph', text: 'This is the core operation in neural networks. When you multiply a matrix by a vector, you get a new vector. Each element of the output is a dot product.' },
        { type: 'formula', formula: 'If A is m×n and x is n×1, then Ax is m×1' },
        { type: 'paragraph', text: 'The key rule: the number of columns in A must equal the number of elements in x. Let\'s see this step by step:' },
        { type: 'code', language: 'python', filename: 'matrix_vector.py', code: `import numpy as np

# Matrix (2x3) and vector (3x1)
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
x = np.array([1, 0, 1])

# Result has shape (2,) - one element per row of A
result = A @ x  # or np.dot(A, x)
print(f"Result: {result}")  # [4, 10]

# How it works:
# Row 1: 1*1 + 2*0 + 3*1 = 1 + 0 + 3 = 4
# Row 2: 4*1 + 5*0 + 6*1 = 4 + 0 + 6 = 10` },
        { type: 'heading', text: 'Matrices as Transformations' },
        { type: 'paragraph', text: 'Here\'s the beautiful insight: a matrix represents a transformation. When you multiply a matrix by a vector, you\'re transforming that vector into a new space.' },
        { type: 'subheading', text: 'Types of Transformations' },
        { type: 'list', items: [
          'Scaling: Stretch or shrink along axes',
          'Rotation: Rotate around the origin',
          'Reflection: Flip across a line',
          'Shearing: Skew the space',
          'Projection: Reduce to lower dimensions'
        ]},
        { type: 'paragraph', text: 'In neural networks, weight matrices learn to perform useful transformations on data. A layer transforms input features into more useful representations.' },
        { type: 'code', language: 'python', filename: 'transformations.py', code: `import numpy as np

# Scaling matrix (doubles x, triples y)
scale = np.array([
    [2, 0],
    [0, 3]
])

# 90-degree rotation matrix
rotate_90 = np.array([
    [0, -1],
    [1,  0]
])

# Apply to a point
point = np.array([1, 1])

scaled = scale @ point
print(f"Scaled: {scaled}")  # [2, 3]

rotated = rotate_90 @ point
print(f"Rotated 90°: {rotated}")  # [-1, 1]` },
        { type: 'heading', text: 'Why This Matters for ML' },
        { type: 'paragraph', text: 'Every layer in a neural network performs a matrix-vector multiplication followed by an activation function:' },
        { type: 'formula', formula: 'output = activation(W · input + b)' },
        { type: 'paragraph', text: 'Where W is the weight matrix, input is your data, and b is the bias vector. The network learns W and b to perform useful transformations.' },
        { type: 'code', language: 'python', filename: 'neural_layer.py', code: `import numpy as np

def relu(x):
    return np.maximum(0, x)

# Simulating a neural network layer
input_features = np.array([0.5, 0.8, 0.2])  # 3 features
W = np.random.randn(4, 3)  # Weight matrix: 4 outputs, 3 inputs
b = np.zeros(4)  # Bias vector

# Forward pass
z = W @ input_features + b  # Linear transformation
output = relu(z)  # Non-linear activation

print(f"Input shape: {input_features.shape}")  # (3,)
print(f"Weight shape: {W.shape}")  # (4, 3)
print(f"Output shape: {output.shape}")  # (4,)` },
        { type: 'keypoints', points: [
          'Matrices are 2D grids of numbers with shape (rows, columns)',
          'Matrix-vector multiplication transforms vectors into new spaces',
          'Each row of the matrix computes one dot product with the input',
          'Neural network layers are matrix-vector multiplications with activation functions'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'If matrix A has shape (3, 4) and vector x has shape (4,), what is the shape of Ax?',
          options: ['(4,)', '(3,)', '(3, 4)', '(4, 3)'],
          correct: 1,
          explanation: 'The output has one element per row of A. With 3 rows, the output shape is (3,).'
        },
        {
          type: 'multiple-choice',
          question: 'What does multiplying a vector by a matrix do geometrically?',
          options: ['Adds a constant', 'Transforms the vector', 'Calculates magnitude', 'Counts elements'],
          correct: 1,
          explanation: 'Matrix multiplication transforms vectors - it can rotate, scale, shear, or project them into new spaces.'
        },
        {
          type: 'multiple-choice',
          question: 'In a neural network layer with 100 inputs and 50 outputs, what is the weight matrix shape?',
          options: ['(100, 50)', '(50, 100)', '(100, 100)', '(50, 50)'],
          correct: 1,
          explanation: 'The weight matrix has shape (outputs, inputs) = (50, 100). When multiplied by a 100-dim input, it produces a 50-dim output.'
        }
      ]
    },
    {
      id: 'matrix-operations',
      title: 'Matrix Operations and Properties',
      duration: '25 min',
      concepts: ['matrix multiplication', 'transpose', 'inverse', 'identity matrix'],
      content: [
        { type: 'heading', text: 'Matrix Multiplication' },
        { type: 'paragraph', text: 'Matrix multiplication combines two matrices into a new one. Unlike regular multiplication, the order matters! A × B is usually NOT equal to B × A.' },
        { type: 'formula', formula: 'If A is m×n and B is n×p, then AB is m×p' },
        { type: 'paragraph', text: 'The key constraint: the number of columns in A must equal the number of rows in B. The result has the rows of A and columns of B.' },
        { type: 'code', language: 'python', filename: 'matrix_mult.py', code: `import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])  # 2x2

B = np.array([
    [5, 6],
    [7, 8]
])  # 2x2

C = A @ B
print(f"A @ B =\\n{C}")
# [[19, 22],
#  [43, 50]]

# Order matters!
D = B @ A
print(f"B @ A =\\n{D}")
# [[23, 34],
#  [31, 46]]

print(f"A @ B == B @ A? {np.array_equal(C, D)}")  # False` },
        { type: 'subheading', text: 'How Matrix Multiplication Works' },
        { type: 'paragraph', text: 'Each element of the result is a dot product: element (i,j) is the dot product of row i from A with column j from B.' },
        { type: 'code', language: 'python', filename: 'matrix_mult_detail.py', code: `import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element (0, 0): row 0 of A · column 0 of B
c_00 = A[0, 0]*B[0, 0] + A[0, 1]*B[1, 0]  # 1*5 + 2*7 = 19
print(f"C[0,0] = {c_00}")

# Element (0, 1): row 0 of A · column 1 of B
c_01 = A[0, 0]*B[0, 1] + A[0, 1]*B[1, 1]  # 1*6 + 2*8 = 22
print(f"C[0,1] = {c_01}")` },
        { type: 'heading', text: 'The Transpose' },
        { type: 'paragraph', text: 'The transpose of a matrix flips it over its diagonal. Rows become columns and columns become rows.' },
        { type: 'formula', formula: '(Aᵀ)ᵢⱼ = Aⱼᵢ' },
        { type: 'code', language: 'python', filename: 'transpose.py', code: `import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # Shape: (2, 3)

A_T = A.T
print(f"Original shape: {A.shape}")  # (2, 3)
print(f"Transposed shape: {A_T.shape}")  # (3, 2)
print(f"Transposed:\\n{A_T}")
# [[1, 4],
#  [2, 5],
#  [3, 6]]` },
        { type: 'paragraph', text: 'Important properties of transpose:' },
        { type: 'list', items: [
          '(Aᵀ)ᵀ = A (transpose twice = original)',
          '(AB)ᵀ = BᵀAᵀ (order reverses!)',
          '(A + B)ᵀ = Aᵀ + Bᵀ'
        ]},
        { type: 'heading', text: 'The Identity Matrix' },
        { type: 'paragraph', text: 'The identity matrix I is the matrix equivalent of 1. Multiplying any matrix by I leaves it unchanged.' },
        { type: 'formula', formula: 'AI = IA = A' },
        { type: 'paragraph', text: 'The identity matrix has 1s on the diagonal and 0s everywhere else.' },
        { type: 'code', language: 'python', filename: 'identity.py', code: `import numpy as np

I = np.eye(3)  # 3x3 identity matrix
print(f"Identity matrix:\\n{I}")
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"A @ I = A? {np.allclose(A @ I, A)}")  # True` },
        { type: 'heading', text: 'The Inverse Matrix' },
        { type: 'paragraph', text: 'The inverse of a matrix A, written A⁻¹, undoes what A does. When you multiply A by its inverse, you get the identity matrix.' },
        { type: 'formula', formula: 'AA⁻¹ = A⁻¹A = I' },
        { type: 'callout', variant: 'warning', text: 'Not all matrices have inverses! A matrix must be square (same rows and columns) and have full rank to be invertible.' },
        { type: 'code', language: 'python', filename: 'inverse.py', code: `import numpy as np

A = np.array([
    [4, 7],
    [2, 6]
])

# Calculate inverse
A_inv = np.linalg.inv(A)
print(f"A inverse:\\n{A_inv}")

# Verify: A @ A_inv should be identity
product = A @ A_inv
print(f"A @ A_inv:\\n{np.round(product, 10)}")
# [[1. 0.]
#  [0. 1.]]` },
        { type: 'subheading', text: 'Why Inverses Matter' },
        { type: 'paragraph', text: 'Matrix inverses are used to solve systems of linear equations. If Ax = b and we want to find x:' },
        { type: 'formula', formula: 'x = A⁻¹b' },
        { type: 'paragraph', text: 'This is the foundation of many ML algorithms, including linear regression!' },
        { type: 'keypoints', points: [
          'Matrix multiplication order matters: AB ≠ BA in general',
          'Transpose flips rows and columns: (Aᵀ)ᵢⱼ = Aⱼᵢ',
          'The identity matrix I leaves other matrices unchanged: AI = A',
          'The inverse A⁻¹ undoes A: AA⁻¹ = I (only for invertible matrices)'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'If A is a 3×2 matrix and B is a 2×4 matrix, what is the shape of AB?',
          options: ['3×4', '2×2', '3×2', 'Cannot multiply'],
          correct: 0,
          explanation: 'The inner dimensions (2) match, so we can multiply. Result shape is outer dimensions: 3×4.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the transpose of [[1, 2], [3, 4]]?',
          options: ['[[4, 3], [2, 1]]', '[[1, 3], [2, 4]]', '[[2, 1], [4, 3]]', '[[1, 2], [3, 4]]'],
          correct: 1,
          explanation: 'Transpose swaps rows and columns. Row [1, 2] becomes column [1, 3], row [3, 4] becomes column [2, 4].'
        },
        {
          type: 'multiple-choice',
          question: 'What is special about the identity matrix?',
          options: ['All elements are 1', 'AI = IA = A for any A', 'It has no inverse', 'Its transpose is different'],
          correct: 1,
          explanation: 'The identity matrix is the multiplicative identity: multiplying any compatible matrix by I leaves it unchanged.'
        }
      ]
    },
    {
      id: 'derivatives',
      title: 'Derivatives and Gradients',
      duration: '30 min',
      concepts: ['derivatives', 'partial derivatives', 'gradients', 'slope'],
      content: [
        { type: 'heading', text: 'What is a Derivative?' },
        { type: 'paragraph', text: 'A derivative measures how much a function\'s output changes when you make a tiny change to its input. It\'s the instantaneous rate of change - the slope at a specific point.' },
        { type: 'formula', formula: 'f\'(x) = lim[h→0] (f(x+h) - f(x)) / h' },
        { type: 'paragraph', text: 'Don\'t worry about the formal definition. Intuitively, the derivative tells you: "If I nudge x by a tiny amount, how much does f(x) change?"' },
        { type: 'callout', variant: 'tip', text: 'In ML, derivatives tell us how to adjust weights to reduce error. If increasing a weight increases error (positive derivative), decrease that weight!' },
        { type: 'subheading', text: 'Visual Intuition' },
        { type: 'paragraph', text: 'Imagine a ball rolling on a curve. The derivative at any point is the slope of the curve there - it tells you which direction is "downhill" and how steep it is.' },
        { type: 'list', items: [
          'Positive derivative: function is increasing (going uphill)',
          'Negative derivative: function is decreasing (going downhill)',
          'Zero derivative: function is flat (at a peak or valley)'
        ]},
        { type: 'heading', text: 'Common Derivative Rules' },
        { type: 'paragraph', text: 'You don\'t need to derive these from scratch. Here are the rules you\'ll use most often:' },
        { type: 'subheading', text: 'Power Rule' },
        { type: 'formula', formula: 'd/dx [xⁿ] = n·xⁿ⁻¹' },
        { type: 'paragraph', text: 'Bring down the exponent, then reduce it by 1.' },
        { type: 'code', language: 'python', filename: 'power_rule.py', code: `# f(x) = x³
# f'(x) = 3x²

import numpy as np

def f(x):
    return x**3

def f_derivative(x):
    return 3 * x**2

x = 2
print(f"f({x}) = {f(x)}")  # 8
print(f"f'({x}) = {f_derivative(x)}")  # 12

# Verify numerically
h = 0.0001
numerical_derivative = (f(x + h) - f(x)) / h
print(f"Numerical approximation: {numerical_derivative}")  # ~12` },
        { type: 'subheading', text: 'Sum Rule' },
        { type: 'formula', formula: 'd/dx [f(x) + g(x)] = f\'(x) + g\'(x)' },
        { type: 'paragraph', text: 'The derivative of a sum is the sum of derivatives.' },
        { type: 'subheading', text: 'Product Rule' },
        { type: 'formula', formula: 'd/dx [f(x) · g(x)] = f\'(x)·g(x) + f(x)·g\'(x)' },
        { type: 'subheading', text: 'Important Functions' },
        { type: 'list', items: [
          'd/dx [eˣ] = eˣ (exponential is its own derivative!)',
          'd/dx [ln(x)] = 1/x',
          'd/dx [sin(x)] = cos(x)',
          'd/dx [cos(x)] = -sin(x)'
        ]},
        { type: 'heading', text: 'Partial Derivatives' },
        { type: 'paragraph', text: 'When a function has multiple inputs, we take partial derivatives with respect to each input, treating others as constants.' },
        { type: 'formula', formula: 'f(x, y) = x² + 3xy + y²\n∂f/∂x = 2x + 3y (treating y as constant)\n∂f/∂y = 3x + 2y (treating x as constant)' },
        { type: 'code', language: 'python', filename: 'partial_derivatives.py', code: `import numpy as np

def f(x, y):
    return x**2 + 3*x*y + y**2

def df_dx(x, y):
    return 2*x + 3*y

def df_dy(x, y):
    return 3*x + 2*y

x, y = 2, 3
print(f"f({x}, {y}) = {f(x, y)}")  # 4 + 18 + 9 = 31
print(f"∂f/∂x at ({x}, {y}) = {df_dx(x, y)}")  # 4 + 9 = 13
print(f"∂f/∂y at ({x}, {y}) = {df_dy(x, y)}")  # 6 + 6 = 12` },
        { type: 'heading', text: 'The Gradient' },
        { type: 'paragraph', text: 'The gradient is a vector of all partial derivatives. It points in the direction of steepest increase.' },
        { type: 'formula', formula: '∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]' },
        { type: 'callout', variant: 'info', text: 'Gradient descent works by moving in the opposite direction of the gradient (steepest decrease) to minimize a loss function.' },
        { type: 'code', language: 'python', filename: 'gradient.py', code: `import numpy as np

def loss(w):
    # Simple quadratic loss: L = w₁² + w₂²
    return w[0]**2 + w[1]**2

def gradient(w):
    # ∇L = [2w₁, 2w₂]
    return np.array([2*w[0], 2*w[1]])

# Start at some point
w = np.array([3.0, 4.0])
learning_rate = 0.1

# Gradient descent step
for i in range(5):
    grad = gradient(w)
    w = w - learning_rate * grad  # Move opposite to gradient
    print(f"Step {i+1}: w = {w}, loss = {loss(w):.4f}")` },
        { type: 'keypoints', points: [
          'Derivatives measure rate of change (slope at a point)',
          'Partial derivatives hold other variables constant',
          'The gradient is a vector of all partial derivatives',
          'Gradient points toward steepest increase; we go opposite to minimize'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the derivative of f(x) = x⁴?',
          options: ['x³', '4x³', '4x⁴', 'x⁵/5'],
          correct: 1,
          explanation: 'Using the power rule: d/dx[xⁿ] = n·xⁿ⁻¹, so d/dx[x⁴] = 4x³.'
        },
        {
          type: 'multiple-choice',
          question: 'If f(x, y) = 2xy + y³, what is ∂f/∂x?',
          options: ['2y', '2x + 3y²', '2xy', '2x'],
          correct: 0,
          explanation: 'Taking the partial derivative with respect to x, we treat y as a constant. d/dx[2xy] = 2y, and d/dx[y³] = 0 (constant).'
        },
        {
          type: 'multiple-choice',
          question: 'In gradient descent, we move in which direction relative to the gradient?',
          options: ['Same direction', 'Opposite direction', 'Perpendicular', 'Random'],
          correct: 1,
          explanation: 'The gradient points toward steepest increase. To minimize, we go in the opposite direction (steepest decrease).'
        }
      ]
    },
    {
      id: 'chain-rule',
      title: 'The Chain Rule',
      duration: '25 min',
      concepts: ['chain rule', 'composite functions', 'backpropagation'],
      content: [
        { type: 'heading', text: 'Why the Chain Rule Matters' },
        { type: 'paragraph', text: 'The chain rule is THE most important concept for understanding how neural networks learn. Without it, there would be no backpropagation, no deep learning, no modern AI.' },
        { type: 'callout', variant: 'info', text: 'Every time a neural network learns, it\'s applying the chain rule thousands or millions of times to calculate how each weight affects the final output.' },
        { type: 'heading', text: 'Composite Functions' },
        { type: 'paragraph', text: 'A composite function is a function of a function. If you have y = f(g(x)), first g transforms x, then f transforms the result.' },
        { type: 'paragraph', text: 'Example: Let f(u) = u² and g(x) = 3x + 1. Then:' },
        { type: 'formula', formula: 'y = f(g(x)) = (3x + 1)²' },
        { type: 'paragraph', text: 'Neural networks are deeply nested composite functions: the output of layer 1 feeds into layer 2, which feeds into layer 3, and so on.' },
        { type: 'heading', text: 'The Chain Rule Formula' },
        { type: 'paragraph', text: 'If y = f(g(x)), the chain rule says:' },
        { type: 'formula', formula: 'dy/dx = (dy/du) · (du/dx)\n\nwhere u = g(x)' },
        { type: 'paragraph', text: 'In words: multiply the derivative of the outer function by the derivative of the inner function.' },
        { type: 'subheading', text: 'Step-by-Step Example' },
        { type: 'paragraph', text: 'Let\'s find the derivative of y = (3x + 1)²' },
        { type: 'list', items: [
          'Let u = 3x + 1 (inner function)',
          'Then y = u² (outer function)',
          'dy/du = 2u (derivative of outer)',
          'du/dx = 3 (derivative of inner)',
          'dy/dx = dy/du · du/dx = 2u · 3 = 6(3x + 1)'
        ]},
        { type: 'code', language: 'python', filename: 'chain_rule.py', code: `import numpy as np

def y(x):
    return (3*x + 1)**2

def dy_dx(x):
    # Chain rule: 2(3x + 1) * 3 = 6(3x + 1)
    return 6 * (3*x + 1)

x = 2
print(f"y({x}) = {y(x)}")  # (7)² = 49
print(f"dy/dx at x={x}: {dy_dx(x)}")  # 6 * 7 = 42

# Verify numerically
h = 0.0001
numerical = (y(x + h) - y(x)) / h
print(f"Numerical check: {numerical:.4f}")  # ~42` },
        { type: 'heading', text: 'Chain Rule with Multiple Variables' },
        { type: 'paragraph', text: 'In neural networks, we have many variables. The chain rule extends naturally:' },
        { type: 'formula', formula: '∂L/∂w = (∂L/∂y) · (∂y/∂z) · (∂z/∂w)' },
        { type: 'paragraph', text: 'Each term represents how one quantity affects the next in the chain.' },
        { type: 'heading', text: 'Backpropagation: Chain Rule in Action' },
        { type: 'paragraph', text: 'In a neural network, we want to know how each weight affects the loss. The chain rule lets us compute this layer by layer, going backward from the output.' },
        { type: 'code', language: 'python', filename: 'backprop_example.py', code: `import numpy as np

# Simple 2-layer network: input -> hidden -> output
# Forward pass
x = 2.0
w1 = 0.5  # weight for layer 1
w2 = 0.3  # weight for layer 2

h = w1 * x        # hidden layer (before activation)
h_activated = max(0, h)  # ReLU activation
y = w2 * h_activated     # output
target = 1.0
loss = (y - target)**2   # squared error loss

print(f"Forward: x={x} -> h={h} -> y={y}")
print(f"Loss: {loss}")

# Backward pass (chain rule)
dL_dy = 2 * (y - target)  # derivative of loss w.r.t. output
dy_dh = w2 if h > 0 else 0  # derivative through ReLU
dL_dh = dL_dy * dy_dh

dL_dw2 = dL_dy * h_activated  # gradient for w2
dL_dw1 = dL_dh * x            # gradient for w1

print(f"Gradient w.r.t w1: {dL_dw1}")
print(f"Gradient w.r.t w2: {dL_dw2}")` },
        { type: 'callout', variant: 'tip', text: 'PyTorch and TensorFlow automatically apply the chain rule for you! That\'s what autograd/automatic differentiation does. But understanding the chain rule helps you debug and design networks.' },
        { type: 'keypoints', points: [
          'The chain rule differentiates composite functions: dy/dx = (dy/du)(du/dx)',
          'Neural networks are deeply nested composite functions',
          'Backpropagation applies the chain rule backward through the network',
          'Each layer\'s gradient depends on all layers after it'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the derivative of (2x)³ using the chain rule?',
          options: ['6x²', '24x²', '3(2x)²', '6(2x)²'],
          correct: 1,
          explanation: 'Let u = 2x, y = u³. dy/du = 3u², du/dx = 2. dy/dx = 3u² · 2 = 6(2x)² = 6·4x² = 24x².'
        },
        {
          type: 'multiple-choice',
          question: 'Why is the chain rule essential for neural networks?',
          options: ['It makes networks faster', 'It enables backpropagation', 'It reduces memory usage', 'It prevents overfitting'],
          correct: 1,
          explanation: 'The chain rule allows us to calculate how the loss changes with respect to each weight by propagating gradients backward through layers.'
        },
        {
          type: 'multiple-choice',
          question: 'In the chain ∂L/∂w = (∂L/∂y)(∂y/∂z)(∂z/∂w), what does ∂y/∂z represent?',
          options: ['How loss changes with w', 'How y changes when z changes', 'How z changes with w', 'The final output'],
          correct: 1,
          explanation: 'Each partial derivative shows how one variable affects the next in the chain. ∂y/∂z shows how y changes when z changes slightly.'
        }
      ]
    },
    {
      id: 'probability',
      title: 'Probability Fundamentals',
      duration: '25 min',
      concepts: ['probability', 'conditional probability', 'bayes theorem', 'independence'],
      content: [
        { type: 'heading', text: 'Why Probability in ML?' },
        { type: 'paragraph', text: 'Machine learning is fundamentally about making predictions under uncertainty. We rarely know things for certain - instead, we estimate probabilities. Is this email spam? Probably (85% chance). Will this customer churn? Likely (72% probability).' },
        { type: 'callout', variant: 'info', text: 'Classification outputs are often probabilities. The softmax layer in neural networks converts raw scores into probabilities that sum to 1.' },
        { type: 'heading', text: 'Basic Probability' },
        { type: 'paragraph', text: 'A probability is a number between 0 and 1 that represents how likely an event is to occur.' },
        { type: 'formula', formula: 'P(A) = favorable outcomes / total outcomes\n0 ≤ P(A) ≤ 1' },
        { type: 'list', items: [
          'P(A) = 0: Event A is impossible',
          'P(A) = 1: Event A is certain',
          'P(A) = 0.5: Event A is equally likely to happen or not'
        ]},
        { type: 'code', language: 'python', filename: 'basic_probability.py', code: `import numpy as np

# Simulating coin flips
np.random.seed(42)
flips = np.random.choice(['heads', 'tails'], size=10000)

p_heads = np.sum(flips == 'heads') / len(flips)
print(f"P(heads) from simulation: {p_heads:.4f}")  # ~0.5

# Simulating die rolls
rolls = np.random.randint(1, 7, size=10000)
p_six = np.sum(rolls == 6) / len(rolls)
print(f"P(rolling 6): {p_six:.4f}")  # ~0.1667 (1/6)` },
        { type: 'heading', text: 'Joint and Conditional Probability' },
        { type: 'subheading', text: 'Joint Probability' },
        { type: 'paragraph', text: 'The probability that both A AND B occur:' },
        { type: 'formula', formula: 'P(A and B) = P(A ∩ B)' },
        { type: 'subheading', text: 'Conditional Probability' },
        { type: 'paragraph', text: 'The probability of A given that B has already occurred:' },
        { type: 'formula', formula: 'P(A|B) = P(A and B) / P(B)' },
        { type: 'paragraph', text: 'Example: What\'s the probability a patient has a disease given they tested positive?' },
        { type: 'code', language: 'python', filename: 'conditional.py', code: `# Medical test example
# P(Disease) = 0.01 (1% of population has disease)
# P(Positive | Disease) = 0.99 (test is 99% accurate for sick people)
# P(Positive | No Disease) = 0.05 (5% false positive rate)

p_disease = 0.01
p_positive_given_disease = 0.99
p_positive_given_healthy = 0.05

# P(Positive) using law of total probability
p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_healthy * (1 - p_disease))
print(f"P(Positive Test): {p_positive:.4f}")  # 0.0594

# We'll use Bayes' theorem in the next section!` },
        { type: 'heading', text: 'Bayes\' Theorem' },
        { type: 'paragraph', text: 'Bayes\' theorem lets us update our beliefs based on new evidence. It\'s the foundation of Bayesian machine learning.' },
        { type: 'formula', formula: 'P(A|B) = P(B|A) · P(A) / P(B)' },
        { type: 'paragraph', text: 'In words: posterior = likelihood × prior / evidence' },
        { type: 'code', language: 'python', filename: 'bayes.py', code: `# Continuing medical test example
# What's P(Disease | Positive)?

p_disease = 0.01
p_positive_given_disease = 0.99
p_positive_given_healthy = 0.05

# P(Positive) from before
p_positive = 0.0594

# Bayes' theorem
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
print(f"P(Disease | Positive): {p_disease_given_positive:.4f}")  # 0.1667

# Surprising! Even with a positive test, only ~17% chance of disease
# This is because the disease is rare (low prior)` },
        { type: 'callout', variant: 'tip', text: 'Naive Bayes classifiers use Bayes\' theorem to classify data. They\'re simple but surprisingly effective for text classification.' },
        { type: 'heading', text: 'Independence' },
        { type: 'paragraph', text: 'Two events are independent if knowing one tells you nothing about the other.' },
        { type: 'formula', formula: 'If A and B are independent:\nP(A and B) = P(A) · P(B)\nP(A|B) = P(A)' },
        { type: 'paragraph', text: 'Example: Flipping a coin twice. The first flip doesn\'t affect the second.' },
        { type: 'code', language: 'python', filename: 'independence.py', code: `import numpy as np

# Two independent coin flips
p_heads = 0.5

# P(both heads) = P(heads) * P(heads)
p_both_heads = p_heads * p_heads
print(f"P(both heads): {p_both_heads}")  # 0.25

# Verify with simulation
np.random.seed(42)
n_trials = 10000
flip1 = np.random.random(n_trials) < 0.5
flip2 = np.random.random(n_trials) < 0.5
both_heads = np.sum(flip1 & flip2) / n_trials
print(f"Simulated: {both_heads:.4f}")  # ~0.25` },
        { type: 'keypoints', points: [
          'Probability quantifies uncertainty (0 to 1)',
          'Conditional probability: P(A|B) is probability of A given B occurred',
          'Bayes\' theorem updates beliefs: posterior = likelihood × prior / evidence',
          'Independent events: P(A and B) = P(A) × P(B)'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'If P(A) = 0.3 and P(B) = 0.4, and A and B are independent, what is P(A and B)?',
          options: ['0.7', '0.12', '0.1', '0.35'],
          correct: 1,
          explanation: 'For independent events, P(A and B) = P(A) × P(B) = 0.3 × 0.4 = 0.12.'
        },
        {
          type: 'multiple-choice',
          question: 'In Bayes\' theorem P(A|B) = P(B|A)P(A)/P(B), what is P(A) called?',
          options: ['Posterior', 'Likelihood', 'Prior', 'Evidence'],
          correct: 2,
          explanation: 'P(A) is the prior - our belief about A before seeing evidence B. P(B|A) is the likelihood, and P(A|B) is the posterior.'
        },
        {
          type: 'multiple-choice',
          question: 'A test has 99% accuracy but the disease is rare (1%). If you test positive, are you likely sick?',
          options: ['Yes, 99% likely', 'Yes, about 50% likely', 'No, only about 17% likely', 'Cannot determine'],
          correct: 2,
          explanation: 'Due to the base rate fallacy, even with a good test and positive result, the low prior (1% disease rate) means only about 17% chance of actually having the disease.'
        }
      ]
    },
    {
      id: 'distributions',
      title: 'Distributions and Expectations',
      duration: '25 min',
      concepts: ['distributions', 'mean', 'variance', 'normal distribution', 'expected value'],
      content: [
        { type: 'heading', text: 'What is a Distribution?' },
        { type: 'paragraph', text: 'A probability distribution describes all possible values a random variable can take and how likely each is. It\'s like a blueprint for randomness.' },
        { type: 'paragraph', text: 'There are two types:' },
        { type: 'list', items: [
          'Discrete distributions: finite or countable values (e.g., dice rolls)',
          'Continuous distributions: infinite possible values (e.g., height, temperature)'
        ]},
        { type: 'heading', text: 'Expected Value (Mean)' },
        { type: 'paragraph', text: 'The expected value is the average outcome if you repeated an experiment infinitely many times. It\'s the "center" of the distribution.' },
        { type: 'formula', formula: 'E[X] = Σ xᵢ · P(xᵢ)    (discrete)\nE[X] = ∫ x · f(x) dx    (continuous)' },
        { type: 'code', language: 'python', filename: 'expected_value.py', code: `import numpy as np

# Expected value of a fair die
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

expected = sum(x * p for x, p in zip(outcomes, probabilities))
print(f"E[die roll] = {expected:.4f}")  # 3.5

# Verify with simulation
rolls = np.random.randint(1, 7, size=100000)
print(f"Simulated mean: {np.mean(rolls):.4f}")  # ~3.5` },
        { type: 'heading', text: 'Variance and Standard Deviation' },
        { type: 'paragraph', text: 'Variance measures how spread out the distribution is. It\'s the expected squared distance from the mean.' },
        { type: 'formula', formula: 'Var(X) = E[(X - μ)²] = E[X²] - (E[X])²' },
        { type: 'paragraph', text: 'Standard deviation is the square root of variance, in the same units as the data:' },
        { type: 'formula', formula: 'σ = √Var(X)' },
        { type: 'code', language: 'python', filename: 'variance.py', code: `import numpy as np

# Compare two distributions with same mean but different variance
np.random.seed(42)

# Low variance
data_low_var = np.random.normal(loc=10, scale=1, size=1000)
# High variance
data_high_var = np.random.normal(loc=10, scale=5, size=1000)

print(f"Low variance - Mean: {np.mean(data_low_var):.2f}, Std: {np.std(data_low_var):.2f}")
print(f"High variance - Mean: {np.mean(data_high_var):.2f}, Std: {np.std(data_high_var):.2f}")` },
        { type: 'heading', text: 'The Normal (Gaussian) Distribution' },
        { type: 'paragraph', text: 'The normal distribution is the most important distribution in statistics. It\'s the famous "bell curve" and appears everywhere in nature.' },
        { type: 'formula', formula: 'f(x) = (1/σ√2π) · e^(-(x-μ)²/2σ²)' },
        { type: 'paragraph', text: 'It\'s completely described by two parameters: mean (μ) and standard deviation (σ).' },
        { type: 'callout', variant: 'info', text: 'The 68-95-99.7 rule: About 68% of data falls within 1 std of mean, 95% within 2 std, and 99.7% within 3 std.' },
        { type: 'code', language: 'python', filename: 'normal_dist.py', code: `import numpy as np

# Generate normal data
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=10000)

# Check the 68-95-99.7 rule
within_1_std = np.sum(np.abs(data) < 1) / len(data)
within_2_std = np.sum(np.abs(data) < 2) / len(data)
within_3_std = np.sum(np.abs(data) < 3) / len(data)

print(f"Within 1 std: {within_1_std*100:.1f}%")  # ~68%
print(f"Within 2 std: {within_2_std*100:.1f}%")  # ~95%
print(f"Within 3 std: {within_3_std*100:.1f}%")  # ~99.7%` },
        { type: 'heading', text: 'Why Normal Distribution Matters in ML' },
        { type: 'list', items: [
          'Weight initialization: Neural network weights are often initialized from a normal distribution',
          'Noise modeling: Regression assumes normally distributed errors',
          'Latent spaces: VAEs use normal distributions in their latent space',
          'Central Limit Theorem: Sums of random variables tend toward normal'
        ]},
        { type: 'code', language: 'python', filename: 'weight_init.py', code: `import numpy as np

# Xavier/Glorot initialization for neural network weights
def xavier_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, size=(fan_out, fan_in))

# Initialize weights for a layer: 100 inputs -> 50 outputs
weights = xavier_init(100, 50)
print(f"Weight shape: {weights.shape}")
print(f"Weight mean: {np.mean(weights):.6f}")  # ~0
print(f"Weight std: {np.std(weights):.4f}")  # ~0.115` },
        { type: 'keypoints', points: [
          'Distributions describe the probabilities of all possible outcomes',
          'Expected value (mean) is the center of a distribution',
          'Variance measures spread; standard deviation is √variance',
          'Normal distribution is central to ML: weight init, noise modeling, latent spaces'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What percentage of data falls within 2 standard deviations of the mean in a normal distribution?',
          options: ['68%', '95%', '99.7%', '50%'],
          correct: 1,
          explanation: 'The 68-95-99.7 rule: 68% within 1 std, 95% within 2 std, 99.7% within 3 std.'
        },
        {
          type: 'multiple-choice',
          question: 'If E[X] = 5 and E[X²] = 29, what is Var(X)?',
          options: ['24', '4', '34', '5'],
          correct: 1,
          explanation: 'Var(X) = E[X²] - (E[X])² = 29 - 25 = 4.'
        },
        {
          type: 'multiple-choice',
          question: 'Why are neural network weights often initialized from a normal distribution?',
          options: ['It\'s faster to compute', 'It prevents symmetry breaking', 'It provides good variance for stable gradients', 'All of the above'],
          correct: 2,
          explanation: 'Careful initialization with proper variance (like Xavier/He init) helps maintain stable gradients during training, preventing vanishing or exploding gradients.'
        }
      ]
    },
    {
      id: 'statistics-ml',
      title: 'Statistics for ML',
      duration: '20 min',
      concepts: ['correlation', 'covariance', 'sampling', 'hypothesis testing'],
      content: [
        { type: 'heading', text: 'Covariance and Correlation' },
        { type: 'paragraph', text: 'Covariance measures how two variables change together. Positive covariance means they tend to increase together; negative means one decreases as the other increases.' },
        { type: 'formula', formula: 'Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]' },
        { type: 'paragraph', text: 'Correlation is covariance normalized to [-1, 1], making it easier to interpret:' },
        { type: 'formula', formula: 'ρ(X, Y) = Cov(X, Y) / (σₓ · σᵧ)' },
        { type: 'list', items: [
          'ρ = 1: Perfect positive correlation',
          'ρ = 0: No linear correlation',
          'ρ = -1: Perfect negative correlation'
        ]},
        { type: 'code', language: 'python', filename: 'correlation.py', code: `import numpy as np

np.random.seed(42)

# Generate correlated data
x = np.random.normal(0, 1, 1000)
y_positive = x + np.random.normal(0, 0.5, 1000)  # Positively correlated
y_negative = -x + np.random.normal(0, 0.5, 1000)  # Negatively correlated
y_uncorrelated = np.random.normal(0, 1, 1000)  # Uncorrelated

print(f"Positive correlation: {np.corrcoef(x, y_positive)[0,1]:.3f}")  # ~0.89
print(f"Negative correlation: {np.corrcoef(x, y_negative)[0,1]:.3f}")  # ~-0.89
print(f"No correlation: {np.corrcoef(x, y_uncorrelated)[0,1]:.3f}")  # ~0` },
        { type: 'callout', variant: 'warning', text: 'Correlation does not imply causation! Two variables can be correlated because of a hidden third variable (confounding) or pure coincidence.' },
        { type: 'heading', text: 'Sampling and Estimation' },
        { type: 'paragraph', text: 'We rarely have access to entire populations. Instead, we work with samples and estimate population parameters.' },
        { type: 'paragraph', text: 'Key concepts:' },
        { type: 'list', items: [
          'Sample mean: Estimate of population mean',
          'Sample variance: Estimate of population variance',
          'Standard error: How much sample means vary'
        ]},
        { type: 'formula', formula: 'Standard Error = σ / √n' },
        { type: 'paragraph', text: 'Larger samples give more precise estimates (smaller standard error).' },
        { type: 'code', language: 'python', filename: 'sampling.py', code: `import numpy as np

np.random.seed(42)

# True population
population_mean = 100
population = np.random.normal(population_mean, 15, size=100000)

# Take samples of different sizes
for n in [10, 100, 1000]:
    sample = np.random.choice(population, size=n)
    sample_mean = np.mean(sample)
    standard_error = np.std(sample) / np.sqrt(n)
    print(f"n={n}: Mean={sample_mean:.2f}, SE={standard_error:.2f}")` },
        { type: 'heading', text: 'Maximum Likelihood Estimation' },
        { type: 'paragraph', text: 'MLE finds parameter values that make the observed data most probable. It\'s the foundation of many ML algorithms.' },
        { type: 'formula', formula: 'θ_MLE = argmax P(data | θ)' },
        { type: 'paragraph', text: 'Example: For a normal distribution, the MLE for the mean is simply the sample mean, and the MLE for variance is the sample variance.' },
        { type: 'code', language: 'python', filename: 'mle.py', code: `import numpy as np
from scipy import stats

np.random.seed(42)

# Generate data from unknown normal distribution
true_mean, true_std = 5.0, 2.0
data = np.random.normal(true_mean, true_std, size=1000)

# MLE estimates
mle_mean = np.mean(data)
mle_std = np.std(data)

print(f"True parameters: μ={true_mean}, σ={true_std}")
print(f"MLE estimates: μ={mle_mean:.3f}, σ={mle_std:.3f}")` },
        { type: 'heading', text: 'Cross-Validation: Statistical Model Selection' },
        { type: 'paragraph', text: 'Cross-validation uses statistical sampling to estimate how well a model generalizes to unseen data.' },
        { type: 'code', language: 'python', filename: 'cross_val.py', code: `import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Generate data
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 0, 0, 3]) + np.random.randn(100) * 0.5

# Cross-validation
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5)

print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")` },
        { type: 'keypoints', points: [
          'Correlation measures linear relationship (-1 to 1)',
          'Standard error decreases with √n (larger samples = more precision)',
          'MLE finds parameters that maximize data probability',
          'Cross-validation estimates generalization performance'
        ]}
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'If the correlation between X and Y is 0.9, what can we conclude?',
          options: ['X causes Y', 'Y causes X', 'X and Y have a strong linear relationship', 'X and Y are independent'],
          correct: 2,
          explanation: 'Correlation only measures linear relationship strength, not causation. A correlation of 0.9 indicates a strong positive linear relationship.'
        },
        {
          type: 'multiple-choice',
          question: 'If you double your sample size, what happens to the standard error?',
          options: ['Halves', 'Decreases by √2', 'Doubles', 'Stays the same'],
          correct: 1,
          explanation: 'Standard error = σ/√n. If n doubles, √n increases by √2, so SE decreases by a factor of √2 ≈ 1.41.'
        },
        {
          type: 'multiple-choice',
          question: 'What does Maximum Likelihood Estimation find?',
          options: ['The maximum possible data', 'Parameters that make data most probable', 'The minimum error', 'The maximum variance'],
          correct: 1,
          explanation: 'MLE finds parameter values θ that maximize the probability P(data|θ), i.e., parameters that make the observed data most likely.'
        }
      ]
    }
  ]
}
