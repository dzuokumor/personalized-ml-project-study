export const pythonforml = {
  id: 'python-for-ml',
  title: 'Python for Machine Learning',
  description: 'Master the essential Python libraries for ML: NumPy, Pandas, Matplotlib, and Scikit-learn. From array operations to building your first models.',
  difficulty: 'beginner',
  estimatedhours: 10,
  lessons: [
    {
      id: 'numpy-essentials',
      title: 'NumPy Essentials',
      duration: '45 min',
      content: [
        {
          type: 'text',
          content: `NumPy (Numerical Python) is the foundation of virtually all machine learning in Python. Before we dive into code, let's understand **why** NumPy exists and what problem it solves.`
        },
        {
          type: 'heading',
          content: 'Why Not Just Use Python Lists?'
        },
        {
          type: 'text',
          content: `Python lists are flexible - they can hold any type of data. But this flexibility comes at a cost:

**The Problem with Python Lists:**
- Each element is a full Python object with overhead (type info, reference count, etc.)
- Elements are scattered in memory (not contiguous)
- Operations require Python loops (slow)

**NumPy Arrays:**
- Elements are stored contiguously in memory
- All elements must be the same type (no overhead per element)
- Operations use optimized C code under the hood

This difference matters enormously. Multiplying two lists of 1 million numbers: Python takes ~500ms, NumPy takes ~2ms. That's 250x faster!`
        },
        {
          type: 'heading',
          content: 'Creating Arrays'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np

# From a Python list
arr = np.array([1, 2, 3, 4, 5])
print(arr)          # [1 2 3 4 5]
print(arr.dtype)    # int64 (NumPy automatically chose the type)

# 2D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix.shape)  # (2, 3) - 2 rows, 3 columns

# Common creation functions
zeros = np.zeros((3, 4))      # 3x4 matrix of zeros
ones = np.ones((2, 2))        # 2x2 matrix of ones
identity = np.eye(3)          # 3x3 identity matrix
random = np.random.randn(3, 3) # 3x3 matrix of random normal values

# Range of values
sequential = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linear = np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]`
        },
        {
          type: 'heading',
          content: 'Array Indexing and Slicing'
        },
        {
          type: 'text',
          content: `NumPy indexing is similar to Python lists but much more powerful for multi-dimensional arrays.`
        },
        {
          type: 'code',
          language: 'python',
          content: `arr = np.array([10, 20, 30, 40, 50])

# Basic indexing (0-indexed)
print(arr[0])      # 10 (first element)
print(arr[-1])     # 50 (last element)
print(arr[1:4])    # [20 30 40] (elements 1, 2, 3)

# 2D indexing
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix[0, 0])     # 1 (row 0, col 0)
print(matrix[1, 2])     # 6 (row 1, col 2)
print(matrix[0])        # [1 2 3] (entire first row)
print(matrix[:, 1])     # [2 5 8] (entire second column)
print(matrix[0:2, 1:3]) # [[2 3] [5 6]] (submatrix)

# Boolean indexing (very powerful!)
arr = np.array([1, 5, 3, 8, 2, 9])
print(arr[arr > 4])     # [5 8 9] (all elements > 4)
print(arr[arr % 2 == 0]) # [8 2] (all even elements)`
        },
        {
          type: 'heading',
          content: 'Vectorized Operations'
        },
        {
          type: 'text',
          content: `This is where NumPy truly shines. Instead of writing loops, you perform operations on entire arrays at once. The operation is applied element-wise.`
        },
        {
          type: 'code',
          language: 'python',
          content: `a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise operations
print(a + b)      # [11 22 33 44]
print(a * b)      # [10 40 90 160]
print(a ** 2)     # [1 4 9 16]
print(np.sqrt(a)) # [1. 1.414 1.732 2.]

# Broadcasting: operations with scalars
print(a * 10)     # [10 20 30 40]
print(a + 100)    # [101 102 103 104]

# Comparison (returns boolean array)
print(a > 2)      # [False False True True]
print(a == b/10)  # [True True True True]

# Mathematical functions
print(np.sum(a))       # 10
print(np.mean(a))      # 2.5
print(np.std(a))       # 1.118...
print(np.max(a))       # 4
print(np.argmax(a))    # 3 (index of max value)`
        },
        {
          type: 'heading',
          content: 'Matrix Operations'
        },
        {
          type: 'text',
          content: `Matrix operations are fundamental to ML. Neural networks are essentially series of matrix multiplications followed by non-linear functions.`
        },
        {
          type: 'code',
          language: 'python',
          content: `A = np.array([[1, 2],
               [3, 4]])
B = np.array([[5, 6],
               [7, 8]])

# Element-wise multiplication (NOT matrix multiplication)
print(A * B)
# [[ 5 12]
#  [21 32]]

# Matrix multiplication (dot product)
print(np.dot(A, B))  # or A @ B in Python 3.5+
# [[19 22]
#  [43 50]]

# How it works:
# [1,2] dot [5,7] = 1*5 + 2*7 = 19
# [1,2] dot [6,8] = 1*6 + 2*8 = 22
# etc.

# Transpose
print(A.T)
# [[1 3]
#  [2 4]]

# Inverse (if matrix is invertible)
print(np.linalg.inv(A))
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verify: A @ A_inv = Identity
print(A @ np.linalg.inv(A))
# [[1. 0.]
#  [0. 1.]]`
        },
        {
          type: 'heading',
          content: 'Reshaping Arrays'
        },
        {
          type: 'text',
          content: `Reshaping is essential in ML - you'll constantly need to transform data between different shapes for different layers and operations.`
        },
        {
          type: 'code',
          language: 'python',
          content: `arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 3x4 matrix
matrix = arr.reshape(3, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Reshape to 2x2x3 (3D array)
tensor = arr.reshape(2, 2, 3)
print(tensor.shape)  # (2, 2, 3)

# -1 means "figure out this dimension automatically"
print(arr.reshape(-1, 3))  # 4x3 matrix (12/3 = 4 rows)
print(arr.reshape(2, -1))  # 2x6 matrix (12/2 = 6 cols)

# Flatten back to 1D
print(matrix.flatten())  # [ 0  1  2 ...11]

# Add a dimension (useful for batch processing)
vec = np.array([1, 2, 3])
print(vec.shape)              # (3,)
print(vec.reshape(1, -1).shape) # (1, 3) - row vector
print(vec.reshape(-1, 1).shape) # (3, 1) - column vector
print(vec[np.newaxis, :].shape) # (1, 3) - same as reshape`
        },
        {
          type: 'heading',
          content: 'Broadcasting Rules'
        },
        {
          type: 'text',
          content: `Broadcasting allows NumPy to perform operations on arrays with different shapes. Understanding this is crucial - it's both powerful and a common source of bugs.

**Broadcasting Rules:**
1. If arrays have different numbers of dimensions, pad the smaller shape with 1s on the left
2. Arrays are compatible in a dimension if they have the same size OR one of them is 1
3. If compatible, the array with size 1 is "stretched" to match the other`
        },
        {
          type: 'code',
          language: 'python',
          content: `# Example 1: scalar + array
a = np.array([1, 2, 3])
print(a + 10)  # [11 12 13]
# 10 is broadcast to [10, 10, 10]

# Example 2: row + column
row = np.array([[1, 2, 3]])     # shape (1, 3)
col = np.array([[10], [20]])    # shape (2, 1)
print(row + col)
# [[11 12 13]
#  [21 22 23]]
# row is broadcast to [[1,2,3], [1,2,3]]
# col is broadcast to [[10,10,10], [20,20,20]]

# Example 3: Common ML pattern - normalizing features
data = np.array([[1, 200, 3000],
                 [2, 400, 6000],
                 [3, 600, 9000]])
mean = data.mean(axis=0)  # [2, 400, 6000] - mean of each column
std = data.std(axis=0)    # [0.816, 163.3, 2449.5]

normalized = (data - mean) / std  # broadcasting in action!
print(normalized.mean(axis=0))   # [0, 0, 0] - each column centered
print(normalized.std(axis=0))    # [1, 1, 1] - each column scaled`
        },
        {
          type: 'text',
          content: `**Key NumPy Takeaways:**
- Always prefer vectorized operations over loops
- Understand shapes - use .shape frequently
- Broadcasting is powerful but can cause subtle bugs
- For ML: reshape, transpose, and matrix multiply are your friends`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the result of np.array([1,2,3]) * np.array([2,2,2])?',
          options: ['[1, 4, 6]', '[2, 4, 6]', '14', 'Error'],
          correct: 1,
          explanation: 'NumPy performs element-wise multiplication: 1*2=2, 2*2=4, 3*2=6, giving [2, 4, 6].'
        },
        {
          type: 'multiple-choice',
          question: 'Which operation performs matrix multiplication in NumPy?',
          options: ['A * B', 'A.multiply(B)', 'np.dot(A, B) or A @ B', 'np.multiply(A, B)'],
          correct: 2,
          explanation: 'np.dot(A, B) or the @ operator perform matrix multiplication. A * B is element-wise.'
        },
        {
          type: 'multiple-choice',
          question: 'What does arr.reshape(-1, 1) do?',
          options: [
            'Removes the last column',
            'Reshapes to a column vector (n rows, 1 column)',
            'Flattens the array',
            'Causes an error because -1 is invalid'
          ],
          correct: 1,
          explanation: 'The -1 tells NumPy to calculate that dimension automatically. With 1 column, it creates a column vector.'
        }
      ]
    },
    {
      id: 'pandas-data-manipulation',
      title: 'Pandas for Data Manipulation',
      duration: '50 min',
      content: [
        {
          type: 'text',
          content: `While NumPy excels at numerical computation, real-world data is messy: missing values, mixed types, labels, dates. **Pandas** builds on NumPy to handle this complexity.`
        },
        {
          type: 'heading',
          content: 'Series and DataFrames'
        },
        {
          type: 'text',
          content: `Pandas has two main data structures:
- **Series**: A 1D labeled array (like a column in a spreadsheet)
- **DataFrame**: A 2D labeled data structure (like a spreadsheet or SQL table)`
        },
        {
          type: 'code',
          language: 'python',
          content: `import pandas as pd
import numpy as np

# Series - 1D with labels (index)
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
# a    10
# b    20
# c    30
# d    40

print(s['b'])      # 20 (access by label)
print(s[1])        # 20 (access by position)
print(s[s > 15])   # b:20, c:30, d:40 (boolean indexing)

# DataFrame - 2D labeled structure
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
print(df)
#       name  age  salary
# 0    Alice   25   50000
# 1      Bob   30   60000
# 2  Charlie   35   70000`
        },
        {
          type: 'heading',
          content: 'Reading and Writing Data'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Reading data
df = pd.read_csv('data.csv')           # CSV file
df = pd.read_excel('data.xlsx')        # Excel file
df = pd.read_json('data.json')         # JSON file
df = pd.read_sql('SELECT * FROM table', connection)  # SQL

# Writing data
df.to_csv('output.csv', index=False)   # Don't save row numbers
df.to_excel('output.xlsx')
df.to_json('output.json')

# Quick exploration
print(df.head())        # First 5 rows
print(df.tail(3))       # Last 3 rows
print(df.shape)         # (rows, columns)
print(df.columns)       # Column names
print(df.dtypes)        # Data types of each column
print(df.info())        # Summary including non-null counts
print(df.describe())    # Statistical summary`
        },
        {
          type: 'heading',
          content: 'Selecting Data'
        },
        {
          type: 'text',
          content: `There are multiple ways to select data in Pandas. Understanding the differences is crucial:`
        },
        {
          type: 'code',
          language: 'python',
          content: `df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}, index=['w', 'x', 'y', 'z'])

# Selecting columns
print(df['A'])           # Single column as Series
print(df[['A', 'B']])    # Multiple columns as DataFrame

# .loc - selection by LABEL
print(df.loc['x'])            # Row with label 'x'
print(df.loc['x', 'B'])       # Specific cell: row 'x', column 'B'
print(df.loc['w':'y', 'A':'B']) # Slice (inclusive on both ends!)

# .iloc - selection by INTEGER position
print(df.iloc[1])             # Second row (index 1)
print(df.iloc[1, 2])          # Row 1, Column 2
print(df.iloc[0:2, 0:2])      # First 2 rows, first 2 columns

# Boolean selection
print(df[df['A'] > 2])        # Rows where A > 2
print(df[(df['A'] > 1) & (df['B'] < 8)])  # Multiple conditions

# Query method (more readable for complex conditions)
print(df.query('A > 2 and B < 8'))`
        },
        {
          type: 'heading',
          content: 'Handling Missing Data'
        },
        {
          type: 'text',
          content: `Missing data is ubiquitous in real datasets. Pandas represents missing values as NaN (Not a Number). How you handle them significantly impacts your model.`
        },
        {
          type: 'code',
          language: 'python',
          content: `df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# Detecting missing values
print(df.isnull())           # Boolean mask
print(df.isnull().sum())     # Count per column
print(df.isnull().any())     # Does column have any missing?

# Removing missing values
df.dropna()                  # Remove any row with NaN
df.dropna(axis=1)            # Remove any column with NaN
df.dropna(thresh=2)          # Keep rows with at least 2 non-NaN

# Filling missing values
df.fillna(0)                 # Fill with constant
df.fillna(df.mean())         # Fill with column means
df.fillna(method='ffill')    # Forward fill (previous value)
df.fillna(method='bfill')    # Backward fill (next value)

# Interpolation
df.interpolate()             # Linear interpolation`
        },
        {
          type: 'heading',
          content: 'Data Transformation'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Adding/modifying columns
df['D'] = df['A'] + df['B']           # From existing columns
df['E'] = df['A'].apply(lambda x: x**2) # Apply function

# Renaming
df.rename(columns={'A': 'alpha', 'B': 'beta'})

# Sorting
df.sort_values('A')                   # Sort by column
df.sort_values(['A', 'B'], ascending=[True, False])

# Grouping and aggregation (very powerful!)
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

grouped = df.groupby('category')
print(grouped.mean())         # Mean per category
print(grouped.agg(['mean', 'sum', 'count']))  # Multiple aggregations

# Pivot tables
df = pd.DataFrame({
    'date': ['Mon', 'Mon', 'Tue', 'Tue'],
    'city': ['NYC', 'LA', 'NYC', 'LA'],
    'sales': [100, 200, 150, 250]
})
print(pd.pivot_table(df, values='sales', index='date', columns='city'))`
        },
        {
          type: 'heading',
          content: 'Merging DataFrames'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Concatenation (stacking)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
pd.concat([df1, df2])                  # Stack vertically
pd.concat([df1, df2], axis=1)          # Stack horizontally

# Merging (like SQL JOIN)
users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})
orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'user_id': [1, 1, 2],
    'amount': [50, 30, 100]
})

# Inner join (only matching rows)
pd.merge(users, orders, on='user_id')

# Left join (all users, matching orders)
pd.merge(users, orders, on='user_id', how='left')

# Types: 'inner', 'left', 'right', 'outer'`
        },
        {
          type: 'text',
          content: `**Pandas Best Practices:**
- Use vectorized operations instead of loops
- Chain operations for readability: df.dropna().groupby('col').mean()
- Be explicit with .loc and .iloc to avoid warnings
- Profile memory with df.info(memory_usage='deep') for large datasets`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the difference between df.loc[0] and df.iloc[0]?',
          options: [
            'They are always the same',
            '.loc uses label-based indexing, .iloc uses integer position',
            '.loc is for columns, .iloc is for rows',
            '.iloc is deprecated'
          ],
          correct: 1,
          explanation: '.loc accesses by label (which could be a string, date, or even 0 if that\'s the label). .iloc always accesses by integer position regardless of what the labels are.'
        },
        {
          type: 'multiple-choice',
          question: 'How do you count missing values per column in a DataFrame?',
          options: ['df.count()', 'df.isnull().sum()', 'df.missing()', 'len(df[df.isnull()])'],
          correct: 1,
          explanation: 'df.isnull() creates a boolean mask, and .sum() counts True values (which represent missing values) for each column.'
        },
        {
          type: 'multiple-choice',
          question: 'Which method groups data and computes aggregate statistics?',
          options: ['df.aggregate()', 'df.pivot()', 'df.groupby()', 'df.split()'],
          correct: 2,
          explanation: 'groupby() splits the data by a column\'s values, then you can apply aggregate functions like mean(), sum(), count() to each group.'
        }
      ]
    },
    {
      id: 'matplotlib-visualization',
      title: 'Matplotlib & Data Visualization',
      duration: '40 min',
      content: [
        {
          type: 'text',
          content: `Visualization is not just for presentation - it's a crucial tool for understanding your data and diagnosing model problems. We'll cover Matplotlib (foundational) and Seaborn (statistical visualizations).`
        },
        {
          type: 'heading',
          content: 'Matplotlib Fundamentals'
        },
        {
          type: 'text',
          content: `Matplotlib has two interfaces:
- **pyplot**: Quick plotting (like MATLAB)
- **Object-oriented**: More control, better for complex plots

Start with pyplot, graduate to OO for anything serious.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.show()

# Multiple lines with legend
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.show()

# Object-oriented approach (preferred for complex plots)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), 'b-', linewidth=2, label='sin')
ax.plot(x, np.cos(x), 'r--', linewidth=2, label='cos')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Trigonometric Functions', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'Essential Plot Types'
        },
        {
          type: 'code',
          language: 'python',
          content: `import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Scatter plot - relationships between variables
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
axes[0, 0].scatter(x, y, alpha=0.6)
axes[0, 0].set_title('Scatter Plot')

# 2. Histogram - distribution of a single variable
data = np.random.randn(1000)
axes[0, 1].hist(data, bins=30, edgecolor='black')
axes[0, 1].set_title('Histogram')

# 3. Bar chart - categorical comparisons
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[0, 2].bar(categories, values)
axes[0, 2].set_title('Bar Chart')

# 4. Box plot - distribution summary
data = [np.random.randn(100) + i for i in range(4)]
axes[1, 0].boxplot(data)
axes[1, 0].set_title('Box Plot')

# 5. Heatmap - matrix visualization
matrix = np.random.rand(10, 10)
im = axes[1, 1].imshow(matrix, cmap='viridis')
plt.colorbar(im, ax=axes[1, 1])
axes[1, 1].set_title('Heatmap')

# 6. Line with error bars - uncertainty
x = np.arange(5)
y = [2, 4, 3, 5, 4]
errors = [0.5, 0.3, 0.4, 0.2, 0.3]
axes[1, 2].errorbar(x, y, yerr=errors, fmt='o-', capsize=5)
axes[1, 2].set_title('Error Bars')

plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'Visualizations for ML'
        },
        {
          type: 'code',
          language: 'python',
          content: `# Loss curve - track training progress
epochs = range(1, 101)
train_loss = 1 / np.sqrt(np.array(epochs)) + np.random.randn(100) * 0.02
val_loss = 1.2 / np.sqrt(np.array(epochs)) + np.random.randn(100) * 0.03

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0]
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature distributions by class
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

plt.figure(figsize=(12, 4))
for i, col in enumerate(iris.feature_names[:4]):
    plt.subplot(1, 4, i+1)
    for species in range(3):
        plt.hist(df[df['species']==species][col], alpha=0.5, label=f'Class {species}')
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.show()`
        },
        {
          type: 'heading',
          content: 'Seaborn for Statistical Plots'
        },
        {
          type: 'code',
          language: 'python',
          content: `import seaborn as sns

# Seaborn works beautifully with Pandas DataFrames
tips = sns.load_dataset('tips')

# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(tips['total_bill'], kde=True, ax=axes[0])
sns.kdeplot(tips['total_bill'], ax=axes[1])
sns.boxplot(x='day', y='total_bill', data=tips, ax=axes[2])
plt.tight_layout()
plt.show()

# Relationship plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(x='total_bill', y='tip', hue='smoker', data=tips, ax=axes[0])
sns.regplot(x='total_bill', y='tip', data=tips, ax=axes[1])
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
plt.show()

# Pair plot - all pairwise relationships
sns.pairplot(tips, hue='smoker')
plt.show()

# Correlation heatmap
corr = tips[['total_bill', 'tip', 'size']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.show()`
        },
        {
          type: 'text',
          content: `**Visualization Best Practices:**
- Always label axes and include titles
- Use appropriate plot types (don't use pie charts)
- Consider colorblind-friendly palettes
- For ML: always plot learning curves, confusion matrices, and feature distributions`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Which plot type is best for showing the distribution of a single continuous variable?',
          options: ['Scatter plot', 'Line plot', 'Histogram', 'Bar chart'],
          correct: 2,
          explanation: 'Histograms show the distribution of a single continuous variable by grouping values into bins and showing the frequency of each bin.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the advantage of using the object-oriented Matplotlib interface?',
          options: [
            'It runs faster',
            'It provides more control over plot elements and is better for complex multi-plot figures',
            'It uses less memory',
            'It is simpler to use'
          ],
          correct: 1,
          explanation: 'The OO interface (fig, ax = plt.subplots()) gives you explicit references to figure and axes objects, making it easier to customize complex plots with multiple subplots.'
        }
      ]
    },
    {
      id: 'scikit-learn-intro',
      title: 'Scikit-learn Introduction',
      duration: '50 min',
      content: [
        {
          type: 'text',
          content: `Scikit-learn is THE library for classical machine learning in Python. It provides a consistent, elegant API for dozens of algorithms. Learn this API once, and you can use almost any ML algorithm.`
        },
        {
          type: 'heading',
          content: 'The Scikit-learn API Pattern'
        },
        {
          type: 'text',
          content: `Every scikit-learn model follows the same pattern:

1. **Import** the model class
2. **Instantiate** the model with hyperparameters
3. **Fit** the model to training data
4. **Predict** on new data (or transform for preprocessing)

This consistency is scikit-learn's superpower.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# All follow the same pattern:

# 1. Regression example
model = LinearRegression()           # Instantiate
model.fit(X_train, y_train)          # Fit
predictions = model.predict(X_test)  # Predict

# 2. Classification example
clf = DecisionTreeClassifier(max_depth=5)  # Instantiate with hyperparams
clf.fit(X_train, y_train)                  # Fit
predictions = clf.predict(X_test)          # Predict
probabilities = clf.predict_proba(X_test)  # Class probabilities

# 3. Preprocessing example
scaler = StandardScaler()                # Instantiate
scaler.fit(X_train)                      # Fit (learn mean, std)
X_train_scaled = scaler.transform(X_train) # Transform
X_test_scaled = scaler.transform(X_test)   # Use same transformation!

# Or combine fit and transform:
X_train_scaled = scaler.fit_transform(X_train)`
        },
        {
          type: 'heading',
          content: 'Your First ML Pipeline'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
print(f"Shape: {X.shape}")

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 3. Preprocessing: scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Important: use same scaler!

# 4. Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))`
        },
        {
          type: 'heading',
          content: 'Common Estimators'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ============ CLASSIFICATION ============
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    SVC(kernel='rbf'),
    KNeighborsClassifier(n_neighbors=5),
    GaussianNB()
]

# ============ REGRESSION ============
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

regressors = [
    LinearRegression(),
    Ridge(alpha=1.0),          # L2 regularization
    Lasso(alpha=1.0),          # L1 regularization
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100),
    SVR(kernel='rbf')
]

# ============ CLUSTERING ============
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

clusterers = [
    KMeans(n_clusters=3),
    DBSCAN(eps=0.5, min_samples=5),
    AgglomerativeClustering(n_clusters=3)
]`
        },
        {
          type: 'heading',
          content: 'Preprocessing and Pipelines'
        },
        {
          type: 'text',
          content: `Proper preprocessing is crucial. Common issues:
- Features on different scales (age: 0-100, salary: 0-1M)
- Categorical variables (colors, categories)
- Missing values

Scikit-learn provides transformers for all of these.`
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.preprocessing import (
    StandardScaler,     # Zero mean, unit variance
    MinMaxScaler,       # Scale to [0, 1]
    RobustScaler,       # Uses median (robust to outliers)
    OneHotEncoder,      # Categorical to binary columns
    LabelEncoder        # Categorical to integers
)
from sklearn.impute import SimpleImputer

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding categorical variables
encoder = OneHotEncoder(sparse=False)
colors = [['red'], ['blue'], ['red'], ['green']]
encoded = encoder.fit_transform(colors)
# [[1, 0, 0],   # red
#  [0, 1, 0],   # blue
#  [1, 0, 0],   # red
#  [0, 0, 1]]   # green

# Handling missing values
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X_with_nan)

# ============ PIPELINES ============
# Combine preprocessing and model into a single object
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Now just:
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

# The pipeline handles everything in the right order!
# Important: transformations learned on train data are applied to test data`
        },
        {
          type: 'heading',
          content: 'Model Selection and Validation'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

# Cross-validation - more reliable than single train/test split
scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Grid search - find best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)`
        },
        {
          type: 'heading',
          content: 'Evaluation Metrics'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score, roc_curve,

    # Regression
    mean_squared_error, mean_absolute_error,
    r2_score
)

# Classification metrics
y_true = [0, 0, 1, 1, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 1, 0]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
print(f"F1: {f1_score(y_true, y_pred):.3f}")

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Regression metrics
y_true = [3.0, 2.5, 4.0, 5.0]
y_pred = [2.8, 2.7, 4.2, 4.5]

print(f"MSE: {mean_squared_error(y_true, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.3f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
print(f"R2: {r2_score(y_true, y_pred):.3f}")`
        },
        {
          type: 'text',
          content: `**Scikit-learn Tips:**
- Always split data before any preprocessing to prevent data leakage
- Use pipelines to keep preprocessing and modeling together
- Cross-validation gives more reliable estimates than a single split
- Choose metrics that match your problem (accuracy isn't always best)`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the correct order of operations for training a model?',
          options: [
            'predict → fit → split data',
            'split data → fit on all data → predict on test',
            'split data → fit on train → predict on test',
            'fit on all data → split → predict'
          ],
          correct: 2,
          explanation: 'You must split data first, then fit ONLY on training data, then predict on test data. Fitting on all data before splitting causes data leakage.'
        },
        {
          type: 'multiple-choice',
          question: 'Why should you use scaler.transform(X_test) instead of scaler.fit_transform(X_test)?',
          options: [
            'It runs faster',
            'To prevent data leakage - the scaler should only learn from training data',
            'It uses less memory',
            'There is no difference'
          ],
          correct: 1,
          explanation: 'The scaler learns the mean and std from training data. Using fit_transform on test data would learn different values, causing data leakage and inconsistent transformations.'
        },
        {
          type: 'multiple-choice',
          question: 'What does cross_val_score with cv=5 do?',
          options: [
            'Trains 5 different models',
            'Splits data into 5 parts, trains on 4, tests on 1, repeats 5 times',
            'Runs the model 5 times with same data',
            'Uses 5% of data for testing'
          ],
          correct: 1,
          explanation: '5-fold cross-validation splits data into 5 folds, uses 4 for training and 1 for testing, rotating which fold is the test set. This gives 5 scores for a more robust evaluation.'
        }
      ]
    },
    {
      id: 'working-with-datasets',
      title: 'Working with Datasets',
      duration: '45 min',
      content: [
        {
          type: 'text',
          content: `Real ML projects spend 80% of time on data - loading, cleaning, exploring, transforming. This lesson covers practical techniques for handling real-world datasets.`
        },
        {
          type: 'heading',
          content: 'Loading Different Data Sources'
        },
        {
          type: 'code',
          language: 'python',
          content: `import pandas as pd
import numpy as np

# ============ BUILT-IN DATASETS ============
from sklearn.datasets import (
    load_iris,           # Classification: flower species
    load_digits,         # Classification: handwritten digits
    load_boston,         # Regression: house prices
    load_wine,           # Classification: wine quality
    fetch_california_housing,  # Regression: larger house dataset
    make_classification, # Generate synthetic classification data
    make_regression      # Generate synthetic regression data
)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# ============ CSV FILES ============
df = pd.read_csv('data.csv')

# Common parameters:
df = pd.read_csv('data.csv',
    sep=',',              # Delimiter (could be '\\t', ';', etc.)
    header=0,             # Row to use as column names (None if no header)
    index_col=0,          # Column to use as index
    usecols=['col1', 'col2'],  # Only load specific columns
    dtype={'col1': int},  # Force data types
    na_values=['NA', 'missing', '-999'],  # What to treat as NaN
    nrows=1000,           # Only read first 1000 rows
    skiprows=10           # Skip first 10 rows
)

# ============ LARGE FILES ============
# Read in chunks for memory-constrained situations
chunks = pd.read_csv('big_file.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# ============ JSON ============
df = pd.read_json('data.json')

# ============ EXCEL ============
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# ============ URLs ============
url = 'https://raw.githubusercontent.com/example/data.csv'
df = pd.read_csv(url)`
        },
        {
          type: 'heading',
          content: 'Exploratory Data Analysis (EDA)'
        },
        {
          type: 'code',
          language: 'python',
          content: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load example dataset
df = sns.load_dataset('titanic')

# ============ FIRST LOOK ============
print(df.shape)                  # (891, 15)
print(df.head())                 # First 5 rows
print(df.info())                 # Column types, non-null counts
print(df.describe())             # Statistics for numerical columns
print(df.describe(include='all')) # Include categorical too

# ============ MISSING VALUES ============
print(df.isnull().sum())         # Count missing per column
print(df.isnull().sum() / len(df) * 100)  # Percent missing

# Visualize missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Values Pattern')
plt.show()

# ============ DISTRIBUTION OF TARGET ============
print(df['survived'].value_counts())
print(df['survived'].value_counts(normalize=True))  # Percentages

# ============ NUMERICAL FEATURES ============
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {list(numerical_cols)}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(numerical_cols[:6]):
    ax = axes[i//3, i%3]
    df[col].hist(ax=ax, bins=30)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# ============ CATEGORICAL FEATURES ============
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"Categorical columns: {list(categorical_cols)}")

for col in categorical_cols:
    print(f"\\n{col}:")
    print(df[col].value_counts())

# ============ CORRELATIONS ============
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# ============ RELATIONSHIPS WITH TARGET ============
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    df.groupby('survived')[col].hist(alpha=0.5, label=['Died', 'Survived'])
    plt.legend()
    plt.title(f'{col} by Survival')
    plt.show()`
        },
        {
          type: 'heading',
          content: 'Feature Engineering'
        },
        {
          type: 'text',
          content: `Feature engineering is the art of creating new features from existing data. Good features can dramatically improve model performance.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import pandas as pd
import numpy as np

# Example dataset
df = pd.DataFrame({
    'date': ['2023-01-15', '2023-03-20', '2023-07-04', '2023-12-25'],
    'price': [100, 150, 200, 180],
    'quantity': [5, 3, 8, 2],
    'category': ['A', 'B', 'A', 'C'],
    'description': ['red apple', 'green banana', 'red cherry', 'blue berry']
})

# ============ DATETIME FEATURES ============
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# ============ MATHEMATICAL COMBINATIONS ============
df['total_value'] = df['price'] * df['quantity']
df['price_log'] = np.log1p(df['price'])  # log(1+x), handles zeros
df['price_squared'] = df['price'] ** 2

# ============ BINNING ============
df['price_bin'] = pd.cut(df['price'], bins=[0, 100, 150, np.inf],
                          labels=['low', 'medium', 'high'])

# ============ ENCODING CATEGORICAL ============
# One-hot encoding
dummies = pd.get_dummies(df['category'], prefix='cat')
df = pd.concat([df, dummies], axis=1)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# ============ TEXT FEATURES ============
df['desc_length'] = df['description'].str.len()
df['word_count'] = df['description'].str.split().str.len()
df['has_red'] = df['description'].str.contains('red').astype(int)

# ============ AGGREGATION FEATURES ============
# When you have groups
df['avg_price_by_category'] = df.groupby('category')['price'].transform('mean')
df['count_by_category'] = df.groupby('category')['price'].transform('count')
df['price_rank_in_category'] = df.groupby('category')['price'].rank()

print(df.head())`
        },
        {
          type: 'heading',
          content: 'Train-Test Split Strategies'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    GroupShuffleSplit
)

# ============ BASIC SPLIT ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============ STRATIFIED SPLIT ============
# Maintains class proportions in both sets
# Essential for imbalanced datasets!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============ TIME SERIES SPLIT ============
# For time-dependent data - train on past, predict future
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # Train is always before test in time!

# ============ GROUP SPLIT ============
# When data has groups (e.g., multiple samples from same patient)
# Ensures all samples from a group are in same set
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(X, y, groups=patient_ids):
    X_train, X_test = X[train_idx], X[test_idx]

# ============ TRAIN/VALIDATION/TEST ============
# Best practice: 60/20/20 or 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
# 0.25 * 0.8 = 0.2, so final split is 60/20/20`
        },
        {
          type: 'heading',
          content: 'Handling Imbalanced Data'
        },
        {
          type: 'code',
          language: 'python',
          content: `from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Check class distribution
print(pd.Series(y).value_counts())

# ============ CLASS WEIGHTS ============
# Tell the model to pay more attention to minority class
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weights_dict = dict(zip(np.unique(y), class_weights))

# Use in model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
# or
model = RandomForestClassifier(class_weight=weights_dict)

# ============ OVERSAMPLING ============
# Create synthetic samples of minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Before: {np.bincount(y_train)}")
print(f"After: {np.bincount(y_resampled)}")

# ============ UNDERSAMPLING ============
# Remove samples from majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# ============ COMBINATION ============
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)`
        },
        {
          type: 'text',
          content: `**Dataset Best Practices:**
- Always explore your data before modeling
- Use stratified splits for classification problems
- Use time-based splits for time series data
- Handle imbalanced data - don't just use accuracy as your metric
- Document your feature engineering steps for reproducibility`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why should you use stratified train-test split for classification?',
          options: [
            'It runs faster',
            'It preserves class proportions in both train and test sets',
            'It prevents overfitting',
            'It improves accuracy'
          ],
          correct: 1,
          explanation: 'Stratified split ensures both train and test sets have the same proportion of each class, which is especially important for imbalanced datasets.'
        },
        {
          type: 'multiple-choice',
          question: 'For time series data, what type of train-test split should you use?',
          options: [
            'Random split',
            'Stratified split',
            'Time-based split where training data comes before test data',
            'Any split is fine'
          ],
          correct: 2,
          explanation: 'Time series data requires chronological splitting - you train on historical data and test on future data to simulate real-world usage and prevent data leakage.'
        }
      ]
    },
    {
      id: 'best-practices-debugging',
      title: 'Best Practices & Debugging',
      duration: '40 min',
      content: [
        {
          type: 'text',
          content: `Writing ML code that works once is easy. Writing ML code that's reproducible, maintainable, and debuggable is an art. This lesson covers practices that separate professional ML engineers from beginners.`
        },
        {
          type: 'heading',
          content: 'Reproducibility'
        },
        {
          type: 'text',
          content: `ML experiments must be reproducible. If you can't get the same results twice, you can't trust your results at all.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import random
import os

# ============ SET ALL RANDOM SEEDS ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For TensorFlow
    # import tensorflow as tf
    # tf.random.set_seed(seed)

    # For PyTorch
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

set_seed(42)

# ============ VERSION TRACKING ============
import sklearn
import pandas as pd
import numpy as np

print(f"sklearn: {sklearn.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# Save to requirements.txt for exact reproduction:
# pip freeze > requirements.txt

# ============ DATA VERSIONING ============
# Save processed data with timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'processed_data_{timestamp}.csv', index=False)

# Or use DVC (Data Version Control) for larger datasets`
        },
        {
          type: 'heading',
          content: 'Code Organization'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ============ PROJECT STRUCTURE ============
# project/
# ├── data/
# │   ├── raw/          # Original, immutable data
# │   └── processed/    # Cleaned data
# ├── notebooks/        # Exploration and experiments
# ├── src/
# │   ├── data/         # Data loading and processing
# │   ├── features/     # Feature engineering
# │   ├── models/       # Model definitions
# │   └── utils/        # Helper functions
# ├── models/           # Saved trained models
# ├── reports/          # Analysis outputs
# ├── config.yaml       # Hyperparameters and settings
# └── requirements.txt

# ============ CONFIGURATION FILES ============
# config.yaml
"""
data:
  train_path: data/processed/train.csv
  test_path: data/processed/test.csv

model:
  name: random_forest
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

training:
  test_size: 0.2
  cv_folds: 5
"""

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = RandomForestClassifier(**config['model']['params'])

# ============ MODULAR CODE ============
# src/data/loader.py
def load_and_preprocess(path, config):
    df = pd.read_csv(path)
    df = handle_missing(df, config['missing_strategy'])
    df = encode_categoricals(df, config['categorical_cols'])
    return df

# src/models/trainer.py
def train_model(X_train, y_train, model_config):
    model = get_model(model_config['name'])
    model.set_params(**model_config['params'])
    model.fit(X_train, y_train)
    return model`
        },
        {
          type: 'heading',
          content: 'Debugging ML Code'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ============ SANITY CHECKS ============

def sanity_check_data(X, y):
    """Run before training any model."""
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y unique values: {np.unique(y)}")

    # Check for NaN
    nan_count = np.isnan(X).sum()
    print(f"NaN count: {nan_count}")

    # Check for infinity
    inf_count = np.isinf(X).sum()
    print(f"Inf count: {inf_count}")

    # Check for extreme values
    print(f"X min: {X.min()}, max: {X.max()}")

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    assert X.shape[0] == y.shape[0], "X and y have different number of samples!"
    assert nan_count == 0, "Data contains NaN!"
    assert inf_count == 0, "Data contains infinity!"

sanity_check_data(X_train, y_train)

# ============ DEBUGGING POOR PERFORMANCE ============

def diagnose_model(model, X_train, y_train, X_test, y_test):
    """Diagnose why a model might be performing poorly."""

    # Training vs test performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")

    gap = train_score - test_score

    if train_score < 0.6:
        print("ISSUE: Model underfitting (high bias)")
        print("TRY: More complex model, more features, less regularization")
    elif gap > 0.1:
        print("ISSUE: Model overfitting (high variance)")
        print("TRY: More data, simpler model, regularization, dropout")
    elif test_score < 0.6:
        print("ISSUE: Data might be noisy or features not informative")
        print("TRY: Better features, data cleaning, different model")
    else:
        print("Model seems OK!")

# ============ LEARNING CURVES ============
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Validation score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Interpretation:
    # - Both low: underfitting
    # - Train high, val low: overfitting
    # - Both high and close: good fit
    # - Gap closing with more data: more data helps`
        },
        {
          type: 'heading',
          content: 'Common Pitfalls'
        },
        {
          type: 'code',
          language: 'python',
          content: `# ============ PITFALL 1: DATA LEAKAGE ============

# WRONG: Fitting scaler on all data before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Learns from test data too!
X_train, X_test = train_test_split(X_scaled, ...)

# RIGHT: Fit only on training data
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
X_test_scaled = scaler.transform(X_test)        # Transform test

# ============ PITFALL 2: FEATURE LEAKAGE ============

# WRONG: Using future information
df['next_day_price'] = df['price'].shift(-1)
df['target'] = (df['next_day_price'] > df['price']).astype(int)
# Then accidentally including next_day_price as a feature!

# ============ PITFALL 3: WRONG METRIC ============

# For imbalanced classification, accuracy can be misleading
# 99% accuracy might just mean predicting majority class

# Use appropriate metrics:
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

# ============ PITFALL 4: NOT USING CROSS-VALIDATION ============

# WRONG: Single train-test split
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Could be lucky/unlucky split

# RIGHT: Cross-validation for reliable estimate
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# ============ PITFALL 5: MODIFYING DATAFRAMES IN PLACE ============

# WRONG: Can cause subtle bugs
df.fillna(0, inplace=True)  # Modifies original

# RIGHT: Create new DataFrame
df_clean = df.fillna(0)
# or be explicit
df = df.fillna(0)`
        },
        {
          type: 'heading',
          content: 'Model Persistence'
        },
        {
          type: 'code',
          language: 'python',
          content: `import joblib
import pickle
from datetime import datetime

# ============ SAVING MODELS ============

# Method 1: joblib (recommended for sklearn)
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')

# Method 2: pickle (standard Python)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ============ SAVING COMPLETE PIPELINES ============

# Best practice: save the entire pipeline including preprocessing
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'pipeline.joblib')

# Now deployment only needs:
pipeline = joblib.load('pipeline.joblib')
predictions = pipeline.predict(new_data)  # Handles scaling automatically

# ============ METADATA TRACKING ============
import json

metadata = {
    'model_name': 'random_forest',
    'timestamp': datetime.now().isoformat(),
    'train_score': float(model.score(X_train, y_train)),
    'test_score': float(model.score(X_test, y_test)),
    'hyperparameters': model.get_params(),
    'feature_names': list(X.columns) if hasattr(X, 'columns') else None,
    'sklearn_version': sklearn.__version__
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)`
        },
        {
          type: 'text',
          content: `**Professional ML Checklist:**
1. Set random seeds for reproducibility
2. Use configuration files for hyperparameters
3. Split data before any preprocessing
4. Use cross-validation for reliable estimates
5. Check for data leakage
6. Save models with their preprocessing pipelines
7. Track metadata and versions
8. Write sanity checks for your data`
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is data leakage?',
          options: [
            'When data files are accidentally deleted',
            'When information from the test set influences training',
            'When the model memorizes the training data',
            'When features have missing values'
          ],
          correct: 1,
          explanation: 'Data leakage occurs when information from outside the training set (like test data or future data) is used during training, leading to overly optimistic performance estimates.'
        },
        {
          type: 'multiple-choice',
          question: 'Why should you save the entire pipeline instead of just the model?',
          options: [
            'It takes less storage space',
            'It ensures preprocessing is applied consistently at prediction time',
            'It runs faster',
            'It is required by scikit-learn'
          ],
          correct: 1,
          explanation: 'The preprocessing (scaling, encoding) must be applied identically to new data. Saving the full pipeline ensures the same transformations are used at both training and prediction time.'
        },
        {
          type: 'multiple-choice',
          question: 'High training accuracy but low test accuracy indicates:',
          options: [
            'Underfitting - the model is too simple',
            'Overfitting - the model is too complex',
            'Data leakage',
            'Perfectly tuned model'
          ],
          correct: 1,
          explanation: 'A large gap between training and test accuracy (high train, low test) indicates overfitting - the model memorized the training data but fails to generalize to new data.'
        }
      ]
    }
  ]
}
