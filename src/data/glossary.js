export const glossary = [
  {
    term: 'Activation Function',
    definition: 'A function applied to the output of a neuron to introduce non-linearity. Common examples include ReLU, sigmoid, and tanh. Without activation functions, neural networks would only learn linear relationships.',
    category: 'deep-learning'
  },
  {
    term: 'Adam Optimizer',
    definition: 'Adaptive Moment Estimation optimizer that combines momentum and RMSprop. Maintains per-parameter learning rates and momentum, making it robust across many problems.',
    category: 'optimization'
  },
  {
    term: 'Attention Mechanism',
    definition: 'A technique that allows models to focus on relevant parts of the input when producing each part of the output. Computes weighted sums where weights indicate importance.',
    category: 'transformers'
  },
  {
    term: 'Autoencoder',
    definition: 'Neural network trained to reconstruct its input through a bottleneck. The encoder compresses data, the decoder reconstructs it. Used for dimensionality reduction and generative models.',
    category: 'generative'
  },
  {
    term: 'Backpropagation',
    definition: 'Algorithm for computing gradients in neural networks by applying the chain rule from output to input. Enables efficient training by propagating error signals backwards through layers.',
    category: 'deep-learning'
  },
  {
    term: 'Batch Normalization',
    definition: 'Technique that normalizes layer inputs to have zero mean and unit variance. Stabilizes training, allows higher learning rates, and acts as regularization.',
    category: 'optimization'
  },
  {
    term: 'Batch Size',
    definition: 'Number of training examples used in one forward/backward pass. Larger batches give stable gradients but use more memory. Smaller batches add noise that can help generalization.',
    category: 'training'
  },
  {
    term: 'BERT',
    definition: 'Bidirectional Encoder Representations from Transformers. Pre-trained language model that reads text in both directions. Foundation for many NLP tasks through fine-tuning.',
    category: 'transformers'
  },
  {
    term: 'Bias (Model)',
    definition: 'A learnable parameter added to neuron outputs before activation. Allows the model to shift the activation function, increasing expressiveness.',
    category: 'deep-learning'
  },
  {
    term: 'Bias-Variance Tradeoff',
    definition: 'Balance between underfitting (high bias) and overfitting (high variance). Simple models have high bias, complex models have high variance. Goal is to minimize total error.',
    category: 'fundamentals'
  },
  {
    term: 'Binary Classification',
    definition: 'Classification task with exactly two classes. Uses sigmoid activation and binary cross-entropy loss. Examples: spam detection, medical diagnosis.',
    category: 'supervised'
  },
  {
    term: 'Categorical Cross-Entropy',
    definition: 'Loss function for multi-class classification with one-hot encoded labels. Measures divergence between predicted probabilities and true class distribution.',
    category: 'loss-functions'
  },
  {
    term: 'Chain Rule',
    definition: 'Calculus rule for computing derivatives of composite functions. Foundation of backpropagation: if y = f(g(x)), then dy/dx = dy/dg * dg/dx.',
    category: 'math'
  },
  {
    term: 'Classification',
    definition: 'Supervised learning task that predicts discrete class labels. Models learn decision boundaries between classes from labeled training data.',
    category: 'supervised'
  },
  {
    term: 'Clustering',
    definition: 'Unsupervised learning task that groups similar data points together. No labels required. K-means and hierarchical clustering are common algorithms.',
    category: 'unsupervised'
  },
  {
    term: 'CNN (Convolutional Neural Network)',
    definition: 'Neural network using convolutional layers to process grid-like data. Exploits spatial structure through local connectivity and weight sharing. Standard for computer vision.',
    category: 'computer-vision'
  },
  {
    term: 'Confusion Matrix',
    definition: 'Table showing true positives, false positives, true negatives, and false negatives. Visualizes classifier performance beyond simple accuracy.',
    category: 'evaluation'
  },
  {
    term: 'Convolution',
    definition: 'Operation that slides a filter across input, computing dot products. Detects local patterns regardless of position. Foundation of CNNs.',
    category: 'computer-vision'
  },
  {
    term: 'Cross-Validation',
    definition: 'Technique for assessing model generalization by training on different data subsets. K-fold CV splits data into k parts, trains on k-1, tests on 1, rotates.',
    category: 'evaluation'
  },
  {
    term: 'Decision Tree',
    definition: 'Model that makes predictions by learning if-then rules from data. Splits data recursively based on features. Interpretable but prone to overfitting.',
    category: 'supervised'
  },
  {
    term: 'Derivative',
    definition: 'Rate of change of a function. In ML, measures how loss changes with respect to parameters. Positive derivative means increasing, negative means decreasing.',
    category: 'math'
  },
  {
    term: 'Diffusion Model',
    definition: 'Generative model that learns to reverse a gradual noising process. Starts with noise, iteratively denoises to generate samples. Powers DALL-E 2, Stable Diffusion.',
    category: 'generative'
  },
  {
    term: 'Dimensionality Reduction',
    definition: 'Reducing the number of features while preserving important information. PCA and autoencoders are common methods. Helps visualization and reduces overfitting.',
    category: 'unsupervised'
  },
  {
    term: 'Dot Product',
    definition: 'Sum of element-wise products of two vectors. Measures similarity between vectors. Foundation of neural network computations and attention.',
    category: 'math'
  },
  {
    term: 'Dropout',
    definition: 'Regularization technique that randomly zeroes neuron outputs during training. Prevents co-adaptation and improves generalization. Disabled during inference.',
    category: 'regularization'
  },
  {
    term: 'Embedding',
    definition: 'Dense vector representation of discrete items like words or categories. Learned during training to capture semantic relationships. Similar items have similar embeddings.',
    category: 'nlp'
  },
  {
    term: 'Encoder-Decoder',
    definition: 'Architecture where encoder processes input into representations, decoder generates output from those representations. Used for translation, summarization, image captioning.',
    category: 'architecture'
  },
  {
    term: 'Ensemble Methods',
    definition: 'Combining multiple models to improve predictions. Bagging trains on different data subsets, boosting trains sequentially on errors. Random forests and gradient boosting are examples.',
    category: 'supervised'
  },
  {
    term: 'Epoch',
    definition: 'One complete pass through the entire training dataset. Models typically train for many epochs, with validation checked after each to monitor overfitting.',
    category: 'training'
  },
  {
    term: 'F1 Score',
    definition: 'Harmonic mean of precision and recall: 2 * (precision * recall) / (precision + recall). Balances both metrics, useful for imbalanced datasets.',
    category: 'evaluation'
  },
  {
    term: 'Feature',
    definition: 'An input variable used to make predictions. Raw data is transformed into features through feature engineering. Good features capture patterns relevant to the task.',
    category: 'fundamentals'
  },
  {
    term: 'Feature Engineering',
    definition: 'Creating new features from raw data to improve model performance. Requires domain knowledge. Examples: extracting date parts, combining features, polynomial features.',
    category: 'fundamentals'
  },
  {
    term: 'Feature Map',
    definition: 'Output of a convolutional layer. Each feature map shows where a particular pattern (learned by one filter) is detected in the input.',
    category: 'computer-vision'
  },
  {
    term: 'Fine-tuning',
    definition: 'Taking a pre-trained model and training it further on a specific task. Often uses lower learning rates. Transfer learning approach that leverages existing knowledge.',
    category: 'training'
  },
  {
    term: 'Forward Pass',
    definition: 'Computing model output from input by passing data through layers sequentially. Activations are stored for use in backward pass during training.',
    category: 'deep-learning'
  },
  {
    term: 'GAN (Generative Adversarial Network)',
    definition: 'Two networks trained adversarially: generator creates fake samples, discriminator distinguishes real from fake. Competition drives both to improve.',
    category: 'generative'
  },
  {
    term: 'Gradient',
    definition: 'Vector of partial derivatives pointing in direction of steepest increase. In ML, we move opposite to gradient to minimize loss. Computed via backpropagation.',
    category: 'math'
  },
  {
    term: 'Gradient Descent',
    definition: 'Optimization algorithm that updates parameters in the direction that reduces loss. Step size controlled by learning rate. Foundation of neural network training.',
    category: 'optimization'
  },
  {
    term: 'GPT',
    definition: 'Generative Pre-trained Transformer. Autoregressive language model trained to predict next token. Foundation of ChatGPT and many text generation systems.',
    category: 'transformers'
  },
  {
    term: 'GRU (Gated Recurrent Unit)',
    definition: 'Simplified LSTM variant with two gates instead of three. Combines forget and input gates into update gate. Often performs similarly with fewer parameters.',
    category: 'rnn'
  },
  {
    term: 'Hidden Layer',
    definition: 'Layer between input and output layers in a neural network. Learns intermediate representations. Deep networks have many hidden layers.',
    category: 'deep-learning'
  },
  {
    term: 'Hyperparameter',
    definition: 'Parameter set before training, not learned from data. Examples: learning rate, batch size, number of layers. Tuned via cross-validation or search.',
    category: 'training'
  },
  {
    term: 'K-Means Clustering',
    definition: 'Clustering algorithm that partitions data into k groups. Iteratively assigns points to nearest centroid, then updates centroids. Fast but requires choosing k.',
    category: 'unsupervised'
  },
  {
    term: 'K-Nearest Neighbors (KNN)',
    definition: 'Classification by majority vote of k closest training examples. Simple, no training required, but slow at inference with large datasets.',
    category: 'supervised'
  },
  {
    term: 'Kernel',
    definition: 'Small filter matrix slid across input in convolution. Also: function measuring similarity in SVMs. Learned kernels detect features like edges, textures.',
    category: 'computer-vision'
  },
  {
    term: 'L1 Regularization',
    definition: 'Adding sum of absolute weights to loss. Encourages sparsity by pushing some weights to exactly zero. Also called Lasso.',
    category: 'regularization'
  },
  {
    term: 'L2 Regularization',
    definition: 'Adding sum of squared weights to loss. Penalizes large weights, encouraging smaller, more distributed weights. Also called Ridge or weight decay.',
    category: 'regularization'
  },
  {
    term: 'Learning Rate',
    definition: 'Hyperparameter controlling step size in gradient descent. Too high causes instability, too low causes slow training. Often decayed during training.',
    category: 'optimization'
  },
  {
    term: 'Linear Regression',
    definition: 'Predicting continuous values by fitting a line (or hyperplane) to data. Minimizes squared error. Simple, interpretable, foundation of many methods.',
    category: 'supervised'
  },
  {
    term: 'Logistic Regression',
    definition: 'Classification algorithm using sigmoid function to predict probabilities. Despite the name, used for classification not regression. Linear decision boundary.',
    category: 'supervised'
  },
  {
    term: 'Loss Function',
    definition: 'Function measuring prediction error. Training minimizes loss. Different tasks use different losses: MSE for regression, cross-entropy for classification.',
    category: 'fundamentals'
  },
  {
    term: 'LSTM (Long Short-Term Memory)',
    definition: 'RNN variant with gated memory cells that can learn long-term dependencies. Gates control information flow: forget, input, and output gates. Solves vanishing gradient.',
    category: 'rnn'
  },
  {
    term: 'MAE (Mean Absolute Error)',
    definition: 'Average absolute difference between predictions and actual values. Less sensitive to outliers than MSE. Common regression metric.',
    category: 'evaluation'
  },
  {
    term: 'Matrix',
    definition: '2D array of numbers. Used to represent data, weights, and transformations. Matrix multiplication is the core operation in neural networks.',
    category: 'math'
  },
  {
    term: 'Max Pooling',
    definition: 'Downsampling operation taking maximum value in each region. Reduces spatial dimensions while preserving strongest activations. Adds translation invariance.',
    category: 'computer-vision'
  },
  {
    term: 'Mini-batch',
    definition: 'Subset of training data used for one gradient update. Balances between full-batch (stable but slow) and single-sample (noisy but fast) gradient descent.',
    category: 'training'
  },
  {
    term: 'MLOps',
    definition: 'Practices for deploying and maintaining ML models in production. Includes versioning, monitoring, CI/CD, and infrastructure management.',
    category: 'mlops'
  },
  {
    term: 'Momentum',
    definition: 'Optimization technique that accumulates gradient direction over time. Helps overcome local minima and accelerates convergence in consistent gradient directions.',
    category: 'optimization'
  },
  {
    term: 'MSE (Mean Squared Error)',
    definition: 'Average squared difference between predictions and actual values. Penalizes large errors heavily. Standard regression loss function.',
    category: 'loss-functions'
  },
  {
    term: 'Multi-Head Attention',
    definition: 'Parallel attention mechanisms attending to different aspects of input. Each head learns different patterns. Outputs concatenated and projected.',
    category: 'transformers'
  },
  {
    term: 'Naive Bayes',
    definition: 'Probabilistic classifier assuming feature independence. Despite naive assumption, works well for text classification. Fast training and inference.',
    category: 'supervised'
  },
  {
    term: 'Neural Network',
    definition: 'Computing system inspired by biological neurons. Layers of nodes with weighted connections learn representations from data. Foundation of deep learning.',
    category: 'deep-learning'
  },
  {
    term: 'Normalization',
    definition: 'Scaling features to standard range. Min-max scales to [0,1], z-score standardizes to mean 0, std 1. Improves training stability and convergence.',
    category: 'preprocessing'
  },
  {
    term: 'Object Detection',
    definition: 'Computer vision task finding and classifying objects in images. Outputs bounding boxes with class labels. YOLO, Faster R-CNN are popular methods.',
    category: 'computer-vision'
  },
  {
    term: 'One-Hot Encoding',
    definition: 'Representing categories as binary vectors with single 1. Category 2 of 5 becomes [0,1,0,0,0]. Creates sparse, high-dimensional representations.',
    category: 'preprocessing'
  },
  {
    term: 'Overfitting',
    definition: 'Model learns training data too well, including noise, failing to generalize. Signs: low training error, high validation error. Address with regularization, more data, simpler model.',
    category: 'fundamentals'
  },
  {
    term: 'Padding',
    definition: 'Adding values (usually zeros) around input borders in convolution. Preserves spatial dimensions. Also: making sequences equal length for batching.',
    category: 'preprocessing'
  },
  {
    term: 'Parameter',
    definition: 'Values learned during training. Weights and biases are parameters. Number of parameters determines model capacity and memory requirements.',
    category: 'fundamentals'
  },
  {
    term: 'PCA (Principal Component Analysis)',
    definition: 'Dimensionality reduction finding directions of maximum variance. Projects data onto principal components. Linear, fast, interpretable.',
    category: 'unsupervised'
  },
  {
    term: 'Perceptron',
    definition: 'Simplest neural network: weighted sum of inputs plus bias, through activation. Building block of neural networks. Single perceptron learns linear boundaries.',
    category: 'deep-learning'
  },
  {
    term: 'Pooling',
    definition: 'Downsampling operation in CNNs reducing spatial dimensions. Max pooling takes maximum, average pooling takes mean. Adds invariance, reduces computation.',
    category: 'computer-vision'
  },
  {
    term: 'Positional Encoding',
    definition: 'Adding position information to transformer inputs since attention has no inherent order. Uses sinusoidal functions or learned embeddings.',
    category: 'transformers'
  },
  {
    term: 'Precision',
    definition: 'Of predicted positives, what fraction is actually positive? TP / (TP + FP). High precision means few false positives.',
    category: 'evaluation'
  },
  {
    term: 'Probability',
    definition: 'Measure of likelihood between 0 and 1. ML models often output probabilities. Understanding probability distributions is essential for ML.',
    category: 'math'
  },
  {
    term: 'Random Forest',
    definition: 'Ensemble of decision trees trained on random data subsets with random feature subsets. Reduces overfitting, robust, handles mixed data types.',
    category: 'supervised'
  },
  {
    term: 'Recall',
    definition: 'Of actual positives, what fraction did we find? TP / (TP + FN). High recall means few missed positives. Also called sensitivity.',
    category: 'evaluation'
  },
  {
    term: 'Regression',
    definition: 'Supervised learning task predicting continuous values. Linear regression, neural networks, gradient boosting can all do regression.',
    category: 'supervised'
  },
  {
    term: 'Regularization',
    definition: 'Techniques to prevent overfitting by constraining model complexity. L1/L2 penalties, dropout, early stopping, data augmentation are examples.',
    category: 'regularization'
  },
  {
    term: 'ReLU (Rectified Linear Unit)',
    definition: 'Activation function f(x) = max(0, x). Simple, fast, avoids vanishing gradients for positive values. Most common hidden layer activation.',
    category: 'deep-learning'
  },
  {
    term: 'ResNet',
    definition: 'Residual Network with skip connections that add input to output. Enables training very deep networks by allowing gradient flow. Won ImageNet 2015.',
    category: 'computer-vision'
  },
  {
    term: 'RNN (Recurrent Neural Network)',
    definition: 'Neural network with connections forming cycles, maintaining hidden state across sequence. Processes sequences of variable length. Basis for LSTM, GRU.',
    category: 'rnn'
  },
  {
    term: 'ROC Curve',
    definition: 'Plot of true positive rate vs false positive rate at various thresholds. Area under curve (AUC) measures classifier quality. 0.5 is random, 1.0 is perfect.',
    category: 'evaluation'
  },
  {
    term: 'Self-Attention',
    definition: 'Attention where queries, keys, and values come from same sequence. Each position attends to all positions. Core mechanism of transformers.',
    category: 'transformers'
  },
  {
    term: 'Semantic Segmentation',
    definition: 'Classifying every pixel in an image. Outputs same-size mask with class per pixel. U-Net, DeepLab are popular architectures.',
    category: 'computer-vision'
  },
  {
    term: 'SGD (Stochastic Gradient Descent)',
    definition: 'Gradient descent using random samples instead of full dataset. Faster updates, noise helps escape local minima. Mini-batch SGD is most common.',
    category: 'optimization'
  },
  {
    term: 'Sigmoid',
    definition: 'Activation function squashing values to (0,1). Output interpretable as probability. Used for binary classification, LSTM gates.',
    category: 'deep-learning'
  },
  {
    term: 'Skip Connection',
    definition: 'Connection bypassing one or more layers, adding input directly to output. Enables gradient flow in deep networks. Key to ResNet success.',
    category: 'architecture'
  },
  {
    term: 'Softmax',
    definition: 'Function converting logits to probability distribution summing to 1. Exponentiates then normalizes. Standard for multi-class classification output.',
    category: 'deep-learning'
  },
  {
    term: 'Stride',
    definition: 'Step size when sliding filter across input in convolution. Stride 2 reduces dimensions by half. Alternative to pooling for downsampling.',
    category: 'computer-vision'
  },
  {
    term: 'Supervised Learning',
    definition: 'Learning from labeled data where correct outputs are known. Model learns mapping from inputs to outputs. Classification and regression are supervised.',
    category: 'fundamentals'
  },
  {
    term: 'SVM (Support Vector Machine)',
    definition: 'Classifier finding maximum-margin hyperplane between classes. Kernel trick enables non-linear boundaries. Effective for high-dimensional data.',
    category: 'supervised'
  },
  {
    term: 'Tanh',
    definition: 'Activation function squashing values to (-1, 1). Zero-centered unlike sigmoid. Used in RNN hidden states.',
    category: 'deep-learning'
  },
  {
    term: 'Tensor',
    definition: 'Multi-dimensional array. Scalar is 0D, vector is 1D, matrix is 2D, and higher. Core data structure in deep learning frameworks.',
    category: 'fundamentals'
  },
  {
    term: 'Tokenization',
    definition: 'Breaking text into units (tokens) for processing. Word-level, subword (BPE), or character-level. First step in NLP pipelines.',
    category: 'nlp'
  },
  {
    term: 'Training Set',
    definition: 'Data used to train the model. Model learns patterns from this data. Should be representative of real-world data.',
    category: 'fundamentals'
  },
  {
    term: 'Transfer Learning',
    definition: 'Using knowledge from one task to help another. Pre-train on large dataset, fine-tune on specific task. Enables good results with less data.',
    category: 'training'
  },
  {
    term: 'Transformer',
    definition: 'Architecture based entirely on attention, no recurrence. Processes all positions in parallel. Foundation of BERT, GPT, modern NLP.',
    category: 'transformers'
  },
  {
    term: 'Underfitting',
    definition: 'Model too simple to capture patterns in data. High training and validation error. Address with more complex model, better features, longer training.',
    category: 'fundamentals'
  },
  {
    term: 'Unsupervised Learning',
    definition: 'Learning from unlabeled data, finding patterns without correct answers. Clustering, dimensionality reduction, generative models are unsupervised.',
    category: 'fundamentals'
  },
  {
    term: 'Validation Set',
    definition: 'Data held out during training to tune hyperparameters and detect overfitting. Not used to update model weights directly.',
    category: 'fundamentals'
  },
  {
    term: 'Vanishing Gradient',
    definition: 'Problem where gradients become extremely small during backpropagation. Early layers stop learning. Solved by ReLU, skip connections, LSTMs.',
    category: 'deep-learning'
  },
  {
    term: 'VAE (Variational Autoencoder)',
    definition: 'Generative model encoding data to probability distribution in latent space. Enables sampling new data points. Combines autoencoders with variational inference.',
    category: 'generative'
  },
  {
    term: 'Vector',
    definition: '1D array of numbers. Represents data points, features, or learned representations. Dot products and norms are key vector operations.',
    category: 'math'
  },
  {
    term: 'Weight',
    definition: 'Learnable parameter multiplied with inputs in neural networks. Weights determine feature importance and are adjusted during training.',
    category: 'deep-learning'
  },
  {
    term: 'Weight Initialization',
    definition: 'Setting initial values for weights before training. Poor initialization causes vanishing/exploding gradients. Xavier and He initialization are popular methods.',
    category: 'training'
  },
  {
    term: 'Word Embedding',
    definition: 'Dense vector representation of words capturing semantic meaning. Similar words have similar vectors. Word2Vec, GloVe, FastText are methods.',
    category: 'nlp'
  },
  {
    term: 'Xavier Initialization',
    definition: 'Weight initialization keeping variance constant across layers. Scales by 1/sqrt(fan_in). Works well with sigmoid/tanh activations.',
    category: 'training'
  }
]
