export const unsupervisedlearning = {
  id: 'unsupervised-learning',
  title: 'Unsupervised Learning',
  description: 'Discover patterns in unlabeled data through clustering, dimensionality reduction, and anomaly detection',
  category: 'Classical ML',
  difficulty: 'Intermediate',
  duration: '5 hours',
  lessons: [
    {
      id: 'clustering-intro',
      title: 'Introduction to Clustering',
      duration: '45 min',
      concepts: ['Clustering', 'Distance Metrics', 'Similarity'],
      content: [
        {
          type: 'heading',
          content: 'What is Clustering?'
        },
        {
          type: 'text',
          content: `Imagine you're organizing a music library with thousands of songs, but none of them have genre labels. How would you group similar songs together? You'd probably listen to each song and notice patterns - tempo, instruments, mood - and naturally group similar-sounding tracks.

This is exactly what clustering does. It's the task of grouping data points so that items in the same group (cluster) are more similar to each other than to items in other groups.`
        },
        {
          type: 'visualization',
          title: 'Clustering Intuition',
          svg: `<svg viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="250" fill="#f8fafc"/>

            <!-- Cluster 1 - Blue -->
            <circle cx="80" cy="70" r="6" fill="#3b82f6"/>
            <circle cx="95" cy="85" r="6" fill="#3b82f6"/>
            <circle cx="70" cy="90" r="6" fill="#3b82f6"/>
            <circle cx="100" cy="60" r="6" fill="#3b82f6"/>
            <circle cx="85" cy="100" r="6" fill="#3b82f6"/>
            <circle cx="60" cy="75" r="6" fill="#3b82f6"/>
            <ellipse cx="82" cy="80" rx="35" ry="30" fill="none" stroke="#3b82f6" stroke-width="2" stroke-dasharray="4"/>

            <!-- Cluster 2 - Green -->
            <circle cx="280" cy="80" r="6" fill="#10b981"/>
            <circle cx="300" cy="95" r="6" fill="#10b981"/>
            <circle cx="265" cy="100" r="6" fill="#10b981"/>
            <circle cx="290" cy="65" r="6" fill="#10b981"/>
            <circle cx="310" cy="85" r="6" fill="#10b981"/>
            <circle cx="275" cy="70" r="6" fill="#10b981"/>
            <ellipse cx="288" cy="82" rx="35" ry="30" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="4"/>

            <!-- Cluster 3 - Purple -->
            <circle cx="180" cy="180" r="6" fill="#8b5cf6"/>
            <circle cx="200" cy="195" r="6" fill="#8b5cf6"/>
            <circle cx="165" cy="200" r="6" fill="#8b5cf6"/>
            <circle cx="190" cy="165" r="6" fill="#8b5cf6"/>
            <circle cx="210" cy="185" r="6" fill="#8b5cf6"/>
            <circle cx="175" cy="170" r="6" fill="#8b5cf6"/>
            <circle cx="195" cy="210" r="6" fill="#8b5cf6"/>
            <ellipse cx="188" cy="185" rx="38" ry="35" fill="none" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="4"/>

            <!-- Labels -->
            <text x="82" y="130" text-anchor="middle" font-size="11" fill="#3b82f6" font-weight="500">Cluster A</text>
            <text x="288" y="130" text-anchor="middle" font-size="11" fill="#10b981" font-weight="500">Cluster B</text>
            <text x="188" y="235" text-anchor="middle" font-size="11" fill="#8b5cf6" font-weight="500">Cluster C</text>
          </svg>`,
          caption: 'Clustering groups similar data points together without labels'
        },
        {
          type: 'heading',
          content: 'Why Unsupervised Learning?'
        },
        {
          type: 'text',
          content: `In supervised learning, we need labeled data - someone has to tell the algorithm "this email is spam" or "this image is a cat." But labeling data is expensive and time-consuming.

Unsupervised learning works with **unlabeled data**. The algorithm discovers structure on its own. This is incredibly powerful because:

**1. Labels are expensive** - Medical experts labeling X-rays, linguists annotating text, or analysts categorizing transactions all cost money and time.

**2. Labels might not exist** - What if you want to discover customer segments you didn't know about? You can't label data for categories you haven't discovered yet.

**3. Hidden patterns** - Sometimes the most valuable insights are patterns humans haven't noticed.`
        },
        {
          type: 'heading',
          content: 'Distance Metrics: Measuring Similarity'
        },
        {
          type: 'text',
          content: `To group similar items, we need to define "similar." This is where **distance metrics** come in. The smaller the distance between two points, the more similar they are.`
        },
        {
          type: 'subheading',
          content: 'Euclidean Distance'
        },
        {
          type: 'text',
          content: `The most intuitive distance - the straight-line distance between two points. It's what you'd measure with a ruler.`
        },
        {
          type: 'formula',
          content: 'd(a, b) = √[(a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²]'
        },
        {
          type: 'diagram',
          svg: `<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="300" height="200" fill="#f8fafc"/>

            <!-- Grid -->
            <g stroke="#e2e8f0" stroke-width="1">
              <line x1="50" y1="30" x2="50" y2="170"/>
              <line x1="100" y1="30" x2="100" y2="170"/>
              <line x1="150" y1="30" x2="150" y2="170"/>
              <line x1="200" y1="30" x2="200" y2="170"/>
              <line x1="250" y1="30" x2="250" y2="170"/>
              <line x1="50" y1="50" x2="250" y2="50"/>
              <line x1="50" y1="90" x2="250" y2="90"/>
              <line x1="50" y1="130" x2="250" y2="130"/>
              <line x1="50" y1="170" x2="250" y2="170"/>
            </g>

            <!-- Points -->
            <circle cx="80" cy="140" r="8" fill="#3b82f6"/>
            <circle cx="220" cy="60" r="8" fill="#10b981"/>

            <!-- Euclidean distance -->
            <line x1="80" y1="140" x2="220" y2="60" stroke="#ef4444" stroke-width="2"/>

            <!-- Right angle lines -->
            <line x1="80" y1="140" x2="220" y2="140" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4"/>
            <line x1="220" y1="140" x2="220" y2="60" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4"/>

            <!-- Labels -->
            <text x="65" y="155" font-size="11" fill="#3b82f6" font-weight="500">A(1,1)</text>
            <text x="225" y="55" font-size="11" fill="#10b981" font-weight="500">B(4,3)</text>
            <text x="145" y="88" font-size="11" fill="#ef4444" font-weight="500">d = 5</text>
            <text x="145" y="155" font-size="10" fill="#64748b">Δx = 3</text>
            <text x="225" y="105" font-size="10" fill="#64748b">Δy = 2</text>
          </svg>`,
          caption: 'Euclidean distance: √(3² + 2²) = √13 ≈ 3.6'
        },
        {
          type: 'subheading',
          content: 'Manhattan Distance'
        },
        {
          type: 'text',
          content: `Also called "city block" distance - imagine navigating a grid of streets where you can only move horizontally or vertically. It's the sum of absolute differences along each dimension.`
        },
        {
          type: 'formula',
          content: 'd(a, b) = |a₁-b₁| + |a₂-b₂| + ... + |aₙ-bₙ|'
        },
        {
          type: 'text',
          content: `**When to use which?**
- **Euclidean**: When features have similar scales and diagonal movement makes sense
- **Manhattan**: When features represent different things (e.g., age and salary), or in high dimensions where Euclidean distance becomes less meaningful`
        },
        {
          type: 'subheading',
          content: 'Cosine Similarity'
        },
        {
          type: 'text',
          content: `Instead of measuring distance, cosine similarity measures the angle between two vectors. Two documents about machine learning will have similar word patterns, regardless of document length.`
        },
        {
          type: 'formula',
          content: 'cos(θ) = (A · B) / (||A|| × ||B||)'
        },
        {
          type: 'text',
          content: `Cosine similarity ranges from -1 (opposite) to 1 (identical). A value of 0 means the vectors are perpendicular (unrelated).

**Perfect for**: Text analysis, recommendation systems, any case where magnitude doesn't matter but direction does.`
        },
        {
          type: 'heading',
          content: 'Types of Clustering Algorithms'
        },
        {
          type: 'table',
          headers: ['Type', 'Algorithm', 'Best For'],
          rows: [
            ['Partitioning', 'K-Means, K-Medoids', 'Spherical clusters, known K'],
            ['Hierarchical', 'Agglomerative, Divisive', 'Exploring cluster structure'],
            ['Density-based', 'DBSCAN, OPTICS', 'Irregular shapes, outliers'],
            ['Distribution', 'Gaussian Mixture', 'Overlapping clusters']
          ],
          caption: 'Overview of clustering algorithm families'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

euclidean_dist = euclidean(a, b)
manhattan_dist = cityblock(a, b)
cosine_sim = 1 - cosine(a, b)

print(f"Euclidean distance: {euclidean_dist:.2f}")
print(f"Manhattan distance: {manhattan_dist}")
print(f"Cosine similarity: {cosine_sim:.4f}")`
        },
        {
          type: 'keypoints',
          points: [
            'Clustering groups similar data points without labels',
            'Distance metrics define similarity - Euclidean for general use, Manhattan for mixed features, Cosine for direction',
            'Different clustering algorithms suit different data shapes and use cases',
            'Unsupervised learning discovers patterns humans might miss'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What makes clustering an unsupervised learning technique?',
          options: [
            'It requires labeled training data',
            'It works with unlabeled data to discover patterns',
            'It uses a supervisor to guide the algorithm',
            'It only works with numerical data'
          ],
          correct: 1,
          explanation: 'Clustering is unsupervised because it discovers structure in data without being told what groups exist - no labels needed.'
        },
        {
          type: 'multiple-choice',
          question: 'Which distance metric would be best for comparing document similarity?',
          options: [
            'Euclidean distance',
            'Manhattan distance',
            'Cosine similarity',
            'Hamming distance'
          ],
          correct: 2,
          explanation: 'Cosine similarity measures the angle between vectors, making it ideal for text where document length varies but topic similarity matters.'
        }
      ]
    },
    {
      id: 'kmeans',
      title: 'K-Means Clustering',
      duration: '55 min',
      concepts: ['K-Means', 'Centroids', 'Elbow Method'],
      content: [
        {
          type: 'heading',
          content: 'The K-Means Algorithm'
        },
        {
          type: 'text',
          content: `K-Means is the most popular clustering algorithm because it's simple, fast, and often works surprisingly well. The idea is beautifully intuitive:

**Goal**: Partition n data points into K clusters, where each point belongs to the cluster with the nearest center (centroid).

Think of it like placing K magnets in a room full of iron filings. Each filing gets attracted to its nearest magnet, and the magnets adjust their positions to be at the center of their filings.`
        },
        {
          type: 'heading',
          content: 'How K-Means Works'
        },
        {
          type: 'text',
          content: `The algorithm alternates between two steps until convergence:

**Step 1: Assignment** - Assign each point to its nearest centroid
**Step 2: Update** - Move each centroid to the mean of its assigned points

Repeat until centroids stop moving (or move very little).`
        },
        {
          type: 'visualization',
          title: 'K-Means Iteration Process',
          svg: `<svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="600" height="200" fill="#f8fafc"/>

            <!-- Stage 1: Initial -->
            <g transform="translate(0,0)">
              <text x="75" y="20" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">1. Initialize</text>
              <rect x="10" y="30" width="130" height="130" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Random points -->
              <circle cx="40" cy="60" r="4" fill="#94a3b8"/>
              <circle cx="55" cy="75" r="4" fill="#94a3b8"/>
              <circle cx="45" cy="90" r="4" fill="#94a3b8"/>
              <circle cx="90" cy="55" r="4" fill="#94a3b8"/>
              <circle cx="100" cy="70" r="4" fill="#94a3b8"/>
              <circle cx="95" cy="85" r="4" fill="#94a3b8"/>
              <circle cx="70" cy="120" r="4" fill="#94a3b8"/>
              <circle cx="85" cy="130" r="4" fill="#94a3b8"/>
              <circle cx="60" cy="135" r="4" fill="#94a3b8"/>

              <!-- Initial centroids (random) -->
              <circle cx="50" cy="100" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>
              <circle cx="110" cy="60" r="6" fill="#3b82f6" stroke="white" stroke-width="2"/>
            </g>

            <!-- Arrow -->
            <path d="M145 95 L165 95" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

            <!-- Stage 2: Assign -->
            <g transform="translate(160,0)">
              <text x="75" y="20" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">2. Assign</text>
              <rect x="10" y="30" width="130" height="130" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Points colored by cluster -->
              <circle cx="40" cy="60" r="4" fill="#fca5a5"/>
              <circle cx="55" cy="75" r="4" fill="#fca5a5"/>
              <circle cx="45" cy="90" r="4" fill="#fca5a5"/>
              <circle cx="90" cy="55" r="4" fill="#93c5fd"/>
              <circle cx="100" cy="70" r="4" fill="#93c5fd"/>
              <circle cx="95" cy="85" r="4" fill="#93c5fd"/>
              <circle cx="70" cy="120" r="4" fill="#fca5a5"/>
              <circle cx="85" cy="130" r="4" fill="#fca5a5"/>
              <circle cx="60" cy="135" r="4" fill="#fca5a5"/>

              <!-- Centroids -->
              <circle cx="50" cy="100" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>
              <circle cx="110" cy="60" r="6" fill="#3b82f6" stroke="white" stroke-width="2"/>
            </g>

            <!-- Arrow -->
            <path d="M305 95 L325 95" stroke="#94a3b8" stroke-width="2"/>

            <!-- Stage 3: Update -->
            <g transform="translate(320,0)">
              <text x="75" y="20" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">3. Update</text>
              <rect x="10" y="30" width="130" height="130" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Points same colors -->
              <circle cx="40" cy="60" r="4" fill="#fca5a5"/>
              <circle cx="55" cy="75" r="4" fill="#fca5a5"/>
              <circle cx="45" cy="90" r="4" fill="#fca5a5"/>
              <circle cx="90" cy="55" r="4" fill="#93c5fd"/>
              <circle cx="100" cy="70" r="4" fill="#93c5fd"/>
              <circle cx="95" cy="85" r="4" fill="#93c5fd"/>
              <circle cx="70" cy="120" r="4" fill="#fca5a5"/>
              <circle cx="85" cy="130" r="4" fill="#fca5a5"/>
              <circle cx="60" cy="135" r="4" fill="#fca5a5"/>

              <!-- New centroid positions (moved) -->
              <circle cx="55" cy="100" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>
              <circle cx="95" cy="70" r="6" fill="#3b82f6" stroke="white" stroke-width="2"/>

              <!-- Movement arrows -->
              <path d="M50 100 L54 100" stroke="#ef4444" stroke-width="1" stroke-dasharray="2"/>
              <path d="M110 60 L96 69" stroke="#3b82f6" stroke-width="1" stroke-dasharray="2"/>
            </g>

            <!-- Arrow -->
            <path d="M465 95 L485 95" stroke="#94a3b8" stroke-width="2"/>

            <!-- Stage 4: Converged -->
            <g transform="translate(480,0)">
              <text x="60" y="20" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">4. Converge</text>
              <rect x="10" y="30" width="110" height="130" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Final clusters -->
              <circle cx="35" cy="60" r="4" fill="#fca5a5"/>
              <circle cx="50" cy="75" r="4" fill="#fca5a5"/>
              <circle cx="40" cy="90" r="4" fill="#fca5a5"/>
              <circle cx="85" cy="55" r="4" fill="#93c5fd"/>
              <circle cx="95" cy="70" r="4" fill="#93c5fd"/>
              <circle cx="90" cy="85" r="4" fill="#93c5fd"/>
              <circle cx="55" cy="115" r="4" fill="#a7f3d0"/>
              <circle cx="70" cy="125" r="4" fill="#a7f3d0"/>
              <circle cx="45" cy="130" r="4" fill="#a7f3d0"/>

              <!-- Final centroids -->
              <circle cx="42" cy="75" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>
              <circle cx="90" cy="70" r="6" fill="#3b82f6" stroke="white" stroke-width="2"/>
              <circle cx="57" cy="123" r="5" fill="#10b981" stroke="white" stroke-width="2"/>

              <text x="60" y="170" text-anchor="middle" font-size="9" fill="#10b981">✓ Done</text>
            </g>

            <defs>
              <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'K-Means alternates between assigning points to nearest centroid and updating centroid positions'
        },
        {
          type: 'heading',
          content: 'The Mathematics'
        },
        {
          type: 'text',
          content: `K-Means minimizes the **within-cluster sum of squares (WCSS)**, also called inertia:`
        },
        {
          type: 'formula',
          content: 'J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²'
        },
        {
          type: 'text',
          content: `Where:
- Cᵢ is cluster i
- μᵢ is the centroid of cluster i
- x is a data point
- ||x - μᵢ||² is the squared Euclidean distance

The algorithm finds a **local minimum** of this objective. Different initializations may find different solutions!`
        },
        {
          type: 'heading',
          content: 'Choosing K: The Elbow Method'
        },
        {
          type: 'text',
          content: `The biggest question: How many clusters should we use?

The **Elbow Method** plots WCSS against K. As K increases, WCSS decreases (more clusters = points closer to centroids). But at some point, adding more clusters gives diminishing returns.

Look for the "elbow" - where the curve bends. That's often a good K.`
        },
        {
          type: 'visualization',
          title: 'The Elbow Method',
          svg: `<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="220" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="60" y1="180" x2="360" y2="180" stroke="#475569" stroke-width="2"/>
            <line x1="60" y1="180" x2="60" y2="30" stroke="#475569" stroke-width="2"/>

            <!-- Y-axis label -->
            <text x="25" y="105" font-size="11" fill="#475569" transform="rotate(-90,25,105)">WCSS (Inertia)</text>

            <!-- X-axis label -->
            <text x="210" y="210" text-anchor="middle" font-size="11" fill="#475569">Number of Clusters (K)</text>

            <!-- X-axis ticks -->
            <g font-size="10" fill="#64748b">
              <text x="90" y="195">1</text>
              <text x="135" y="195">2</text>
              <text x="180" y="195">3</text>
              <text x="225" y="195">4</text>
              <text x="270" y="195">5</text>
              <text x="315" y="195">6</text>
            </g>

            <!-- Elbow curve -->
            <path d="M90,50 Q120,80 150,120 Q170,140 180,150 L225,160 L270,165 L315,168"
                  fill="none" stroke="#3b82f6" stroke-width="3"/>

            <!-- Data points -->
            <circle cx="90" cy="50" r="5" fill="#3b82f6"/>
            <circle cx="135" cy="90" r="5" fill="#3b82f6"/>
            <circle cx="180" cy="145" r="5" fill="#10b981" stroke="#10b981" stroke-width="3"/>
            <circle cx="225" cy="160" r="5" fill="#3b82f6"/>
            <circle cx="270" cy="165" r="5" fill="#3b82f6"/>
            <circle cx="315" cy="168" r="5" fill="#3b82f6"/>

            <!-- Elbow indicator -->
            <path d="M180,145 L180,185" stroke="#10b981" stroke-width="1" stroke-dasharray="4"/>
            <text x="180" y="135" text-anchor="middle" font-size="10" fill="#10b981" font-weight="600">Elbow</text>
            <text x="185" y="75" font-size="9" fill="#10b981">← Optimal K = 3</text>
          </svg>`,
          caption: 'The elbow point suggests K=3 as the optimal number of clusters'
        },
        {
          type: 'heading',
          content: 'K-Means in Practice'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

print(f"Cluster centers:\\n{kmeans.cluster_centers_}")
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")

wcss = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)`
        },
        {
          type: 'heading',
          content: 'K-Means Limitations'
        },
        {
          type: 'callout',
          variant: 'warning',
          content: 'K-Means assumes spherical clusters of similar size. It struggles with elongated clusters, clusters of different densities, and non-convex shapes.'
        },
        {
          type: 'visualization',
          title: 'When K-Means Fails',
          svg: `<svg viewBox="0 0 400 150" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="150" fill="#f8fafc"/>

            <!-- Good case -->
            <g transform="translate(10,10)">
              <text x="80" y="10" text-anchor="middle" font-size="10" fill="#10b981" font-weight="500">✓ Works Well</text>
              <rect x="10" y="20" width="140" height="110" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Spherical clusters -->
              <circle cx="45" cy="50" r="4" fill="#3b82f6"/>
              <circle cx="55" cy="60" r="4" fill="#3b82f6"/>
              <circle cx="40" cy="65" r="4" fill="#3b82f6"/>
              <circle cx="50" cy="75" r="4" fill="#3b82f6"/>

              <circle cx="110" cy="55" r="4" fill="#ef4444"/>
              <circle cx="120" cy="65" r="4" fill="#ef4444"/>
              <circle cx="105" cy="70" r="4" fill="#ef4444"/>
              <circle cx="115" cy="80" r="4" fill="#ef4444"/>

              <circle cx="75" cy="105" r="4" fill="#10b981"/>
              <circle cx="85" cy="110" r="4" fill="#10b981"/>
              <circle cx="70" cy="115" r="4" fill="#10b981"/>
              <circle cx="80" cy="95" r="4" fill="#10b981"/>
            </g>

            <!-- Bad case - elongated -->
            <g transform="translate(200,10)">
              <text x="90" y="10" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="500">✗ Struggles</text>
              <rect x="10" y="20" width="170" height="110" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Elongated cluster -->
              <circle cx="30" cy="45" r="3" fill="#94a3b8"/>
              <circle cx="45" cy="50" r="3" fill="#94a3b8"/>
              <circle cx="60" cy="55" r="3" fill="#94a3b8"/>
              <circle cx="75" cy="60" r="3" fill="#94a3b8"/>
              <circle cx="90" cy="65" r="3" fill="#94a3b8"/>
              <circle cx="105" cy="70" r="3" fill="#94a3b8"/>
              <circle cx="120" cy="75" r="3" fill="#94a3b8"/>
              <circle cx="135" cy="80" r="3" fill="#94a3b8"/>
              <circle cx="150" cy="85" r="3" fill="#94a3b8"/>

              <!-- Another elongated -->
              <circle cx="30" cy="90" r="3" fill="#94a3b8"/>
              <circle cx="45" cy="95" r="3" fill="#94a3b8"/>
              <circle cx="60" cy="100" r="3" fill="#94a3b8"/>
              <circle cx="75" cy="105" r="3" fill="#94a3b8"/>
              <circle cx="90" cy="110" r="3" fill="#94a3b8"/>
              <circle cx="105" cy="115" r="3" fill="#94a3b8"/>

              <text x="90" y="135" text-anchor="middle" font-size="8" fill="#64748b">Non-spherical clusters</text>
            </g>
          </svg>`,
          caption: 'K-Means works best with spherical, similarly-sized clusters'
        },
        {
          type: 'heading',
          content: 'Improving K-Means: K-Means++'
        },
        {
          type: 'text',
          content: `Standard K-Means initializes centroids randomly, which can lead to poor results. **K-Means++** is a smarter initialization:

1. Choose first centroid randomly from data points
2. For each subsequent centroid, choose a point with probability proportional to its squared distance from the nearest existing centroid
3. This spreads out initial centroids, leading to faster convergence and better results

Scikit-learn uses K-Means++ by default (\`init='k-means++'\`).`
        },
        {
          type: 'keypoints',
          points: [
            'K-Means partitions data into K clusters by minimizing within-cluster variance',
            'The algorithm alternates between assignment and update steps until convergence',
            'Use the Elbow Method to help choose K',
            'K-Means++ initialization improves results',
            'Works best for spherical clusters of similar size'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does K-Means optimize?',
          options: [
            'Number of clusters',
            'Within-cluster sum of squares (WCSS)',
            'Distance between cluster centers',
            'Number of iterations'
          ],
          correct: 1,
          explanation: 'K-Means minimizes the within-cluster sum of squares (WCSS), also called inertia - the sum of squared distances from points to their cluster centroids.'
        },
        {
          type: 'multiple-choice',
          question: 'What is the purpose of the Elbow Method?',
          options: [
            'To initialize centroids',
            'To determine the optimal number of clusters K',
            'To measure cluster quality',
            'To speed up convergence'
          ],
          correct: 1,
          explanation: 'The Elbow Method helps determine the optimal K by plotting WCSS vs K and looking for the "elbow" where adding more clusters gives diminishing returns.'
        }
      ]
    },
    {
      id: 'hierarchical-clustering',
      title: 'Hierarchical Clustering',
      duration: '50 min',
      concepts: ['Dendrogram', 'Agglomerative', 'Linkage'],
      content: [
        {
          type: 'heading',
          content: 'Beyond Flat Clustering'
        },
        {
          type: 'text',
          content: `K-Means gives you K clusters - a flat partition of the data. But what if you want to explore clustering at different levels of granularity?

**Hierarchical clustering** builds a tree of clusters. You can cut this tree at any level to get the desired number of clusters. It's like a family tree for your data points.`
        },
        {
          type: 'heading',
          content: 'Two Approaches'
        },
        {
          type: 'text',
          content: `**Agglomerative (Bottom-Up)**: Start with each point as its own cluster. Repeatedly merge the two closest clusters until only one remains.

**Divisive (Top-Down)**: Start with all points in one cluster. Repeatedly split clusters until each point is its own cluster.

Agglomerative is far more common because it's computationally simpler.`
        },
        {
          type: 'visualization',
          title: 'Agglomerative Clustering Process',
          svg: `<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="500" height="220" fill="#f8fafc"/>

            <!-- Step 1 -->
            <g transform="translate(10,30)">
              <text x="55" y="-10" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Step 1</text>
              <rect x="5" y="0" width="100" height="160" fill="white" stroke="#e2e8f0" rx="4"/>
              <circle cx="25" cy="30" r="8" fill="#ef4444" stroke="white" stroke-width="2"/>
              <circle cx="55" cy="40" r="8" fill="#3b82f6" stroke="white" stroke-width="2"/>
              <circle cx="35" cy="70" r="8" fill="#10b981" stroke="white" stroke-width="2"/>
              <circle cx="70" cy="90" r="8" fill="#f59e0b" stroke="white" stroke-width="2"/>
              <circle cx="45" cy="120" r="8" fill="#8b5cf6" stroke="white" stroke-width="2"/>
              <text x="55" y="150" text-anchor="middle" font-size="8" fill="#64748b">5 clusters</text>
            </g>

            <!-- Arrow -->
            <text x="130" y="110" font-size="16" fill="#94a3b8">→</text>

            <!-- Step 2 -->
            <g transform="translate(140,30)">
              <text x="55" y="-10" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Step 2</text>
              <rect x="5" y="0" width="100" height="160" fill="white" stroke="#e2e8f0" rx="4"/>
              <ellipse cx="40" cy="35" rx="25" ry="20" fill="none" stroke="#64748b" stroke-dasharray="3"/>
              <circle cx="25" cy="30" r="8" fill="#ec4899" stroke="white" stroke-width="2"/>
              <circle cx="55" cy="40" r="8" fill="#ec4899" stroke="white" stroke-width="2"/>
              <circle cx="35" cy="70" r="8" fill="#10b981" stroke="white" stroke-width="2"/>
              <circle cx="70" cy="90" r="8" fill="#f59e0b" stroke="white" stroke-width="2"/>
              <circle cx="45" cy="120" r="8" fill="#8b5cf6" stroke="white" stroke-width="2"/>
              <text x="55" y="150" text-anchor="middle" font-size="8" fill="#64748b">4 clusters</text>
            </g>

            <!-- Arrow -->
            <text x="260" y="110" font-size="16" fill="#94a3b8">→</text>

            <!-- Step 3 -->
            <g transform="translate(270,30)">
              <text x="55" y="-10" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Step 3</text>
              <rect x="5" y="0" width="100" height="160" fill="white" stroke="#e2e8f0" rx="4"/>
              <ellipse cx="40" cy="35" rx="25" ry="20" fill="none" stroke="#64748b" stroke-dasharray="3"/>
              <circle cx="25" cy="30" r="8" fill="#ec4899" stroke="white" stroke-width="2"/>
              <circle cx="55" cy="40" r="8" fill="#ec4899" stroke="white" stroke-width="2"/>
              <ellipse cx="52" cy="95" rx="30" ry="35" fill="none" stroke="#64748b" stroke-dasharray="3"/>
              <circle cx="35" cy="70" r="8" fill="#0ea5e9" stroke="white" stroke-width="2"/>
              <circle cx="70" cy="90" r="8" fill="#0ea5e9" stroke="white" stroke-width="2"/>
              <circle cx="45" cy="120" r="8" fill="#0ea5e9" stroke="white" stroke-width="2"/>
              <text x="55" y="150" text-anchor="middle" font-size="8" fill="#64748b">2 clusters</text>
            </g>

            <!-- Arrow -->
            <text x="390" y="110" font-size="16" fill="#94a3b8">→</text>

            <!-- Final -->
            <g transform="translate(400,30)">
              <text x="45" y="-10" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Final</text>
              <rect x="5" y="0" width="85" height="160" fill="white" stroke="#e2e8f0" rx="4"/>
              <ellipse cx="45" cy="75" rx="35" ry="65" fill="none" stroke="#64748b" stroke-dasharray="3"/>
              <circle cx="25" cy="30" r="7" fill="#6366f1" stroke="white" stroke-width="2"/>
              <circle cx="55" cy="40" r="7" fill="#6366f1" stroke="white" stroke-width="2"/>
              <circle cx="35" cy="70" r="7" fill="#6366f1" stroke="white" stroke-width="2"/>
              <circle cx="60" cy="90" r="7" fill="#6366f1" stroke="white" stroke-width="2"/>
              <circle cx="40" cy="115" r="7" fill="#6366f1" stroke="white" stroke-width="2"/>
              <text x="45" y="150" text-anchor="middle" font-size="8" fill="#64748b">1 cluster</text>
            </g>
          </svg>`,
          caption: 'Agglomerative clustering progressively merges closest clusters'
        },
        {
          type: 'heading',
          content: 'Dendrograms: The Cluster Tree'
        },
        {
          type: 'text',
          content: `A **dendrogram** is a tree diagram that shows the sequence of cluster merges. The height of each merge indicates how far apart the clusters were.

You can "cut" the dendrogram at any height to get a specific number of clusters. Cut high = few clusters. Cut low = many clusters.`
        },
        {
          type: 'visualization',
          title: 'Reading a Dendrogram',
          svg: `<svg viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="250" fill="#f8fafc"/>

            <!-- Y-axis -->
            <line x1="50" y1="220" x2="50" y2="30" stroke="#475569" stroke-width="2"/>
            <text x="20" y="125" font-size="10" fill="#475569" transform="rotate(-90,20,125)">Distance</text>

            <!-- Y-axis ticks -->
            <g font-size="9" fill="#64748b">
              <line x1="45" y1="200" x2="50" y2="200" stroke="#64748b"/>
              <text x="40" y="203" text-anchor="end">0</text>
              <line x1="45" y1="150" x2="50" y2="150" stroke="#64748b"/>
              <text x="40" y="153" text-anchor="end">2</text>
              <line x1="45" y1="100" x2="50" y2="100" stroke="#64748b"/>
              <text x="40" y="103" text-anchor="end">4</text>
              <line x1="45" y1="50" x2="50" y2="50" stroke="#64748b"/>
              <text x="40" y="53" text-anchor="end">6</text>
            </g>

            <!-- Leaf nodes -->
            <g font-size="10" fill="#475569">
              <text x="90" y="235" text-anchor="middle">A</text>
              <text x="130" y="235" text-anchor="middle">B</text>
              <text x="200" y="235" text-anchor="middle">C</text>
              <text x="240" y="235" text-anchor="middle">D</text>
              <text x="310" y="235" text-anchor="middle">E</text>
            </g>

            <!-- Vertical lines from leaves -->
            <line x1="90" y1="220" x2="90" y2="180" stroke="#3b82f6" stroke-width="2"/>
            <line x1="130" y1="220" x2="130" y2="180" stroke="#3b82f6" stroke-width="2"/>
            <line x1="200" y1="220" x2="200" y2="140" stroke="#10b981" stroke-width="2"/>
            <line x1="240" y1="220" x2="240" y2="140" stroke="#10b981" stroke-width="2"/>
            <line x1="310" y1="220" x2="310" y2="80" stroke="#8b5cf6" stroke-width="2"/>

            <!-- First merge: A-B -->
            <line x1="90" y1="180" x2="130" y2="180" stroke="#3b82f6" stroke-width="2"/>
            <line x1="110" y1="180" x2="110" y2="100" stroke="#3b82f6" stroke-width="2"/>

            <!-- Second merge: C-D -->
            <line x1="200" y1="140" x2="240" y2="140" stroke="#10b981" stroke-width="2"/>
            <line x1="220" y1="140" x2="220" y2="100" stroke="#10b981" stroke-width="2"/>

            <!-- Third merge: (A-B)-(C-D) -->
            <line x1="110" y1="100" x2="220" y2="100" stroke="#f59e0b" stroke-width="2"/>
            <line x1="165" y1="100" x2="165" y2="60" stroke="#f59e0b" stroke-width="2"/>

            <!-- Fourth merge: all -->
            <line x1="165" y1="60" x2="310" y2="60" stroke="#ef4444" stroke-width="2"/>
            <line x1="237" y1="60" x2="237" y2="50" stroke="#ef4444" stroke-width="2"/>
            <line x1="310" y1="80" x2="310" y2="60" stroke="#8b5cf6" stroke-width="2"/>

            <!-- Cut line for 3 clusters -->
            <line x1="55" y1="120" x2="350" y2="120" stroke="#ef4444" stroke-width="1" stroke-dasharray="5"/>
            <text x="355" y="123" font-size="9" fill="#ef4444">Cut → 3 clusters</text>

            <!-- Cut line for 2 clusters -->
            <line x1="55" y1="80" x2="350" y2="80" stroke="#10b981" stroke-width="1" stroke-dasharray="5"/>
            <text x="355" y="83" font-size="9" fill="#10b981">Cut → 2 clusters</text>
          </svg>`,
          caption: 'Cut the dendrogram at different heights to get different numbers of clusters'
        },
        {
          type: 'heading',
          content: 'Linkage Methods'
        },
        {
          type: 'text',
          content: `When merging clusters, how do we measure the distance between two clusters? Different **linkage** methods give different results:`
        },
        {
          type: 'table',
          headers: ['Linkage', 'Distance Definition', 'Characteristics'],
          rows: [
            ['Single', 'Min distance between any two points', 'Can create long, chain-like clusters'],
            ['Complete', 'Max distance between any two points', 'Tends to create compact clusters'],
            ['Average', 'Average of all pairwise distances', 'Balance between single and complete'],
            ['Ward', 'Minimizes variance increase', 'Creates spherical, similar-sized clusters']
          ],
          caption: 'Common linkage methods for hierarchical clustering'
        },
        {
          type: 'code',
          language: 'python',
          content: `from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')

clusters = fcluster(Z, t=3, criterion='maxclust')
print(f"Cluster assignments: {clusters}")`
        },
        {
          type: 'heading',
          content: 'When to Use Hierarchical Clustering'
        },
        {
          type: 'text',
          content: `**Advantages:**
- No need to specify K upfront
- Dendrogram provides insights into data structure
- Can reveal hierarchical relationships (taxonomy, evolution)

**Disadvantages:**
- O(n²) memory and O(n³) time - doesn't scale well
- Once a merge happens, it can't be undone
- Sensitive to noise and outliers`
        },
        {
          type: 'callout',
          variant: 'tip',
          content: 'Use hierarchical clustering for small datasets (< 10,000 points) when you want to explore the clustering structure, or when the data has natural hierarchies.'
        },
        {
          type: 'keypoints',
          points: [
            'Hierarchical clustering builds a tree of clusters',
            'Agglomerative (bottom-up) is more common than divisive (top-down)',
            'Dendrograms visualize the cluster hierarchy',
            'Different linkage methods produce different cluster shapes',
            'Ward linkage often works well for general-purpose clustering'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does cutting a dendrogram horizontally represent?',
          options: [
            'Removing outliers',
            'Choosing the number of clusters',
            'Calculating cluster distances',
            'Merging all data points'
          ],
          correct: 1,
          explanation: 'Cutting a dendrogram at a specific height determines the number of clusters. Higher cuts give fewer clusters, lower cuts give more clusters.'
        },
        {
          type: 'multiple-choice',
          question: 'Which linkage method tends to create compact, spherical clusters?',
          options: [
            'Single linkage',
            'Complete linkage',
            'Ward linkage',
            'Average linkage'
          ],
          correct: 2,
          explanation: 'Ward linkage minimizes the increase in total within-cluster variance, which tends to produce compact, spherical clusters of similar sizes.'
        }
      ]
    },
    {
      id: 'dimensionality-reduction',
      title: 'Dimensionality Reduction',
      duration: '50 min',
      concepts: ['Curse of Dimensionality', 'Feature Compression', 'Visualization'],
      content: [
        {
          type: 'heading',
          content: 'The Curse of Dimensionality'
        },
        {
          type: 'text',
          content: `Modern datasets often have hundreds or thousands of features. This seems like more information should be better, right? Unfortunately, high dimensions create serious problems:

**1. Sparse Data**: As dimensions increase, data points become increasingly far apart. In 1D, 10 points might cover the space well. In 100D, you'd need astronomically more points for the same coverage.

**2. Distance Meaninglessness**: In high dimensions, the difference between the nearest and farthest point approaches zero. All points seem equally far away!

**3. Computational Cost**: More dimensions = more computation, more memory, longer training times.`
        },
        {
          type: 'visualization',
          title: 'The Curse: Points Spread Out',
          svg: `<svg viewBox="0 0 450 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="180" fill="#f8fafc"/>

            <!-- 1D -->
            <g transform="translate(20,30)">
              <text x="50" y="-5" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">1D</text>
              <line x1="10" y1="30" x2="90" y2="30" stroke="#475569" stroke-width="2"/>
              <circle cx="20" cy="30" r="5" fill="#3b82f6"/>
              <circle cx="35" cy="30" r="5" fill="#3b82f6"/>
              <circle cx="45" cy="30" r="5" fill="#3b82f6"/>
              <circle cx="60" cy="30" r="5" fill="#3b82f6"/>
              <circle cx="80" cy="30" r="5" fill="#3b82f6"/>
              <text x="50" y="60" text-anchor="middle" font-size="9" fill="#64748b">Dense coverage</text>
            </g>

            <!-- 2D -->
            <g transform="translate(140,30)">
              <text x="50" y="-5" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">2D</text>
              <rect x="10" y="10" width="80" height="80" fill="white" stroke="#475569" stroke-width="2"/>
              <circle cx="25" cy="25" r="4" fill="#3b82f6"/>
              <circle cx="70" cy="35" r="4" fill="#3b82f6"/>
              <circle cx="40" cy="60" r="4" fill="#3b82f6"/>
              <circle cx="60" cy="75" r="4" fill="#3b82f6"/>
              <circle cx="75" cy="55" r="4" fill="#3b82f6"/>
              <text x="50" y="110" text-anchor="middle" font-size="9" fill="#64748b">Sparser</text>
            </g>

            <!-- 3D -->
            <g transform="translate(260,30)">
              <text x="55" y="-5" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">3D</text>
              <!-- Cube outline -->
              <path d="M30,80 L30,30 L70,10 L110,30 L110,80 L70,100 L30,80 L70,60 L70,100 M70,60 L110,80 M70,10 L70,60 M30,30 L70,60"
                fill="none" stroke="#475569" stroke-width="1.5"/>
              <circle cx="45" cy="50" r="3" fill="#3b82f6"/>
              <circle cx="85" cy="40" r="3" fill="#3b82f6"/>
              <circle cx="65" cy="70" r="3" fill="#3b82f6"/>
              <circle cx="55" cy="35" r="3" fill="#3b82f6"/>
              <circle cx="90" cy="65" r="3" fill="#3b82f6"/>
              <text x="55" y="120" text-anchor="middle" font-size="9" fill="#64748b">Even sparser</text>
            </g>

            <!-- High D -->
            <g transform="translate(350,30)">
              <text x="45" y="-5" text-anchor="middle" font-size="11" fill="#475569" font-weight="500">100D</text>
              <rect x="10" y="10" width="70" height="80" fill="white" stroke="#475569" stroke-width="2"/>
              <circle cx="45" cy="50" r="3" fill="#3b82f6"/>
              <text x="45" y="65" font-size="8" fill="#64748b">?</text>
              <text x="45" y="110" text-anchor="middle" font-size="9" fill="#ef4444">Extremely sparse!</text>
            </g>
          </svg>`,
          caption: 'Same number of points becomes increasingly sparse as dimensions increase'
        },
        {
          type: 'heading',
          content: 'What is Dimensionality Reduction?'
        },
        {
          type: 'text',
          content: `Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation while preserving as much important information as possible.

**Two main goals:**
1. **Visualization**: Reduce to 2D or 3D for plotting
2. **Feature extraction**: Create compressed features for ML models

**Two main approaches:**
1. **Feature selection**: Choose a subset of existing features
2. **Feature extraction**: Create new features from combinations of old ones (PCA, etc.)`
        },
        {
          type: 'visualization',
          title: 'Dimensionality Reduction Concept',
          svg: `<svg viewBox="0 0 400 160" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="160" fill="#f8fafc"/>

            <!-- High-D representation -->
            <g transform="translate(30,20)">
              <text x="50" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Original (High-D)</text>
              <rect x="0" y="10" width="100" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Feature bars -->
              <rect x="10" y="25" width="80" height="8" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="25" width="65" height="8" fill="#3b82f6" rx="2"/>

              <rect x="10" y="40" width="80" height="8" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="40" width="30" height="8" fill="#3b82f6" rx="2"/>

              <rect x="10" y="55" width="80" height="8" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="55" width="55" height="8" fill="#3b82f6" rx="2"/>

              <rect x="10" y="70" width="80" height="8" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="70" width="45" height="8" fill="#3b82f6" rx="2"/>

              <rect x="10" y="85" width="80" height="8" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="85" width="70" height="8" fill="#3b82f6" rx="2"/>

              <text x="50" y="105" text-anchor="middle" font-size="9" fill="#64748b">100 features</text>
            </g>

            <!-- Arrow -->
            <g transform="translate(150,65)">
              <path d="M0,20 L60,20" stroke="#10b981" stroke-width="2" marker-end="url(#greenarrow)"/>
              <text x="30" y="10" text-anchor="middle" font-size="9" fill="#10b981">Reduce</text>
            </g>

            <!-- Low-D representation -->
            <g transform="translate(230,20)">
              <text x="50" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Reduced (Low-D)</text>
              <rect x="0" y="10" width="100" height="100" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Fewer feature bars -->
              <rect x="10" y="35" width="80" height="12" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="35" width="75" height="12" fill="#10b981" rx="2"/>

              <rect x="10" y="55" width="80" height="12" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="55" width="50" height="12" fill="#10b981" rx="2"/>

              <rect x="10" y="75" width="80" height="12" fill="#e2e8f0" rx="2"/>
              <rect x="10" y="75" width="60" height="12" fill="#10b981" rx="2"/>

              <text x="50" y="105" text-anchor="middle" font-size="9" fill="#64748b">3 components</text>
            </g>

            <defs>
              <marker id="greenarrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#10b981"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Dimensionality reduction compresses many features into fewer components'
        },
        {
          type: 'heading',
          content: 'Linear vs Non-Linear Methods'
        },
        {
          type: 'text',
          content: `**Linear methods** (PCA, LDA) find linear combinations of features. They work well when the data lies near a linear subspace.

**Non-linear methods** (t-SNE, UMAP) can capture complex, curved structures in data. They're better for visualization but harder to interpret.`
        },
        {
          type: 'table',
          headers: ['Method', 'Type', 'Best For'],
          rows: [
            ['PCA', 'Linear', 'General purpose, feature extraction'],
            ['LDA', 'Linear (supervised)', 'Classification preprocessing'],
            ['t-SNE', 'Non-linear', 'Visualization, clusters'],
            ['UMAP', 'Non-linear', 'Visualization, preserves structure'],
            ['Autoencoders', 'Non-linear (neural)', 'Complex patterns, images']
          ],
          caption: 'Popular dimensionality reduction techniques'
        },
        {
          type: 'heading',
          content: 'Information Loss Trade-off'
        },
        {
          type: 'text',
          content: `Dimensionality reduction always involves some information loss. The goal is to lose the least important information - typically noise - while keeping the signal.

Think of it like summarizing a book. A good summary captures the main ideas while leaving out minor details. A bad summary loses crucial plot points.

The key question: **How many dimensions do we need?** This depends on:
- How much variance we want to preserve (usually 90-99%)
- Downstream task requirements
- Visualization constraints (2D or 3D)`
        },
        {
          type: 'keypoints',
          points: [
            'High dimensions cause problems: sparsity, meaningless distances, computational cost',
            'Dimensionality reduction compresses data while preserving important information',
            'Linear methods (PCA) are simple and interpretable',
            'Non-linear methods (t-SNE, UMAP) capture complex structures',
            'Always consider the trade-off between compression and information loss'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the "curse of dimensionality"?',
          options: [
            'Having too few features',
            'Problems that arise as the number of dimensions increases',
            'Difficulty in visualizing data',
            'Slow computer processing'
          ],
          correct: 1,
          explanation: 'The curse of dimensionality refers to problems that arise in high-dimensional spaces: data becomes sparse, distances become meaningless, and computation becomes expensive.'
        },
        {
          type: 'multiple-choice',
          question: 'Which dimensionality reduction method is best for visualization of complex clusters?',
          options: [
            'PCA',
            'LDA',
            't-SNE',
            'Random projection'
          ],
          correct: 2,
          explanation: 't-SNE is specifically designed for visualization, especially for revealing cluster structure. It uses non-linear mappings to preserve local neighborhood relationships.'
        }
      ]
    },
    {
      id: 'pca',
      title: 'Principal Component Analysis (PCA)',
      duration: '60 min',
      concepts: ['PCA', 'Eigenvalues', 'Variance', 'Principal Components'],
      content: [
        {
          type: 'heading',
          content: 'The Intuition Behind PCA'
        },
        {
          type: 'text',
          content: `Imagine you're photographing a 3D object. Different angles give different views - some show the object's shape clearly, others are just confusing silhouettes.

PCA finds the "best angle" to view your data. It identifies the directions along which data varies the most, called **principal components**. The first principal component captures the most variance, the second captures the most of what's left, and so on.

**Key insight**: Directions with high variance contain signal. Directions with low variance are often just noise.`
        },
        {
          type: 'visualization',
          title: 'Finding the Direction of Maximum Variance',
          svg: `<svg viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="250" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="50" y1="200" x2="350" y2="200" stroke="#cbd5e1" stroke-width="1"/>
            <line x1="50" y1="200" x2="50" y2="30" stroke="#cbd5e1" stroke-width="1"/>
            <text x="355" y="205" font-size="10" fill="#64748b">x₁</text>
            <text x="45" y="25" font-size="10" fill="#64748b">x₂</text>

            <!-- Data points in elongated cluster -->
            <g fill="#3b82f6">
              <circle cx="100" cy="170" r="4"/>
              <circle cx="120" cy="155" r="4"/>
              <circle cx="135" cy="148" r="4"/>
              <circle cx="150" cy="135" r="4"/>
              <circle cx="165" cy="128" r="4"/>
              <circle cx="180" cy="115" r="4"/>
              <circle cx="195" cy="108" r="4"/>
              <circle cx="210" cy="95" r="4"/>
              <circle cx="225" cy="88" r="4"/>
              <circle cx="240" cy="75" r="4"/>
              <circle cx="255" cy="68" r="4"/>
              <circle cx="270" cy="55" r="4"/>
              <circle cx="285" cy="48" r="4"/>

              <!-- Some scatter -->
              <circle cx="145" cy="160" r="4"/>
              <circle cx="175" cy="100" r="4"/>
              <circle cx="200" cy="120" r="4"/>
              <circle cx="230" cy="65" r="4"/>
              <circle cx="160" cy="140" r="4"/>
            </g>

            <!-- PC1 - main axis of variance -->
            <line x1="80" y1="185" x2="300" y2="35" stroke="#10b981" stroke-width="3"/>
            <text x="305" y="40" font-size="11" fill="#10b981" font-weight="500">PC1</text>

            <!-- PC2 - perpendicular -->
            <line x1="130" y1="55" x2="250" y2="165" stroke="#ef4444" stroke-width="2" stroke-dasharray="5"/>
            <text x="255" y="170" font-size="11" fill="#ef4444" font-weight="500">PC2</text>

            <!-- Center point -->
            <circle cx="190" cy="110" r="5" fill="#f59e0b" stroke="white" stroke-width="2"/>
            <text x="195" y="125" font-size="9" fill="#f59e0b">mean</text>

            <!-- Legend -->
            <g transform="translate(60,220)">
              <line x1="0" y1="0" x2="20" y2="0" stroke="#10b981" stroke-width="3"/>
              <text x="25" y="4" font-size="9" fill="#475569">PC1: Maximum variance direction</text>
            </g>
          </svg>`,
          caption: 'PC1 points in the direction of maximum data variance'
        },
        {
          type: 'heading',
          content: 'The Mathematics of PCA'
        },
        {
          type: 'text',
          content: `PCA works by finding the eigenvectors of the covariance matrix. Here's what that means:

**Step 1: Center the data** - Subtract the mean from each feature so data is centered at origin.

**Step 2: Compute covariance matrix** - This captures how features vary together.`
        },
        {
          type: 'formula',
          content: 'Σ = (1/n) XᵀX  (for centered X)'
        },
        {
          type: 'text',
          content: `**Step 3: Find eigenvectors and eigenvalues** - The eigenvectors point in the principal component directions. The eigenvalues tell us how much variance each direction captures.`
        },
        {
          type: 'formula',
          content: 'Σv = λv'
        },
        {
          type: 'text',
          content: `**Step 4: Sort and select** - Sort eigenvectors by eigenvalue (descending). Keep the top k to reduce to k dimensions.

**Step 5: Transform** - Project data onto the selected eigenvectors.`
        },
        {
          type: 'formula',
          content: 'X_reduced = X × W  (W = matrix of top k eigenvectors)'
        },
        {
          type: 'heading',
          content: 'Understanding Explained Variance'
        },
        {
          type: 'text',
          content: `Each eigenvalue λᵢ tells us how much variance that component explains. The **explained variance ratio** is:`
        },
        {
          type: 'formula',
          content: 'Explained variance ratio = λᵢ / Σλⱼ'
        },
        {
          type: 'text',
          content: `The cumulative explained variance helps us decide how many components to keep. Common rule: keep enough to explain 90-95% of variance.`
        },
        {
          type: 'visualization',
          title: 'Explained Variance by Component',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Axes -->
            <line x1="60" y1="160" x2="350" y2="160" stroke="#475569" stroke-width="2"/>
            <line x1="60" y1="160" x2="60" y2="30" stroke="#475569" stroke-width="2"/>

            <text x="200" y="190" text-anchor="middle" font-size="10" fill="#475569">Principal Component</text>
            <text x="25" y="95" font-size="10" fill="#475569" transform="rotate(-90,25,95)">Variance %</text>

            <!-- Bars - individual variance -->
            <rect x="80" y="45" width="30" height="115" fill="#3b82f6" rx="2"/>
            <rect x="130" y="85" width="30" height="75" fill="#3b82f6" rx="2"/>
            <rect x="180" y="115" width="30" height="45" fill="#3b82f6" rx="2"/>
            <rect x="230" y="135" width="30" height="25" fill="#3b82f6" rx="2"/>
            <rect x="280" y="148" width="30" height="12" fill="#3b82f6" rx="2"/>

            <!-- Cumulative line -->
            <polyline points="95,45 145,70 195,90 245,105 295,112"
                      fill="none" stroke="#10b981" stroke-width="2"/>
            <circle cx="95" cy="45" r="4" fill="#10b981"/>
            <circle cx="145" cy="70" r="4" fill="#10b981"/>
            <circle cx="195" cy="90" r="4" fill="#10b981"/>
            <circle cx="245" cy="105" r="4" fill="#10b981"/>
            <circle cx="295" cy="112" r="4" fill="#10b981"/>

            <!-- 95% line -->
            <line x1="60" y1="95" x2="350" y2="95" stroke="#ef4444" stroke-width="1" stroke-dasharray="4"/>
            <text x="355" y="98" font-size="9" fill="#ef4444">95%</text>

            <!-- X labels -->
            <g font-size="9" fill="#64748b" text-anchor="middle">
              <text x="95" y="175">1</text>
              <text x="145" y="175">2</text>
              <text x="195" y="175">3</text>
              <text x="245" y="175">4</text>
              <text x="295" y="175">5</text>
            </g>

            <!-- Legend -->
            <rect x="100" y="10" width="12" height="12" fill="#3b82f6" rx="2"/>
            <text x="117" y="20" font-size="9" fill="#475569">Individual</text>
            <line x1="180" y1="16" x2="200" y2="16" stroke="#10b981" stroke-width="2"/>
            <text x="205" y="20" font-size="9" fill="#475569">Cumulative</text>
          </svg>`,
          caption: 'First 3 components capture 95% of variance - we can discard the rest'
        },
        {
          type: 'heading',
          content: 'PCA in Practice'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.random.randn(100, 10)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

pca_reduced = PCA(n_components=n_components_95)
X_reduced = pca_reduced.fit_transform(X_scaled)
print(f"Reduced shape: {X_reduced.shape}")`
        },
        {
          type: 'heading',
          content: 'When to Use PCA'
        },
        {
          type: 'text',
          content: `**Good use cases:**
- Reducing features before training ML models
- Visualizing high-dimensional data (reduce to 2-3D)
- Removing noise (low-variance components)
- Finding latent structure in data
- Preprocessing for algorithms that struggle with correlated features

**Limitations:**
- Assumes linear relationships
- Sensitive to feature scaling (always standardize first!)
- Principal components may not be interpretable
- Information loss is inevitable`
        },
        {
          type: 'callout',
          variant: 'warning',
          content: 'Always scale your features before PCA! PCA is sensitive to feature scales - a feature measured in millions will dominate one measured in decimals.'
        },
        {
          type: 'keypoints',
          points: [
            'PCA finds directions of maximum variance in data',
            'Eigenvectors of the covariance matrix give principal component directions',
            'Eigenvalues indicate how much variance each component explains',
            'Keep enough components to explain 90-95% of variance',
            'Always standardize features before applying PCA'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What does the first principal component represent?',
          options: [
            'The mean of the data',
            'The direction of maximum variance',
            'The smallest eigenvalue',
            'The most important feature'
          ],
          correct: 1,
          explanation: 'The first principal component points in the direction along which the data varies the most - the direction of maximum variance.'
        },
        {
          type: 'multiple-choice',
          question: 'Why should you standardize features before PCA?',
          options: [
            'To make computation faster',
            'Because PCA only works with normalized data',
            'Because PCA is sensitive to feature scales',
            'To remove outliers'
          ],
          correct: 2,
          explanation: 'PCA is sensitive to feature scales. Without standardization, features with larger scales will dominate the principal components, regardless of their actual importance.'
        }
      ]
    },
    {
      id: 'anomaly-detection',
      title: 'Anomaly Detection',
      duration: '50 min',
      concepts: ['Outliers', 'Statistical Methods', 'Isolation Forest'],
      content: [
        {
          type: 'heading',
          content: 'What is Anomaly Detection?'
        },
        {
          type: 'text',
          content: `Anomaly detection finds data points that are "different" from the majority. These outliers might be:

**Bad data**: Sensor errors, data entry mistakes, corrupted records
**Interesting events**: Fraud transactions, network intrusions, equipment failures
**Rare but normal**: Unusual but legitimate customer behavior

The challenge: What makes something "anomalous"? It's often context-dependent and subjective.`
        },
        {
          type: 'visualization',
          title: 'Spotting Anomalies',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Normal cluster -->
            <g fill="#3b82f6">
              <circle cx="180" cy="100" r="4"/>
              <circle cx="195" cy="110" r="4"/>
              <circle cx="170" cy="95" r="4"/>
              <circle cx="200" cy="90" r="4"/>
              <circle cx="185" cy="120" r="4"/>
              <circle cx="160" cy="105" r="4"/>
              <circle cx="210" cy="100" r="4"/>
              <circle cx="175" cy="115" r="4"/>
              <circle cx="190" cy="85" r="4"/>
              <circle cx="165" cy="90" r="4"/>
              <circle cx="205" cy="115" r="4"/>
              <circle cx="195" cy="95" r="4"/>
              <circle cx="180" cy="105" r="4"/>
              <circle cx="170" cy="110" r="4"/>
              <circle cx="200" cy="105" r="4"/>
            </g>

            <!-- Anomalies -->
            <circle cx="60" cy="50" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>
            <circle cx="320" cy="160" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>
            <circle cx="350" cy="45" r="6" fill="#ef4444" stroke="white" stroke-width="2"/>

            <!-- Labels -->
            <text x="60" y="35" text-anchor="middle" font-size="9" fill="#ef4444">Anomaly</text>
            <text x="320" y="180" text-anchor="middle" font-size="9" fill="#ef4444">Anomaly</text>
            <text x="350" y="30" text-anchor="middle" font-size="9" fill="#ef4444">Anomaly</text>

            <text x="185" y="145" text-anchor="middle" font-size="9" fill="#3b82f6">Normal data</text>

            <!-- Boundary -->
            <ellipse cx="185" cy="100" rx="50" ry="35" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="5"/>
          </svg>`,
          caption: 'Anomalies are data points that deviate significantly from the normal pattern'
        },
        {
          type: 'heading',
          content: 'Statistical Approaches'
        },
        {
          type: 'subheading',
          content: 'Z-Score Method'
        },
        {
          type: 'text',
          content: `For univariate data, points more than 2-3 standard deviations from the mean are often considered anomalies.`
        },
        {
          type: 'formula',
          content: 'z = (x - μ) / σ'
        },
        {
          type: 'text',
          content: `If |z| > 3, the point is unusual (occurs < 0.3% of the time in normal data).

**Limitation**: Assumes normal distribution. Fails for multimodal or skewed data.`
        },
        {
          type: 'subheading',
          content: 'IQR Method'
        },
        {
          type: 'text',
          content: `More robust to non-normal distributions. Uses quartiles:

- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1

Points below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are outliers.`
        },
        {
          type: 'heading',
          content: 'Isolation Forest'
        },
        {
          type: 'text',
          content: `**Isolation Forest** is a powerful tree-based method. The key insight: anomalies are easier to isolate.

Imagine randomly splitting data with if-then rules. Normal points in dense regions need many splits to isolate. Anomalies in sparse regions get isolated quickly.

**How it works:**
1. Build many random decision trees
2. For each tree, randomly select a feature and split value
3. Repeat until each point is isolated
4. Anomalies have shorter average path lengths`
        },
        {
          type: 'visualization',
          title: 'Isolation Forest Intuition',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Normal point - many splits needed -->
            <g transform="translate(20,20)">
              <text x="80" y="0" text-anchor="middle" font-size="10" fill="#3b82f6" font-weight="500">Normal Point</text>
              <rect x="10" y="10" width="140" height="130" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Dense cluster -->
              <g fill="#94a3b8">
                <circle cx="60" cy="60" r="3"/>
                <circle cx="70" cy="70" r="3"/>
                <circle cx="55" cy="75" r="3"/>
                <circle cx="75" cy="55" r="3"/>
                <circle cx="65" cy="65" r="3"/>
                <circle cx="80" cy="65" r="3"/>
                <circle cx="50" cy="60" r="3"/>
                <circle cx="70" cy="80" r="3"/>
                <circle cx="85" cy="75" r="3"/>
                <circle cx="65" cy="50" r="3"/>
              </g>

              <!-- Target point -->
              <circle cx="65" cy="65" r="4" fill="#3b82f6" stroke="white" stroke-width="2"/>

              <!-- Splits -->
              <line x1="40" y1="10" x2="40" y2="140" stroke="#10b981" stroke-width="1" stroke-dasharray="3"/>
              <line x1="40" y1="90" x2="140" y2="90" stroke="#10b981" stroke-width="1" stroke-dasharray="3"/>
              <line x1="90" y1="10" x2="90" y2="90" stroke="#10b981" stroke-width="1" stroke-dasharray="3"/>
              <line x1="40" y1="45" x2="90" y2="45" stroke="#10b981" stroke-width="1" stroke-dasharray="3"/>

              <text x="80" y="150" text-anchor="middle" font-size="9" fill="#64748b">Many splits needed</text>
            </g>

            <!-- Anomaly - few splits needed -->
            <g transform="translate(210,20)">
              <text x="80" y="0" text-anchor="middle" font-size="10" fill="#ef4444" font-weight="500">Anomaly</text>
              <rect x="10" y="10" width="140" height="130" fill="white" stroke="#e2e8f0" rx="4"/>

              <!-- Dense cluster -->
              <g fill="#94a3b8">
                <circle cx="60" cy="100" r="3"/>
                <circle cx="70" cy="110" r="3"/>
                <circle cx="55" cy="115" r="3"/>
                <circle cx="75" cy="95" r="3"/>
                <circle cx="65" cy="105" r="3"/>
                <circle cx="80" cy="105" r="3"/>
                <circle cx="50" cy="100" r="3"/>
                <circle cx="70" cy="120" r="3"/>
              </g>

              <!-- Anomaly point (isolated) -->
              <circle cx="120" cy="35" r="4" fill="#ef4444" stroke="white" stroke-width="2"/>

              <!-- Quick splits -->
              <line x1="100" y1="10" x2="100" y2="140" stroke="#10b981" stroke-width="1" stroke-dasharray="3"/>
              <line x1="100" y1="55" x2="150" y2="55" stroke="#10b981" stroke-width="1" stroke-dasharray="3"/>

              <text x="80" y="150" text-anchor="middle" font-size="9" fill="#64748b">Few splits needed</text>
            </g>
          </svg>`,
          caption: 'Anomalies are isolated quickly because they are in sparse regions'
        },
        {
          type: 'heading',
          content: 'Implementation'
        },
        {
          type: 'code',
          language: 'python',
          content: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_anomalies = np.random.uniform(low=-4, high=4, size=(10, 2))
X = np.vstack([X_normal, X_anomalies])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

predictions = iso_forest.fit_predict(X_scaled)

scores = iso_forest.decision_function(X_scaled)

anomalies = X[predictions == -1]
print(f"Detected {len(anomalies)} anomalies")`
        },
        {
          type: 'heading',
          content: 'Other Approaches'
        },
        {
          type: 'table',
          headers: ['Method', 'Approach', 'Best For'],
          rows: [
            ['Local Outlier Factor', 'Density-based, compares local density', 'Varying density clusters'],
            ['One-Class SVM', 'Learns boundary around normal data', 'High-dimensional data'],
            ['DBSCAN', 'Points not in any cluster are outliers', 'When clustering anyway'],
            ['Autoencoders', 'High reconstruction error = anomaly', 'Complex patterns, images']
          ],
          caption: 'Different anomaly detection methods suit different scenarios'
        },
        {
          type: 'heading',
          content: 'Practical Considerations'
        },
        {
          type: 'callout',
          variant: 'tip',
          content: 'The "contamination" parameter in Isolation Forest is crucial. Set it based on domain knowledge about the expected anomaly rate, not just trial and error.'
        },
        {
          type: 'text',
          content: `**Challenges in real-world anomaly detection:**

1. **Class imbalance**: Anomalies are rare by definition
2. **Novel anomalies**: New types of anomalies may look different
3. **Concept drift**: What's "normal" may change over time
4. **Ground truth**: Often we don't know what the real anomalies are
5. **False positives**: Too many alerts causes alert fatigue`
        },
        {
          type: 'keypoints',
          points: [
            'Anomaly detection finds data points that deviate from normal patterns',
            'Statistical methods (Z-score, IQR) work for simple univariate cases',
            'Isolation Forest is a powerful, scalable method for multivariate data',
            'The "contamination" parameter should reflect expected anomaly rate',
            'Real-world challenges include class imbalance and concept drift'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why are anomalies isolated quickly in an Isolation Forest?',
          options: [
            'They have higher values',
            'They are in sparse regions of the feature space',
            'They are always at the edges of the data',
            'They have more features'
          ],
          correct: 1,
          explanation: 'Anomalies exist in sparse regions where there are few other data points. Random splits can isolate them quickly because there are no neighboring points to separate them from.'
        },
        {
          type: 'multiple-choice',
          question: 'What does the contamination parameter control in Isolation Forest?',
          options: [
            'The number of trees',
            'The maximum depth of trees',
            'The expected proportion of anomalies',
            'The feature importance'
          ],
          correct: 2,
          explanation: 'The contamination parameter specifies the expected proportion of outliers in the dataset. It helps set the threshold for classifying points as anomalies.'
        }
      ]
    }
  ]
}
