export const learningpaths = [
  {
    id: 'foundations',
    title: 'ML Foundations',
    description: 'Essential mathematics, Python, and core concepts',
    icon: 'foundation',
    color: 'blue',
    courses: ['math-for-ml', 'python-for-ml', 'core-ml-concepts'],
    estimatedhours: 25,
    difficulty: 'beginner',
    order: 1
  },
  {
    id: 'classical-ml',
    title: 'Classical ML',
    description: 'Traditional algorithms and model evaluation',
    icon: 'chart',
    color: 'purple',
    courses: ['supervised-learning', 'unsupervised-learning', 'model-evaluation'],
    estimatedhours: 35,
    difficulty: 'intermediate',
    order: 2,
    prerequisites: ['foundations']
  },
  {
    id: 'deep-learning',
    title: 'Deep Learning',
    description: 'Neural networks, training, and computer vision',
    icon: 'layers',
    color: 'emerald',
    courses: ['neural-networks', 'training-deep-networks', 'cnns-computer-vision'],
    estimatedhours: 40,
    difficulty: 'intermediate',
    order: 3,
    prerequisites: ['foundations', 'classical-ml']
  },
  {
    id: 'sequence-nlp',
    title: 'Sequence & NLP',
    description: 'RNNs, transformers, and language processing',
    icon: 'text',
    color: 'amber',
    courses: ['rnns-lstms', 'attention-transformers', 'nlp-applications'],
    estimatedhours: 35,
    difficulty: 'advanced',
    order: 4,
    prerequisites: ['deep-learning']
  },
  {
    id: 'advanced-production',
    title: 'Advanced & Production',
    description: 'Generative models, MLOps, and real projects',
    icon: 'rocket',
    color: 'rose',
    courses: ['generative-models', 'mlops-curriculum', 'capstone-projects'],
    estimatedhours: 30,
    difficulty: 'advanced',
    order: 5,
    prerequisites: ['sequence-nlp']
  }
]

export const pathcolors = {
  blue: {
    bg: 'bg-blue-50',
    text: 'text-blue-600',
    border: 'border-blue-200',
    progress: 'bg-blue-500'
  },
  purple: {
    bg: 'bg-purple-50',
    text: 'text-purple-600',
    border: 'border-purple-200',
    progress: 'bg-purple-500'
  },
  emerald: {
    bg: 'bg-emerald-50',
    text: 'text-emerald-600',
    border: 'border-emerald-200',
    progress: 'bg-emerald-500'
  },
  amber: {
    bg: 'bg-amber-50',
    text: 'text-amber-600',
    border: 'border-amber-200',
    progress: 'bg-amber-500'
  },
  rose: {
    bg: 'bg-rose-50',
    text: 'text-rose-600',
    border: 'border-rose-200',
    progress: 'bg-rose-500'
  }
}
