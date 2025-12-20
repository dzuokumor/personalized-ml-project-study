export const audioprocessingcourse = {
  id: 'audio-processing-ml',
  title: 'Audio Processing & Feature Extraction',
  icon: 'audio',
  description: 'Master audio feature extraction with MFCC, spectral analysis, and XGBoost classification.',
  difficulty: 'advanced',
  sourceproject: 'multimodel-data-preprocessing',
  lessons: [
    {
      id: 'audio-fundamentals',
      title: 'Audio Signal Fundamentals',
      duration: '15 min read',
      concepts: ['Waveform', 'Sampling', 'Frequency'],
      content: [
        { type: 'heading', text: 'What is Audio Data?' },
        { type: 'paragraph', text: 'Audio is a continuous pressure wave through air. Computers represent this by sampling the wave at regular intervals. A sample rate of 22,050 Hz means 22,050 measurements per second—enough to capture frequencies up to 11,025 Hz (Nyquist theorem).' },

        { type: 'heading', text: 'Time Domain vs Frequency Domain' },
        { type: 'paragraph', text: 'The raw waveform is in the time domain—amplitude over time. But many audio characteristics are better understood in the frequency domain—which frequencies are present and how strong they are.' },
        { type: 'paragraph', text: 'The Fourier Transform converts between these domains. Most audio ML features are extracted from frequency representations.' },

        { type: 'heading', text: 'Your Audio Loader' },
        { type: 'code', language: 'python', filename: 'audio_processor.py', fromproject: 'multimodel-data-preprocessing',
          code: `class AudioProcessor:
    def __init__(self, audio_dir: str, output_csv: str, sample_rate: int = 22050):
        self.audio_dir = Path(audio_dir)
        self.output_csv = Path(output_csv)
        self.sample_rate = sample_rate
        self.audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr` },

        { type: 'paragraph', text: 'Librosa handles format conversion and resampling automatically. Setting sr=22050 ensures consistent sample rate across all files, important for comparing features.' },

        { type: 'keypoints', points: [
          'Audio is sampled at regular intervals to create digital representation',
          'Sample rate determines maximum capturable frequency (Nyquist)',
          'Frequency domain features are more useful than raw waveforms',
          'Consistent sample rate is essential for comparable features'
        ]}
      ],
      quiz: [
        {
          question: 'What does a sample rate of 22,050 Hz mean?',
          options: ['Maximum frequency', '22,050 amplitude measurements per second', 'File size in bytes', 'Number of channels'],
          correct: 1,
          explanation: 'Sample rate is the number of amplitude measurements taken per second to digitize the audio signal.'
        }
      ]
    },
    {
      id: 'mfcc-features',
      title: 'MFCC: The Gold Standard',
      duration: '20 min read',
      concepts: ['MFCC', 'Mel Scale', 'Cepstrum'],
      content: [
        { type: 'heading', text: 'Why MFCC?' },
        { type: 'paragraph', text: 'Mel-Frequency Cepstral Coefficients (MFCCs) are the most widely used audio features in speech and music processing. They approximate human auditory perception and compress spectral information into a compact representation.' },

        { type: 'heading', text: 'The Mel Scale' },
        { type: 'paragraph', text: 'Humans perceive pitch logarithmically—the difference between 100 Hz and 200 Hz sounds larger than between 1000 Hz and 1100 Hz, even though both are 100 Hz apart. The Mel scale models this perception.' },
        { type: 'formula', formula: 'mel = 2595 × log₁₀(1 + f/700)' },

        { type: 'heading', text: 'MFCC Extraction Pipeline' },
        { type: 'list', items: [
          '1. Frame the audio into overlapping windows (25ms typical)',
          '2. Apply FFT to get frequency spectrum',
          '3. Apply Mel filterbank to get Mel spectrum',
          '4. Take log of Mel energies',
          '5. Apply DCT to decorrelate and compress'
        ]},

        { type: 'heading', text: 'Your MFCC Implementation' },
        { type: 'code', language: 'python', filename: 'audio_processor.py', fromproject: 'multimodel-data-preprocessing',
          code: `def extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])` },

        { type: 'paragraph', text: 'You extract 13 MFCCs and compute both mean and standard deviation across time frames. This gives 26 features capturing both the average spectral shape and its variation over the clip.' },

        { type: 'callout', variant: 'tip', text: 'The first MFCC (C0) represents overall energy. Some practitioners exclude it; others find it useful. Your choice to include it is valid.' },

        { type: 'keypoints', points: [
          'MFCCs model human auditory perception',
          'The Mel scale compresses higher frequencies matching perception',
          'DCT decorrelates features for compact representation',
          'Mean and std across frames capture temporal dynamics'
        ]}
      ],
      quiz: [
        {
          question: 'Why use the Mel scale instead of linear frequency?',
          options: ['Faster computation', 'Matches human perception of pitch', 'Reduces file size', 'Required by librosa'],
          correct: 1,
          explanation: 'The Mel scale models how humans perceive pitch—logarithmically, with finer resolution at lower frequencies.'
        }
      ]
    },
    {
      id: 'spectral-features',
      title: 'Spectral Audio Features',
      duration: '15 min read',
      concepts: ['Spectral Centroid', 'Zero Crossing', 'Rolloff'],
      content: [
        { type: 'heading', text: 'Beyond MFCCs' },
        { type: 'paragraph', text: 'While MFCCs capture spectral shape, other features describe different aspects of audio. Combining multiple feature types often improves classification performance.' },

        { type: 'heading', text: 'Your Feature Extractors' },
        { type: 'code', language: 'python', filename: 'audio_processor.py', fromproject: 'multimodel-data-preprocessing',
          code: `def extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> float:
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return np.mean(rolloff)

def extract_energy(self, y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y)
    return np.mean(rms)

def extract_zero_crossing_rate(self, y: np.ndarray) -> float:
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)

def extract_spectral_centroid(self, y: np.ndarray, sr: int) -> float:
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)` },

        { type: 'subheading', text: 'What Each Feature Captures' },
        { type: 'list', items: [
          'Spectral Centroid: "Center of mass" of the spectrum. Higher = brighter sound.',
          'Spectral Rolloff: Frequency below which 85% of energy lies. Distinguishes harmonic vs noisy sounds.',
          'Zero Crossing Rate: How often signal crosses zero. Higher = more noise/percussion.',
          'RMS Energy: Overall loudness of the signal.'
        ]},

        { type: 'heading', text: 'Combined Feature Vector' },
        { type: 'code', language: 'python', filename: 'audio_processor.py', fromproject: 'multimodel-data-preprocessing',
          code: `def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
    mfcc_features = self.extract_mfcc(y, sr)
    spectral_rolloff = self.extract_spectral_rolloff(y, sr)
    energy = self.extract_energy(y)
    zcr = self.extract_zero_crossing_rate(y)
    spectral_centroid = self.extract_spectral_centroid(y, sr)

    return np.concatenate([
        mfcc_features,
        [spectral_rolloff, energy, zcr, spectral_centroid]
    ])` },

        { type: 'paragraph', text: 'Your final feature vector has 30 dimensions: 26 from MFCCs (13 mean + 13 std) plus 4 spectral features. This compact representation captures most information needed for audio classification.' },

        { type: 'keypoints', points: [
          'Spectral centroid indicates brightness of sound',
          'Zero crossing rate distinguishes percussive from tonal sounds',
          'Combining multiple feature types improves classification',
          'Averaging across frames produces fixed-size feature vectors'
        ]}
      ],
      quiz: [
        {
          question: 'What does a high zero crossing rate indicate?',
          options: ['Loud sound', 'Low pitch', 'Noisy or percussive content', 'Long duration'],
          correct: 2,
          explanation: 'Rapid zero crossings indicate high-frequency content or noise, common in percussion and consonants.'
        }
      ]
    },
    {
      id: 'audio-augmentation',
      title: 'Audio Data Augmentation',
      duration: '12 min read',
      concepts: ['Augmentation', 'Pitch Shift', 'Time Stretch'],
      content: [
        { type: 'heading', text: 'Why Augment Audio?' },
        { type: 'paragraph', text: 'Audio datasets are often small due to collection costs. Augmentation creates realistic variations of existing samples, effectively expanding the training set and improving model robustness.' },

        { type: 'heading', text: 'Your Augmentation Pipeline' },
        { type: 'code', language: 'python', filename: 'audio_processor.py', fromproject: 'multimodel-data-preprocessing',
          code: `def apply_pitch_shift(self, y: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def apply_time_stretch(self, y: np.ndarray, rate: float = 1.2) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)

def add_background_noise(self, y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def apply_augmentations(self, y: np.ndarray, sr: int) -> List[Tuple[np.ndarray, str]]:
    augmentations = []
    augmentations.append((y.copy(), 'original'))
    pitch_shifted = self.apply_pitch_shift(y, sr, n_steps=2.0)
    augmentations.append((pitch_shifted, 'pitch_shift'))
    time_stretched = self.apply_time_stretch(y, rate=1.2)
    augmentations.append((time_stretched, 'time_stretch'))
    noisy = self.add_background_noise(y, noise_factor=0.005)
    augmentations.append((noisy, 'background_noise'))
    return augmentations` },

        { type: 'subheading', text: 'Augmentation Techniques' },
        { type: 'list', items: [
          'Pitch Shift: Change pitch without changing speed. Simulates different speakers/instruments.',
          'Time Stretch: Change speed without changing pitch. Simulates tempo variations.',
          'Background Noise: Add random noise. Improves robustness to real-world conditions.',
          'Volume Change: Scale amplitude. Handles recording level variations.'
        ]},

        { type: 'callout', variant: 'warning', text: 'Augmentation parameters matter. Too extreme (e.g., n_steps=12) creates unrealistic samples that hurt performance.' },

        { type: 'keypoints', points: [
          'Augmentation expands small datasets with realistic variations',
          'Pitch and time can be modified independently',
          'Adding noise improves real-world robustness',
          'Keep augmentation parameters within realistic ranges'
        ]}
      ],
      quiz: [
        {
          question: 'Why apply augmentations during audio preprocessing?',
          options: ['Reduces file size', 'Expands training data and improves model robustness', 'Required by librosa', 'Speeds up inference'],
          correct: 1,
          explanation: 'Augmentation creates realistic variations, effectively increasing dataset size and training models to be invariant to common variations.'
        }
      ]
    },
    {
      id: 'xgboost-classification',
      title: 'XGBoost for Audio Classification',
      duration: '15 min read',
      concepts: ['XGBoost', 'Gradient Boosting', 'Hyperparameters'],
      content: [
        { type: 'heading', text: 'Why XGBoost?' },
        { type: 'paragraph', text: 'XGBoost (eXtreme Gradient Boosting) is often the best choice for tabular data. Unlike neural networks, it requires less data and tuning while providing excellent performance. For audio features (tabular after extraction), XGBoost is highly competitive.' },

        { type: 'heading', text: 'Gradient Boosting Explained' },
        { type: 'paragraph', text: 'Gradient boosting builds trees sequentially. Each tree learns to correct the errors of previous trees by fitting to the residuals. XGBoost optimizes this with regularization, parallel processing, and clever handling of missing values.' },

        { type: 'heading', text: 'Handling Class Imbalance with SMOTE' },
        { type: 'paragraph', text: 'Your product recommendation model used SMOTE (Synthetic Minority Oversampling Technique) to handle imbalanced classes. SMOTE creates synthetic examples of minority classes by interpolating between existing samples.' },

        { type: 'subheading', text: 'Key XGBoost Hyperparameters' },
        { type: 'list', items: [
          'n_estimators: Number of trees. More = better fit, slower training.',
          'max_depth: Tree depth. Deeper = more complex, risk of overfitting.',
          'learning_rate: Shrinks tree contribution. Lower = more trees needed.',
          'subsample: Fraction of samples per tree. Adds randomness.',
          'colsample_bytree: Fraction of features per tree. Decorrelates trees.'
        ]},

        { type: 'callout', variant: 'tip', text: 'Your GridSearchCV tested 108 parameter combinations. For faster tuning, try RandomizedSearchCV or Bayesian optimization.' },

        { type: 'keypoints', points: [
          'XGBoost excels on tabular data with less tuning than neural networks',
          'Gradient boosting fits residuals sequentially',
          'SMOTE handles class imbalance by creating synthetic samples',
          'Cross-validation prevents overfitting during hyperparameter search'
        ]}
      ],
      quiz: [
        {
          question: 'What does SMOTE do?',
          options: ['Removes outliers', 'Creates synthetic minority class samples', 'Normalizes features', 'Prunes decision trees'],
          correct: 1,
          explanation: 'SMOTE generates synthetic examples by interpolating between existing minority class samples, balancing the dataset.'
        }
      ]
    }
  ]
}
