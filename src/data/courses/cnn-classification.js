export const cnncourse = {
  id: 'cnn-image-classification',
  title: 'CNNs for Image Classification',
  icon: 'satellite',
  description: 'Learn convolutional neural networks through satellite image land cover classification.',
  difficulty: 'intermediate',
  sourceproject: 'lagos-urban-classification-ml-summative',
  lessons: [
    {
      id: 'cnn-history',
      title: 'The Rise of Convolutional Networks',
      duration: '15 min read',
      concepts: ['CNN', 'History', 'Computer Vision'],
      content: [
        { type: 'heading', text: 'From Biological Inspiration to ImageNet' },
        { type: 'paragraph', text: 'CNNs are inspired by the visual cortex, where neurons respond to stimuli in small regions of the visual field. Hubel and Wiesel won the Nobel Prize in 1981 for discovering this, providing the biological foundation for convolutional networks.' },
        { type: 'paragraph', text: 'Yann LeCun applied these ideas in 1989, creating LeNet for digit recognition. But CNNs remained niche until 2012 when AlexNet won the ImageNet competition by a huge margin, triggering the deep learning revolution.' },

        { type: 'heading', text: 'Why Convolutions for Images?' },
        { type: 'paragraph', text: 'Images have special properties that fully connected networks ignore:' },
        { type: 'list', items: [
          'Spatial locality: Nearby pixels are more related than distant ones',
          'Translation invariance: A cat is a cat regardless of position',
          'Hierarchical features: Edges combine into shapes, shapes into objects'
        ]},
        { type: 'paragraph', text: 'Convolutions exploit these properties, making CNNs parameter-efficient and effective for visual tasks.' },

        { type: 'heading', text: 'The Classification Task' },
        { type: 'paragraph', text: 'This project classifies satellite images into 10 land cover types using the ESA WorldCover 2021 dataset. This real-world application demonstrates CNNs on geospatial data.' },
        { type: 'code', language: 'python', filename: 'app.py', fromproject: 'lagos-urban-classification-ml-summative',
          code: `classes = {
    10: 'Trees',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves'
}` },

        { type: 'keypoints', points: [
          'CNNs are inspired by the hierarchical structure of biological vision',
          'AlexNet in 2012 proved deep CNNs could dominate visual tasks',
          'Convolutions exploit spatial locality and translation invariance',
          'Satellite imagery classification is a real-world CNN application'
        ]}
      ],
      quiz: [
        {
          question: 'What event triggered the deep learning revolution in computer vision?',
          options: ['LeNet in 1989', 'AlexNet winning ImageNet 2012', 'ResNet in 2015', 'VGG in 2014'],
          correct: 1,
          explanation: 'AlexNet won ImageNet 2012 with a 10% improvement over traditional methods, proving deep learning viability.'
        }
      ]
    },
    {
      id: 'convolution-operation',
      title: 'The Convolution Operation',
      duration: '18 min read',
      concepts: ['Convolution', 'Filters', 'Feature Maps'],
      content: [
        { type: 'heading', text: 'What is Convolution?' },
        { type: 'paragraph', text: 'Convolution slides a small filter (kernel) across the image, computing dot products at each position. The output, called a feature map, highlights where the filter pattern appears in the image.' },
        { type: 'paragraph', text: 'A 3x3 edge detection filter might have weights that respond strongly to horizontal intensity changes, producing high values wherever horizontal edges exist.' },

        { type: 'heading', text: 'Kernels and Feature Maps' },
        { type: 'paragraph', text: 'Each convolutional layer has multiple filters, each learning to detect different features. Early layers detect edges and colors; deeper layers detect complex patterns like textures and object parts.' },
        { type: 'formula', formula: 'Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m,n]' },

        { type: 'subheading', text: 'Key Hyperparameters' },
        { type: 'list', items: [
          'Kernel size: Typically 3x3 or 5x5. Larger sees more context but uses more parameters',
          'Stride: Step size when sliding. Stride 2 halves spatial dimensions',
          'Padding: Add zeros around edges to control output size',
          'Number of filters: More filters = more features, but more computation'
        ]},

        { type: 'heading', text: 'Why Convolution Works for Images' },
        { type: 'paragraph', text: 'Parameter sharing: The same filter weights are used at every position, dramatically reducing parameters. A 3x3 filter has only 9 weights regardless of image size.' },
        { type: 'paragraph', text: 'Local connectivity: Each output depends only on a small region, matching how visual features are local.' },

        { type: 'keypoints', points: [
          'Convolution applies learned filters across spatial positions',
          'Feature maps highlight where patterns appear in the input',
          'Parameter sharing makes CNNs efficient regardless of image size',
          'Hierarchical features emerge: edges → textures → parts → objects'
        ]}
      ],
      quiz: [
        {
          question: 'Why is parameter sharing important in CNNs?',
          options: ['Faster training only', 'Same features detected everywhere with fewer parameters', 'Reduces overfitting only', 'Required by GPU'],
          correct: 1,
          explanation: 'The same filter weights apply at every position, enabling translation invariance with far fewer parameters than fully connected layers.'
        }
      ]
    },
    {
      id: 'pooling-layers',
      title: 'Pooling and Downsampling',
      duration: '10 min read',
      concepts: ['Pooling', 'Max Pool', 'Downsampling'],
      content: [
        { type: 'heading', text: 'Why Downsample?' },
        { type: 'paragraph', text: 'Raw images are high resolution. A 224x224 RGB image has 150,528 values. Processing this directly would require massive computation. Pooling progressively reduces spatial dimensions while retaining important features.' },

        { type: 'heading', text: 'Max Pooling' },
        { type: 'paragraph', text: 'Max pooling takes the maximum value in each region. A 2x2 max pool with stride 2 reduces each dimension by half. The intuition: if a feature is detected anywhere in the region, preserve that information.' },
        { type: 'paragraph', text: 'Max pooling also provides some translation invariance—small shifts in feature position dont change the pooled output.' },

        { type: 'subheading', text: 'Average Pooling' },
        { type: 'paragraph', text: 'Average pooling takes the mean instead of maximum. Its smoother but may dilute strong activations. Often used at the end of networks (global average pooling) to collapse spatial dimensions entirely.' },

        { type: 'heading', text: 'The Image Pipeline' },
        { type: 'code', language: 'python', filename: 'app.py', fromproject: 'lagos-urban-classification-ml-summative',
          code: `img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
img_resized = img.resize((64, 64))
img_array = np.array(img_resized).astype('float32') / 255.0
img_input = img_array.reshape(1, 64, 64, 3)` },
        { type: 'paragraph', text: 'This model uses 64x64 images—already downsampled from satellite resolution. The model then further pools internally to extract compact feature representations.' },

        { type: 'keypoints', points: [
          'Pooling reduces spatial dimensions while preserving features',
          'Max pooling retains strongest activations in each region',
          'Provides translation invariance and reduces computation',
          '2x2 pooling with stride 2 halves each spatial dimension'
        ]}
      ],
      quiz: [
        {
          question: 'What does 2x2 max pooling with stride 2 do to a 64x64 feature map?',
          options: ['No change', 'Reduces to 32x32', 'Reduces to 16x16', 'Doubles to 128x128'],
          correct: 1,
          explanation: 'With stride 2, each dimension is halved: 64/2 = 32.'
        }
      ]
    },
    {
      id: 'softmax-multiclass',
      title: 'Multi-class Classification',
      duration: '12 min read',
      concepts: ['Softmax', 'Cross-Entropy', 'Multi-class'],
      content: [
        { type: 'heading', text: 'From Binary to Multi-class' },
        { type: 'paragraph', text: 'This land cover model classifies into 10 categories, not just 2. This requires softmax activation and categorical cross-entropy loss instead of sigmoid and binary cross-entropy.' },

        { type: 'heading', text: 'The Softmax Function' },
        { type: 'paragraph', text: 'Softmax converts raw scores (logits) into a probability distribution. All outputs sum to 1, and higher logits get exponentially more probability.' },
        { type: 'formula', formula: 'softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ' },

        { type: 'heading', text: 'Prediction and Confidence' },
        { type: 'code', language: 'python', filename: 'app.py', fromproject: 'lagos-urban-classification-ml-summative',
          code: `pred_probs = model_rgb.predict(img_input, verbose=0)[0]
pred_class_idx = np.argmax(pred_probs)
pred_class_code = reverse_mapping_rgb[pred_class_idx]
pred_class_name = classes[pred_class_code]
confidence = pred_probs[pred_class_idx]

all_idx = np.argsort(pred_probs)[::-1]
all_predictions = [(classes[reverse_mapping_rgb[idx]], float(pred_probs[idx])) for idx in all_idx]` },

        { type: 'paragraph', text: 'The code extracts not just the top prediction but all predictions sorted by probability. This allows showing the user how confident the model is across all classes.' },

        { type: 'callout', variant: 'info', text: 'High confidence in wrong predictions indicates the model learned wrong patterns. Low confidence on correct predictions suggests the model needs more training data.' },

        { type: 'keypoints', points: [
          'Softmax outputs probability distribution over all classes',
          'argmax selects the predicted class',
          'Confidence scores help assess prediction reliability',
          'Top-k predictions provide more nuanced output'
        ]}
      ],
      quiz: [
        {
          question: 'What guarantees that softmax outputs sum to 1?',
          options: ['Normalization in the formula', 'Special loss function', 'Gradient clipping', 'Batch normalization'],
          correct: 0,
          explanation: 'The division by Σe^xⱼ normalizes outputs to sum to 1, creating a valid probability distribution.'
        }
      ]
    }
  ]
}
