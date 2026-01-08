export const cnnscomputervision = {
  id: 'cnns-computer-vision',
  title: 'CNNs & Computer Vision',
  description: 'Master Convolutional Neural Networks - the backbone of modern computer vision - from basic operations to advanced architectures like ResNet',
  category: 'Deep Learning',
  difficulty: 'Advanced',
  duration: '7 hours',
  lessons: [
    {
      id: 'convolution-operation',
      title: 'The Convolution Operation',
      duration: '50 min',
      concepts: ['Convolution', 'Filters', 'Feature Maps'],
      content: [
        {
          type: 'heading',
          content: 'Why MLPs Fail for Images'
        },
        {
          type: 'text',
          content: `A 256×256 RGB image has 196,608 input features. A single hidden layer with 1000 neurons would need ~200 million weights. This is:

1. **Computationally expensive**: Too many parameters
2. **Data hungry**: Needs massive datasets to avoid overfitting
3. **Ignores structure**: Treats pixels as unrelated features

Images have **spatial structure** - nearby pixels are related. We need an architecture that exploits this.`
        },
        {
          type: 'heading',
          content: 'The Convolution Operation'
        },
        {
          type: 'text',
          content: `Convolution slides a small **filter** (kernel) across the image, computing dot products at each position. This produces a **feature map** that detects where the pattern exists.

Key insight: The same filter is applied everywhere (weight sharing), dramatically reducing parameters.`
        },
        {
          type: 'visualization',
          title: 'Convolution: Filter Sliding Over Image',
          svg: `<svg viewBox="0 0 450 220" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="220" fill="#f8fafc"/>

            <!-- Input image -->
            <g transform="translate(30,30)">
              <text x="45" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Input Image (5×5)</text>

              <!-- Grid -->
              <g fill="white" stroke="#94a3b8" stroke-width="1">
                <rect x="0" y="10" width="18" height="18"/>
                <rect x="18" y="10" width="18" height="18"/>
                <rect x="36" y="10" width="18" height="18"/>
                <rect x="54" y="10" width="18" height="18"/>
                <rect x="72" y="10" width="18" height="18"/>

                <rect x="0" y="28" width="18" height="18"/>
                <rect x="18" y="28" width="18" height="18"/>
                <rect x="36" y="28" width="18" height="18"/>
                <rect x="54" y="28" width="18" height="18"/>
                <rect x="72" y="28" width="18" height="18"/>

                <rect x="0" y="46" width="18" height="18"/>
                <rect x="18" y="46" width="18" height="18"/>
                <rect x="36" y="46" width="18" height="18"/>
                <rect x="54" y="46" width="18" height="18"/>
                <rect x="72" y="46" width="18" height="18"/>

                <rect x="0" y="64" width="18" height="18"/>
                <rect x="18" y="64" width="18" height="18"/>
                <rect x="36" y="64" width="18" height="18"/>
                <rect x="54" y="64" width="18" height="18"/>
                <rect x="72" y="64" width="18" height="18"/>

                <rect x="0" y="82" width="18" height="18"/>
                <rect x="18" y="82" width="18" height="18"/>
                <rect x="36" y="82" width="18" height="18"/>
                <rect x="54" y="82" width="18" height="18"/>
                <rect x="72" y="82" width="18" height="18"/>
              </g>

              <!-- Filter highlight -->
              <rect x="0" y="10" width="54" height="54" fill="#3b82f6" fill-opacity="0.2" stroke="#3b82f6" stroke-width="2"/>

              <!-- Numbers -->
              <g font-size="8" fill="#64748b" text-anchor="middle">
                <text x="9" y="23">1</text><text x="27" y="23">0</text><text x="45" y="23">2</text><text x="63" y="23">1</text><text x="81" y="23">0</text>
                <text x="9" y="41">0</text><text x="27" y="41">1</text><text x="45" y="41">1</text><text x="63" y="41">0</text><text x="81" y="41">1</text>
                <text x="9" y="59">1</text><text x="27" y="59">2</text><text x="45" y="59">0</text><text x="63" y="59">1</text><text x="81" y="59">0</text>
                <text x="9" y="77">0</text><text x="27" y="77">1</text><text x="45" y="77">1</text><text x="63" y="77">2</text><text x="81" y="77">1</text>
                <text x="9" y="95">1</text><text x="27" y="95">0</text><text x="45" y="95">0</text><text x="63" y="95">1</text><text x="81" y="95">0</text>
              </g>
            </g>

            <!-- Filter -->
            <g transform="translate(150,50)">
              <text x="27" y="-10" text-anchor="middle" font-size="10" fill="#3b82f6" font-weight="500">Filter (3×3)</text>

              <g fill="#dbeafe" stroke="#3b82f6" stroke-width="1">
                <rect x="0" y="0" width="18" height="18"/>
                <rect x="18" y="0" width="18" height="18"/>
                <rect x="36" y="0" width="18" height="18"/>
                <rect x="0" y="18" width="18" height="18"/>
                <rect x="18" y="18" width="18" height="18"/>
                <rect x="36" y="18" width="18" height="18"/>
                <rect x="0" y="36" width="18" height="18"/>
                <rect x="18" y="36" width="18" height="18"/>
                <rect x="36" y="36" width="18" height="18"/>
              </g>

              <g font-size="9" fill="#1e40af" text-anchor="middle" font-weight="500">
                <text x="9" y="13">1</text><text x="27" y="13">0</text><text x="45" y="13">-1</text>
                <text x="9" y="31">2</text><text x="27" y="31">0</text><text x="45" y="31">-2</text>
                <text x="9" y="49">1</text><text x="27" y="49">0</text><text x="45" y="49">-1</text>
              </g>
            </g>

            <!-- Arrow -->
            <path d="M220,80 L260,80" stroke="#64748b" stroke-width="2" marker-end="url(#convarrow)"/>
            <text x="240" y="95" text-anchor="middle" font-size="8" fill="#64748b">convolve</text>

            <!-- Output -->
            <g transform="translate(280,40)">
              <text x="27" y="0" text-anchor="middle" font-size="10" fill="#10b981" font-weight="500">Output (3×3)</text>

              <g fill="#d1fae5" stroke="#10b981" stroke-width="1">
                <rect x="0" y="10" width="18" height="18"/>
                <rect x="18" y="10" width="18" height="18"/>
                <rect x="36" y="10" width="18" height="18"/>
                <rect x="0" y="28" width="18" height="18"/>
                <rect x="18" y="28" width="18" height="18"/>
                <rect x="36" y="28" width="18" height="18"/>
                <rect x="0" y="46" width="18" height="18"/>
                <rect x="18" y="46" width="18" height="18"/>
                <rect x="36" y="46" width="18" height="18"/>
              </g>

              <!-- First output highlighted -->
              <rect x="0" y="10" width="18" height="18" fill="#10b981" stroke="#059669" stroke-width="2"/>
              <text x="9" y="23" text-anchor="middle" font-size="8" fill="white" font-weight="500">-1</text>
            </g>

            <!-- Calculation -->
            <text x="225" y="160" text-anchor="middle" font-size="8" fill="#64748b">1×1 + 0×0 + 2×(-1) + 0×2 + 1×0 + 1×(-2) + 1×1 + 2×0 + 0×(-1) = -1</text>

            <!-- Legend -->
            <text x="225" y="200" text-anchor="middle" font-size="9" fill="#475569">Element-wise multiply, then sum</text>

            <defs>
              <marker id="convarrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Convolution: slide filter over input, compute dot product at each position'
        },
        {
          type: 'heading',
          content: 'The Mathematics'
        },
        {
          type: 'formula',
          content: '(I * K)[i,j] = ΣₘΣₙ I[i+m, j+n] × K[m,n]'
        },
        {
          type: 'text',
          content: `Where I is the input, K is the kernel (filter), and * denotes convolution.

For each output position [i,j]:
1. Place the filter at position [i,j]
2. Multiply corresponding elements
3. Sum all products → one output value`
        },
        {
          type: 'heading',
          content: 'What Filters Detect'
        },
        {
          type: 'text',
          content: `Different filters detect different patterns:

**Edge detection** (Sobel, etc.): High response where pixel intensity changes rapidly
**Blur**: Averaging filter smooths the image
**Sharpen**: Emphasizes differences from neighbors

Early CNN layers learn simple patterns (edges, textures). Deeper layers combine these into complex features (eyes, wheels, faces).`
        },
        {
          type: 'visualization',
          title: 'Common Filter Patterns',
          svg: `<svg viewBox="0 0 400 160" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="160" fill="#f8fafc"/>

            <!-- Vertical edge -->
            <g transform="translate(30,30)">
              <text x="27" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Vertical Edge</text>
              <g fill="#dbeafe" stroke="#3b82f6" stroke-width="1">
                <rect x="0" y="10" width="18" height="18"/><rect x="18" y="10" width="18" height="18"/><rect x="36" y="10" width="18" height="18"/>
                <rect x="0" y="28" width="18" height="18"/><rect x="18" y="28" width="18" height="18"/><rect x="36" y="28" width="18" height="18"/>
                <rect x="0" y="46" width="18" height="18"/><rect x="18" y="46" width="18" height="18"/><rect x="36" y="46" width="18" height="18"/>
              </g>
              <g font-size="8" fill="#1e40af" text-anchor="middle">
                <text x="9" y="23">-1</text><text x="27" y="23">0</text><text x="45" y="23">1</text>
                <text x="9" y="41">-2</text><text x="27" y="41">0</text><text x="45" y="41">2</text>
                <text x="9" y="59">-1</text><text x="27" y="59">0</text><text x="45" y="59">1</text>
              </g>
            </g>

            <!-- Horizontal edge -->
            <g transform="translate(130,30)">
              <text x="27" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Horizontal Edge</text>
              <g fill="#d1fae5" stroke="#10b981" stroke-width="1">
                <rect x="0" y="10" width="18" height="18"/><rect x="18" y="10" width="18" height="18"/><rect x="36" y="10" width="18" height="18"/>
                <rect x="0" y="28" width="18" height="18"/><rect x="18" y="28" width="18" height="18"/><rect x="36" y="28" width="18" height="18"/>
                <rect x="0" y="46" width="18" height="18"/><rect x="18" y="46" width="18" height="18"/><rect x="36" y="46" width="18" height="18"/>
              </g>
              <g font-size="8" fill="#065f46" text-anchor="middle">
                <text x="9" y="23">-1</text><text x="27" y="23">-2</text><text x="45" y="23">-1</text>
                <text x="9" y="41">0</text><text x="27" y="41">0</text><text x="45" y="41">0</text>
                <text x="9" y="59">1</text><text x="27" y="59">2</text><text x="45" y="59">1</text>
              </g>
            </g>

            <!-- Blur -->
            <g transform="translate(230,30)">
              <text x="27" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Blur (Average)</text>
              <g fill="#fef3c7" stroke="#f59e0b" stroke-width="1">
                <rect x="0" y="10" width="18" height="18"/><rect x="18" y="10" width="18" height="18"/><rect x="36" y="10" width="18" height="18"/>
                <rect x="0" y="28" width="18" height="18"/><rect x="18" y="28" width="18" height="18"/><rect x="36" y="28" width="18" height="18"/>
                <rect x="0" y="46" width="18" height="18"/><rect x="18" y="46" width="18" height="18"/><rect x="36" y="46" width="18" height="18"/>
              </g>
              <g font-size="7" fill="#92400e" text-anchor="middle">
                <text x="9" y="22">1/9</text><text x="27" y="22">1/9</text><text x="45" y="22">1/9</text>
                <text x="9" y="40">1/9</text><text x="27" y="40">1/9</text><text x="45" y="40">1/9</text>
                <text x="9" y="58">1/9</text><text x="27" y="58">1/9</text><text x="45" y="58">1/9</text>
              </g>
            </g>

            <!-- Sharpen -->
            <g transform="translate(320,30)">
              <text x="27" y="0" text-anchor="middle" font-size="9" fill="#475569" font-weight="500">Sharpen</text>
              <g fill="#fee2e2" stroke="#ef4444" stroke-width="1">
                <rect x="0" y="10" width="18" height="18"/><rect x="18" y="10" width="18" height="18"/><rect x="36" y="10" width="18" height="18"/>
                <rect x="0" y="28" width="18" height="18"/><rect x="18" y="28" width="18" height="18"/><rect x="36" y="28" width="18" height="18"/>
                <rect x="0" y="46" width="18" height="18"/><rect x="18" y="46" width="18" height="18"/><rect x="36" y="46" width="18" height="18"/>
              </g>
              <g font-size="8" fill="#dc2626" text-anchor="middle">
                <text x="9" y="23">0</text><text x="27" y="23">-1</text><text x="45" y="23">0</text>
                <text x="9" y="41">-1</text><text x="27" y="41">5</text><text x="45" y="41">-1</text>
                <text x="9" y="59">0</text><text x="27" y="59">-1</text><text x="45" y="59">0</text>
              </g>
            </g>

            <!-- Note -->
            <text x="200" y="130" text-anchor="middle" font-size="9" fill="#64748b">In CNNs, filters are learned from data, not hand-designed</text>
          </svg>`,
          caption: 'Classic hand-designed filters. CNNs learn filter values automatically.'
        },
        {
          type: 'heading',
          content: 'Output Size Calculation'
        },
        {
          type: 'formula',
          content: 'output_size = (input_size - filter_size + 2×padding) / stride + 1'
        },
        {
          type: 'text',
          content: `**Stride**: How many pixels to move the filter each step
- Stride 1: Move by 1 pixel → large output
- Stride 2: Move by 2 pixels → output half the size

**Padding**: Add zeros around the input edge
- "Valid" (no padding): Output shrinks
- "Same" padding: Output same size as input`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

x = torch.randn(1, 3, 32, 32)
output = conv(x)
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")`
        },
        {
          type: 'keypoints',
          points: [
            'Convolution slides a filter over the image, computing dot products',
            'Weight sharing dramatically reduces parameters vs fully connected',
            'Filters learn to detect patterns (edges, textures, objects)',
            'Output size depends on input size, filter size, stride, and padding',
            'Early layers detect simple patterns, deeper layers detect complex ones'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why does convolution reduce parameters compared to fully connected layers?',
          options: [
            'It uses smaller filters',
            'Weight sharing - same filter applied everywhere',
            'It removes some pixels',
            'It uses less memory'
          ],
          correct: 1,
          explanation: 'Convolution uses weight sharing - the same small filter (e.g., 3×3 = 9 weights) is applied at every position. A fully connected layer would need separate weights for every input-output connection.'
        },
        {
          type: 'multiple-choice',
          question: 'If input is 32×32 and filter is 5×5 with stride 1 and no padding, what is the output size?',
          options: [
            '32×32',
            '28×28',
            '27×27',
            '30×30'
          ],
          correct: 1,
          explanation: 'output = (input - filter + 2×padding) / stride + 1 = (32 - 5 + 0) / 1 + 1 = 28. The output is 28×28.'
        }
      ]
    },
    {
      id: 'pooling-stride',
      title: 'Pooling and Stride',
      duration: '40 min',
      concepts: ['Max Pooling', 'Average Pooling', 'Downsampling'],
      content: [
        {
          type: 'heading',
          content: 'Why Downsample?'
        },
        {
          type: 'text',
          content: `As we process an image, we want to:
1. Build increasingly abstract representations
2. Reduce spatial dimensions (smaller feature maps = less computation)
3. Achieve translation invariance (detect patterns regardless of exact position)

**Pooling** achieves all three by summarizing regions.`
        },
        {
          type: 'heading',
          content: 'Max Pooling'
        },
        {
          type: 'text',
          content: `For each region, output the maximum value. This keeps the strongest activations - "if this pattern was detected anywhere in this region, keep it."`
        },
        {
          type: 'visualization',
          title: 'Max Pooling 2×2 with Stride 2',
          svg: `<svg viewBox="0 0 400 180" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="180" fill="#f8fafc"/>

            <!-- Input -->
            <g transform="translate(50,40)">
              <text x="36" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Input (4×4)</text>

              <!-- Grid -->
              <g stroke="#94a3b8" stroke-width="1">
                <rect x="0" y="10" width="18" height="18" fill="#fee2e2"/>
                <rect x="18" y="10" width="18" height="18" fill="#fee2e2"/>
                <rect x="36" y="10" width="18" height="18" fill="#d1fae5"/>
                <rect x="54" y="10" width="18" height="18" fill="#d1fae5"/>

                <rect x="0" y="28" width="18" height="18" fill="#fee2e2"/>
                <rect x="18" y="28" width="18" height="18" fill="#fee2e2"/>
                <rect x="36" y="28" width="18" height="18" fill="#d1fae5"/>
                <rect x="54" y="28" width="18" height="18" fill="#d1fae5"/>

                <rect x="0" y="46" width="18" height="18" fill="#dbeafe"/>
                <rect x="18" y="46" width="18" height="18" fill="#dbeafe"/>
                <rect x="36" y="46" width="18" height="18" fill="#fef3c7"/>
                <rect x="54" y="46" width="18" height="18" fill="#fef3c7"/>

                <rect x="0" y="64" width="18" height="18" fill="#dbeafe"/>
                <rect x="18" y="64" width="18" height="18" fill="#dbeafe"/>
                <rect x="36" y="64" width="18" height="18" fill="#fef3c7"/>
                <rect x="54" y="64" width="18" height="18" fill="#fef3c7"/>
              </g>

              <!-- Numbers -->
              <g font-size="8" fill="#475569" text-anchor="middle">
                <text x="9" y="23">1</text><text x="27" y="23" font-weight="bold" fill="#dc2626">5</text><text x="45" y="23">2</text><text x="63" y="23" font-weight="bold" fill="#059669">8</text>
                <text x="9" y="41">3</text><text x="27" y="41">2</text><text x="45" y="41" font-weight="bold" fill="#059669">6</text><text x="63" y="41">1</text>
                <text x="9" y="59">0</text><text x="27" y="59" font-weight="bold" fill="#1e40af">4</text><text x="45" y="59">3</text><text x="63" y="59" font-weight="bold" fill="#92400e">7</text>
                <text x="9" y="77">1</text><text x="27" y="77">2</text><text x="45" y="77">5</text><text x="63" y="77">2</text>
              </g>
            </g>

            <!-- Arrow -->
            <path d="M145,85 L200,85" stroke="#64748b" stroke-width="2" marker-end="url(#poolarrow)"/>
            <text x="172" y="75" text-anchor="middle" font-size="9" fill="#64748b">max</text>
            <text x="172" y="100" text-anchor="middle" font-size="8" fill="#64748b">2×2, stride 2</text>

            <!-- Output -->
            <g transform="translate(220,50)">
              <text x="18" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Output (2×2)</text>

              <g stroke="#94a3b8" stroke-width="1">
                <rect x="0" y="10" width="24" height="24" fill="#fee2e2"/>
                <rect x="24" y="10" width="24" height="24" fill="#d1fae5"/>
                <rect x="0" y="34" width="24" height="24" fill="#dbeafe"/>
                <rect x="24" y="34" width="24" height="24" fill="#fef3c7"/>
              </g>

              <g font-size="11" fill="#475569" text-anchor="middle" font-weight="bold">
                <text x="12" y="28" fill="#dc2626">5</text>
                <text x="36" y="28" fill="#059669">8</text>
                <text x="12" y="52" fill="#1e40af">4</text>
                <text x="36" y="52" fill="#92400e">7</text>
              </g>
            </g>

            <!-- Legend -->
            <text x="200" y="145" text-anchor="middle" font-size="9" fill="#64748b">Max pooling keeps the maximum from each 2×2 region</text>

            <defs>
              <marker id="poolarrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>
              </marker>
            </defs>
          </svg>`,
          caption: 'Max pooling with 2×2 window and stride 2 reduces dimensions by half'
        },
        {
          type: 'heading',
          content: 'Average Pooling'
        },
        {
          type: 'text',
          content: `Instead of max, take the average of each region. Less aggressive than max pooling - keeps more information but less "sharp."

**Global Average Pooling** averages the entire feature map to a single value. Often used before the final classification layer.`
        },
        {
          type: 'heading',
          content: 'Strided Convolution Alternative'
        },
        {
          type: 'text',
          content: `Instead of pooling, use convolution with stride > 1. The convolution learns what to keep while downsampling.

**Modern practice**: Many architectures now prefer strided convolutions over pooling, giving the network more control.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

pool = nn.AvgPool2d(kernel_size=2, stride=2)

gap = nn.AdaptiveAvgPool2d(1)

strided_conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

x = torch.randn(1, 64, 32, 32)
print(f"Max pool: {pool(x).shape}")
print(f"GAP: {gap(x).shape}")
print(f"Strided conv: {strided_conv(x).shape}")`
        },
        {
          type: 'heading',
          content: 'Translation Invariance'
        },
        {
          type: 'text',
          content: `Pooling provides **translation invariance** - the ability to recognize a pattern regardless of its exact position.

If a feature appears anywhere within a pooling region, the output is similar. This makes CNNs robust to small shifts in the input.`
        },
        {
          type: 'keypoints',
          points: [
            'Pooling reduces spatial dimensions while keeping important features',
            'Max pooling keeps the strongest activation in each region',
            'Average pooling keeps the average (smoother)',
            'Global Average Pooling reduces entire feature map to single value',
            'Strided convolutions are an alternative to pooling'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the output size after 2×2 max pooling with stride 2 on a 16×16 input?',
          options: [
            '16×16',
            '14×14',
            '8×8',
            '4×4'
          ],
          correct: 2,
          explanation: '2×2 pooling with stride 2 divides each dimension by 2. So 16×16 becomes 8×8.'
        },
        {
          type: 'multiple-choice',
          question: 'What is Global Average Pooling used for?',
          options: [
            'Downsampling feature maps',
            'Reducing each feature map to a single value',
            'Increasing resolution',
            'Adding more channels'
          ],
          correct: 1,
          explanation: 'Global Average Pooling averages all spatial locations in each feature map, producing a single value per channel. This is often used before the final classifier.'
        }
      ]
    },
    {
      id: 'cnn-architectures',
      title: 'CNN Architecture Design',
      duration: '55 min',
      concepts: ['Layer Stacking', 'Receptive Field', 'Feature Hierarchy'],
      content: [
        {
          type: 'heading',
          content: 'Building Blocks of a CNN'
        },
        {
          type: 'text',
          content: `A typical CNN has:

1. **Convolutional blocks**: Conv → BatchNorm → ReLU (repeated)
2. **Downsampling**: Pooling or strided conv
3. **Fully connected head**: Flatten → Dense → Output

The pattern: spatial dimensions decrease, number of channels increases.`
        },
        {
          type: 'visualization',
          title: 'Typical CNN Architecture',
          svg: `<svg viewBox="0 0 450 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="200" fill="#f8fafc"/>

            <!-- Input -->
            <rect x="20" y="50" width="30" height="80" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" rx="2"/>
            <text x="35" y="145" text-anchor="middle" font-size="8" fill="#64748b">Input</text>
            <text x="35" y="157" text-anchor="middle" font-size="7" fill="#64748b">224×224×3</text>

            <!-- Conv Block 1 -->
            <rect x="65" y="55" width="35" height="70" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="2"/>
            <text x="82" y="145" text-anchor="middle" font-size="8" fill="#3b82f6">Conv1</text>
            <text x="82" y="157" text-anchor="middle" font-size="7" fill="#64748b">112×112×64</text>

            <!-- Conv Block 2 -->
            <rect x="115" y="60" width="35" height="60" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="2"/>
            <text x="132" y="145" text-anchor="middle" font-size="8" fill="#3b82f6">Conv2</text>
            <text x="132" y="157" text-anchor="middle" font-size="7" fill="#64748b">56×56×128</text>

            <!-- Conv Block 3 -->
            <rect x="165" y="65" width="35" height="50" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="2"/>
            <text x="182" y="145" text-anchor="middle" font-size="8" fill="#3b82f6">Conv3</text>
            <text x="182" y="157" text-anchor="middle" font-size="7" fill="#64748b">28×28×256</text>

            <!-- Conv Block 4 -->
            <rect x="215" y="70" width="35" height="40" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="2"/>
            <text x="232" y="145" text-anchor="middle" font-size="8" fill="#3b82f6">Conv4</text>
            <text x="232" y="157" text-anchor="middle" font-size="7" fill="#64748b">14×14×512</text>

            <!-- Conv Block 5 -->
            <rect x="265" y="75" width="35" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="2"/>
            <text x="282" y="145" text-anchor="middle" font-size="8" fill="#3b82f6">Conv5</text>
            <text x="282" y="157" text-anchor="middle" font-size="7" fill="#64748b">7×7×512</text>

            <!-- GAP -->
            <rect x="315" y="83" width="20" height="14" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="2"/>
            <text x="325" y="145" text-anchor="middle" font-size="8" fill="#10b981">GAP</text>
            <text x="325" y="157" text-anchor="middle" font-size="7" fill="#64748b">1×1×512</text>

            <!-- FC -->
            <rect x="350" y="78" width="25" height="24" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5" rx="2"/>
            <text x="362" y="145" text-anchor="middle" font-size="8" fill="#f59e0b">FC</text>
            <text x="362" y="157" text-anchor="middle" font-size="7" fill="#64748b">1000</text>

            <!-- Output -->
            <rect x="390" y="82" width="20" height="16" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" rx="2"/>
            <text x="400" y="145" text-anchor="middle" font-size="8" fill="#ef4444">Out</text>

            <!-- Arrows -->
            <g stroke="#94a3b8" stroke-width="1">
              <line x1="50" y1="90" x2="65" y2="90"/>
              <line x1="100" y1="90" x2="115" y2="90"/>
              <line x1="150" y1="90" x2="165" y2="90"/>
              <line x1="200" y1="90" x2="215" y2="90"/>
              <line x1="250" y1="90" x2="265" y2="90"/>
              <line x1="300" y1="90" x2="315" y2="90"/>
              <line x1="335" y1="90" x2="350" y2="90"/>
              <line x1="375" y1="90" x2="390" y2="90"/>
            </g>

            <!-- Annotations -->
            <text x="175" y="25" text-anchor="middle" font-size="9" fill="#475569">↓ Spatial size decreases</text>
            <text x="175" y="185" text-anchor="middle" font-size="9" fill="#475569">↑ Channels increase</text>
          </svg>`,
          caption: 'Typical CNN: spatial dimensions decrease, channels increase'
        },
        {
          type: 'heading',
          content: 'Receptive Field'
        },
        {
          type: 'text',
          content: `The **receptive field** is the region of the input that influences a neuron's output.

- Early layers: Small receptive field (local patterns)
- Deep layers: Large receptive field (global context)

Stacking 3×3 convolutions is more efficient than large filters:
- Two 3×3 = 5×5 receptive field
- Three 3×3 = 7×7 receptive field
- But with fewer parameters and more non-linearity!`
        },
        {
          type: 'heading',
          content: 'The 1×1 Convolution'
        },
        {
          type: 'text',
          content: `A 1×1 convolution doesn't look at neighbors - it's point-wise. Uses:

1. **Change channel count**: 256 → 64 or 64 → 256
2. **Add non-linearity**: Insert ReLU between operations
3. **Reduce computation**: Compress channels before expensive 3×3 convs

The "bottleneck" design uses 1×1 to reduce, 3×3 to process, 1×1 to expand.`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x`
        },
        {
          type: 'heading',
          content: 'Design Guidelines'
        },
        {
          type: 'table',
          headers: ['Aspect', 'Common Choice', 'Notes'],
          rows: [
            ['Filter size', '3×3', 'Stack for larger receptive field'],
            ['Stride', '1 (conv), 2 (downsample)', 'Strided conv or pooling'],
            ['Channels', 'Double after downsample', '64→128→256→512'],
            ['BatchNorm', 'After every conv', 'Essential for training'],
            ['Activation', 'ReLU', 'After BN']
          ],
          caption: 'CNN design rules of thumb'
        },
        {
          type: 'keypoints',
          points: [
            'CNNs follow: Conv blocks → Downsample → Classifier',
            'Spatial size decreases, channels increase through the network',
            'Receptive field grows with depth - deep layers see more context',
            '1×1 convolutions change channel count efficiently',
            'Stack 3×3 convs instead of using large filters'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'Why stack multiple 3×3 convolutions instead of one large filter?',
          options: [
            'Larger filters are not supported',
            'More parameters and non-linearity with same receptive field',
            'Fewer parameters and more non-linearity with same receptive field',
            'It is faster on GPUs'
          ],
          correct: 2,
          explanation: 'Two 3×3 convs have 2×9=18 params for a 5×5 receptive field. A single 5×5 has 25 params. Stacking also adds extra non-linearity (ReLU between layers).'
        }
      ]
    },
    {
      id: 'classic-architectures',
      title: 'Classic CNN Architectures',
      duration: '50 min',
      concepts: ['LeNet', 'VGG', 'GoogLeNet'],
      content: [
        {
          type: 'heading',
          content: 'LeNet-5 (1998)'
        },
        {
          type: 'text',
          content: `The original CNN by Yann LeCun for digit recognition. Simple but groundbreaking:

- Input: 32×32 grayscale
- Conv → Pool → Conv → Pool → FC → FC → Output
- ~60K parameters

Established the fundamental CNN pattern we still use today.`
        },
        {
          type: 'heading',
          content: 'AlexNet (2012)'
        },
        {
          type: 'text',
          content: `Won ImageNet 2012 by a huge margin, kickstarting the deep learning revolution.

**Innovations**:
- ReLU activation (faster than sigmoid)
- Dropout for regularization
- Data augmentation
- GPU training

8 layers, 60M parameters. Showed that bigger, deeper networks with more data beat hand-crafted features.`
        },
        {
          type: 'heading',
          content: 'VGGNet (2014)'
        },
        {
          type: 'text',
          content: `Showed that depth matters. Very uniform architecture:

**Key principle**: Only 3×3 convs, double channels after each pooling.

VGG-16: 13 conv + 3 FC = 16 layers, 138M parameters
VGG-19: 16 conv + 3 FC = 19 layers

Simple but effective. Still used as a feature extractor today.`
        },
        {
          type: 'visualization',
          title: 'VGG-16 Architecture',
          svg: `<svg viewBox="0 0 450 120" xmlns="http://www.w3.org/2000/svg">
            <rect width="450" height="120" fill="#f8fafc"/>

            <!-- Blocks representation -->
            <g transform="translate(20,25)">
              <!-- Block 1: 2 convs -->
              <rect x="0" y="20" width="25" height="50" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="28" y="20" width="25" height="50" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <text x="26" y="80" text-anchor="middle" font-size="7" fill="#64748b">64</text>

              <!-- Pool -->
              <rect x="56" y="30" width="8" height="30" fill="#ef4444" rx="1"/>

              <!-- Block 2: 2 convs -->
              <rect x="68" y="25" width="22" height="40" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="93" y="25" width="22" height="40" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <text x="91" y="80" text-anchor="middle" font-size="7" fill="#64748b">128</text>

              <!-- Pool -->
              <rect x="118" y="32" width="8" height="26" fill="#ef4444" rx="1"/>

              <!-- Block 3: 3 convs -->
              <rect x="130" y="30" width="18" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="151" y="30" width="18" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="172" y="30" width="18" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <text x="160" y="80" text-anchor="middle" font-size="7" fill="#64748b">256</text>

              <!-- Pool -->
              <rect x="193" y="35" width="8" height="20" fill="#ef4444" rx="1"/>

              <!-- Block 4: 3 convs -->
              <rect x="205" y="33" width="15" height="24" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="223" y="33" width="15" height="24" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="241" y="33" width="15" height="24" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <text x="232" y="80" text-anchor="middle" font-size="7" fill="#64748b">512</text>

              <!-- Pool -->
              <rect x="259" y="38" width="8" height="14" fill="#ef4444" rx="1"/>

              <!-- Block 5: 3 convs -->
              <rect x="271" y="36" width="12" height="18" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="286" y="36" width="12" height="18" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <rect x="301" y="36" width="12" height="18" fill="#dbeafe" stroke="#3b82f6" stroke-width="1" rx="2"/>
              <text x="293" y="80" text-anchor="middle" font-size="7" fill="#64748b">512</text>

              <!-- Pool -->
              <rect x="316" y="40" width="8" height="10" fill="#ef4444" rx="1"/>

              <!-- FC layers -->
              <rect x="330" y="38" width="20" height="14" fill="#fef3c7" stroke="#f59e0b" stroke-width="1" rx="2"/>
              <rect x="355" y="38" width="20" height="14" fill="#fef3c7" stroke="#f59e0b" stroke-width="1" rx="2"/>
              <rect x="380" y="40" width="15" height="10" fill="#fee2e2" stroke="#ef4444" stroke-width="1" rx="2"/>

              <text x="360" y="80" text-anchor="middle" font-size="7" fill="#64748b">FC 4096</text>
              <text x="387" y="80" text-anchor="middle" font-size="7" fill="#64748b">1000</text>
            </g>

            <!-- Legend -->
            <g transform="translate(100,100)">
              <rect x="0" y="0" width="10" height="10" fill="#dbeafe" stroke="#3b82f6" rx="1"/>
              <text x="15" y="9" font-size="7" fill="#64748b">3×3 Conv + ReLU</text>

              <rect x="90" y="0" width="10" height="10" fill="#ef4444" rx="1"/>
              <text x="105" y="9" font-size="7" fill="#64748b">MaxPool</text>

              <rect x="160" y="0" width="10" height="10" fill="#fef3c7" stroke="#f59e0b" rx="1"/>
              <text x="175" y="9" font-size="7" fill="#64748b">FC + ReLU</text>
            </g>
          </svg>`,
          caption: 'VGG-16: uniform 3×3 convs, channels double after pooling'
        },
        {
          type: 'heading',
          content: 'GoogLeNet/Inception (2014)'
        },
        {
          type: 'text',
          content: `Introduced the **Inception module**: multiple parallel paths with different filter sizes, concatenated together.

**Key insight**: Let the network choose what scale of features to use at each location by running 1×1, 3×3, and 5×5 convolutions in parallel.

Only 5M parameters (vs VGG's 138M) but similar accuracy!`
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch.nn as nn
import torchvision.models as models

vgg16 = models.vgg16(pretrained=True)

googlenet = models.googlenet(pretrained=True)

alexnet = models.alexnet(pretrained=True)

for param in vgg16.features.parameters():
    param.requires_grad = False

vgg16.classifier[-1] = nn.Linear(4096, num_classes)`
        },
        {
          type: 'keypoints',
          points: [
            'LeNet established the basic CNN pattern',
            'AlexNet proved deep learning works on large-scale vision',
            'VGG showed depth and uniform architecture matters',
            'Inception/GoogLeNet introduced multi-scale parallel processing',
            'These models are available pre-trained in all major frameworks'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What was the key innovation of VGGNet?',
          options: [
            'Using very large filters',
            'Using only 3×3 filters with increased depth',
            'Using 1×1 convolutions',
            'Using skip connections'
          ],
          correct: 1,
          explanation: 'VGG showed that using only 3×3 convolutions but stacking many layers (16-19) achieves excellent results. This uniform, deep architecture became a template for future CNNs.'
        }
      ]
    },
    {
      id: 'resnet',
      title: 'ResNet and Skip Connections',
      duration: '55 min',
      concepts: ['Residual Learning', 'Skip Connections', 'Deep Networks'],
      content: [
        {
          type: 'heading',
          content: 'The Degradation Problem'
        },
        {
          type: 'text',
          content: `As networks get deeper, accuracy gets saturated then degrades rapidly. Surprisingly, this isn't just overfitting - even training error gets worse!

**The problem**: Optimizing very deep networks is hard. Gradients vanish, and it's difficult to learn identity mappings when needed.`
        },
        {
          type: 'heading',
          content: 'The Residual Learning Solution'
        },
        {
          type: 'text',
          content: `Instead of learning H(x), learn F(x) = H(x) - x, where H(x) = F(x) + x.

**Key insight**: If the optimal mapping is close to identity, it's easier to push F(x) to zero than to learn H(x) = x with a stack of non-linear layers.`
        },
        {
          type: 'visualization',
          title: 'Residual Block',
          svg: `<svg viewBox="0 0 400 170" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="170" fill="#f8fafc"/>

            <!-- Input -->
            <rect x="30" y="65" width="40" height="30" fill="#e2e8f0" stroke="#64748b" stroke-width="1.5" rx="4"/>
            <text x="50" y="84" text-anchor="middle" font-size="10" fill="#475569">x</text>

            <!-- Split point -->
            <line x1="70" y1="80" x2="100" y2="80" stroke="#64748b" stroke-width="1.5"/>
            <circle cx="100" cy="80" r="3" fill="#64748b"/>

            <!-- Main path (F(x)) -->
            <line x1="100" y1="80" x2="100" y2="35" stroke="#64748b" stroke-width="1.5"/>
            <line x1="100" y1="35" x2="130" y2="35" stroke="#64748b" stroke-width="1.5"/>

            <rect x="130" y="20" width="50" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="4"/>
            <text x="155" y="40" text-anchor="middle" font-size="9" fill="#1e40af">Conv+BN</text>

            <line x1="180" y1="35" x2="200" y2="35" stroke="#64748b" stroke-width="1.5"/>

            <rect x="200" y="20" width="35" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="4"/>
            <text x="217" y="40" text-anchor="middle" font-size="9" fill="#065f46">ReLU</text>

            <line x1="235" y1="35" x2="255" y2="35" stroke="#64748b" stroke-width="1.5"/>

            <rect x="255" y="20" width="50" height="30" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="4"/>
            <text x="280" y="40" text-anchor="middle" font-size="9" fill="#1e40af">Conv+BN</text>

            <line x1="305" y1="35" x2="320" y2="35" stroke="#64748b" stroke-width="1.5"/>
            <line x1="320" y1="35" x2="320" y2="80" stroke="#64748b" stroke-width="1.5"/>

            <!-- Skip connection -->
            <line x1="100" y1="80" x2="100" y2="125" stroke="#10b981" stroke-width="2.5"/>
            <line x1="100" y1="125" x2="320" y2="125" stroke="#10b981" stroke-width="2.5"/>
            <line x1="320" y1="125" x2="320" y2="80" stroke="#10b981" stroke-width="2.5"/>
            <text x="210" y="145" text-anchor="middle" font-size="9" fill="#10b981" font-weight="500">Identity shortcut (skip)</text>

            <!-- Add -->
            <circle cx="320" cy="80" r="12" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
            <text x="320" y="85" text-anchor="middle" font-size="14" fill="#92400e">+</text>

            <!-- After add -->
            <line x1="332" y1="80" x2="345" y2="80" stroke="#64748b" stroke-width="1.5"/>

            <rect x="345" y="65" width="30" height="30" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="4"/>
            <text x="360" y="84" text-anchor="middle" font-size="9" fill="#065f46">ReLU</text>

            <!-- F(x) label -->
            <text x="217" y="12" text-anchor="middle" font-size="9" fill="#64748b">F(x) = Weight layers</text>
          </svg>`,
          caption: 'Residual block: output = F(x) + x, where F(x) is the learned residual'
        },
        {
          type: 'heading',
          content: 'Why Skip Connections Work'
        },
        {
          type: 'text',
          content: `**1. Easy to learn identity**: If identity is optimal, just push F(x) → 0

**2. Better gradient flow**: Gradients can flow directly through the skip connection

**3. Implicit ensemble**: ResNet can be viewed as an ensemble of shallower networks

**4. Feature reuse**: Information from earlier layers is preserved`
        },
        {
          type: 'heading',
          content: 'ResNet Variants'
        },
        {
          type: 'text',
          content: `**ResNet-18, 34**: Basic blocks (two 3×3 convs)
**ResNet-50, 101, 152**: Bottleneck blocks (1×1 → 3×3 → 1×1)

Bottleneck design reduces parameters: squeeze channels with 1×1, process with 3×3, expand with 1×1.`
        },
        {
          type: 'visualization',
          title: 'Basic vs Bottleneck Block',
          svg: `<svg viewBox="0 0 400 150" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="150" fill="#f8fafc"/>

            <!-- Basic Block -->
            <g transform="translate(50,25)">
              <text x="45" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Basic Block</text>

              <rect x="20" y="15" width="50" height="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="45" y="32" text-anchor="middle" font-size="8" fill="#1e40af">3×3, 64</text>

              <rect x="20" y="50" width="50" height="25" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="45" y="67" text-anchor="middle" font-size="8" fill="#1e40af">3×3, 64</text>

              <!-- Skip -->
              <line x1="5" y1="40" x2="5" y2="80" stroke="#10b981" stroke-width="2"/>
              <line x1="5" y1="80" x2="20" y2="80" stroke="#10b981" stroke-width="2"/>

              <text x="45" y="105" text-anchor="middle" font-size="8" fill="#64748b">~70K params</text>
            </g>

            <!-- Bottleneck Block -->
            <g transform="translate(220,25)">
              <text x="50" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Bottleneck Block</text>

              <rect x="25" y="10" width="50" height="22" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="50" y="25" text-anchor="middle" font-size="8" fill="#1e40af">1×1, 64</text>

              <rect x="25" y="37" width="50" height="22" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="50" y="52" text-anchor="middle" font-size="8" fill="#1e40af">3×3, 64</text>

              <rect x="25" y="64" width="50" height="22" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="3"/>
              <text x="50" y="79" text-anchor="middle" font-size="8" fill="#1e40af">1×1, 256</text>

              <!-- Skip -->
              <line x1="10" y1="35" x2="10" y2="85" stroke="#10b981" stroke-width="2"/>
              <line x1="10" y1="85" x2="25" y2="85" stroke="#10b981" stroke-width="2"/>

              <text x="50" y="105" text-anchor="middle" font-size="8" fill="#64748b">~70K params</text>
              <text x="50" y="118" text-anchor="middle" font-size="7" fill="#64748b">(but 4× output channels)</text>
            </g>
          </svg>`,
          caption: 'Bottleneck blocks achieve similar computation with more output channels'
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch
import torch.nn as nn
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, num_classes)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out`
        },
        {
          type: 'keypoints',
          points: [
            'Skip connections solve the degradation problem in very deep networks',
            'Learning residual F(x) = H(x) - x is easier than learning H(x) directly',
            'Gradients can flow directly through skip connections',
            'ResNet enabled training of 100+ layer networks',
            'Bottleneck blocks (1×1→3×3→1×1) are efficient for deeper variants'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'What is the main benefit of residual connections?',
          options: [
            'Reduce the number of parameters',
            'Enable training of very deep networks',
            'Speed up inference',
            'Reduce memory usage'
          ],
          correct: 1,
          explanation: 'Residual connections solve the degradation problem where very deep networks perform worse than shallower ones. They allow gradients to flow directly through the network and make it easier to learn identity mappings.'
        }
      ]
    },
    {
      id: 'transfer-learning-cnn',
      title: 'Transfer Learning',
      duration: '50 min',
      concepts: ['Feature Extraction', 'Fine-tuning', 'Domain Adaptation'],
      content: [
        {
          type: 'heading',
          content: 'Why Transfer Learning?'
        },
        {
          type: 'text',
          content: `Training a CNN from scratch needs:
- Lots of labeled data (millions of images)
- Lots of compute (days on GPUs)
- Lots of expertise (hyperparameter tuning)

**Transfer learning** uses a pre-trained model as a starting point. Models trained on ImageNet (14M images, 1000 classes) have learned general visual features that transfer to new tasks.`
        },
        {
          type: 'heading',
          content: 'Two Approaches'
        },
        {
          type: 'text',
          content: `**1. Feature Extraction**
- Freeze pre-trained layers
- Only train new classifier head
- Fast, works with small datasets

**2. Fine-tuning**
- Start from pre-trained weights
- Train entire network (or later layers)
- Better performance, needs more data`
        },
        {
          type: 'visualization',
          title: 'Transfer Learning Approaches',
          svg: `<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="200" fill="#f8fafc"/>

            <!-- Feature Extraction -->
            <g transform="translate(30,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Feature Extraction</text>

              <!-- Frozen layers -->
              <rect x="0" y="15" width="100" height="70" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" rx="4"/>
              <text x="50" y="45" text-anchor="middle" font-size="9" fill="#1e40af">Pre-trained</text>
              <text x="50" y="58" text-anchor="middle" font-size="9" fill="#1e40af">CNN</text>
              <text x="50" y="75" text-anchor="middle" font-size="8" fill="#3b82f6">🔒 FROZEN</text>

              <!-- New head -->
              <rect x="110" y="30" width="40" height="40" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="4"/>
              <text x="130" y="52" text-anchor="middle" font-size="8" fill="#065f46">New</text>
              <text x="130" y="63" text-anchor="middle" font-size="8" fill="#065f46">Head</text>

              <line x1="100" y1="50" x2="110" y2="50" stroke="#64748b" stroke-width="1.5"/>

              <text x="75" y="100" text-anchor="middle" font-size="8" fill="#64748b">Train only classifier</text>
            </g>

            <!-- Fine-tuning -->
            <g transform="translate(220,30)">
              <text x="70" y="0" text-anchor="middle" font-size="10" fill="#475569" font-weight="500">Fine-tuning</text>

              <!-- Trainable layers -->
              <rect x="0" y="15" width="100" height="70" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5" rx="4"/>
              <text x="50" y="45" text-anchor="middle" font-size="9" fill="#92400e">Pre-trained</text>
              <text x="50" y="58" text-anchor="middle" font-size="9" fill="#92400e">CNN</text>
              <text x="50" y="75" text-anchor="middle" font-size="8" fill="#f59e0b">🔓 TRAINABLE</text>

              <!-- New head -->
              <rect x="110" y="30" width="40" height="40" fill="#d1fae5" stroke="#10b981" stroke-width="1.5" rx="4"/>
              <text x="130" y="52" text-anchor="middle" font-size="8" fill="#065f46">New</text>
              <text x="130" y="63" text-anchor="middle" font-size="8" fill="#065f46">Head</text>

              <line x1="100" y1="50" x2="110" y2="50" stroke="#64748b" stroke-width="1.5"/>

              <text x="75" y="100" text-anchor="middle" font-size="8" fill="#64748b">Train all (lower LR for CNN)</text>
            </g>

            <!-- When to use -->
            <g transform="translate(30, 145)">
              <text x="0" y="0" font-size="8" fill="#3b82f6">• Small dataset</text>
              <text x="0" y="12" font-size="8" fill="#3b82f6">• Similar to ImageNet</text>
              <text x="0" y="24" font-size="8" fill="#3b82f6">• Limited compute</text>
            </g>

            <g transform="translate(220, 145)">
              <text x="0" y="0" font-size="8" fill="#f59e0b">• More data available</text>
              <text x="0" y="12" font-size="8" fill="#f59e0b">• Domain differs from ImageNet</text>
              <text x="0" y="24" font-size="8" fill="#f59e0b">• Best performance needed</text>
            </g>
          </svg>`,
          caption: 'Feature extraction freezes CNN, fine-tuning trains everything'
        },
        {
          type: 'heading',
          content: 'Implementation'
        },
        {
          type: 'code',
          language: 'python',
          content: `import torch
import torch.nn as nn
import torchvision.models as models

def feature_extractor(num_classes):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

def fine_tune(num_classes, freeze_until='layer3'):
    model = models.resnet50(pretrained=True)

    freeze = True
    for name, child in model.named_children():
        if name == freeze_until:
            freeze = False
        for param in child.parameters():
            param.requires_grad = not freeze

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

feature_params = list(model.fc.parameters())
backbone_params = [p for p in model.parameters() if p not in feature_params]

optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': feature_params, 'lr': 1e-3}
])`
        },
        {
          type: 'heading',
          content: 'When to Use What'
        },
        {
          type: 'table',
          headers: ['Dataset Size', 'Similarity to ImageNet', 'Approach'],
          rows: [
            ['Small', 'Similar', 'Feature extraction'],
            ['Small', 'Different', 'Feature extraction + augmentation'],
            ['Large', 'Similar', 'Fine-tune later layers'],
            ['Large', 'Different', 'Fine-tune entire network']
          ],
          caption: 'Transfer learning strategy selection'
        },
        {
          type: 'callout',
          variant: 'tip',
          content: 'When fine-tuning, use a smaller learning rate for pre-trained layers (10-100x smaller) than for the new head. This preserves learned features while adapting to the new task.'
        },
        {
          type: 'keypoints',
          points: [
            'Transfer learning uses pre-trained models as starting points',
            'Feature extraction freezes the CNN, only trains the classifier',
            'Fine-tuning trains the entire network with lower learning rates',
            'Pre-trained ImageNet features transfer well to many vision tasks',
            'Use differential learning rates: smaller for pre-trained, larger for new layers'
          ]
        }
      ],
      quiz: [
        {
          type: 'multiple-choice',
          question: 'When should you use feature extraction over fine-tuning?',
          options: [
            'When you have lots of data',
            'When you have very little data',
            'When your task is very different from ImageNet',
            'When you have unlimited compute'
          ],
          correct: 1,
          explanation: 'Feature extraction works well with small datasets because you are only training a small classifier head. Fine-tuning the entire network with little data risks overfitting.'
        }
      ]
    }
  ]
}
