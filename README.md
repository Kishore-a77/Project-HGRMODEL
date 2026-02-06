# Hand Gesture Recognition Model (HGRMODEL)

A Convolutional Neural Network (CNN) implementation for recognizing hand-written digits using the MNIST dataset, built with TensorFlow and Keras.

## 📋 Overview

This project implements a deep learning model capable of recognizing and classifying hand-written digits (0-9). Using a Convolutional Neural Network architecture, the model achieves high accuracy in digit recognition, which can be applied to various applications including handwriting recognition, automated form processing, and gesture-based interfaces.

## ✨ Features

- **CNN Architecture**: Multi-layer convolutional neural network for robust feature extraction
- **MNIST Dataset**: Training and testing on 70,000 hand-written digit images
- **High Accuracy**: Optimized model achieving excellent classification performance
- **Custom Callbacks**: Clean training output with step-by-step progress tracking
- **Efficient Training**: Fast convergence with minimal epochs
- **TensorFlow/Keras**: Built using industry-standard deep learning frameworks

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed along with the following libraries:

- TensorFlow (2.x or higher)
- NumPy
- Keras (included with TensorFlow)

### Installation

Install required packages using pip:

```bash
pip install tensorflow numpy
```

### Dataset

The project uses the **MNIST dataset** (`mnist.npz`), which contains:

| Component | Description | Shape |
|-----------|-------------|-------|
| Training Images | 60,000 hand-written digit images | (60000, 28, 28) |
| Training Labels | Corresponding digit labels (0-9) | (60000,) |
| Test Images | 10,000 test images | (10000, 28, 28) |
| Test Labels | Corresponding test labels | (10000,) |

Each image is a 28×28 pixel grayscale representation of a hand-written digit.

## 📁 Project Structure

```
.
├── Prodigy_4.py        # Main Python script with CNN implementation
├── mnist.npz           # MNIST dataset file
└── README.md           # Project documentation
```

## 🔧 Usage

1. **Ensure the dataset** `mnist.npz` is available in your local directory
2. **Update the filepath** in the script to match your dataset location:
   ```python
   path = r'path/to/your/mnist.npz'
   ```
3. **Run the script**:
   ```bash
   python Prodigy_4.py
   ```
4. **Monitor training progress** and final test accuracy

### Expected Output

```
Step 1 completed
Step 2 completed
Test accuracy: 0.98XX
```

## 🧠 Model Architecture

The CNN model consists of the following layers:

```
Input Layer (28, 28, 1)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
Flatten
    ↓
Dense (64 units, ReLU)
    ↓
Dense (10 units, Softmax)
    ↓
Output (10 classes)
```

### Layer Details

| Layer Type | Parameters | Purpose |
|------------|------------|---------|
| **Conv2D (32)** | 3×3 kernel, ReLU | Extract basic features (edges, corners) |
| **MaxPooling2D** | 2×2 pool | Downsample and reduce spatial dimensions |
| **Conv2D (64)** | 3×3 kernel, ReLU | Extract mid-level features |
| **MaxPooling2D** | 2×2 pool | Further dimensionality reduction |
| **Conv2D (64)** | 3×3 kernel, ReLU | Extract high-level features |
| **Flatten** | - | Convert 2D features to 1D vector |
| **Dense (64)** | ReLU activation | Fully connected layer for learning |
| **Dense (10)** | Softmax activation | Output probabilities for 10 classes |

## 📊 Training Configuration

- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 2 (configurable)
- **Batch Size**: Default (32)
- **Validation**: Test set used for validation during training

## 📈 Data Preprocessing

1. **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
   ```python
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```

2. **Reshaping**: Images reshaped to include channel dimension
   ```python
   x_train = x_train.reshape((-1, 28, 28, 1))
   ```

3. **Label Format**: Using sparse categorical format (integer labels 0-9)

## 🎯 Performance

The model typically achieves:

- **Test Accuracy**: ~98-99% after just 2 epochs
- **Training Time**: Fast convergence due to efficient architecture
- **Inference Speed**: Real-time prediction capability

## 🔮 Future Enhancements

Potential improvements for the project:

### Model Improvements
- **Data Augmentation**: Rotation, scaling, shifting for better generalization
- **Batch Normalization**: Add batch norm layers for faster convergence
- **Dropout Layers**: Include dropout to prevent overfitting
- **Deeper Architecture**: Experiment with more convolutional layers
- **Transfer Learning**: Use pre-trained models like ResNet or VGG

### Functionality Extensions
- **Real-time Recognition**: Implement webcam-based digit recognition
- **Model Deployment**: Create REST API for inference
- **Mobile App**: Deploy model on mobile devices using TensorFlow Lite
- **Extended Datasets**: Train on custom hand gesture datasets beyond digits
- **Confusion Matrix**: Visualize prediction errors and misclassifications
- **Model Saving**: Save trained model for future use without retraining

### Code Enhancements
- **Hyperparameter Tuning**: Implement grid search or random search
- **Early Stopping**: Add callbacks to prevent overfitting
- **Learning Rate Scheduling**: Dynamic learning rate adjustment
- **Model Checkpointing**: Save best model during training

## 🛠️ Technologies Used

- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural networks API
- **NumPy**: Numerical computing library
- **Python**: Core programming language

## 📝 Code Highlights

### Custom Callback for Clean Output

```python
class QuietCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Step {epoch + 1} completed')
```

This custom callback provides clean, step-by-step progress updates during training without verbose output.

### Environment Configuration

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

Suppresses TensorFlow informational messages for cleaner console output.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open-source and available under the MIT License.

## 📧 Contact

For questions, suggestions, or feedback, please open an issue in the repository.

## 🎓 Acknowledgments

This project demonstrates:
- Convolutional Neural Network architecture for image classification
- Deep learning best practices with TensorFlow/Keras
- MNIST dataset processing and model training
- Real-world application of computer vision techniques

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

---

**Note**: This is an educational project showcasing deep learning and computer vision techniques. The architecture and methodology can be extended for more complex hand gesture recognition systems and real-world applications.
