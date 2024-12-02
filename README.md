# Arabic Text Sentiment Analysis

This Jupyter notebook implements a sentiment analysis model for Arabic text using deep learning techniques. The model is designed to classify text into multiple sentiment categories.

## Dataset
- The dataset contains Arabic tweets with sentiment labels
- Total number of samples: 9,694
- Features: tweet (text) and label (sentiment category)
- Labels are encoded as numbers (0-3)

## Implementation Details

### Dependencies
- sentence_transformers
- tensorflow
- scikit-learn
- imbalanced-learn (imblearn)
- numpy
- pandas

### Model Architecture
The implementation uses a deep learning approach with the following components:
1. Text preprocessing and tokenization
2. SMOTE for handling class imbalance
3. Neural network with multiple layers for classification
4. Learning rate scheduling for optimization

### Training Process
- The model is trained for 200 epochs
- Uses a learning rate of 1e-4 initially, with rate adjustment during training
- Implements validation split to monitor model performance
- Achieves approximately:
  - Training accuracy: ~96%
  - Validation accuracy: ~85-87%

## Performance
The model shows good performance metrics with:
- High training accuracy (>95%)
- Good validation accuracy (>85%)
- Handles class imbalance effectively using SMOTE
- Shows consistent improvement in both accuracy and loss during training

## Usage
The notebook can be run in Google Colab or any Jupyter environment with the required dependencies installed.