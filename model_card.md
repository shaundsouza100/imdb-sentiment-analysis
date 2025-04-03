
# Model Card: IMDb Sentiment Classifier

## Model Details
This model is a fine-tuned version of `distilbert-base-uncased` on the IMDb dataset using Hugging Face AutoTrain. It classifies movie reviews into positive or negative sentiment.

## Training Data
- Dataset: IMDb (50,000 movie reviews)
- Format: Binary classification (0 = Negative, 1 = Positive)
- Split: 80% training, 20% validation

## Training Configuration
- Platform: Hugging Face AutoTrain
- Epochs: 3
- Batch Size: 8
- Learning Rate: 5e-5

## Evaluation Metrics
- Accuracy: ~93%
- F1 Score: ~92%

## Intended Use
This model is intended for educational and research purposes to demonstrate text classification. It is not intended for production use without further testing.

## Limitations
- May struggle with sarcasm or ambiguous sentiment.
- Trained only on movie reviews in English.

## Ethical Considerations
- No demographic-specific data or filtering applied.
- May inherit biases present in IMDb reviews.

## Authors
[Your Name] – April 2025
