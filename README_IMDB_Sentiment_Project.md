
# IMDb Sentiment Analysis using DistilBERT

This project fine-tunes the `distilbert-base-uncased` model on the IMDb movie reviews dataset to classify sentiment as Positive or Negative.

## 🚀 Project Overview

- **Task**: Binary Sentiment Classification (Positive / Negative)
- **Model**: DistilBERT (`distilbert-base-uncased`)
- **Dataset**: IMDb (from Hugging Face Hub)
- **Platform**: Hugging Face AutoTrain
- **Accuracy**: ~93% on the validation set

## 📊 Dataset Details

- 50,000 reviews total
- Balanced between positive and negative sentiment
- Clean and pre-tokenized using Hugging Face AutoTrain

## 🧪 Inference Pipeline

You can use the following Python snippet to make predictions with your trained model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your trained model from Hugging Face
model_name = "username/imdb-sentiment-model"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment

# Example usage
print(predict_sentiment("This movie was surprisingly good!"))  # Positive
print(predict_sentiment("I wasted two hours of my life."))     # Negative
```

## 📈 Results

- **Accuracy**: ~93%
- **F1 Score**: ~92%
- Evaluated on test split of IMDb

## 📎 Future Work

- Tune hyperparameters with Optuna
- Add multi-class emotion detection
- Improve robustness on sarcastic or ambiguous reviews

## 📄 License

MIT License.
