
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
if __name__ == "__main__":
    print(predict_sentiment("This movie was surprisingly good!"))  # Output: Positive
    print(predict_sentiment("I wasted two hours of my life."))     # Output: Negative
