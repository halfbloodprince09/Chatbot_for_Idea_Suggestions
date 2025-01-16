import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the saved model and tokenizer
MODEL_PATH = "./app_ideas_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set the model to evaluation mode
model.eval()

# Function to make predictions
def predict_idea(idea, explanation):
    # Combine idea and explanation
    combined_text = f"{idea} {explanation}"
    
    # Tokenize input
    inputs = tokenizer(
        combined_text,
        truncation=True,      # Truncate input if it's too long
        padding=True,         # Pad input
        max_length=128,       # Match training token length
        return_tensors="pt"   # Return PyTorch tensors
    )
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()  # Get the predicted label
    
    return predictions

# Test the function with an example
idea = "A mobile app to monitor sleep patterns"
explanation = "This app would use accelerometer data to track movement during sleep and provide insights for improving sleep quality."
predicted_label = predict_idea(idea, explanation)

# Map the label to its meaning (assuming 0 = Negative, 1 = Positive)
label_map = {0: "Negative", 1: "Positive"}
print(f"Predicted label: {label_map[predicted_label]}")

# Batch testing function
def batch_predict(test_data):
    predictions = []
    for item in test_data:
        idea = item["idea"]
        explanation = item["explanation"]
        pred_label = predict_idea(idea, explanation)
        predictions.append({
            "idea": idea,
            "explanation": explanation,
            "predicted_label": label_map[pred_label]
        })
    return predictions

# Example test data
test_data = [
    {"idea": "A fitness tracker app", "explanation": "Tracks daily steps and calories burned."},
    {"idea": "An AI chatbot", "explanation": "Helps answer common questions for customer support."},
]

# Get predictions for the batch
batch_results = batch_predict(test_data)

# Print results
for result in batch_results:
    print(f"Idea: {result['idea']}\nExplanation: {result['explanation']}\nPredicted Label: {result['predicted_label']}\n")
