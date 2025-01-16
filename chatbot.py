import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# Path to the locally stored model
MODEL_PATH = r"C:\Users\AJAY\Desktop\Chatbot_for_Idea_Suggestions-main\AppIdeasModel\app_ideas_model"

# Load the model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Function to generate random prompts for diversity
def get_random_prompt():
    prompts = [
        "Generate creative app ideas.",
        "What are some unique app ideas?",
        "Suggest innovative app concepts.",
        "List a few interesting app suggestions.",
        "Create some unique app concepts.",
    ]
    return random.choice(prompts)

# Expanded idea pool for mapping
idea_mapping = {
    0: "A meditation app with gamified rewards.",
    1: "A grocery delivery app for people with specific dietary needs.",
    2: "A subscription service for AI-generated art.",
    3: "A fitness app with personalized meal plans.",
    4: "An educational app for learning new languages with gamification.",
    5: "A task management app with AI-driven reminders.",
    6: "A mental health app with daily mood tracking.",
    7: "An e-commerce platform for eco-friendly products.",
    8: "A food waste reduction app for donating excess food.",
    9: "A social networking app for niche hobbies.",
}

# Function to generate app ideas using the local model
def generate_app_ideas():
    # Use a random prompt to introduce variability
    prompt = get_random_prompt()

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze()

    # Handle edge cases where the number of classes is less than the desired samples
    num_classes = probabilities.size(0)  # Total number of classes in the model
    num_samples = min(3, num_classes)  # Generate up to 3 ideas, but not more than available classes

    # Ensure sampling works even if there are fewer classes
    sampled_classes = (
        torch.multinomial(probabilities, num_samples=num_samples, replacement=False).tolist()
        if num_samples > 0
        else []
    )

    # Map sampled classes to ideas
    all_ideas = [idea_mapping.get(label, f"Unknown idea for label {label}") for label in sampled_classes]

    # Shuffle ideas to add extra randomness
    random.shuffle(all_ideas)

    return all_ideas

# Function to provide detailed explanation for selected app ideas
def generate_detailed_explanation(selected_ideas):
    explanations = {}
    for idea in selected_ideas:
        prompt = f"{idea} Provide a detailed explanation of this idea."
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode the explanation
        explanation = tokenizer.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        explanations[idea] = explanation.strip()

    return explanations

# CLI-based application
def chatbot_cli():
    print("\nWelcome to the App Idea Generator!\n")

    # Generate app ideas
    print("Generating app ideas...\n")
    app_ideas = generate_app_ideas()

    if not app_ideas:
        print("No app ideas could be generated. Please check the model and try again.")
        return

    print("Here are some app ideas:")
    for i, idea in enumerate(app_ideas, start=1):
        print(f"{i}. {idea}")

    # Let the user select 2 ideas
    while True:
        user_choice = input("\nSelect two ideas by their numbers (e.g., '1 and 3'): ").strip()
        try:
            selected_numbers = [int(num) for num in user_choice.split() if int(num) in range(1, len(app_ideas) + 1)]
            if len(selected_numbers) == 2:
                break
            else:
                print("Invalid selection. Please choose exactly two ideas.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")

    # Get the selected ideas
    selected_ideas = [app_ideas[i - 1] for i in selected_numbers]

    # Generate detailed explanations
    print("\nGenerating detailed suggestions for your chosen ideas...\n")
    explanations = generate_detailed_explanation(selected_ideas)
    for idea, explanation in explanations.items():
        print(f"Idea: {idea}")
        print(f"Explanation: {explanation}\n")

if __name__ == "__main__":
    chatbot_cli()
