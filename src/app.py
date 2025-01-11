from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("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")
API_URL = "https://api-inference.huggingface.co/models/gpt2"

def generate_ideas(query):
    """Generates three unique ideas using Hugging Face API."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": f"Suggest 3 unique app ideas for: {query}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for HTTP issues
        suggestions = response.json()
        if isinstance(suggestions, list) and suggestions:
            return suggestions[0]["generated_text"].split("\n")[:3]
        return ["Error: No suggestions generated."]
    except requests.exceptions.RequestException as e:
        return [f"Error: {str(e)}"]

@app.route('/')
def home():
    """Root route to verify app is running."""
    return "Welcome to the Idea Generator! Use POST /generate to get ideas."

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint to generate ideas."""
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    ideas = generate_ideas(query)
    return jsonify({'ideas': ideas})

if __name__ == '__main__':
    app.run(debug=True)
