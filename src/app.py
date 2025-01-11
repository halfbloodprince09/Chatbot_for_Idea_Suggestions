from flask import Flask, request, jsonify
import os
import openai
from dotenv import load_dotenv

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_ideas(query):
    """Generates three unique ideas using OpenAI API."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Suggest 3 unique ideas for: {query}",
            max_tokens=100
        )
        ideas = response.choices[0].text.strip().split('\n')
        return [idea.strip() for idea in ideas if idea.strip()]
    except Exception as e:
        return str(e)

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
    if isinstance(ideas, str):  # If error occurred in generation
        return jsonify({'error': ideas}), 500
    return jsonify({'ideas': ideas})

if __name__ == '__main__':
    app.run(debug=True)
