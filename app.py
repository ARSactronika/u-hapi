from flask import Flask, request, jsonify, render_template_string, send_from_directory
import aiohttp
import asyncio
import os

# Define the classes for text classification
main_classes = [
    'air', 'avatar', 'cinematic', 'earth', 'electrical', 'essentials', 'fire',
    'game', 'materials', 'mechanics', 'spells', 'supernatural', 'user interface',
    'water', 'weapons'
]

# Define the classes for position classification
position_classes = ["front", "back", "right", "left"]

# Define the Hugging Face API endpoint and your API key
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
API_KEY = "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"  # Hugging Face API key

# Initialize Flask app
app = Flask(__name__)

# Define your authentication key
AUTH_KEY = 'my_secret_auth_key'

# Define the directory containing the .wav files
AUDIO_DIR = "audio_files"

# Ensure the directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Define the HTML template for the interactive interface
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Text to Haptics</title>
</head>
<body>
    <h1>Text to Haptics</h1>
    <form id="classify-form">
        <label for="text">Enter text:</label><br><br>
        <input type="text" id="text" name="text" size="50"><br><br>
        <input type="submit" value="Classify">
    </form>
    <h2>Result:</h2>
    <p id="result"></p>
    <audio id="audio-player" controls></audio>
    <script>
        document.getElementById('classify-form').onsubmit = async function(event) {
            event.preventDefault();
            const text = document.getElementById('text').value;
            const response = await fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': '{{ auth_key }}'
                },
                body: JSON.stringify({ text: text })
            });
            const result = await response.json();
            const resultElement = document.getElementById('result');
            const audioPlayer = document.getElementById('audio-player');
            
            if (result.class && result.position) {
                resultElement.innerText = `Class: ${result.class}, Position: ${result.position}`;
                audioPlayer.src = `/audio/${result.class}.wav`;
                audioPlayer.play();
                console.log(`Classified as: ${result.class}, Position: ${result.position}`);
            } else {
                resultElement.innerText = `Error: ${result.error}`;
                audioPlayer.src = '';
            }
        }
    </script>
</body>
</html>
"""

async def classify_text(session, text, candidate_labels):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": candidate_labels
        }
    }
    async with session.post(API_URL, headers=headers, json=payload) as response:
        if response.status == 200:
            result = await response.json()
            return result['labels'][0], result['scores'][0]
        else:
            raise Exception(f"API request failed with status code {response.status}: {await response.text()}")

async def classify_all(text):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(main_classes), 10):
            chunk = main_classes[i:i+10]
            tasks.append(classify_text(session, text, chunk))
        
        results = await asyncio.gather(*tasks)
        best_class, best_score = max(results, key=lambda x: x[1])
        
        position = await classify_text(session, text, position_classes)
        
        return best_class, position[0]

@app.route('/')
def index():
    return render_template_string(html_template, auth_key=AUTH_KEY)

@app.route('/classify', methods=['POST'])
async def classify():
    auth_key = request.headers.get('Authorization')
    if auth_key != AUTH_KEY:
        return jsonify({'error': 'Unauthorized access'}), 401

    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        best_class, position = await classify_all(text)
        return jsonify({'class': best_class, 'position': position})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
