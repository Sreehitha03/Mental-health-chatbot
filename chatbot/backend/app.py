import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_inference import generate_response

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure 'llama_inference' folder is created in D drive
LLAMA_INFERENCE_DIR = "D:/llama_inference"
os.makedirs(LLAMA_INFERENCE_DIR, exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "No prompt provided"}), 400
            
        prompt = data["prompt"]
        
        # Generate response using LLaMA model
        response = generate_response(prompt)
        
        # Save response to a file
        output_file = os.path.join(LLAMA_INFERENCE_DIR, "response_log.txt")
        with open(output_file, "a", encoding='utf-8') as file:
            file.write(f"Prompt: {prompt}\nResponse: {response}\n\n")

        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print(f"Starting server... Responses will be saved in '{LLAMA_INFERENCE_DIR}'")
    app.run(debug=True, host='0.0.0.0', port=5000)