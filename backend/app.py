from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
from backend.evaluation.rag_evaluation import RAGEvaluationPipeline

# Create Flask app
app = Flask(__name__)

# Enable CORS for all routes with all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    api_key = os.getenv('OPENAI_API_KEY_TEG')
if not api_key:
    raise ValueError("OpenAI API key not found. Please set either OPENAI_API_KEY or OPENAI_API_KEY_TEG in your .env file.")

# Initialize the evaluation pipeline
pipeline = RAGEvaluationPipeline(api_key)

@app.route('/', methods=['GET'])
def home():
    """Root endpoint to verify server is running"""
    return "Server is running! Available endpoints: /, /test, /api/evaluate"

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server is running"""
    return jsonify({
        "status": "ok", 
        "message": "Server is running",
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Endpoint to evaluate a single question"""
    print("\nReceived request to /api/evaluate")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Request data: {request.get_data()}")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        print(f"Parsed JSON data: {data}")
        
        question = data.get('question')
        ground_truth = data.get('ground_truth')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
            
        print(f"Processing question: {question}")
        print(f"Ground truth: {ground_truth}")
        
        # Run evaluation
        results_df = pipeline.run_evaluation(question=question, ground_truth=ground_truth)
        print(f"Evaluation completed. DataFrame shape: {results_df.shape}")
        
        # Convert DataFrame to list of dictionaries
        results = results_df.to_dict('records')
        print(f"Number of results: {len(results)}")
        
        response_data = {
            "results": results,
            "ground_truth": ground_truth
        }
        
        print("Sending response...")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in evaluation endpoint: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

def create_app():
    return app
"""
if __name__ == '__main__':
    PORT = 5001
    print("\n=== Starting Flask Server ===")
    print(f"Server will run on http://localhost:{PORT}")
    print("Available endpoints:")
    print("  - GET  /")
    print("  - GET  /test")
    print("  - POST /api/evaluate")
    print("\nRegistered routes:")
    print([str(rule) for rule in app.url_map.iter_rules()])
    app.run(debug=True, host='0.0.0.0', port=PORT) 
"""
if __name__ == '__main__':
    import os
    PORT = int(os.environ.get("PORT", 8080))  # GCP przekazuje PORT jako zmienną środowiskową

    print("\n=== Starting Flask Server ===")
    print(f"Server will run on http://localhost:{PORT}")
    print("Available endpoints:")
    print("  - GET  /")
    print("  - GET  /test")
    print("  - POST /api/evaluate")
    print("\nRegistered routes:")
    print([str(rule) for rule in app.url_map.iter_rules()])

    app.run(debug=True, host='0.0.0.0', port=PORT)