import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google.generativeai.errors import APIError

# --- Configuration and Initialization ---

# Azure App Service will securely provide the API key via Application Settings.
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Hardcoded model name as requested
MODEL_TO_USE = 'gemini-2.5-flash' 

if not API_KEY:
    # In a cloud environment, print a fatal message and allow the host 
    # (like Gunicorn/Azure) to handle the startup failure.
    print("FATAL: GOOGLE_API_KEY environment variable not found. The application cannot start.")

try:
    # Only configure if the key is available to avoid runtime errors on startup
    if API_KEY:
        genai.configure(api_key=API_KEY)
        # The GenerativeModel instance explicitly uses gemini-2.5-flash
        model = genai.GenerativeModel(MODEL_TO_USE)
    else:
        # Create a placeholder for the model if the API key is missing
        model = None 
except Exception as e:
    # Handle configuration failure if key is present but invalid
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    model = None

# Initialize Flask app
# The name must be 'app' for Azure/Gunicorn to easily find it.
app = Flask(__name__)

# Configure CORS (use specific origins in production)
CORS(app, resources={r"/api/*": {"origins": "*", "supports_credentials": True}})

# --- Core LLM Logic ---

def execute_workout(query: str):
    """
    Skill: Workout Plan Generator
    Generates a workout plan using AI and Google Search for exercise info.
    """
    if not model:
        raise Exception("AI model failed to initialize due to missing or invalid API key.")

    # System instruction defines the AI's persona and task
    system_prompt = f"""
You are a certified personal trainer and exercise physiologist.

TASK:
1. Read the user's parameters: situation, available equipment, timeframe.
2. Search online for exercises that best fit the user's situation and equipment, supplementing or replacing the starter pack as needed.
3. Generate a week-by-week workout plan for the specified timeframe.
4. For each workout, select exercises feasible with the available equipment and situation, ensuring balanced total-body coverage.
5. Clearly list each day's exercises, sets, reps (or time), and any necessary instructions or substitutions.
6. Include guidance on active recovery or cardio for non-workout days.
7. Include online links for exercise instructions wherever possible using Google Search.
8. Output only the workout plan, clearly formatted, without extra commentary.
"""
    
    # Use Google Search grounding to find exercises and links
    response = model.generate_content(
        query,  # The user's context (e.g., "at home, 30 mins, dumbells") is the main prompt
        system_instruction=system_prompt,
        tools=[{"google_search": {}}]
    )
    return response.text

# --- Health Check Route ---

@app.route('/check', methods=['GET'])
def check():
    """
    Simple health check route used to confirm the backend server is running and accessible.
    """
    status_code = 200
    if not model:
        # Return 503 if the core dependency (AI model) failed to initialize
        status_code = 503
        message = "backend is running, but AI model failed to initialize."
    else:
        message = "backend is running"

    return jsonify({
        'status': 'ok' if status_code == 200 else 'error', 
        'message': message, 
        'model': MODEL_TO_USE
    }), status_code

# --- Main API Route ---

@app.route('/api/execute', methods=['POST'])
def execute():
    # 1. Input Validation
    if not model:
        return jsonify({'error': 'AI service not initialized. Check API key configuration.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload.'}), 400

    query = data.get('query')
    # Validate that query is a non-empty string
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'Missing or empty "query" field in the request.'}), 400
    
    # 2. Execution and Specific Error Handling
    try:
        result = execute_workout(query)
        
        return jsonify({'success': True, 'result': result})
        
    except APIError as e:
        # Handle specific Gemini API errors
        print(f"Gemini API Error: {e}")
        return jsonify({'error': f'AI Service Unavailable or request error: {e}'}), 503
        
    except Exception as e:
        # Catch all other unexpected errors
        print(f"Internal Server Error: {e}")
        return jsonify({'error': 'An unexpected internal server error occurred.'}), 500

