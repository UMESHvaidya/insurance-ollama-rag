
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.insurance_rag.rag_pipeline import InsurancePolicyRAG
from src.insurance_rag.models import CoverageResponse

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = "data/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

rag_pipeline = None

@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def query():
    """Handle file upload and query"""
    global rag_pipeline

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    query = request.form.get("query")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if file and file.filename.endswith(".pdf"):
        filename = secure_filename(file.filename)
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(filepath)

        try:
            # Initialize RAG pipeline and load the document
            rag_pipeline = InsurancePolicyRAG()
            rag_pipeline.load_policy(filepath)

            # Perform query
            response = rag_pipeline.query(query)
            return jsonify(response.to_dict())

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

def main():
    """Run the Flask app"""
    app.run(debug=True, port=5001)

if __name__ == "__main__":
    main()
