from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import uuid

# Load env
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

# Store extracted text
extracted_text_store = {}

# -------------------
# Upload route
# -------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    extracted_text = ""

    if file.filename.endswith(".pdf"):
        # Try digital PDF first
        with pdfplumber.open(filepath) as pdf:
            extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        # If scanned PDF (no text), use OCR
        if not extracted_text.strip():
            doc = fitz.open(filepath)
            for i, page in enumerate(doc):
                pix = page.get_pixmap()
                img_file = os.path.join(app.config["UPLOAD_FOLDER"], f"page_{i}_{uuid.uuid4().hex}.png")
                pix.save(img_file)

                # OCR via Gemini
                img_upload = genai.upload_file(img_file)
                result = model.generate_content([f"Extract text from this image.", img_upload])
                extracted_text += result.text + "\n"

                # Cleanup temp image
                os.remove(img_file)

    else:
        # For images
        img_upload = genai.upload_file(filepath)
        result = model.generate_content(["Extract text from this image.", img_upload])
        extracted_text = result.text

    extracted_text_store["text"] = extracted_text.strip()

    return jsonify({
        "message": "File uploaded & text extracted successfully!",
        "text_preview": extracted_text[:1000]  # preview first 1000 chars
    })

# -------------------
# Ask question route
# -------------------
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    context_text = extracted_text_store.get("text", "")

    if not context_text:
        return jsonify({"answer": "No file uploaded yet."})

    prompt = f"""
    The following text was extracted from a file:

    {context_text}

    User question: {question}

    Answer appropriately:
    - If translation is asked, translate.
    - If summarization is asked, summarize.
    - If Q&A, answer based on context.
    - If general query, act as AI assistant.
    """

    response = model.generate_content(prompt)
    return jsonify({"answer": response.text})

# -------------------
# Run server
# -------------------
if __name__ == "__main__":
    app.run(debug=True)
