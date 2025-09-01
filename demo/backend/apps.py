import os
import cv2
from PIL import Image
import fitz
import pdfplumber
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import uuid
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------
# Config & app setup
# --------------------------
load_dotenv()
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # from Google AI Studio
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------
# Global state
# --------------------------
selected_language = "english"
extracted_text = ""
vector_store = None
OCR_LANG_MAP = {
    "english": ["en"], "tamil": ["ta"], "hindi": ["hi"], "malayalam": ["ml"],
    "telugu": ["te"], "spanish": ["es"], "french": ["fr"], "german": ["de"]
}
ocr_reader = None

# --------------------------
# Gemini query
# --------------------------
def query_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "No answer found."

# --------------------------
# OCR helpers
# --------------------------
def get_ocr_reader(lang):
    global ocr_reader
    langs = OCR_LANG_MAP.get(lang, ["en"])
    if ocr_reader is None or getattr(ocr_reader, "lang_list", None) != langs:
        ocr_reader = easyocr.Reader(langs, gpu=False)
    return ocr_reader

def preprocess_image(image_path: str) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return image_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    h, w = thresh.shape
    base_width = 1500
    scale = base_width / max(w, 1)
    resized = cv2.resize(thresh, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"processed_{uuid.uuid4().hex}.png")
    cv2.imwrite(temp_path, resized)
    return temp_path

def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()

def translate_text(text: str, target_lang: str = "english") -> str:
    try:
        tl = (target_lang or "english").lower()
        if tl == "english":
            return text
        return GoogleTranslator(source="auto", target=tl).translate(text)
    except Exception:
        return text

def to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="english").translate(text)
    except Exception:
        return text

# --------------------------
# Extraction functions
# --------------------------
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") or ""
        doc.close()
        if text.strip():
            return text
    except Exception:
        pass
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
        if text.strip():
            return text
    except Exception:
        pass
    return extract_text_from_scanned_pdf(file_path)

def extract_text_from_scanned_pdf(file_path: str) -> str:
    pages = convert_from_path(file_path, dpi=300)
    all_text = []
    reader = get_ocr_reader(selected_language)
    for page in pages:
        tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_page_{uuid.uuid4().hex}.png")
        page.save(tmp_path)
        processed = preprocess_image(tmp_path)
        results = reader.readtext(processed, detail=0)
        all_text.append(" ".join(results))
    return "\n".join(all_text)

def extract_text_from_image(file_path: str) -> str:
    _ = Image.open(file_path)
    processed = preprocess_image(file_path)
    reader = get_ocr_reader(selected_language)
    results = reader.readtext(processed, detail=0)
    return clean_text(" ".join(results))

# --------------------------
# Vector store
# --------------------------
def build_vector_store(text: str):
    global vector_store
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBEDDING_MODEL"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = splitter.split_text(text or "")
    if not chunks:
        vector_store = None
        return
    vector_store = FAISS.from_texts(chunks, embeddings)

# --------------------------
# Routes
# --------------------------
@app.route("/set_language", methods=["POST"])
def set_language():
    global selected_language
    data = request.get_json(force=True, silent=True) or {}
    lang_key = (data.get("language") or "english").lower()
    selected_language = lang_key
    return jsonify({"status": "success", "language": selected_language})

@app.route("/upload", methods=["POST"])
def upload_file():
    global extracted_text, vector_store
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{f.filename}")
    f.save(file_path)
    try:
        if f.filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
        else:
            extracted_text = extract_text_from_image(file_path)
        extracted_text = clean_text(extracted_text)
        english_text = to_english(extracted_text)
        build_vector_store(english_text)
        return jsonify({
            "status": "success",
            "language": selected_language,
            "text_preview": extracted_text[:300]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store, selected_language, extracted_text
    data = request.get_json(force=True, silent=True) or {}
    question_raw = (data.get("question") or "").strip()
    if not question_raw:
        return jsonify({"answer": ""})

    question_en = to_english(question_raw)
    context = ""
    if vector_store:
        try:
            results = vector_store.similarity_search_with_score(question_en, k=3)
            top_chunks = [getattr(doc, "page_content", str(doc)) for (doc, _score) in results]
            context = "\n".join(top_chunks[:3]).strip()
        except Exception:
            context = ""

    prompt = (
        "You are an assistant answering based on the provided context when relevant.\n"
        f"Context:\n{context or extracted_text}\n\n"
        f"Question: {question_en}\nAnswer clearly and concisely:\n"
    )

    try:
        answer_en = query_gemini(prompt)
    except Exception as e:
        answer_en = f"Error: {str(e)}"

    final_answer = translate_text(answer_en, selected_language)
    return jsonify({"answer": final_answer})

@app.route("/summarize", methods=["POST"])
def summarize():
    global extracted_text, selected_language
    if not extracted_text.strip():
        return jsonify({"error": "No text available for summarization"}), 400
    base_en = to_english(extracted_text)
    prompt = (
        "Summarize the following content in 5-7 concise bullet points.\n"
        "Include key definitions if present.\n\n"
        f"{base_en}\n\nSummary:"
    )
    try:
        summary_en = query_gemini(prompt)
    except Exception as e:
        summary_en = f"Error: {str(e)}"

    summary_out = translate_text(summary_en, selected_language)
    return jsonify({"summary": summary_out})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(force=True, silent=True) or {}
    target_lang = (data.get("language") or "english").lower()
    text_to_translate = data.get("text") or extracted_text
    translated = translate_text(text_to_translate, target_lang)
    return jsonify({"translated_text": translated, "language": target_lang})

@app.route("/clear", methods=["POST"])
def clear():
    global extracted_text, vector_store
    extracted_text = ""
    vector_store = None
    return jsonify({"status": "cleared"})

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
