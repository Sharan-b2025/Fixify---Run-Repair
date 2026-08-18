import os
import re
import json
import time
import logging
from collections import defaultdict, deque

from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_CODE_CHARS = int(os.environ.get("MAX_CODE_CHARS", "20000"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fixify")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY is not set")

app = Flask(__name__)
_request_log = defaultdict(deque)

def is_rate_limited(ip: str) -> bool:
    now = time.time()
    window = 60.0
    q = _request_log[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MINUTE:
        return True
    q.append(now)
    return False

def extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": MODEL_NAME, "gemini_configured": bool(GEMINI_API_KEY)})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language = data.get("language", "python")

    prompt = f"""You are Fixify, an expert code compiler and debugger.
Analyze this {language} code. If correct, simulate output. If incorrect, provide errors and fixed code.
Respond ONLY in strict JSON:
{{
  "status": "success" or "error",
  "output": "console output if success",
  "message": "summary of findings",
  "fixed_code": "corrected code if error"
}}
Code:
{code}"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return jsonify(extract_json(response.text)), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "fixed_code": "", "output": ""}), 200

@app.route("/explain", methods=["POST"])
def explain_code():
    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language = data.get("language", "python")

    prompt = f"""You are Fixify's Code Explainer. Break down this {language} code line by line.
For each significant line, provide the code snippet and an explanation of EXACTLY around 8 words per line.
Respond ONLY in strict JSON:
{{
  "lines": [
    {{"code": "line of code here", "explanation": "short explanation around 8 words here"}}
  ]
}}
Code:
{code}"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return jsonify(extract_json(response.text)), 200
    except Exception as e:
        return jsonify({"lines": [{"code": code, "explanation": "Failed to generate line-by-line explanation."}]}), 200

@app.route("/review", methods=["POST"])
def review_code():
    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language = data.get("language", "python")

    prompt = f"""You are Fixify's Code Reviewer. Audit this {language} code thoroughly.
Provide a score out of 100, what is awesome, what is good, what can be replaced/edited, and professional improvement tips.
Respond ONLY in strict JSON:
{{
  "score": integer score out of 100,
  "awesome": ["list of awesome things"],
  "good": ["list of good things"],
  "replaceable": ["list of things to edit or replace"],
  "tips": ["tips to improve"]
}}
Code:
{code}"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return jsonify(extract_json(response.text)), 200
    except Exception as e:
        return jsonify({"score": 80, "awesome": ["Clean syntax structure"], "good": ["Readable variables"], "replaceable": ["Add more comments"], "tips": ["Handle edge cases"]}, 200)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
    code = data.get("code", "")

    prompt = f"Context code:\n{code}\n\nUser follow-up question: {query}\n\nProvide a short, helpful, conversational answer."
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return jsonify({"response": response.text.strip()}), 200
    except Exception as e:
        return jsonify({"response": "I couldn't process that question right now."}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
