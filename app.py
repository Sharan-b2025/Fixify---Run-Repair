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
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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


SYSTEM_PROMPT = """You are Fixify, an expert multi-language code debugger and compiler simulator.
You will be given source code (language may or may not be specified) and must:

1. Detect the programming language if not given.
2. Carefully read the ENTIRE code as a real compiler/interpreter would.
3. If the code is 100% correct and would run without errors, simulate and return
   its exact expected console output.
4. If the code has ANY error (syntax, logical, runtime, type, indentation,
   missing import, etc.), find the FIRST major error, and report:
   - the exact line number it occurs on
   - a short, clear explanation of what the error is
   - a short 10-15 word message explaining WHY this should be fixed
   - a corrected version of the FULL code with the fix applied

Respond ONLY with strict, valid JSON (no markdown fences, no extra text) in
EXACTLY this schema:

{
  "language": "detected language name",
  "status": "success" or "error",
  "output": "expected program output if status is success, else empty string",
  "error_line": integer line number if status is error else null,
  "error_title": "short name of the error, e.g. 'SyntaxError: missing colon'",
  "error_message": "clear explanation of what is wrong and where",
  "fix_reason": "10-15 word sentence on why fixing this matters",
  "fixed_code": "full corrected code if status is error, else empty string",
  "suggestions": [
    {"tip": "short 4-8 word tip title", "why": "one brief sentence (10-18 words) on what it does and why it's useful"},
    {"tip": "short 4-8 word tip title", "why": "one brief sentence (10-18 words) on what it does and why it's useful"}
  ]
}

Be precise about line numbers (1-indexed, matching the given code exactly).
Keep error_message under 40 words. Give 2-4 suggestions, each genuinely useful
for this specific code (style, performance, edge cases, readability, etc.),
never generic filler.
"""


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def error_response(title: str, message: str, fix_reason: str, language: str = "auto"):
    return {
        "status": "error",
        "error_title": title,
        "error_message": message,
        "error_line": None,
        "fix_reason": fix_reason,
        "fixed_code": "",
        "output": "",
        "language": language,
        "suggestions": [],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "gemini_configured": bool(GEMINI_API_KEY),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    client_ip = client_ip.split(",")[0].strip()

    if is_rate_limited(client_ip):
        return jsonify(error_response(
            "Rate Limit Reached",
            "Too many requests in a short time. Please wait a moment.",
            "Slow down a bit so Fixify can keep serving everyone fairly.",
        )), 200

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language_hint = data.get("language", "auto")

    if not code:
        return jsonify(error_response(
            "Empty Input",
            "No code was provided to analyze.",
            "Write some code first so Fixify has something to check.",
            language_hint,
        )), 200

    if len(code) > MAX_CODE_CHARS:
        return jsonify(error_response(
            "Code Too Long",
            f"Submitted code exceeds the {MAX_CODE_CHARS}-character limit.",
            "Trim the snippet so it can be analyzed reliably and quickly.",
            language_hint,
        )), 200

    if not GEMINI_API_KEY:
        return jsonify(error_response(
            "Server Not Configured",
            "GEMINI_API_KEY is missing on the server.",
            "Add your Gemini API key as an environment variable to enable analysis.",
            language_hint,
        )), 200

    try:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        prompt = f"Language hint: {language_hint}\n\nCode:\n{code}"
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
        result = extract_json(response.text)
        result.setdefault("suggestions", [])
        result.setdefault("language", language_hint)
        logger.info("Analyzed %d chars for %s -> status=%s", len(code), client_ip, result.get("status"))
        return jsonify(result), 200

    except json.JSONDecodeError:
        logger.exception("Gemini returned non-JSON output")
        return jsonify(error_response(
            "Analysis Failed",
            "The AI response could not be parsed. Please try again.",
            "A retry usually resolves temporary formatting hiccups from the model.",
            language_hint,
        )), 200

    except Exception as exc:
        logger.exception("Gemini request failed")
        return jsonify(error_response(
            "Analysis Failed",
            f"Fixify couldn't reach the AI engine: {exc}",
            "Check your API key, quota, and internet connection.",
            language_hint,
        )), 200


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
