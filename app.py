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


def client_ip_of(req) -> str:
    ip = req.headers.get("X-Forwarded-For", req.remote_addr) or "unknown"
    return ip.split(",")[0].strip()


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.15) -> dict:
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=system_prompt)
    response = model.generate_content(
        user_prompt,
        generation_config={"temperature": temperature, "response_mime_type": "application/json"},
    )
    return extract_json(response.text)


ANALYZE_SYSTEM_PROMPT = """You are Fixify, a friendly expert multi-language code debugger and compiler simulator.
Explain things simply enough that a 15-year-old beginner can understand and fix their own code.
You will be given source code (language may or may not be specified) and must:

1. Detect the programming language if not given.
2. Carefully read the ENTIRE code as a real compiler/interpreter would, top to bottom.
3. If the code is 100% correct and would run without errors, simulate and return
   its exact expected console output.
4. If the code has ANY errors, find EVERY error in the code (not just the first
   one) — syntax, logical, runtime, type, indentation, missing import, typos,
   missing semicolons/braces, undefined variables, etc. For EACH error report:
   - the exact line number it occurs on
   - a short, simple title for the error a beginner would understand
   - a plain-English explanation of what is wrong on that line
   - a short 10-15 word message explaining WHY fixing it matters
   List errors in the order they appear, top to bottom.

Respond ONLY with strict, valid JSON (no markdown fences, no extra text) in
EXACTLY this schema:

{
  "language": "detected language name",
  "status": "success" or "error",
  "output": "expected program output if status is success, else empty string",
  "errors": [
    {
      "line": integer line number,
      "title": "short simple error name",
      "message": "plain-English explanation of what is wrong on this line",
      "fix_reason": "10-15 word sentence on why fixing this matters"
    }
  ],
  "fixed_code": "full corrected code with ALL errors fixed, else empty string if status is success",
  "suggestions": [
    {"tip": "short 4-8 word tip title", "why": "one brief sentence (10-18 words) on what it does and why it's useful"},
    {"tip": "short 4-8 word tip title", "why": "one brief sentence (10-18 words) on what it does and why it's useful"}
  ]
}

Be precise about line numbers (1-indexed, matching the given code exactly).
"errors" must be an empty array when status is "success". Keep each error's
message under 35 words, written simply, no jargon a beginner wouldn't know.
Give 2-4 suggestions, each genuinely useful for this specific code (style,
performance, edge cases, readability, etc.), never generic filler.
"""

FOLLOWUP_ERROR_SYSTEM_PROMPT = """You are Fixify, a friendly coding tutor answering a follow-up question
about ONE specific error a student is stuck on. You will get the full code, that one
error's line/title/message, and the student's question. Answer ONLY their question,
in 2-4 short sentences, in plain language a 15-year-old beginner would understand.
Do not restate the whole error or reprint the code. Respond ONLY with strict JSON:
{"answer": "your explanation"}
"""

INSIGHT_SYSTEM_PROMPT = """You are Fixify Insight, a patient coding tutor who walks a beginner through
code one line at a time. You will be given source code. For EVERY line that contains
real code (skip blank lines), write an extremely short explanation of what that line
does — no more than 8 words, simple language, no jargon. Respond ONLY with strict,
valid JSON in EXACTLY this schema:

{
  "language": "detected language name",
  "summary": "one short sentence (under 20 words) on what the whole program does",
  "lines": [
    {"line": integer 1-indexed line number, "explanation": "8 words or fewer explaining this line"}
  ]
}

Only include lines with actual code in the "lines" array. Keep every explanation
under 8 words — this is a hard limit.
"""

INSIGHT_MORE_SYSTEM_PROMPT = """You are Fixify Insight, a patient coding tutor. A student didn't fully
understand one line of code and wants a deeper explanation. You'll get the full code,
the specific line, and the short explanation already given. Provide a slightly deeper,
still simple explanation in 2-3 short sentences, plain language, no jargon a 15-year-old
wouldn't know. Respond ONLY with strict JSON: {"more": "your deeper explanation"}
"""

REVIEW_SYSTEM_PROMPT = """You are Fixify Review, an expert code reviewer who evaluates code quality
fairly and constructively, the way a senior engineer would review a junior's pull request,
but explained simply enough for a beginner to learn from. You will be given source code.
Evaluate it on correctness, readability, structure, naming, efficiency, and best practices
for its language. Then respond ONLY with strict, valid JSON in EXACTLY this schema:

{
  "language": "detected language name",
  "score": integer from 0 to 100 rating overall code quality,
  "summary": "one or two sentence overall verdict, encouraging but honest",
  "awesome": ["specific genuinely great thing about this code", "..."],
  "good": ["specific thing that's fine/solid but not exceptional", "..."],
  "improve": [
    {"issue": "specific thing that could be better", "tip": "concrete, actionable suggestion to fix it"}
  ]
}

Be specific to THIS code, never generic filler. "awesome" and "good" can be empty
arrays if genuinely nothing qualifies, but try to find at least one real positive.
Give 2-5 "improve" items ordered by importance. Score should reflect real quality —
don't inflate it, a working but messy script should score in the 40-65 range, clean
professional code in the 80-95 range, perfect production-grade code above 95.
"""


def analyze_error(title, message, fix_reason, language="auto"):
    return {
        "status": "error",
        "errors": [{"line": None, "title": title, "message": message, "fix_reason": fix_reason}],
        "fixed_code": "",
        "output": "",
        "language": language,
        "suggestions": [],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/explainer")
def explainer_page():
    return render_template("explainer.html")


@app.route("/reviewer")
def reviewer_page():
    return render_template("reviewer.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip()
        message = (data.get("message") or "").strip()
        if not name or not email or not message:
            return jsonify({"status": "error", "message": "Please fill in every field."}), 200
        logger.info("Contact message from %s <%s>: %s", name, email, message[:300])
        return jsonify({"status": "ok", "message": "Thanks — your message has been received."}), 200
    return render_template("contact.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": MODEL_NAME, "gemini_configured": bool(GEMINI_API_KEY)})


@app.route("/analyze", methods=["POST"])
def analyze():
    ip = client_ip_of(request)
    if is_rate_limited(ip):
        return jsonify(analyze_error("Rate Limit Reached", "Too many requests in a short time. Please wait a moment.",
                                      "Slow down a bit so Fixify can keep serving everyone fairly.")), 200

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language_hint = data.get("language", "auto")

    if not code:
        return jsonify(analyze_error("Empty Input", "No code was provided to analyze.",
                                      "Write some code first so Fixify has something to check.", language_hint)), 200
    if len(code) > MAX_CODE_CHARS:
        return jsonify(analyze_error("Code Too Long", f"Submitted code exceeds the {MAX_CODE_CHARS}-character limit.",
                                      "Trim the snippet so it can be analyzed reliably and quickly.", language_hint)), 200
    if not GEMINI_API_KEY:
        return jsonify(analyze_error("Server Not Configured", "GEMINI_API_KEY is missing on the server.",
                                      "Add your Gemini API key as an environment variable to enable analysis.", language_hint)), 200

    try:
        result = call_gemini(ANALYZE_SYSTEM_PROMPT, f"Language hint: {language_hint}\n\nCode:\n{code}")
        result.setdefault("suggestions", [])
        result.setdefault("errors", [])
        result.setdefault("language", language_hint)
        logger.info("analyze ip=%s chars=%d status=%s errors=%d", ip, len(code), result.get("status"), len(result.get("errors", [])))
        return jsonify(result), 200
    except json.JSONDecodeError:
        logger.exception("Gemini returned non-JSON output")
        return jsonify(analyze_error("Analysis Failed", "The AI response could not be parsed. Please try again.",
                                      "A retry usually resolves temporary formatting hiccups from the model.", language_hint)), 200
    except Exception as exc:
        logger.exception("Gemini request failed")
        return jsonify(analyze_error("Analysis Failed", f"Fixify couldn't reach the AI engine: {exc}",
                                      "Check your API key, quota, and internet connection.", language_hint)), 200


@app.route("/explain", methods=["POST"])
def explain():
    ip = client_ip_of(request)
    if is_rate_limited(ip):
        return jsonify({"answer": "Too many requests right now — please wait a moment and try again."}), 200

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    error = data.get("error") or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Type a question first so Fixify knows what to explain."}), 200
    if not GEMINI_API_KEY:
        return jsonify({"answer": "The server is missing its Gemini API key, so Fixify can't answer right now."}), 200

    try:
        prompt = (
            f"Code:\n{code}\n\n"
            f"Error on line {error.get('line')}: {error.get('title')} — {error.get('message')}\n\n"
            f"Student question: {question}"
        )
        result = call_gemini(FOLLOWUP_ERROR_SYSTEM_PROMPT, prompt, temperature=0.25)
        answer = (result.get("answer") or "").strip() or "Fixify couldn't come up with an answer for that — try rephrasing."
        return jsonify({"answer": answer}), 200
    except Exception as exc:
        logger.exception("Explain request failed")
        return jsonify({"answer": f"Something went wrong reaching the AI: {exc}"}), 200


@app.route("/insight", methods=["POST"])
def insight():
    ip = client_ip_of(request)
    if is_rate_limited(ip):
        return jsonify({"summary": "Too many requests right now — please wait a moment.", "lines": []}), 200

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language_hint = data.get("language", "auto")

    if not code:
        return jsonify({"summary": "Paste some code first so Fixify has something to explain.", "lines": []}), 200
    if not GEMINI_API_KEY:
        return jsonify({"summary": "The server is missing its Gemini API key.", "lines": []}), 200

    try:
        result = call_gemini(INSIGHT_SYSTEM_PROMPT, f"Language hint: {language_hint}\n\nCode:\n{code}", temperature=0.15)
        result.setdefault("lines", [])
        result.setdefault("summary", "")
        result.setdefault("language", language_hint)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Insight request failed")
        return jsonify({"summary": f"Something went wrong reaching the AI: {exc}", "lines": []}), 200


@app.route("/insight/more", methods=["POST"])
def insight_more():
    ip = client_ip_of(request)
    if is_rate_limited(ip):
        return jsonify({"more": "Too many requests right now — please wait a moment."}), 200

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    line_number = data.get("line")
    explanation = (data.get("explanation") or "").strip()

    if not GEMINI_API_KEY:
        return jsonify({"more": "The server is missing its Gemini API key."}), 200

    try:
        prompt = f"Code:\n{code}\n\nLine {line_number}, short explanation already given: {explanation}\n\nGive a deeper explanation of this specific line."
        result = call_gemini(INSIGHT_MORE_SYSTEM_PROMPT, prompt, temperature=0.25)
        more = (result.get("more") or "").strip() or "No extra detail available for this line."
        return jsonify({"more": more}), 200
    except Exception as exc:
        logger.exception("Insight-more request failed")
        return jsonify({"more": f"Something went wrong reaching the AI: {exc}"}), 200


@app.route("/review", methods=["POST"])
def review():
    ip = client_ip_of(request)
    if is_rate_limited(ip):
        return jsonify({"score": 0, "summary": "Too many requests right now — please wait a moment.",
                         "awesome": [], "good": [], "improve": []}), 200

    data = request.get_json(force=True, silent=True) or {}
    code = (data.get("code") or "").strip()
    language_hint = data.get("language", "auto")

    if not code:
        return jsonify({"score": 0, "summary": "Paste some code first so Fixify has something to review.",
                         "awesome": [], "good": [], "improve": []}), 200
    if not GEMINI_API_KEY:
        return jsonify({"score": 0, "summary": "The server is missing its Gemini API key.",
                         "awesome": [], "good": [], "improve": []}), 200

    try:
        result = call_gemini(REVIEW_SYSTEM_PROMPT, f"Language hint: {language_hint}\n\nCode:\n{code}", temperature=0.2)
        result.setdefault("awesome", [])
        result.setdefault("good", [])
        result.setdefault("improve", [])
        result.setdefault("score", 0)
        result.setdefault("summary", "")
        result.setdefault("language", language_hint)
        try:
            result["score"] = max(0, min(100, int(result["score"])))
        except (TypeError, ValueError):
            result["score"] = 0
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Review request failed")
        return jsonify({"score": 0, "summary": f"Something went wrong reaching the AI: {exc}",
                         "awesome": [], "good": [], "improve": []}), 200


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
