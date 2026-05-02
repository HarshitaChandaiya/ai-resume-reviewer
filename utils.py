"""
utils.py - Helper functions for AI Resume Reviewer Bot
Uses the new google-genai library
"""

import os
import json
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ─────────────────────────────────────────────
# 1. API SETUP
# ─────────────────────────────────────────────

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it with: set GEMINI_API_KEY=your_key")
    client = genai.Client(api_key=api_key)
    return client

MODEL = "gemini-2.5-flash"
# ─────────────────────────────────────────────
# 2. PDF PARSING
# ─────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    text_pages = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_pages.append(page_text.strip())
                except Exception as e:
                    print(f"Warning: Could not extract page {page_num + 1}: {e}")
                    continue
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")

    if not text_pages:
        raise ValueError("No readable text found in PDF. Ensure it is not scanned/image-based.")

    full_text = "\n\n".join(text_pages)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    return full_text

# ─────────────────────────────────────────────
# 3. RESUME ANALYSIS
# ─────────────────────────────────────────────

RESUME_ANALYSIS_PROMPT = """
You are an expert ATS (Applicant Tracking System) and resume coach.

Analyze the following resume and return a JSON response ONLY — no markdown, no extra text.

Resume:
\"\"\"
{resume_text}
\"\"\"

Return this exact JSON structure:
{{
  "ats_score": <integer 0-100>,
  "ats_score_reason": "<one sentence explaining the score>",
  "skills_detected": ["skill1", "skill2"],
  "missing_keywords": ["keyword1", "keyword2"],
  "section_feedback": {{
    "education": "<feedback>",
    "experience": "<feedback>",
    "projects": "<feedback>",
    "skills": "<feedback>",
    "summary": "<feedback>"
  }},
  "strengths": ["strength1", "strength2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "overall_impression": "<2-3 sentence assessment>"
}}

Return ONLY valid JSON, nothing else.
"""

def analyze_resume(resume_text: str, client) -> dict:
    prompt = RESUME_ANALYSIS_PROMPT.format(resume_text=resume_text)
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        raw_output = response.text.strip()
        raw_output = re.sub(r'^```json\s*', '', raw_output)
        raw_output = re.sub(r'\s*```$', '', raw_output)
        return json.loads(raw_output.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"LLM analysis failed: {e}")

# ─────────────────────────────────────────────
# 4. JOB DESCRIPTION MATCHING
# ─────────────────────────────────────────────

JOB_MATCH_PROMPT = """
You are an expert recruiter and ATS specialist.

Compare the resume to the job description and return JSON ONLY — no extra text.

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Return this exact JSON structure:
{{
  "match_score": <integer 0-100>,
  "match_score_reason": "<one sentence>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "missing_keywords": ["keyword1", "keyword2"],
  "experience_gap": "<assessment>",
  "suggestions": ["suggestion1", "suggestion2"],
  "tailoring_tips": "<paragraph on how to tailor this resume>"
}}

Return ONLY valid JSON.
"""

def match_with_job(resume_text: str, job_description: str, client) -> dict:
    prompt = JOB_MATCH_PROMPT.format(resume_text=resume_text, job_description=job_description)
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        raw_output = response.text.strip()
        raw_output = re.sub(r'^```json\s*', '', raw_output)
        raw_output = re.sub(r'\s*```$', '', raw_output)
        return json.loads(raw_output.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Job match failed: {e}")

# ─────────────────────────────────────────────
# 5. TF-IDF SIMILARITY
# ─────────────────────────────────────────────

def compute_tfidf_similarity(resume_text: str, job_description: str) -> float:
    if not resume_text.strip() or not job_description.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(float(similarity[0][0]), 4)
    except Exception as e:
        print(f"TF-IDF error: {e}")
        return 0.0

def tfidf_score_to_percent(score: float) -> int:
    return int(round(min(score * 250, 100)))

# ─────────────────────────────────────────────
# 6. HELPERS
# ─────────────────────────────────────────────

def highlight_keywords(text: str, keywords: list) -> str:
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(
            lambda m: f'<mark style="background:#ffd700;padding:1px 3px;border-radius:3px;">{m.group()}</mark>',
            text
        )
    return text

def get_score_color(score: int) -> str:
    if score >= 80: return "#22c55e"
    elif score >= 60: return "#f59e0b"
    elif score >= 40: return "#f97316"
    else: return "#ef4444"

def get_score_label(score: int) -> str:
    if score >= 80: return "Excellent ✨"
    elif score >= 60: return "Good 👍"
    elif score >= 40: return "Average ⚠️"
    else: return "Needs Work 🔧"