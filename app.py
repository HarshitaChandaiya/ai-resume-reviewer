"""
app.py - AI Resume Reviewer Bot
"""
import streamlit as st
from utils import (
    setup_gemini, extract_text_from_pdf, analyze_resume,
    match_with_job, compute_tfidf_similarity, tfidf_score_to_percent,
    highlight_keywords, get_score_color, get_score_label,
)

st.set_page_config(page_title="AI Resume Reviewer", page_icon="📄", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    :root { --bg:#0f0f13; --surface:#1a1a24; --surface2:#22222f; --border:#2e2e3e; --accent:#7c6cfc; --accent2:#a78bfa; --text:#e2e8f0; --muted:#94a3b8; }
    html,body,.stApp { background-color:var(--bg)!important; color:var(--text)!important; font-family:'DM Sans',sans-serif!important; }
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:2rem!important;max-width:860px!important;}
    .hero-header{text-align:center;padding:2rem 1rem 1rem;margin-bottom:1.5rem;}
    .hero-header h1{font-family:'Syne',sans-serif!important;font-size:2.8rem!important;font-weight:800!important;background:linear-gradient(135deg,#7c6cfc,#a78bfa,#c4b5fd);-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;margin-bottom:0.5rem!important;}
    .hero-header p{color:var(--muted)!important;font-size:1rem!important;}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:1rem;}
    .card-title{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;}
    .score-display{display:flex;align-items:center;gap:1.5rem;padding:1.2rem;background:var(--surface2);border-radius:12px;margin-bottom:1rem;}
    .score-circle{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;line-height:1;min-width:90px;text-align:center;}
    .score-label{font-size:1.1rem;font-weight:600;margin-bottom:0.3rem;}
    .score-reason{color:var(--muted);font-size:0.88rem;}
    .tag{display:inline-block;padding:0.25rem 0.75rem;border-radius:99px;font-size:0.8rem;font-weight:500;margin:0.2rem;}
    .tag-skill{background:rgba(124,108,252,0.15);color:#a78bfa;border:1px solid rgba(124,108,252,0.3);}
    .tag-missing{background:rgba(248,113,113,0.1);color:#fca5a5;border:1px solid rgba(248,113,113,0.2);}
    .tag-strength{background:rgba(34,197,94,0.1);color:#86efac;border:1px solid rgba(34,197,94,0.2);}
    .tag-matched{background:rgba(34,197,94,0.1);color:#86efac;border:1px solid rgba(34,197,94,0.2);}
    .section-block{background:var(--surface2);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.6rem;}
    .section-block-title{font-weight:600;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--accent2);margin-bottom:0.3rem;}
    .section-block-text{color:var(--text);font-size:0.9rem;line-height:1.5;}
    .suggestion-item{display:flex;gap:0.75rem;align-items:flex-start;padding:0.7rem 0;border-bottom:1px solid var(--border);}
    .suggestion-item:last-child{border-bottom:none;}
    .suggestion-num{background:var(--accent);color:white;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;margin-top:1px;}
    .stFileUploader>div{border:2px dashed var(--border)!important;border-radius:14px!important;background:var(--surface)!important;padding:1rem!important;}
    .stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:12px!important;padding:0.3rem!important;gap:0.3rem!important;border:1px solid var(--border)!important;}
    .stTabs [data-baseweb="tab"]{border-radius:8px!important;color:var(--muted)!important;font-family:'Syne',sans-serif!important;font-weight:600!important;padding:0.5rem 1.2rem!important;}
    .stTabs [aria-selected="true"]{background:var(--accent)!important;color:white!important;}
    .stTabs [data-baseweb="tab-panel"]{padding-top:1.2rem!important;}
    .stButton>button{background:linear-gradient(135deg,#7c6cfc,#a78bfa)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;padding:0.65rem 2rem!important;font-size:0.95rem!important;width:100%!important;}
    .stTextArea textarea{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;}
    .success-badge{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:8px;padding:0.6rem 1rem;color:#86efac;font-size:0.88rem;margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)

if "resume_text" not in st.session_state: st.session_state.resume_text = None
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "match_result" not in st.session_state: st.session_state.match_result = None
if "tfidf_score" not in st.session_state: st.session_state.tfidf_score = None

st.markdown('<div class="hero-header"><h1>📄 AI Resume Reviewer</h1><p>Upload your resume · Get your ATS score · Match against job descriptions · Land interviews</p></div>', unsafe_allow_html=True)

# UPLOAD ON MAIN PAGE
st.markdown('<div class="card"><div class="card-title">📎 Upload Your Resume (PDF)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
if uploaded_file:
    with st.spinner("Reading resume..."):
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
            st.session_state.resume_text = resume_text
            st.markdown(f'<div class="success-badge">✅ Resume uploaded! ({len(resume_text):,} characters extracted)</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ {e}")
            st.session_state.resume_text = None
st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Resume Analysis", "🎯 Job Match"])

with tab1:
    if not st.session_state.resume_text:
        st.markdown('<div style="text-align:center;padding:2rem;color:#64748b;">⬆️ Upload your resume above to get started</div>', unsafe_allow_html=True)
    else:
        if st.button("🚀 Analyze Resume", key="analyze_btn"):
            with st.spinner("🤖 AI is analyzing your resume… (10–15 seconds)"):
                try:
                    model = setup_gemini()
                    st.session_state.analysis_result = analyze_resume(st.session_state.resume_text, model)
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state.analysis_result:
        r = st.session_state.analysis_result
        score = r.get("ats_score", 0)
        color = get_score_color(score)
        label = get_score_label(score)

        st.markdown(f'<div class="card"><div class="card-title">ATS Score</div><div class="score-display"><div class="score-circle" style="color:{color};">{score}</div><div><div class="score-label" style="color:{color};">{label}</div><div class="score-reason">{r.get("ats_score_reason","")}</div></div></div></div>', unsafe_allow_html=True)
        st.progress(score/100)
        st.markdown("<br>", unsafe_allow_html=True)

        skills = r.get("skills_detected", [])
        if skills:
            st.markdown('<div class="card"><div class="card-title">✅ Skills Detected</div>' + "".join([f'<span class="tag tag-skill">{s}</span>' for s in skills]) + "</div>", unsafe_allow_html=True)

        missing = r.get("missing_keywords", [])
        if missing:
            st.markdown('<div class="card"><div class="card-title">⚠️ Missing Keywords</div>' + "".join([f'<span class="tag tag-missing">{k}</span>' for k in missing]) + "</div>", unsafe_allow_html=True)

        strengths = r.get("strengths", [])
        if strengths:
            st.markdown('<div class="card"><div class="card-title">💪 Strengths</div>' + "".join([f'<span class="tag tag-strength">{s}</span>' for s in strengths]) + "</div>", unsafe_allow_html=True)

        impression = r.get("overall_impression", "")
        if impression:
            st.markdown(f'<div class="card"><div class="card-title">🌟 Overall Impression</div><div style="color:#e2e8f0;font-size:0.92rem;line-height:1.6;">{impression}</div></div>', unsafe_allow_html=True)

        section_feedback = r.get("section_feedback", {})
        if section_feedback:
            icons = {"education":"🎓","experience":"💼","projects":"🛠️","skills":"⚡","summary":"📝"}
            st.markdown('<div class="card"><div class="card-title">📋 Section-wise Feedback</div>', unsafe_allow_html=True)
            for section, feedback in section_feedback.items():
                if feedback:
                    icon = icons.get(section.lower(), "📌")
                    st.markdown(f'<div class="section-block"><div class="section-block-title">{icon} {section.title()}</div><div class="section-block-text">{feedback}</div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        suggestions = r.get("suggestions", [])
        if suggestions:
            st.markdown('<div class="card"><div class="card-title">💡 Actionable Suggestions</div>', unsafe_allow_html=True)
            for i, s in enumerate(suggestions, 1):
                st.markdown(f'<div class="suggestion-item"><div class="suggestion-num">{i}</div><div style="color:#e2e8f0;font-size:0.92rem;line-height:1.5;">{s}</div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    if not st.session_state.resume_text:
        st.markdown('<div style="text-align:center;padding:2rem;color:#64748b;">⬆️ Upload your resume above first</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card-title">📋 Paste Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area("JD", placeholder="Paste the full job description here…", height=200, label_visibility="collapsed")
        if st.button("🎯 Match with Job", key="match_btn"):
            if not job_description.strip():
                st.warning("⚠️ Please paste a job description first.")
            else:
                with st.spinner("🤖 Comparing… (10–15 seconds)"):
                    try:
                        model = setup_gemini()
                        st.session_state.match_result = match_with_job(st.session_state.resume_text, job_description, model)
                        raw_sim = compute_tfidf_similarity(st.session_state.resume_text, job_description)
                        st.session_state.tfidf_score = tfidf_score_to_percent(raw_sim)
                    except Exception as e:
                        st.error(f"❌ {e}")

    if st.session_state.match_result and st.session_state.resume_text:
        m = st.session_state.match_result
        match_score = m.get("match_score", 0)
        match_color = get_score_color(match_score)
        match_label = get_score_label(match_score)
        tfidf_pct = st.session_state.get("tfidf_score", 0)
        tfidf_color = get_score_color(tfidf_pct)

        st.markdown(f'<div class="card"><div class="card-title">AI Match Score</div><div class="score-display"><div class="score-circle" style="color:{match_color};">{match_score}%</div><div><div class="score-label" style="color:{match_color};">{match_label}</div><div class="score-reason">{m.get("match_score_reason","")}</div></div></div></div>', unsafe_allow_html=True)
        st.progress(match_score/100)
        st.markdown(f'<div class="card" style="text-align:center;"><div class="card-title" style="text-align:center;">📊 Keyword Similarity (TF-IDF)</div><div style="font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;color:{tfidf_color};">{tfidf_pct}%</div><div style="color:#64748b;font-size:0.82rem;">Mathematical keyword overlap</div></div>', unsafe_allow_html=True)

        matched = m.get("matched_skills", [])
        if matched:
            st.markdown('<div class="card"><div class="card-title">✅ Matched Skills</div>' + "".join([f'<span class="tag tag-matched">{s}</span>' for s in matched]) + "</div>", unsafe_allow_html=True)

        missing_skills = m.get("missing_skills", [])
        if missing_skills:
            st.markdown('<div class="card"><div class="card-title">❌ Missing Skills</div>' + "".join([f'<span class="tag tag-missing">{s}</span>' for s in missing_skills]) + "</div>", unsafe_allow_html=True)

        missing_kw = m.get("missing_keywords", [])
        if missing_kw:
            st.markdown('<div class="card"><div class="card-title">🔑 Missing Keywords</div>' + "".join([f'<span class="tag tag-missing">{k}</span>' for k in missing_kw]) + "</div>", unsafe_allow_html=True)

        exp_gap = m.get("experience_gap", "")
        if exp_gap:
            st.markdown(f'<div class="card"><div class="card-title">📊 Experience Alignment</div><div style="color:#e2e8f0;font-size:0.92rem;line-height:1.6;">{exp_gap}</div></div>', unsafe_allow_html=True)

        suggestions = m.get("suggestions", [])
        if suggestions:
            st.markdown('<div class="card"><div class="card-title">💡 Improvement Suggestions</div>', unsafe_allow_html=True)
            for i, s in enumerate(suggestions, 1):
                st.markdown(f'<div class="suggestion-item"><div class="suggestion-num">{i}</div><div style="color:#e2e8f0;font-size:0.92rem;line-height:1.5;">{s}</div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        tips = m.get("tailoring_tips", "")
        if tips:
            st.markdown(f'<div class="card" style="border-left:3px solid #7c6cfc;"><div class="card-title">✍️ How to Tailor This Resume</div><div style="color:#e2e8f0;font-size:0.92rem;line-height:1.7;">{tips}</div></div>', unsafe_allow_html=True)

        if missing_kw:
            with st.expander("🔍 Resume with missing keywords highlighted"):
                highlighted = highlight_keywords(st.session_state.resume_text, missing_kw)
                st.markdown(f'<div style="font-size:0.85rem;line-height:1.7;white-space:pre-wrap;color:#cbd5e1;">{highlighted}</div>', unsafe_allow_html=True)
