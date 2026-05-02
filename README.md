# 📄 AI Resume Reviewer

A free, AI-powered web app that analyzes your resume, gives you an ATS score, and matches it against job descriptions — built with Python, Streamlit, and Google Gemini API.

🔗 **Live App:** https://ai-resume-reviewer-zarfqywbrgcufzcwydhcq3.streamlit.app/

---

## 🎯 What It Does

- Upload your resume as a PDF
- Get an **ATS Score (0–100)** with detailed explanation
- See **skills detected** and **missing keywords**
- Get **section-wise feedback** on Education, Experience, Projects, and Skills
- Paste any job description and get a **match score**
- See exactly which skills are missing from the job description
- **TF-IDF cosine similarity** score for mathematical keyword matching

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web app UI |
| Google Gemini API | AI analysis and scoring |
| scikit-learn | TF-IDF cosine similarity |
| pdfplumber | PDF text extraction |

---

## 🚀 Run Locally

```bash
git clone https://github.com/HarshitaChandaiya/ai-resume-reviewer.git
cd ai-resume-reviewer
pip install -r requirements.txt
set GEMINI_API_KEY=your_key_here
streamlit run app.py


