import os
import re
import json
from dotenv import load_dotenv
from llama_parse import LlamaParse
from pdfminer.high_level import extract_text
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize APIs
def initialize_apis():
    """Initialize LlamaParse and Gemini with error handling"""
    try:
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True
        )
    except Exception as e:
        st.error(f"LlamaParse initialization failed: {str(e)}")
        parser = None

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        st.error(f"Gemini initialization failed: {str(e)}")
        model = None

    return parser, model

# Parse resume document
def parse_resume(file_path: str, parser) -> str:
    """Extract text from resume with fallback"""
    try:
        if parser:
            documents = parser.load_data(file_path)
            return documents[0].text
    except Exception as e:
        st.warning(f"LlamaParse failed: {str(e)}. Using PDFMiner fallback.")

    try:
        return extract_text(file_path)
    except Exception as e:
        st.error(f"PDF parsing failed: {str(e)}")
        return ""

# Analyze resume with Gemini
def analyze_resume(resume_text: str, jd_text: str, model) -> dict:
    """Get structured analysis from Gemini with robust JSON handling"""
    prompt = f"""
    Analyze this resume against the job description. Return STRICT JSON with:
    - match_score (0-100)
    - matching_skills (list)
    - missing_skills (list)
    - experience (years)
    - suggestions (list)
    - summary (str)

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Example Output:
    {{
        "match_score": 75,
        "matching_skills": ["Python", "ML"],
        "missing_skills": ["AWS"],
        "experience": 2,
        "suggestions": ["Add AWS certification"],
        "summary": "Strong technical skills but lacks cloud experience"
    }}

    Return ONLY valid JSON with no additional text or formatting.
    """

    try:
        response = model.generate_content(prompt)
        return parse_gemini_response(response.text)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# Parse Gemini's response
def parse_gemini_response(response_text: str) -> dict:
    """Handle various response formats from Gemini"""
    try:
        # Clean response text
        clean_text = response_text.strip()
        
        # Handle markdown code blocks
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3].strip()
        
        # Parse JSON
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Fallback to regex parsing
        return {
            "match_score": extract_value(response_text, "match_score", int),
            "matching_skills": extract_list(response_text, "matching_skills"),
            "missing_skills": extract_list(response_text, "missing_skills"),
            "experience": extract_value(response_text, "experience", int),
            "suggestions": extract_list(response_text, "suggestions"),
            "summary": extract_value(response_text, "summary", str)
        }


# Helper functions
def extract_value(text: str, key: str, type_func):
    """Extract value by key"""
    match = re.search(f'"{key}":\s*([^,\n}}]+)', text)
    if match:
        try:
            return type_func(match.group(1).strip(' "\''))
        except:
            return type_func()
    return type_func()

def extract_list(text: str, key: str) -> list:
    """Extract list by key"""
    match = re.search(f'"{key}":\s*\[([^\]]+)\]', text)
    if match:
        return [item.strip(' "\'') for item in match.group(1).split(",") if item.strip()]
    return []

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("📄 AI Resume Screener")
    st.markdown("Upload a resume and job description to analyze compatibility")

    # Initialize APIs
    parser, model = initialize_apis()
    if not model:
        st.stop()

    # File upload
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    with col2:
        jd_text = st.text_area("Paste Job Description", height=200)

    if uploaded_file and jd_text:
        # Save uploaded file temporarily
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Parse resume
        with st.spinner("Parsing resume..."):
            resume_text = parse_resume(temp_file, parser)
            if not resume_text:
                st.error("Failed to parse resume")
                os.remove(temp_file)
                return

        # Analyze
        with st.spinner("Analyzing with AI..."):
            analysis = analyze_resume(resume_text, jd_text, model)

        # Display results
        if analysis:
            st.subheader("Analysis Results")
            
            # Score card
            st.metric("Match Score", f"{analysis.get('match_score', 0)}%")
            
            # Columns layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Matching Skills")
                for skill in analysis.get("matching_skills", []):
                    st.success(f"- {skill}")
                
            with col2:
                st.subheader("❌ Missing Skills")
                for skill in analysis.get("missing_skills", []):
                    st.error(f"- {skill}")
            
            # Suggestions
            st.subheader("💡 Improvement Suggestions")
            for suggestion in analysis.get("suggestions", []):
                st.info(f"- {suggestion}")
            
            # Summary
            if analysis.get("summary"):
                st.subheader("📝 Summary")
                st.write(analysis["summary"])

        # Clean up
        os.remove(temp_file)

if __name__ == "__main__":
    main()