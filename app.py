import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# Download NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------

st.set_page_config(
    page_title="SmartHire | Resume Screening",
    page_icon="💼",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Header */
.hero {
    padding: 25px 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f4e79, #2878b5);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    margin-top: 5px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Info cards */
.info-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e6e9ef;
    margin-bottom: 15px;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: 600;
}

/* Score */
.score-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: white;
    border: 1px solid #e6e9ef;
    margin-top: 20px;
}

.score-box h2 {
    font-size: 42px;
    margin: 5px;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>💼 SmartHire</h1>
    <p>AI-Powered Resume Screening & Job Matching System</p>
    <p style="font-size:15px;">
        Analyze how closely a resume matches a job description using NLP,
        TF-IDF and Cosine Similarity.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("💼 SmartHire")

    st.info("""
    SmartHire helps you analyze a resume against a job description.

    **Features**
    - 📄 PDF Resume Analysis
    - 🔍 Text Processing
    - 🧠 NLP Techniques
    - 📊 Match Score
    - 🎯 Job Matching
    """)

    st.subheader("⚙️ How It Works")

    st.write("""
    **1.** Upload your resume

    **2.** Paste the job description

    **3.** Click Analyze Match

    **4.** View your match score
    """)

    st.divider()

    st.caption("Developed for College Project")
    st.caption("Created by Karan Dev")


# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

st.markdown(
    '<div class="section-title">📋 Resume & Job Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# Resume Upload
with col1:

    st.markdown(
        '<div class="info-card"><h3>📄 Upload Resume</h3>'
        '<p>Upload your resume in PDF format.</p></div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose your resume",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} uploaded successfully")


# Job Description
with col2:

    st.markdown(
        '<div class="info-card"><h3>💼 Job Description</h3>'
        '<p>Paste the job requirements below.</p></div>',
        unsafe_allow_html=True
    )

    job_description = st.text_area(
        "Job Description",
        height=180,
        placeholder="Example: Python developer with experience in Machine Learning, SQL, Pandas and Scikit-learn...",
        label_visibility="collapsed"
    )


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def extract_text_from_pdf(uploaded_file):

    try:

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        return text

    except Exception as e:

        st.error(f"Error reading PDF: {e}")

        return ""


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words("english"))

    words = word_tokenize(text)

    return " ".join(
        [word for word in words if word not in stop_words]
    )


def calculate_similarity(resume_text, job_description):

    resume_processed = remove_stopwords(
        clean_text(resume_text)
    )

    job_processed = remove_stopwords(
        clean_text(job_description)
    )

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [resume_processed, job_processed]
    )

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0] * 100

    return (
        round(score, 2),
        resume_processed,
        job_processed
    )


def extract_keywords(text, num_keywords=10):

    words = word_tokenize(text)

    words = [
        w for w in words
        if len(w) > 2
    ]

    tagged_words = pos_tag(words)

    nouns = [
        w for w, pos in tagged_words
        if pos.startswith("NN") or pos.startswith("JJ")
    ]

    word_freq = Counter(nouns)

    return word_freq.most_common(num_keywords)


# -------------------------------------------------
# ANALYZE BUTTON
# -------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "🚀 Analyze Resume Match"
)


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

if analyze:

    if not uploaded_file:

        st.warning("⚠️ Please upload your resume PDF first.")

    elif not job_description:

        st.warning("⚠️ Please paste the job description first.")

    else:

        with st.spinner("🔍 Analyzing your resume..."):

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

            if not resume_text:

                st.error(
                    "❌ Could not extract text from this PDF. "
                    "Please try another PDF."
                )

            else:

                similarity_score, resume_processed, job_processed = (
                    calculate_similarity(
                        resume_text,
                        job_description
                    )
                )

                # -------------------------------------------------
                # RESULTS
                # -------------------------------------------------

                st.markdown(
                    '<div class="section-title">📊 Analysis Results</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="score-box">
                        <p>Resume–Job Match Score</p>
                        <h2>{similarity_score:.2f}%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # -------------------------------------------------
                # MATCH BAR
                # -------------------------------------------------

                fig, ax = plt.subplots(figsize=(8, 0.7))

                if similarity_score < 40:

                    bar_color = "#ff4b4b"

                elif similarity_score < 70:

                    bar_color = "#ffa726"

                else:

                    bar_color = "#0f9d58"

                ax.barh(
                    [0],
                    [similarity_score],
                    color=bar_color,
                    height=0.5
                )

                ax.set_xlim(0, 100)

                ax.set_yticks([])

                ax.set_xlabel("Match Percentage")

                ax.set_title(
                    "Resume vs Job Description"
                )

                st.pyplot(fig)


                # -------------------------------------------------
                # RESULT MESSAGE
                # -------------------------------------------------

                if similarity_score < 40:

                    st.warning(
                        "🔴 Low Match — Consider tailoring "
                        "your resume more closely to the job requirements."
                    )

                elif similarity_score < 70:

                    st.info(
                        "🟠 Good Match — Your resume aligns "
                        "reasonably well with the job description."
                    )

                else:

                    st.success(
                        "🟢 Excellent Match — Your resume strongly "
                        "aligns with the job requirements."
                    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("""
<div class="footer">
    <b>SmartHire</b> — AI-Powered Resume Screening System<br>
    Built using Python • Streamlit • NLP • Machine Learning
</div>
""", unsafe_allow_html=True)