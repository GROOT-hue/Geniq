from flask import Flask, request, render_template, send_file, Response
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os
from gtts import gTTS
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Ensure NLTK data is available at startup
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# PyMuPDF check
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

app = Flask(__name__)

# API Key (Hugging Face)
hf_api_key = os.getenv("HF_API_KEY")

# Home page with links to functionalities
@app.route('/')
def home():
    return render_template('index.html')

# 1. Text-to-Image
@app.route('/text_to_image', methods=['GET', 'POST'])
def text_to_image():
    if request.method == 'POST':
        prompt = request.form.get('prompt', 'A futuristic city')
        if not hf_api_key:
            return "Hugging Face API key missing.", 400
        url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        payload = {"inputs": prompt}
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_io = BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')
            else:
                return f"API error: {response.status_code}", 500
        except Exception as e:
            return f"Error: {str(e)}", 500
    return render_template('text_to_image.html')

# 2. Text-to-Audio
@app.route('/text_to_audio', methods=['GET', 'POST'])
def text_to_audio():
    if request.method == 'POST':
        text = request.form.get('text', 'Hello, this is a test.')
        lang = request.form.get('lang', 'en')
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(tmp.name)
                return send_file(tmp.name, mimetype='audio/mp3', as_attachment=True, download_name='output.mp3')
        except Exception as e:
            return f"Error: {str(e)}", 500
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    return render_template('text_to_audio.html')

# 3. Summarization
@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        text = request.form.get('text', '')
        summary_sentences = int(request.form.get('summary_sentences', 2))
        if not text.strip():
            return "Please enter some text.", 400
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return "Please enter at least two sentences.", 400
        if len(sentences) <= summary_sentences:
            return "\n".join(sentences), 200
        stop_words = set(stopwords.words("english"))
        words = [w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in stop_words]
        word_freq = Counter(words)
        sentence_scores = {}
        for i, sent in enumerate(sentences):
            score = sum(word_freq[w.lower()] for w in word_tokenize(sent) if w.isalnum() and w.lower() in word_freq)
            sentence_scores[i] = score / (len(word_tokenize(sent)) + 1)
        top_sentences = sorted(sorted(sentence_scores.items(), key=lambda x: x[0])[:summary_sentences], key=lambda x: x[1], reverse=True)
        summary = [sentences[i] for i, _ in top_sentences]
        return "\n".join(summary), 200
    return render_template('summarize.html')

# 4. Code Debugger
@app.route('/debug', methods=['GET', 'POST'])
def debug():
    if request.method == 'POST':
        code = request.form.get('code', '')
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            output = StringIO()
            reporter = TextReporter(output)
            Run([tmp_path, "--reports=n"], reporter=reporter, exit=False)
            lint_output = output.getvalue()
            output.close()
            os.unlink(tmp_path)
            if lint_output.strip():
                return lint_output, 200
            return "No issues detected by pylint.", 200
        except Exception as e:
            return f"Error: {str(e)}", 500
    return render_template('debug.html')

# 5. ATS Score Checker
@app.route('/ats_score', methods=['GET', 'POST'])
def ats_score():
    if request.method == 'POST':
        if not fitz:
            return "ATS unavailable due to missing 'pymupdf'.", 400
        resume = request.files.get('resume')
        job_desc = request.form.get('job_desc', '')
        if not resume or not job_desc:
            return "Upload a resume and enter a job description.", 400
        try:
            pdf = fitz.open(stream=resume.read(), filetype="pdf")
            resume_text = "".join(page.get_text() for page in pdf)
            resume_words = set(resume_text.lower().split())
            job_words = set(job_desc.lower().split())
            common = resume_words.intersection(job_words)
            score = min(len(common) / len(job_words) * 100, 100)
            return f"ATS Score: {score:.2f}%\nMatches: {', '.join(common)}", 200
        except Exception as e:
            return f"Error: {str(e)}", 500
    return render_template('ats_score.html')

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
