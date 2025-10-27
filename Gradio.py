import re
import gradio as gr
import pandas as pd
import numpy as np
from bertopic import BERTopic
import docx
import PyPDF2
import logging
import os
import time

# -----------------------------
# ✅ Logging setup
# -----------------------------
LOG_FILE = "gradio_app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("🚀 Starting Gradio BERTopic + OpenAI app...")

# -----------------------------
# ✅ Load pretrained BERTopic model
# -----------------------------
try:
    logging.info("Loading pretrained BERTopic model...")
    topic_model = BERTopic.load("bertopic_presidents")
    logging.info(f"✅ Model loaded with {len(topic_model.get_topic_info())} topics.")
except Exception as e:
    logging.exception("❌ Error loading model.")
    raise RuntimeError(f"Error loading BERTopic model: {e}")

# -----------------------------
# ✅ Helper functions
# -----------------------------
def extract_text_from_file(file):
    if file is None:
        return ""
    filepath = getattr(file, "name", None)
    if filepath is None or not os.path.exists(filepath):
        return f"⚠️ Could not read uploaded file path: {file}"
    name = os.path.basename(filepath).lower()
    try:
        if name.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif name.endswith(".csv"):
            df = pd.read_csv(filepath)
            return df.to_string(index=False)
        elif name.endswith(".pdf"):
            text = []
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        elif name.endswith(".docx"):
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return f"⚠️ Unsupported file format: {name}"
    except Exception as e:
        return f"⚠️ Error reading file {name}: {e}"

def split_into_chunks(text, max_words=100):
    paragraphs = re.split(r"\n+", text)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) > 5:
                chunks.append(chunk)
    return chunks

def merge_files_text(files):
    all_texts = []
    for file in files:
        text = extract_text_from_file(file)
        if isinstance(text, str):
            all_texts.append(text)
    return "\n\n".join(all_texts)

def summarize_files_metadata(files):
    if not files:
        return "⚠️ No files uploaded yet."
    metadata = []
    for file in files:
        filepath = getattr(file, "name", None)
        if filepath is None or not os.path.exists(filepath):
            metadata.append(f"⚠️ Could not read file path: {file}")
            continue
        name = os.path.basename(filepath)
        size_kb = round(os.path.getsize(filepath) / 1024, 2)
        ext = os.path.splitext(name)[1].lower()
        word_count = "N/A"
        page_count = "N/A"
        try:
            if ext in [".txt", ".csv"]:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    word_count = len(content.split())
            elif ext == ".pdf":
                reader = PyPDF2.PdfReader(filepath)
                page_count = len(reader.pages)
                word_count = sum(len(page.extract_text().split()) for page in reader.pages if page.extract_text())
            elif ext == ".docx":
                doc = docx.Document(filepath)
                word_count = sum(len(p.text.split()) for p in doc.paragraphs)
        except Exception as e:
            metadata.append(f"⚠️ Error reading {name}: {e}")
            continue
        metadata.append(f"📄 **{name}** | Type: {ext} | Size: {size_kb} KB | Words: {word_count} | Pages: {page_count}")
    return "\n\n".join(metadata)

def download_metadata_csv(files):
    if not files:
        return None
    data = []
    for file in files:
        filepath = getattr(file, "name", None)
        if filepath is None or not os.path.exists(filepath):
            continue
        name = os.path.basename(filepath)
        size_kb = round(os.path.getsize(filepath) / 1024, 2)
        ext = os.path.splitext(name)[1].lower()
        word_count = None
        page_count = None
        try:
            if ext in [".txt", ".csv"]:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    word_count = len(content.split())
            elif ext == ".pdf":
                reader = PyPDF2.PdfReader(filepath)
                page_count = len(reader.pages)
                word_count = sum(len(page.extract_text().split()) for page in reader.pages if page.extract_text())
            elif ext == ".docx":
                doc = docx.Document(filepath)
                word_count = sum(len(p.text.split()) for p in doc.paragraphs)
        except Exception as e:
            continue
        data.append({
            "Filename": name,
            "Type": ext,
            "Size_KB": size_kb,
            "Word_Count": word_count,
            "Page_Count": page_count
        })
    if not data:
        return None
    df = pd.DataFrame(data)
    csv_path = "metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def analyze_multiple_files(files, num_topics):
    try:
        if not files:
            return "⚠️ Please upload at least one file."
        text = merge_files_text(files)
        if not text.strip():
            return "⚠️ Uploaded files contain no readable text."
        chunks = split_into_chunks(text)
        if len(chunks) == 0:
            return "⚠️ Input too short — try longer documents."
        topics, probs = topic_model.transform(chunks)
        if probs is None:
            probs = np.zeros((len(chunks), len(set(topics))))
        probs_mean = np.mean(probs, axis=1) if probs.ndim == 2 else probs
        topic_df = pd.DataFrame({"topic": topics, "prob": probs_mean})
        summary = (
            topic_df.groupby("topic")["prob"]
            .mean()
            .sort_values(ascending=False)
            .head(num_topics)
            .reset_index()
        )
        output_lines = []
        for _, row in summary.iterrows():
            tid = int(row["topic"])
            prob = round(row["prob"], 3)
            if tid == -1:
                output_lines.append(f"🌀 Outlier topic (−1): confidence {prob}")
            else:
                words = ", ".join([w for w, _ in topic_model.get_topic(tid)[:5]])
                output_lines.append(f"🔹 **Topic {tid}** — confidence {prob}\n   Keywords: {words}")
        if not output_lines:
            return "⚠️ No recognizable topics found."
        return "\n\n".join(output_lines)
    except Exception as e:
        logging.exception("Error during topic analysis.")
        return f"❌ **Error during topic analysis:**\n```\n{e}\n```"

def openai_prompt_analysis_with_key(files, prompt, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    text = merge_files_text(files)
    if not text.strip():
        return "⚠️ Uploaded files contain no readable text."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\n\nUse the following text:\n{text}"}
            ],
            max_tokens=500,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.exception("OpenAI API call failed.")
        return f"❌ Error calling OpenAI API:\n{e}"

# -----------------------------
# ✅ Gradio Interface
# -----------------------------
with gr.Blocks(title="🎙️ Text Analyzer (BERTopic + OpenAI)") as demo:
    gr.Markdown("## 🎙️ Text Analyzer")
    gr.Markdown(
        "Upload files, preview metadata, then either analyze topics with BERTopic or run a custom OpenAI prompt."
    )

    # File upload
    file_input = gr.Files(label="Upload your files", file_types=[".txt", ".csv", ".pdf", ".docx"])

    # Metadata preview and download
    preview_button = gr.Button("📝 Preview Metadata")
    metadata_md = gr.Markdown()
    download_button = gr.Button("💾 Download Metadata CSV")
    download_file = gr.File(label="Download CSV", file_types=[".csv"])
    preview_button.click(summarize_files_metadata, inputs=file_input, outputs=metadata_md)
    download_button.click(download_metadata_csv, inputs=file_input, outputs=download_file)

    # Choose analysis type
    analysis_type = gr.Radio(
        ["BERTopic Topic Analysis", "OpenAI Prompt Analysis"],
        label="Select Analysis Type",
        value=None  # <-- nothing selected initially
    )

    # BERTopic controls (hidden initially)
    num_topics_slider = gr.Slider(
        minimum=1, maximum=10, value=3, step=1, label="Number of Topics to Display",
        visible=False
    )

    # OpenAI prompt controls (hidden initially)
    prompt_input = gr.Textbox(
        lines=3, placeholder="Write your prompt here (e.g., summarize the files)",
        label="OpenAI Prompt",
        visible=False
    )
    api_key_input = gr.Textbox(
        type="password", placeholder="Enter your OpenAI API Key",
        label="OpenAI API Key",
        visible=False
    )

    # Output
    output_md = gr.Markdown(label="Analysis Output")

    # Analyze and Clear buttons
    analyze_button = gr.Button("🔍 Run Analysis")
    clear_button = gr.Button("🧹 Clear Result")

    # Function to handle both analysis types with progress
    def run_analysis_with_progress(files, analysis_type, num_topics, prompt, api_key, progress=gr.Progress()):
        if not files:
            return "⚠️ Please upload files first."
        if analysis_type == "BERTopic Topic Analysis":
            text = merge_files_text(files)
            chunks = split_into_chunks(text)
            total_chunks = max(len(chunks), 1)
            for i, chunk in enumerate(chunks):
                progress((i+1)/total_chunks, desc=f"Processing chunk {i+1}/{total_chunks}")
            return analyze_multiple_files(files, num_topics)
        elif analysis_type == "OpenAI Prompt Analysis":
            progress(0.5, desc="Sending request to OpenAI...")
            time.sleep(0.5)
            result = openai_prompt_analysis_with_key(files, prompt, api_key)
            progress(1.0, desc="Done")
            return result
        else:
            return "⚠️ Please select an analysis type."

    analyze_button.click(
        run_analysis_with_progress,
        inputs=[file_input, analysis_type, num_topics_slider, prompt_input, api_key_input],
        outputs=output_md
    )

    # Clear output
    def clear_output():
        return ""
    clear_button.click(clear_output, inputs=None, outputs=output_md)

    # Dynamic UI: show/hide controls based on task selection
    def toggle_controls(choice):
        if choice == "BERTopic Topic Analysis":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif choice == "OpenAI Prompt Analysis":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    analysis_type.change(
        toggle_controls,
        inputs=[analysis_type],
        outputs=[num_topics_slider, prompt_input, api_key_input]
    )

# -----------------------------
# ✅ Launch app
# -----------------------------
if __name__ == "__main__":
    demo.launch(share=True)
