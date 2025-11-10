import re
import gradio as gr
import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import docx
import PyPDF2
import logging
import os
import time
from collections import Counter, defaultdict


# -----------------------------
# ✅ Helpers
# -----------------------------
def ensure_list(x):
    if isinstance(x, list):
        return x
    elif x is None:
        return []
    else:
        return [x]

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
    files = ensure_list(files)
    all_texts = []
    for file in files:
        text = extract_text_from_file(file)
        if isinstance(text, str):
            all_texts.append(text)
    return "\n\n".join(all_texts)

def summarize_files_metadata(files):
    files = ensure_list(files)
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
    files = ensure_list(files)
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
        except Exception:
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

# -----------------------------
# ✅ BERTopic Logic
# -----------------------------
def parse_seed_words(seed_words_text):
    seed_topic_list = []
    for line in seed_words_text.strip().split("\n"):
        words = [w.strip() for w in line.split(",") if w.strip()]
        if words:
            seed_topic_list.append(words)
    return seed_topic_list

def train_or_load_topic_model(seed_words_text, uploaded_chunks):
    try:
        seed_words_text = (seed_words_text or "").strip()

        # 1️⃣ Load pretrained model
        if not os.path.exists("bertopic_presidents"):
            return "⚠️ Pretrained model not found. Please ensure 'bertopic_presidents' exists."
        base_model = BERTopic.load("bertopic_presidents")

        # CASE 1: No seed words → just transform uploaded chunks
        if not seed_words_text:
            logging.info("🧠 Using pretrained model without seeds.")
            topics, probs = base_model.transform(uploaded_chunks)
            topic_info = base_model.get_topic_info()
            return base_model, topic_info, topics

        # CASE 2: Seed words provided → fine-tune on full pretrained corpus
        logging.info("🌱 Fine-tuning pretrained BERTopic with seed words...")

        # 2️⃣ Load full original corpus (must exist)
        if not os.path.exists("chunks.csv"):
            return "⚠️ Original training chunks CSV not found ('chunks.csv')."
        df_chunks = pd.read_csv("chunks.csv")
        all_docs = df_chunks["speech_chunk"].tolist()

        # 3️⃣ Parse seed words
        seed_topic_list = parse_seed_words(seed_words_text)

        # 4️⃣ Reuse pretrained components
        vectorizer_model = base_model.vectorizer_model
        umap_model = base_model.umap_model
        hdbscan_model = base_model.hdbscan_model

        # 5️⃣ Initialize new BERTopic with seeds
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=KeyBERTInspired(),
            seed_topic_list=seed_topic_list,  # ✅ seeds go here
            nr_topics="auto",
            min_topic_size=30
        )

        # 6️⃣ Fit on full original corpus
        topics, probs = topic_model.fit_transform(all_docs)

        # 7️⃣ Save fine-tuned model
        topic_model.save("bertopic_presidents_seeded")

        # 8️⃣ Transform uploaded chunks
        new_topics, _ = topic_model.transform(uploaded_chunks)
        topic_info = topic_model.get_topic_info()

        logging.info(f"✅ Fine-tuned model trained on full corpus with {len(seed_topic_list)} seed topics, "
                     f"produced {len(topic_info)} total topics.")

        return topic_model, topic_info, new_topics

    except Exception as e:
        logging.exception("Error during BERTopic fine-tuning.")
        return f"❌ Error during BERTopic fine-tuning:\n{e}"



def summarize_topics_with_chunk_counts(topic_model, topic_info, topics, chunks, num_topics=10):
    """
    Summarize topics sorted by relevance in uploaded chunks,
    showing number of chunks and average words per chunk.
    """
    topic_counts = Counter([t for t in topics if t != -1])
    if not topic_counts:
        return "⚠️ No meaningful topics found in uploaded text."

    # Count total non-outlier topics
    total_topics = len([t for t in topic_info["Topic"] if t != -1])

    # Map topic ID -> list of word counts of assigned chunks
    topic_chunk_word_counts = defaultdict(list)
    for t, chunk in zip(topics, chunks):
        if t != -1:
            topic_chunk_word_counts[t].append(len(chunk.split()))

    # Sort topics by frequency in uploaded text
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    # Start output with total topics summary
    output_lines = [f"### 🧩 Total Topics Found: {total_topics}\n"]

    for tid, count in sorted_topics[:num_topics]:
        topic_name = topic_info.loc[topic_info["Topic"] == tid, "Name"].values[0]
        words = ", ".join([w for w, _ in topic_model.get_topic(tid)[:5]])
        word_counts = topic_chunk_word_counts[tid]
        avg_words = round(np.mean(word_counts))
        output_lines.append(
            f"🔹 **Topic {tid}** — {topic_name} "
            f"({count} chunks, avg. {avg_words} words per chunk)\n"
            f"   Keywords: {words}"
        )

    return "\n\n".join(output_lines)


def train_and_analyze(files, seed_words_text, num_topics):
    files = ensure_list(files)
    if not files:
        return "⚠️ Please upload at least one file."
    text = merge_files_text(files)
    chunks = split_into_chunks(text)
    if len(chunks) == 0:
        return "⚠️ Uploaded files contain no readable text."
    result = train_or_load_topic_model(seed_words_text, chunks)
    if isinstance(result, str):
        return result
    topic_model, topic_info, topics = result
    return summarize_topics_with_chunk_counts(topic_model, topic_info, topics, chunks, num_topics)

# -----------------------------
# ✅ OpenAI Prompt Analysis
# -----------------------------
def openai_prompt_analysis_with_key(files, prompt, api_key):
    from openai import OpenAI
    files = ensure_list(files)
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
    gr.Markdown("Upload files, preview metadata, then either analyze topics with BERTopic or run a custom OpenAI prompt.")

    file_input = gr.Files(label="Upload your files", file_types=[".txt", ".csv", ".pdf", ".docx"])
    preview_button = gr.Button("📝 Preview Metadata")
    metadata_md = gr.Markdown()
    download_button = gr.Button("💾 Download Metadata CSV")
    download_file = gr.File(label="Download CSV", file_types=[".csv"])
    preview_button.click(summarize_files_metadata, inputs=file_input, outputs=metadata_md)
    download_button.click(download_metadata_csv, inputs=file_input, outputs=download_file)

    analysis_type = gr.Radio(
        ["BERTopic Topic Analysis", "OpenAI Prompt Analysis"],
        label="Select Analysis Type",
        value=None
    )

    num_topics_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Topics", visible=False)
    prompt_input = gr.Textbox(lines=3, placeholder="Write your prompt here", label="OpenAI Prompt", visible=False)
    api_key_input = gr.Textbox(type="password", placeholder="Enter your OpenAI API Key", label="OpenAI API Key", visible=False)
    seed_input = gr.Textbox(lines=4, placeholder="Enter seed words (each line: comma-separated keywords)", label="Seed Words (Optional)", visible=False)

    output_md = gr.Markdown(label="Analysis Output")
    analyze_button = gr.Button("🔍 Run Analysis")
    clear_button = gr.Button("🧹 Clear Result")

    def run_analysis_with_progress(files, analysis_type, num_topics, prompt, api_key, seed_words, progress=gr.Progress()):
        files = ensure_list(files)
        if not files:
            return "⚠️ Please upload files first."
        if analysis_type == "BERTopic Topic Analysis":
            progress(0.3, desc="Training or loading BERTopic model...")
            result = train_and_analyze(files, seed_words, num_topics)
            progress(1.0, desc="Done")
            return result
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
        inputs=[file_input, analysis_type, num_topics_slider, prompt_input, api_key_input, seed_input],
        outputs=output_md
    )

    clear_button.click(lambda: "", None, output_md)

    def toggle_controls(choice):
        if choice == "BERTopic Topic Analysis":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        elif choice == "OpenAI Prompt Analysis":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    analysis_type.change(
        toggle_controls,
        inputs=[analysis_type],
        outputs=[num_topics_slider, prompt_input, api_key_input, seed_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)
