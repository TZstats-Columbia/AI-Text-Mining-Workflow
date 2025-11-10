# president-speeches-analysis

This repository contains tools and resources for analyzing U.S. Presidential inaugural speeches. The project includes data preprocessing, exploratory data analysis, and visualizations inside a Jupyter Notebook, as well as a Gradio app for topic modeling and OpenAI-powered analysis.

---

## Project Structure

    speeches_data/                                 # Extracted speeches: processed text files + cleaned speeches dataset (filename + president + speech_num + year + speech)
    dockerfile                                     # Docker configuration to run the project in a container
    InauguralSpeeches.zip                          # Zipped dataset of speeches
    main.ipynb                                     # Notebook for preprocessing, EDA, BERTopic training
    README.md                                      # Project documentation
    requirements.txt                               # Python dependencies
    bertopic_presidents                            # Pretrained BERTopic model (full corpus)
    bertopic_presidents_seeded                     # Seeded/fine-tuned BERTopic model
    chunks.csv                                     # Speech chunks (preprocessed for BERTopic)
    Gradio.py                                      # Notebook for Gradio interface


---
## Features
- Upload multiple text files (TXT, CSV, PDF, DOCX)
- Preview metadata and download CSV of file details
- BERTopic Topic Analysis: identify and explore topics in uploaded texts
- Seed word guidance: narrow topics based on user-provided keywords
- OpenAI Prompt Analysis: run custom GPT-4 prompts on uploaded texts


---
## Run with Docker

- **Clone this repository**
  
      git clone https://github.com/TZstats-Columbia/president-speeches-analysis.git

- **Navigate to the folder**
  
      cd president-speeches-analysis
  
- **Build the image**

      docker build -t psa .

- **Run the container**

      docker run -p 8888:8888 psa

Then access the Jupyter Notebook at:  
[http://localhost:8888](http://localhost:8888)

---
## Run locally

- **Clone this repository**
  
      git clone https://github.com/TZstats-Columbia/president-speeches-analysis.git

- **Navigate to the folder**
  
      cd president-speeches-analysis

- **Create a virtual environment**
  
      python3.13 -m venv venv
      source venv/bin/activate

- **Install required packages**
  
      pip install -r requirements.txt
  
- **Run the project**
  
      python Gradio.py
      python main.ipynb

---
## Workflow Overview

**1. Preprocessing & Chunking**
- Raw speeches are extracted from `InauguralSpeeches.zip` into `speeches_data/`.

- Each speech is split into 100-word chunks for better topic modeling.

- Chunks are saved in `chunks.csv`

**2. BERTopic Training**
- Custom vectorizer: `CountVectorizer(stop_words="english", min_df=2)`

- Dimensionality reduction: `UMAP (n_neighbors=5, n_components=5, metric='cosine')`

- Clustering: `HDBSCAN (min_cluster_size=5, min_samples=2)`

- Model is trained on `chunks.csv` to produce `bertopic_presidents`.

Visualization options in `main.ipynb`:
- `visualize_topics()`
- `visualize_barchart()`
- `visualize_hierarchy()`

**3. Handling Seed Words**

The Gradio app supports user-provided seed words to narrow and guide topic generation. The workflow is as follows:

a. Load pretrained BERTopic model (`bertopic_presidents`) trained on the full corpus of speech chunks. (The model is trained and saved in 'main.ipynb')

b. Parse seed words from the Gradio interface. Each line represents a seed topic with comma-separated keywords.

c. Fine-tune on the full original corpus (`chunks.csv`) using the pretrained model:
  - Reuse components from the pretrained model: `vectorizer_model`; `umap_model; `hdbscan_model`
  - Pass the seed topics via the `seed_topic_list` argument in the constructor
  - Fits on the full corpus while merging topics guided by seeds
    
d. Transform uploaded user files to assign them to the refined topics.

**4. OpenAI Prompt Analysis**
- Users can upload files and provide a custom prompt.
- The app sends text (up to ~1500 characters per chunk) to the OpenAI API and returns a GPT-generated summary.
- Example: summarizing economic topics across speeches.


---
## Using Gradio Interface
1. Upload one or multiple files (`.txt`, `.csv`, `.pdf`, `.docx`).
2. Preview file metadata (size, words/pages).
3. Choose analysis type:
   - BERTopic Topic Analysis → optionally add seed words, select number of topics
   - OpenAI Prompt Analysis → enter API key and prompt
4. Click Run Analysis to view results.
5. Optionally, download metadata CSV.



  

