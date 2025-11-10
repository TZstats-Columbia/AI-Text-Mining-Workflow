# president-speeches-analysis

This repository contains tools and resources for analyzing U.S. Presidential inaugural speeches. The project includes data preprocessing, exploratory data analysis, and visualizations inside a Jupyter Notebook, as well as a Gradio app for topic modeling and OpenAI-powered analysis.

---

## Project Structure

    speeches_data/                                 # Extracted speeches (raw/processed text files + cleaned csv file)
    dockerfile                                     # Docker configuration to run the project in a container
    InauguralSpeeches.zip                          # Zipped dataset of speeches
    main.ipynb                                     # Jupyter Notebook for analysis
    README.md                                      # Project documentation
    requirements.txt                               # Python dependencies
    jam_2025_06_03_bassell_jam-d-25-00057_sdc1     # Appendix file describing the data pipeline 
    bertopic_presidents                            # Trained modles
    bertopic_presidents_seeded                     # Fine-tuned BERTopic model with seed words
    chunks.csv                                     # Full corpus of speech chunks used for training
    Gradio.py                                      # Gradio Interaction

---
## Features
- Upload multiple text files (TXT, CSV, PDF, DOCX)
- Preview metadata and download CSV of file details
- BERTopic Topic Analysis: identify and explore topics in uploaded texts
- Seed word guidance: narrow topics based on user-provided keywords
- OpenAI Prompt Analysis: run custom GPT-4 prompts on uploaded texts



---
## How Seed Words Are Handled

The Gradio app supports user-provided seed words to narrow and guide topic generation. The workflow is as follows:

1. Load pretrained BERTopic model (`bertopic_presidents`) trained on the full corpus of speech chunks. (The model is trained and saved in 'main.ipynb')
2. Parse seed words from the Gradio interface. Each line represents a seed topic with comma-separated keywords.
3. Fine-tune on the full original corpus (`chunks.csv`) using the pretrained model:
  - Reuse components from the pretrained model: `vectorizer_model`; `umap_model; `hdbscan_model`
  - Pass the seed topics via the `seed_topic_list` argument in the constructor
  - Fits on the full corpus while merging topics guided by seeds
4. Transform uploaded user files to assign them to the refined topics.

 
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

  

