# president-speeches-analysis

This repository contains tools and resources for analyzing U.S. Presidential inaugural speeches. The project includes data preprocessing, exploratory data analysis, and visualizations inside a Jupyter Notebook.

---

## Project Structure

    speeches_data/           # Extracted speeches (raw/processed text files)
    dockerfile               # Docker configuration to run the project in a container
    InauguralSpeeches.zip    # Zipped dataset of speeches
    main.ipynb               # Jupyter Notebook for analysis
    README.md                # Project documentation
    requirements.txt         # Python dependencies

---

## Project Workflow

Below is a **step-by-step workflow** for analyzing U.S. presidential speeches using topic modeling, clustering, and LLM-based thematic analysis. The workflow leverages **ChatGPT/OpenAI API** and Python-based preprocessing to uncover how topics (e.g., *Economy*) evolve over time.

### 1. Data Ingestion & Topic Modeling Setup
1. Download the dataset:  
   `jam_2025_06_03_bassell_jam-d-25-00057_sdc1`  
2. Unzip the file:  
   ```bash
   unzip jam_2025_06_03_bassell_jam-d-25-00057_sdc1.zip
   ```
3. Start ChatGPT (or your LLM of choice):
   - Uplaod the file 'jam_2025_06_03_bassell_jam-d-25-00057_sdc1' and run the followig prompt: Follow the workflow here to analyze the presidential speech data.
   - When encountering errors (e.g., ValueError), paste them into ChatGPT.
   - ChatGPT should add HDBSCAN and UMAP manually to ensure clustering works.

### 2. Initial Topic Table




   

## Run with Docker

Alternatively, you can run the project inside Docker.

- **Build the image**

      docker build -t psa .

- **Run the container**

      docker run -p 8888:8888 psa

Then access the Jupyter Notebook at:  
[http://localhost:8888](http://localhost:8888)
