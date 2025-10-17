# president-speeches-analysis

This repository contains tools and resources for analyzing U.S. Presidential inaugural speeches. The project includes data preprocessing, exploratory data analysis, and visualizations inside a Jupyter Notebook.

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
    Gradio.ipynb                                   # Jupyter Notebook for Gradio Interaction

---
## Run with Docker

- **Clone this repository**
  
      git clone https://github.com/zzziminger/president-speeches-analysis.git

- **Navigate to the folder**
  
      cd president-speeches-analysis
  
- **Build the image**

      docker build -t psa .

- **Run the container**

      docker run -p 8888:8888 psa

Then access the Jupyter Notebook at:  
[http://localhost:8888](http://localhost:8888)



## Project Workflow

Below is a **step-by-step workflow** for analyzing U.S. presidential speeches using topic modeling, clustering, and LLM-based thematic analysis. The workflow leverages **ChatGPT/OpenAI API** and Python-based preprocessing to uncover how topics (e.g., *Economy*) evolve over time.

### 1. Data Ingestion & Topic Modeling Setup
1. Download the dataset:  
   `InauguralSpeeches.zip`  
2. Unzip the file:  
   ```bash
   # path to your zipped dataset
   zip_path = "InauguralSpeeches.zip"

   # extract contents
   with zipfile.ZipFile(zip_path, 'r') as z:
       z.extractall("speeches_data")
   ```
3. Start ChatGPT (or your LLM of choice):
Prompt: Uplaod the file 'jam_2025_06_03_bassell_jam-d-25-00057_sdc1' and run the followig prompt: "Follow the workflow here to analyze the presidential speech data."

Observation:
   - When encountering errors (e.g., ValueError), paste them into ChatGPT.
   - ChatGPT should add HDBSCAN and UMAP manually to ensure clustering works.
   - ChatGPT should generate an Intertiopic Distance Map, a Topic SWord cores barchart, and a Hierarchical Clustering plot. 

### 2. Initial Topic Table
* **Prompt:**
"Create a table with the following columns: Topic, Name, Topic Definition, and Percentages of Sentences."

* **Observations**
The initial topic names were mostly generic stopwords (e.g., the, and, of).

* **Next Prompt:**
Paste the first few lines of the table into ChatGPT and prompt "Improve the table."

* **Observations**
Removing stopwords before topic modeling using nltk did not solve the issue. ChatGPT suggested an alternative approach — splitting long sentences into smaller chunks.

* **Next Prompt:**
"Split long sentences into smaller chunks for better topic detection."

* **Observations**
ChatGPT provided a Python function using .strip() and .split(). After applying this function, the resulting topic table contained meaningful topics (e.g., instead of the, and, of, topics like liberty, freedom, etc. appeared).


### 3. LLM-Based Thematic Analysis

* **Prompt:**
  "Follow this workflow to continue the LLM process: Use the OpenAI API to ask the question — *‘How does the topic of Economy evolve in U.S. politics?’* Then, draw a topic prevalence plot for every 20 years."

* **Observations and Steps:**

  1. **Add Inauguration Dates:**

     * Create a dictionary mapping each president/speech to their inauguration year.
     * Paste the list of speech names into ChatGPT and let it complete the dictionary.
     * Check for missing values.
     * If any presidents are missing, specifically ask ChatGPT for their inauguration years.

  2. **Define `summarize_economy` Function:**

     * Implement a function that uses the OpenAI API to summarize how the *Economy* topic is represented across speeches.

  3. **Keyword Identification and Grouping:**

     * Identify economy-related keywords.
     * Group speeches into 20-year intervals.
     * Generate a topic prevalence plot to visualize the evolution of economic themes over time.

* **Next Prompt:**
  "Print the summary."

* **Observation:**
  Generate a DataFrame using a `for` loop, where:

  * Column 1 = *Period*
  * Column 2 = *Summary*

### 4. Gradio

* **Prompt:**
  Paste the BERTopic-related code and ask: “How to implement Gradio for a project using BERTopic?”
* **Observation:**
  UMAP caused an error when processing a single speech instead of the entire dataset.
A single speech produced only 2–5 text chunks, so `UMAP(n_neighbors=5)` attempted to build a 5-nearest-neighbor graph on fewer than 5 samples, leading to an input size error.

* **Debugging Process:**
  1. Pasted the error traceback into ChatGPT.
  2. Solution: Pretrain BERTopic on the full dataset, then load the saved model in the Gradio app.
  3. New error occurred: the pretrained HDBSCAN model inside BERTopic was not initialized with `prediction_data=True`, preventing `.transform()` from running on new inputs.
     - Fix: Add `prediction_data=True` to the `HDBSCAN(...)` call during model training.
  4. After fixing this, the Gradio app loaded successfully, but outputs remained identical for all inputs.
     - Resolution: Added error handling, probability validation, and detailed progress logging. The app now handles irregular outputs from `BERTopic.transform()`, formats results cleanly in Markdown, and runs publicly with `share=True`.
 
* **Additional Implementations:**
  Built two additional Gradio apps using similar techniques:
  - A topic summarization app analyzing how U.S. presidents discussed the economy over a user-selected time range.
  - A custom prompt app that allows users to run OpenAI API-based analyses on inaugural speeches interactively.




Then access the Jupyter Notebook at:  
[http://localhost:8888](http://localhost:8888)
