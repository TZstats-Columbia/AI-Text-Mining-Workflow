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

## Getting Started

1. **Clone the repository**

    git clone <your-repo-url>
    cd <your-repo-name>

2. **Install dependencies** 

    pip install -r requirements.txt

3. **Extract the dataset**

    unzip InauguralSpeeches.zip -d speeches_data

4. **Run the notebook**

    jupyter notebook main.ipynb

---

## Run with Docker

Alternatively, you can run the project inside Docker.

- **Build the image**

      docker build -t psa .

- **Run the container**

      docker run -p 8888:8888 psa

Then access the Jupyter Notebook at:  
[http://localhost:8888](http://localhost:8888)
