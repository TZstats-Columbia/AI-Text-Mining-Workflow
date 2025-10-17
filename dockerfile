# Use full Python 3.13 image
FROM python:3.13

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent hanging
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Copy project files
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install BERTopic with optional dependencies
RUN pip install --no-cache-dir "bertopic[all]"

# Expose Jupyter Notebook port
EXPOSE 8888

# Launch Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
