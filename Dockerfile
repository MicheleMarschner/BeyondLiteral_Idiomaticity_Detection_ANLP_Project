FROM python:3.11-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python libraries
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data
COPY configs /app/configs

RUN pip install --no-cache-dir -e .


# GGF. 5. Pre-download NLP models (Crucial for reproducibility!)
# This ensures the model is inside the image and doesn't download every time you run it.
RUN python -c "import nltk; nltk.download('punkt')"
# If using HuggingFace:
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

# Default command: show help
# TODO: WHY not entrypoint in CMD? Shouldn't display help??
ENTRYPOINT ["python", "main"]
CMD ["--help"]
