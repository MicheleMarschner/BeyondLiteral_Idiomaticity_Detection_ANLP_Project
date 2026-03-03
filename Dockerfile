FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy project metadata first
COPY pyproject.toml README.md /app/    

# Copy source code
COPY src /app/src

# maybe somewhere else
#COPY data /app/data

# Installs project’s dependencies
RUN pip install --no-cache-dir -e .

# NLTK data
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

RUN mkdir -p /app/experiments /app/results

# If using HuggingFace:
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

# Default command: show help
# TODO: WHY not entrypoint in CMD? Shouldn't display help??
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]