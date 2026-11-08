# RAG Project

A **Retrieval-Augmented Generation (RAG)** application that combines document retrieval with large language models to provide accurate, context-aware responses.

## Overview

RAG enhances LLM responses by retrieving relevant documents from a knowledge base before generating answers — reducing hallucinations and improving factual accuracy.

## Project Structure

```
RAG Project/
│
├── app.py            # Main application entry point
├── README.md         # Project documentation
├── requirements.txt  # Python dependencies
└── .gitignore        # Git ignore rules
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd RAG-Project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

```bash
python app.py
```

## License

MIT License
