# NEO RAG - Retrieval Augmented Generation System

## Overview
NEO RAG is a Retrieval Augmented Generation (RAG) system that enhances language model responses by retrieving relevant information from a knowledge base. The system is designed to provide accurate and contextually relevant answers by combining the power of language models with efficient information retrieval.

## Project Structure
```
NEO_RAG/
├── data/                  # Directory containing data files
├── src/                   # Source code directory
│   ├── data_processors/   # Data processing and preparation modules
│   ├── utils/            # Utility functions and helper modules
│   ├── embeddings/       # Text embedding generation and management
│   ├── database/        # Database operations and management
│   ├── models/          # Language model integration and management
│   └── retrieval/       # Information retrieval system components
├── requirements.txt      # Python dependencies
└── .python-version      # Python version specification
```

## Components

### Data Processors (`src/data_processors/`)
- Handles data ingestion, cleaning, and preprocessing
- Prepares documents for embedding and storage
- Manages data formats and transformations

### Utils (`src/utils/`)
- Common utility functions
- Helper modules for various operations
- Shared tools and constants

### Embeddings (`src/embeddings/`)
- Text embedding generation
- Vector storage and management
- Embedding model integration

### Database (`src/database/`)
- Database operations and management
- Vector store integration
- Data persistence and retrieval

### Models (`src/models/`)
- Language model integration
- Model management and configuration
- Response generation

### Retrieval (`src/retrieval/`)
- Information retrieval system
- Query processing
- Context assembly

## Setup and Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
[To be added based on specific usage instructions]

## Dependencies
See `requirements.txt` for a complete list of dependencies.

## License
[To be added based on your license choice]

## Contributing
[To be added based on contribution guidelines]

# NEO_RAG Setup Guide

## Local Development Setup

1. Clone repository:
```bash
git clone <repository_url>
cd NEO_RAG
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis server:
```bash
redis-server
```

3. Run the application:
```bash
python gradio_app.py
```

## Server Deployment

1. SSH into your server:
```bash
ssh username@server_ip
```

2. Clone the repository:
```bash
git clone <repository_url>
cd NEO_RAG
```

3. Make setup script executable and run:
```bash
chmod +x setup_server.sh
./setup_server.sh
```

4. Configure environment variables:
- Edit ~/.bashrc
- Set appropriate values for:
  - OPENAI_API_KEY
  - MODEL_PATH
  - DATA_PATH

5. Apply changes:
```bash
source ~/.bashrc
```

## Project Structure
```
NEO_RAG/
├── data/           # Data storage
├── logs/           # Log files
├── models/         # Model files
├── venv/           # Virtual environment
├── requirements.txt
└── setup_server.sh
```

## Important Notes
- Always test changes locally before deploying to server
- Keep environment variables secure
- Regular backups of data directory recommended
- Check logs for any issues after deployment

# Legal Chatbot

A legal chatbot built with Gradio and LangChain.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis server:
```bash
redis-server
```

3. Run the application:
```bash
python gradio_app.py
```

## Deployment

### Option 1: Temporary Public URL
Run the app locally and Gradio will generate a public URL valid for 72 hours:
```bash
python gradio_app.py
```

### Option 2: Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Choose Gradio as the SDK
3. Upload the following files:
   - gradio_app.py
   - requirements.txt
   - All files in src/ directory
4. Configure environment variables if needed
5. Deploy!