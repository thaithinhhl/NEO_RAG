# NEO RAG - Retrieval Augmented Generation System
![image](https://github.com/user-attachments/assets/aec09554-7f6e-4861-9cd2-7f653883fc93)

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
