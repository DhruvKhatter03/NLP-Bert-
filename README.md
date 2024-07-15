# PDF Text Extraction and RAPTOR Indexing Project

This project focuses on extracting text from PDF files, organizing them into a structured directory, and creating a RAPTOR indexing system for efficient text retrieval.

## Project Overview

The project involves extracting content from selected textbooks in PDF format, chunking the text, embedding it using SBERT (Sentence-BERT), and creating a hierarchical RAPTOR index for fast and scalable retrieval. It integrates natural language processing techniques to enable advanced querying and question-answering capabilities on the indexed text data.

## Features

- Extract text from PDF files.
- Chunk and embed text using SBERT for vector representations.
- Implement RAPTOR indexing with Gaussian Mixture Models (GMMs) for hierarchical clustering.
- Implement query expansion techniques for enhanced retrieval.
- Integrate a language model for question answering on the indexed data.

## Directory Structure

project/
│
├── data/
│ ├── textbooks/
│ │ ├── textbook1.pdf
│ │
│ │ 
│ └── outputs/
│ ├── embeddings/
│ └── raptor_index/
│
├── scripts/
│ ├── extract_text.py
│ ├── chunk_embed.py
│ ├── cluster_summarize.py
│ ├── milvus_operations.py
│ ├── retrieval.py
│ └── question_answering.py
│
├── main.py
├── README.md
└── requirements.txt

- **data/**: Directory for storing input PDF files and output data.
  - **textbooks/**: Contains the PDF files to be processed.
  - **outputs/**:
    - **embeddings/**: Stores embedded vectors of chunked text.
    - **raptor_index/**: Stores the RAPTOR index data.

- **scripts/**: Contains Python scripts for different stages of the project workflow.
- **main.py**: Main entry point for running the project.
- **README.md**: This file, containing project overview, setup instructions, and usage details.
- **requirements.txt**: List of Python dependencies for easy installation.

## Setup Instructions

### Prerequisites

- Python 3.6+
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DhruvKhatter03/NLP-Bert-.git
   cd your-repository
   
2. pip install -r requirements.txt
3. Usage
Place your PDF files in data/textbooks/.

Run the main script to extract text, embed chunks, and create the RAPTOR index:

4. bash
Copy code
5. python main.py,
Follow the prompts or integrate your own queries for text retrieval and question answering.

6. Dependencies
PyMuPDF,
Sentence Transformers (SBERT),
Scikit-learn,
Milvus,
Other dependencies listed in requirements.txt
