# FluoroCOFs

This repository contains the scripts used for extracting aldehyde and amine precursors from PDFs using large language models (LLMs), specifically OpenAI's GPT models. The workflow automates the process of mining relevant chemical precursors from research papers, particularly for Covalent Organic Framework (COF) synthesis research.

## Overview

The script performs the following tasks:

1. **PDF Parsing**  
   The PDFs are parsed using `PDFminer` from LangChain. The text from multiple pages is extracted and then rejoined into a coherent format for further processing.

2. **Function Call / Structured Output**  
   The extraction of aldehyde and amine precursors is handled via OpenAI’s API, leveraging a structured output format. This format is defined using `pydantic` models to meet OpenAI's API guidelines, ensuring precise and structured data extraction.

3. **Custom GPT-4o-Mini Model Prompt Design**  
   A custom prompt is designed for the `GPT-4o` model. The prompt directs the model to read the research paper, identify aldehyde and amine precursors, and return the results in a structured format, including the paper title for referencing. This approach is tailored to optimize accuracy and relevance in COF synthesis research.

## Components

- **PDF Parsing**  
   The PDFs are loaded and parsed using `PyPDFLoader` from LangChain, which effectively handles scientific text extraction from PDFs.

- **Model Integration**  
   The script integrates the GPT-4o-mini model from OpenAI with function calls designed to extract aldehyde and amine precursors. The structured output format is defined using `pydantic`, ensuring the response follows a well-defined schema.

- **Response Format**  
   The structured response format includes fields for:
   - Aldehyde precursors
   - Amine precursors
   - Research paper title  
   The extracted data is stored in pandas DataFrames for further analysis.

## How to Use

### 1. Install Requirements

Clone the repository and install the required Python libraries:

```bash
git clone https://github.com/your-repository/fluorocofs.git
cd fluorocofs
pip install -r requirements.txt
```

### 2. Run the Script

Place your PDF files in the specified directory and run the extraction script:

```bash
python fluorocofs-mining.py
```
### 3. Customize
Modify the source paths and model prompt settings in the script to suit your specific needs for PDF extraction and precursor analysis.
