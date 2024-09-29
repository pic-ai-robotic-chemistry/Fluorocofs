# FluoroCOFs

This repository contains two parts:

1. COF Mining: Scripts to mine COFs from the literature and store them in a database.
2. COF Prediction: Scripts to predict COFs using Siamese neural networks and optimization algorithms.


## How to Use

### 1. Install Requirements

Clone the repository and install the required Python libraries:

```bash
git clone https://github.com/pic-ai-robotic-chemistry/Fluorocofs.git
cd fluorocofs
pip install -r requirements.txt
```

### 2. Run the Script

Place your PDF files in the specified directory and run the extraction script:

```bash
python cof_mining/fluorocofs-mining.py
```

Or

```bash
python cof_recomendation/example.py
```

### 3. Customize
Modify the source paths and model prompt settings in the script to suit your specific needs for PDF extraction and precursor analysis.