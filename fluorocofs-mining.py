# install the following library
"""
%pip install -qU langchain_community
%pip install pydantic
%pip install pandas
%pip install openai

"""

import os
import time
import pandas as pd
from typing import List
from tqdm.auto import tqdm
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader

# If you are using PyPDFLoader, make sure to import it from the correct library
# For example:
# from langchain.document_loaders import PyPDFLoader

# Initialize OpenAI client
client = OpenAI()

# Define the function use of openai API
# Define the model for each precursor and its usage
class PrecursorUsage(BaseModel):
    precursor: str
    used_for: str

# Define the model for extracting information from the paper
class ResearchPaperExtraction(BaseModel):
    Aldehyde_precursors: List[PrecursorUsage]
    Amine_precursors: List[PrecursorUsage]
    paper_title: str

def content_completion(content: str, ResearchPaperExtraction):
    """
    Uses the OpenAI API to parse the content and extract aldehyde and amine precursors used for COF synthesis.

    Args:
        content (str): The text content of the paper.
        ResearchPaperExtraction: The Pydantic model defining the expected response format.

    Returns:
        research_paper: An instance of ResearchPaperExtraction containing the extracted data.
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "Read the paper and extract the aldehyde and amine precursors used for COF synthesis. Please cite the paper title first.",
            },
            {"role": "user", "content": content},
        ],
        response_format=ResearchPaperExtraction,  # Use predefined Pydantic model as response format
    )

    # Parse the response and extract the required data
    research_paper = completion.choices[0].message.parsed

    return research_paper

# Set the source path where the papers are located
source_path = r"/Volumes/Elements/cof mining/your documents"
files = os.listdir(source_path)

# Initialize dataframes to store the extracted data
Aldehyde_precursor_used_for = pd.DataFrame(
    [], columns=["Aldehyde_precursor", "Aldehyde_precursor_used_for", "references"]
)
Amine_precursor_used_for = pd.DataFrame(
    [], columns=["Amine_precursor", "Amine_precursor_used_for", "references"]
)

num = 0  # Initialize a counter for the number of papers

# Iterate over each file in the source directory
for index, file in enumerate(tqdm(files)):
    file_path = os.path.join(source_path, file)

    if file.endswith(".pdf"):
        # If the file is a PDF, load and extract its content
        loader_PyPDF = PyPDFLoader(file_path)
        docs_PyPDF = loader_PyPDF.load()
        # Extract text content from each page
        content = [page.page_content for page in docs_PyPDF]
        content = "\n\n".join(content)
        num += 1

    elif os.path.isdir(file_path):
        # If the file is a directory, process the PDF files inside
        zongshu_files = os.listdir(file_path)
        for zongshu_file in tqdm(zongshu_files[0:6]):
            if zongshu_file.endswith(".pdf"):
                zongshu_file_path = os.path.join(file_path, zongshu_file)
                loader_PyPDF = PyPDFLoader(zongshu_file_path)
                docs_PyPDF = loader_PyPDF.load()
                content = [page.page_content for page in docs_PyPDF]
                content = "\n\n".join(content)
                num += 1
                print(f"{zongshu_file_path} has been extracted")

    # Use the OpenAI API to extract information from the content
    research_paper = content_completion(content, ResearchPaperExtraction)

    # Create temporary dataframes for aldehyde and amine precursors
    temp_Aldehyde = pd.DataFrame(
        [
            {
                "Aldehyde_precursor": item.precursor,
                "Aldehyde_precursor_used_for": item.used_for,
                "references": research_paper.paper_title,
            }
            for item in research_paper.Aldehyde_precursors
        ]
    )

    temp_Amine = pd.DataFrame(
        [
            {
                "Amine_precursor": item.precursor,
                "Amine_precursor_used_for": item.used_for,
                "references": research_paper.paper_title,
            }
            for item in research_paper.Amine_precursors
        ]
    )

    # Append the temporary dataframes to the main dataframes
    Aldehyde_precursor_used_for = pd.concat(
        [Aldehyde_precursor_used_for, temp_Aldehyde], axis=0
    )
    Amine_precursor_used_for = pd.concat(
        [Amine_precursor_used_for, temp_Amine], axis=0
    )

    # Control the frequency of API calls
    time.sleep(0.2)

# Optionally, print the total number of papers processed
# print(f"Total of {num} papers")
