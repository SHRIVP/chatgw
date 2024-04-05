# Function to extract text from a PDF file
import subprocess
import time
import re
import shutil
import string

# Function to clean and format the text data
def preprocess_text(text):
    print("Cleaning text from pdf")
    # Remove non-alphanumeric characters except whitespace
    text = re.sub(f"[^{string.whitespace}\\w]+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Replace consecutive whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)

    # Return the cleaned and formatted text
    return text

def extract_text_from_pdf(file_path):
    print("Reading from PDF")
    # Use pdftotext to extract the text from the PDF file
    command = f"pdftotext -layout {file_path} -"
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, _ = proc.communicate()

    # Decode the output from bytes to string
    text = output.decode().strip()

    # Clean and format the text data
    text = preprocess_text(text)

    # Return the cleaned and formatted text
    return text