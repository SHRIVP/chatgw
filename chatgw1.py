import os
import random
import re
import shutil
import string
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import AutoModelForMaskedLM, AdamW, BertTokenizerFast
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Path to the directory containing the PDF files
pdf_dir = "/Users/pushkar/projects/chatbot/data"

# Path to the output directory where the fine-tuned model and log files will be saved
output_dir = "/Users/pushkar/projects/chatbot/logs"
os.makedirs(output_dir, exist_ok=True)

# Maximum number of tokens allowed per batch
max_seq_length = 512

# Batch size
batch_size = 8

# Learning rate
learning_rate = 1e-5

# Number of epochs
num_epochs = 3
model_name = 'gpt2'

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)


# Initialize the model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

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

# Function to extract text from a PDF file
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

# Function to tokenize the text data
def tokenize_data(text_data):
    # Tokenize the text data using the ChatGPT tokenizer
    inputs = tokenizer(text_data, truncation=True, padding="max_length", max_length=max_seq_length, return_tensors="pt")
    
    # Convert the inputs dictionary to a list of tuples
    inputs = list(inputs.items())
    
    # Add special attention mask tokens
    attention_masks = [[float(i != 1)] * len(x[0]) for x in inputs[1]['input_ids']]
    
    # Create a DataFrame from the inputs and attention masks
    df = pd.DataFrame({k: v for k, v in zip(['input_ids', 'token_type_ids', 'attention_mask'], inputs)}).astype(int)
    df['attention_mask'] = attention_masks
    
    # Return the DataFrame
    return df

# Function to fine-tune the ChatGPT model
def fine_tune_model(train_df, val_df):
    # Define the loss function and optimizer
    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Shuffle the training data
        train_df = train_df.sample(frac=1, random_state=42)
        
        # Reset the gradients
        optimizer.zero_grad()
        
        # Variables to track the total loss and gradient updates
        total_loss = 0
        num_updates = 0
        
        # Iterate over batches of training data
        for i in tqdm(range(0, len(train_df), batch_size)):
            # Select a batch of input sequences and corresponding labels
            batch_inputs = train_df.iloc[i : i + batch_size][["input_ids", "token_type_ids", "attention_mask"]].values
            
            # Convert the batch inputs to Tensor objects
            batch_inputs = tuple([torch.tensor(x, dtype=torch.long, device=device) for x in batch_inputs])
            
            # Feed the batch inputs through the model
            outputs = model(*batch_inputs)
            
            # Calculate the loss
            loss = outputs.loss
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Track the total loss and gradient updates
            total_loss += loss.item()
            num_updates += 1
        
        # Print the average loss for the epoch
        avg_loss = total_loss / num_updates
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Validate the model
        validate_model(val_df)

# Function to validate the model
def validate_model(val_df):
    # Switch the model to evaluation mode
    model.eval()
    
    # Variable to track the total loss
    total_loss = 0
    
    # Iterate over batches of validation data
    for i in tqdm(range(0, len(val_df), batch_size)):
        # Select a batch of input sequences and corresponding labels
        batch_inputs = val_df.iloc[i : i + batch_size][["input_ids", "token_type_ids", "attention_mask"]].values
        
        # Convert the batch inputs to Tensor objects
        batch_inputs = tuple([torch.tensor(x, dtype=torch.long, device=device) for x in batch_inputs])
        
        # Feed the batch inputs through the model
        with torch.no_grad():
            outputs = model(*batch_inputs)
        
        # Calculate the loss
        loss = outputs.loss
        
        # Track the total loss
        total_loss += loss.item()
    
    # Print the average loss for the validation set
    avg_loss = total_loss / (len(val_df) / batch_size)
    print(f"Validation Loss: {avg_loss:.4f}")
    
    # Switch the model back to training mode
    model.train()

# Main function
if __name__ == "__main__":
    # Extract text from each PDF file in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    text_data = []
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        text_data.append(extract_text_from_pdf(file_path))
    
    # Concatenate the text data from all PDF files
    full_text = " ".join(text_data)
    print(full_text)
    print("Start Tokenize")
    data = tokenize_data(full_text)
    print(data)