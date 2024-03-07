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

def tokenize_data(text_data):
    # Tokenize the text data using the ChatGPT tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text_data, truncation=True, padding="max_length", max_length=max_seq_length, return_tensors="pt")

    # Convert the inputs dictionary to a list of tuples
    inputs = list(inputs.items())

    # Add special attention mask tokens
    attention_masks = [float(i != tokenizer.pad_token_id) for x in inputs[1] for i in x[0]]

    # Create a DataFrame from the inputs and attention masks
    df = pd.DataFrame({k: v.numpy().flatten() for k, v in zip(['input_ids', 'token_type_ids', 'attention_mask'], [x[1] for x in inputs])})
    df['attention_mask'] = attention_masks[0:512]
    # Create labels by shifting the input_ids one position to the right
    labels = df['input_ids'].copy()
    labels = labels.shift(1).fillna(tokenizer.pad_token_id).astype(int)


    # Return the DataFrame with the labels
    return df, labels


# Function to fine-tune the ChatGPT model
def fine_tune_model(train_df, val_df, train_labels, val_labels):
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
            batch_data = train_df.iloc[i : i + batch_size]
            input_ids = torch.tensor(batch_data["input_ids"].tolist(), dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_data["attention_mask"].tolist(), dtype=torch.long, device=device)
            labels = torch.tensor(train_labels.iloc[i:i+batch_size].tolist(), dtype=torch.long, device=device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate the loss
            loss = outputs.loss

            # Backward pass
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
        validate_model(val_df, val_labels)


# Function to validate the model
def validate_model(val_df, val_labels):
    # Switch the model to evaluation mode
    model.eval()

    # Variable to track the total loss
    total_loss = 0

    # Iterate over batches of validation data
    for i in tqdm(range(0, len(val_df), batch_size)):
        # Select a batch of input sequences and corresponding labels
        batch_data = val_df.iloc[i : i + batch_size]
        input_ids = torch.tensor(batch_data["input_ids"].tolist(), dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch_data["attention_mask"].tolist(), dtype=torch.long, device=device)
        labels = torch.tensor(val_labels.iloc[i:i+batch_size].tolist(), dtype=torch.long, device=device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

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

    # Tokenize the text data
    data, labels = tokenize_data(full_text)
    # Split the data into training and validation sets
    train_df, val_df, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, random_state=42)


    # Fine-tune the model on the training data
    fine_tune_model(train_df, val_df, train_labels, val_labels)
