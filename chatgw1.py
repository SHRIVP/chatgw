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
max_seq_length = 4

# Batch size
batch_size = 8

# Learning rate
learning_rate = 1e-5

# Number of epochs
num_epochs = 3
model_name = 'gpt2'

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Initialize the model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move the model to GPU if available
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
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
def chunk_text(text, max_length):
    """
    Splits the text into chunks of `max_length`.
    
    Args:
    - text (str): The text to be chunked.
    - max_length (int): The maximum length of each chunk.
    
    Returns:
    - List[str]: A list of text chunks, each with a length up to `max_length`.
    """
    # Split the text into words to avoid cutting in the middle of words
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        # Check if adding the next word would exceed the max_length
        if len(' '.join(current_chunk + [word])) > max_length:
            # If so, add the current chunk to the chunks list and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            # Otherwise, add the word to the current chunk
            current_chunk.append(word)
    
    # Don't forget to add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# def tokenize_data(text_data):
#     # Tokenize the text data using the ChatGPT tokenizer
#     # text_data = "Tokenize the text data using the ChatGPT tokenizer"
#     tokenizer.pad_token = 'PAD'
#     text_chunks = chunk_text(text_data, max_seq_length)
#     encoded_chunks = [tokenizer.encode(chunk, truncation=True, max_length=max_seq_length, return_tensors="pt") for chunk in text_chunks]
#     inputs = tokenizer(text_data, truncation=True, padding="max_length", max_length=max_seq_length, return_tensors="pt")
#     # Convert the inputs dictionary to a list of tuples
#     inputs = list(inputs.items())

#     # Add special attention mask tokens
#     # attention_masks = [float(i != tokenizer.pad_token_id) for x in inputs[1] for i in x[0]]

#     # print(attention_masks)

#     # Create a DataFrame from the inputs and attention masks
#     df = pd.DataFrame({k: v.numpy().flatten() for k, v in zip(['input_ids', 'attention_mask'], [x[1] for x in inputs])})
#     print(df)
#     # df['attention_mask'] = attention_masks[0:512]
#     # Create labels by shifting the input_ids one position to the right
#     labels = df['input_ids'].copy()
#     labels = labels.shift(1).fillna(tokenizer.pad_token_id).astype(int)
#     tokenizer.save_pretrained(output_dir)


#     # Return the DataFrame with the labels
#     return df, labels

def shift_and_insert(row):
    # Shift elements one position to the right and insert [123] at the beginning
    shifted_list = [tokenizer.pad_token_id]
    new_list = row['input_ids'][:-1]
    shifted_list.extend(new_list)
    return shifted_list


def tokenize_data(text_data):
    # Break the text into chunks of up to max_seq_length, ignoring the last chunk if it's too small
    # text_data = "hello world is"
    encoded_chunks = []
    for i in range(0, len(text_data), max_seq_length):
        # Encode each chunk
        chunk = text_data[i:i+max_seq_length]
        encoded_chunk = tokenizer(chunk, truncation=True, padding="max_length", max_length=max_seq_length, return_tensors="pt")
        encoded_chunks.append(encoded_chunk)

    # Combine the encoded chunks
    input_ids = []
    attention_masks = []
    for chunk in encoded_chunks:
        input_ids.extend(chunk['input_ids'])
        attention_masks.extend(chunk['attention_mask'])

    # Convert to DataFrame
    df = pd.DataFrame({
        'input_ids': [ids.numpy().flatten() for ids in input_ids],
        'attention_mask': [masks.numpy().flatten() for masks in attention_masks]
    })

    print(df)

    # Create labels by shifting the input_ids one position to the right
    # df['labels'] = df['input_ids'].apply(lambda x: [1] + (x[:-1]))
    df['labels'] = df.apply(lambda row: shift_and_insert(row), axis=1)

    print(df)

    return df


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
    model.save_pretrained(output_dir)


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
    print("Length of text data")
    print(len(text_data[0]))
    full_text = " ".join(text_data)
    print(full_text[:1000])

    # Tokenize the text data
    data = tokenize_data(full_text)
    print(data.shape)
    print(data.head())
    # Split the data into training and validation sets
    train_df, val_df, train_labels, val_labels = train_test_split(data, data['labels'], test_size=0.1, random_state=42)


    # Fine-tune the model on the training data
    fine_tune_model(train_df, val_df, train_labels, val_labels)
