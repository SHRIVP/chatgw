from transformers import GPT2LMHeadModel, GPT2Tokenizer

output_dir = "/Users/pushkar/projects/chatbot/logs"

# Step 1: Load the fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(output_dir)

# Step 2: Prepare input text
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Step 3: Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

# Step 4: Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
