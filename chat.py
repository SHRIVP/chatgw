from transformers import GPT2LMHeadModel, GPT2Tokenizer

output_dir = "/Users/pushkar/projects/chatbot/logs"

# Step 1: Load the fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(output_dir)

# Step 2: Prepare input text
input_text = "guidewire  configure workflows"
inputs = tokenizer.encode_plus(input_text, return_tensors="pt", add_special_tokens=True, max_length=512, padding="max_length", truncation=True)

# Step 3: Generate text
output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=1024, num_return_sequences=1, temperature=1.0, repetition_penalty=1.1)

print(output)

# Step 4: Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
