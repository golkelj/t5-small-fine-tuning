from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch 

# Define the path to the downloaded and unzipped model folder
# Make sure this path is correct for your local file system
local_model_path = "./t5-small-finetuned-cnn_dailymail/final_peft_model"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
print("Model loaded successfully!")

# You can now use the loaded model for inference, similar to how you did in Colab.
# For example:

def generate_summary_local(text_to_summarize, model, tokenizer, max_length=128, num_beams=4):
    inputs = tokenizer(
        f"summarize: {text_to_summarize}",
        return_tensors="pt",
        max_length=512, # Use the same max input length as during training
        truncation=True
    )

    # Move inputs to the correct device if you have a GPU locally
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device) # Move model to device as well

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage:
sample_article = """
The annual report released by the World Health Organization highlights the significant \
progress made in combating infectious diseases globally. However, it also warns about \
the increasing threat of non-communicable diseases such as diabetes and heart disease, \
urging governments to invest more in preventative healthcare and public health campaigns. \
The report emphasizes the need for international cooperation to address global health challenges \
effectively.
"""

generated_summary = generate_summary_local(sample_article, model, tokenizer)
# print(f"Original Art
print(len(sample_article))
print(len(generated_summary))