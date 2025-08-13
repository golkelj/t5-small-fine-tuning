import torch
import transformers
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import nltk
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model and dataset names
MODEL_CHECKPOINT = "t5-small"
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
TOKENIZED_DATA_SAVE_PATH = "./tokenized_cnn_dailymail"

# Data preprocessing parameters
MAX_INPUT_LENGTH = 512  # Max token length for input articles
MAX_TARGET_LENGTH = 128 # Max token length for generated summaries

# Training arguments
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 100 # Adjust based on GPU memory. Start small.
PER_DEVICE_EVAL_BATCH_SIZE = 100  # Adjust based on GPU memory.
NUM_TRAIN_EPOCHS = 3 # Number of training epochs (1 is usually too short for good results)
SAVE_TOTAL_LIMIT = 1 # Only save the best model checkpoint

# PEFT (LoRA) config
LORA_R = 8 
LORA_ALPHA = 32 
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q", "v"] 

torch.manual_seed(42)
np.random.seed(42)



tokenized_datasets = DatasetDict.load_from_disk(TOKENIZED_DATA_SAVE_PATH)

if 'validation' in tokenized_datasets:
    tokenized_datasets['eval'] = tokenized_datasets['validation']
    del tokenized_datasets['validation']

print(tokenized_datasets.keys())

# load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(device)

# Parameter-Efficient Fine-Tuning (PEFT)
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Shows how few parameters are actually trainable!

# --- Define Metrics and Data Collator ---
# Download punkt tokenizer for NLTK (used by ROUGE)
nltk.download('punkt')
nltk.download('punkt_tab')

# Load ROUGE metric
rouge_metric = evaluate.load("rouge") # Renamed to avoid conflict with 'rouge' variable in compute_metrics

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()} # Convert to percentage

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- Configure and Initialize Trainer ---
model_output_dir = f"{MODEL_CHECKPOINT.split('/')[-1]}-finetuned-cnn_dailymail" 
training_args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,
    eval_strategy="epoch", 
    save_strategy="epoch", 
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=0.01,
    save_total_limit=SAVE_TOTAL_LIMIT,
    predict_with_generate=True,
    fp16=True, 
    logging_dir=f"./{model_output_dir}_logs",
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    push_to_hub=False,
    report_to="none", 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
print("Model training complete!")

final_model_path = os.path.join(model_output_dir, "final_peft_model")
trainer.save_model(final_model_path)
