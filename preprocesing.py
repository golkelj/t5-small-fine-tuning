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

if not os.path.exists(TOKENIZED_DATA_SAVE_PATH):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

        labels = tokenizer(text_target=examples["highlights"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "highlights", "id"],
        num_proc=os.cpu_count() 
    )

    tokenized_datasets.save_to_disk(TOKENIZED_DATA_SAVE_PATH)
