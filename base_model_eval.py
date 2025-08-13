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


MODEL_CHECKPOINT = "t5-small"
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
TOKENIZED_DATA_SAVE_PATH = "./tokenized_cnn_dailymail"


MAX_INPUT_LENGTH = 512  
MAX_TARGET_LENGTH = 128 

LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 100 
PER_DEVICE_EVAL_BATCH_SIZE = 100 
NUM_TRAIN_EPOCHS = 3 
SAVE_TOTAL_LIMIT = 1

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

tokenized_datasets = DatasetDict.load_from_disk(TOKENIZED_DATA_SAVE_PATH)

if 'validation' in tokenized_datasets:
    tokenized_datasets['eval'] = tokenized_datasets['validation']
    del tokenized_datasets['validation']

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

nltk.download('punkt')
nltk.download('punkt_tab')

rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(device)
base_model.eval()

base_data_collator = DataCollatorForSeq2Seq(tokenizer, model=base_model)

base_eval_args = Seq2SeqTrainingArguments(
    output_dir="./base_model_eval_results",
    eval_strategy="epoch",
    save_strategy="no",
    num_train_epochs=0,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./base_model_eval_logs",
    report_to="none",
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
)

base_trainer = Seq2SeqTrainer(
    model=base_model,
    args=base_eval_args,
    eval_dataset=tokenized_datasets["eval"],
    tokenizer=tokenizer,
    data_collator=base_data_collator,
    compute_metrics=compute_metrics
)

base_eval_results = base_trainer.evaluate()
print(f"Base Model ROUGE Scores: {base_eval_results}")