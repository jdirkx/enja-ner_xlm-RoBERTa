"""
xlm_RoBERTa_NER_training.py 
Author is JACOB DIRKX, last updated July 2025

aaaaaaaaaaa

pip install datasets transformers seqeval
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import evaluate
import yaml

# Load dataset, tokenizer & model checkpoint
combined_dataset = load_dataset("jdirkx/enja-ner")

use_small_dataset = True  # Toggle smaller sample for prototyping 

if use_small_dataset:
    train_dataset = combined_dataset["train"].train_test_split(test_size=0.9, seed=42)["train"]
    val_dataset = combined_dataset["validation"].train_test_split(test_size=0.9, seed=42)["train"]
    test_dataset = combined_dataset["test"].train_test_split(test_size=0.9, seed=42)["train"]
else:
    train_dataset = combined_dataset["train"]
    val_dataset = combined_dataset["validation"]
    test_dataset = combined_dataset["test"]

model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Prepare label list and label to ID mapping
label_list = combined_dataset["train"].features["ner_tags"].feature.names
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

# Tokenize and align labels function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="longest"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply preprocessing
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
)

# Prepare metrics (using seqeval)
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training arguments
with open("../config/training_args.yaml") as f:
    args_dict = yaml.safe_load(f)
training_args = TrainingArguments(**args_dict)

# Trainer setup
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training and evaulation on test set
trainer.train()
metrics = trainer.evaluate(tokenized_test)
print(metrics)

# Finally, push to HuggingFace for cloud accessibility 
model.push_to_hub("your-username/xlmroberta-enja-ner")
tokenizer.push_to_hub("your-username/xlmroberta-enja-ner")
