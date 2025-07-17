"""
WikiANN_enja.py 
Author is JACOB DIRKX, last updated July 2025
Creates bilingual English+Japanese NER datasets for fine-tuning

Uses datasets module from https://huggingface.co/docs/datasets 
to load WikiANN datasets (English + Japanese) and concatenates + saves

pip install datasets transformers seqeval
"""
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets

# Load datasets
wikiann_en = load_dataset("wikiann", "en")  
wikiann_ja = load_dataset("wikiann", "ja") 

# Combine datasets (training + validation sets)
combined_train = concatenate_datasets([
    wikiann_en["train"],
    wikiann_ja["train"],
])
combined_val = concatenate_datasets([
    wikiann_en["validation"],
    wikiann_ja["validation"],
])
combined_test = concatenate_datasets([ 
    wikiann_en["test"], 
    wikiann_ja["test"], 
])

# Package and save
combined_dataset = DatasetDict({
    "train": combined_train,
    "validation": combined_val,
    "test": combined_test
})
combined_dataset.save_to_disk("enja_ner_dataset")

# Upload to HuggingFace for cloud-accessibility 
upload = load_from_disk("enja_ner_dataset")
upload.push_to_hub("jdirkx/enja-ner")

#dataset = load_dataset("your-username/enja-ner")
