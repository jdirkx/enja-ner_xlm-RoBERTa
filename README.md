enja-ner_xlm-RoBERTa
====================

This is a repository for the training of [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) for Named Entity Recognition in English and Japanese. Trained with concatenated English and Japanese [WikiANN](https://huggingface.co/datasets/unimelb-nlp/wikiann) datasets, which contain labels for NER. Purpose is to examine text generated via Optical Character Recognition and return likely client/organization names for a next.js-based mail system web app, created in collaboration with [Matthew Nguyen](https://github.com/matthewnguyen1230). Created for [The DECK](https://thedeck.jp/) during a summer internship.

Last updated by Jacob Dirkx, July 2025

Setup and execution [bash]
======================================
git clone https://github.com/jdirkx/enja-ner_xlm-RoBERTa \
cd enja-ner_xlm-RoBERTa \
pip install -r requirements.txt \
huggingface-cli login \
wandb login  \
python scripts/xlm_RoBERTa_NER_Training.py
