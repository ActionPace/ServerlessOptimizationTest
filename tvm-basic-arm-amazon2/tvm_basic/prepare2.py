import os
from transformers import DistilBertTokenizer

os.environ['TRANSFORMERS_CACHE'] = '/home/app'
name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(name)