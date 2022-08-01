from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = '/home/app'
name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(name, torchscript=True)
tokenizer = DistilBertTokenizer.from_pretrained(name)

input_text="This movie was really horrible and I won't come again!"
inputs = tokenizer(input_text, padding="max_length", max_length=512, return_tensors="pt")
input_ids=torch.tensor(inputs["input_ids"].numpy())
attention_mask=torch.tensor(inputs["attention_mask"].numpy())
model.eval()
traced_model = torch.jit.trace(model, [input_ids, attention_mask])
torch.jit.save(traced_model, "traced_bert.pt")