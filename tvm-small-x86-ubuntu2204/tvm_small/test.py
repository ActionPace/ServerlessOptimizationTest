#https://huggingface.co/docs/transformers/serialization

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import time

name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(name, torchscript=True)
tokenizer = DistilBertTokenizer.from_pretrained(name)

input_text="This movie was really horrible and I won't come again!"
inputs = tokenizer(input_text, padding="max_length", max_length=512, return_tensors="pt")
input_ids=torch.tensor(inputs["input_ids"].numpy())
attention_mask=torch.tensor(inputs["attention_mask"].numpy())

# The model needs to be in evaluation mode
model.eval()

# Creating the trace
print("Test Converted JIT Model")
traced_model = torch.jit.trace(model, [input_ids, attention_mask])
start = time.time()
test_out=traced_model(input_ids, attention_mask)
print("CPU Inference Time= {} ms".format((time.time() - start)*1000))
print(test_out[0][0].detach().numpy())

print("Test Reloaded from File JIT Model")
torch.jit.save(traced_model, "traced_bert.pt")
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()
start = time.time()
test_out = loaded_model(input_ids,attention_mask)
print("CPU Inference Time= {} ms".format((time.time() - start)*1000))
print(test_out[0][0].detach().numpy())

#CPU Inference Time= 45.290523529052734 s
#[ 4.1452837 -3.447971 ]
#Test Reloaded from File JIT Model
#CPU Inference Time= 9.137372732162476 s
#[ 4.1452837 -3.447971 ]