import time, json, os, subprocess
import torch
#from transformers import DistilBertTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import intel_extension_for_pytorch as ipex

from optimum.onnxruntime import ORTModelForSequenceClassification

def runprocess(command):
    return subprocess.Popen( command, shell=True, stdout=subprocess.PIPE ).communicate()[0].decode('unicode_escape').strip()

#os.environ['TRANSFORMERS_CACHE'] = '/home/app'
name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(name)
loaded_model = torch.jit.load("traced_bert.pt")

#IPEX Model - workaround = put this in function2
#loaded_model2 = torch.jit.load("traced_bert2.pt")

#Optimum model
onnx_path = "onnx"
model2 = ORTModelForSequenceClassification.from_pretrained(onnx_path)

#Standard Huggingface
model3 = DistilBertForSequenceClassification.from_pretrained(name)

def lambda_handler(event, context):
     
    body = json.loads(event["body"])

    input_text = body["input_text"]
    
    if "max_tokens" in body:
        maxtokens = body["max_tokens"]
        if maxtokens=="":
            maxtokens="512"
    else:
        maxtokens="512"

    #type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
    type = runprocess("cat /proc/cpuinfo | grep 'model name' | tail -1 | cut -d':' -f2 | xargs")
    #type = runprocess("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
    cores = runprocess("getconf _NPROCESSORS_ONLN")
    osname=runprocess("cat /etc/os-release | grep '^NAME=' | cut -d'=' -f2 | tr -d '\"'")
    osversion=runprocess("cat /etc/os-release | grep 'VERSION=' | cut -d'=' -f2 | tr -d '\"'")
    functionname=os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    functionversion=os.environ.get('AWS_LAMBDA_FUNCTION_VERSION')
    awsregion=os.environ.get('AWS_REGION')
    memorysize=os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
    
    start = time.time()
    
    #input_text="This movie was really horrible and I won't come again!"
    inputs = tokenizer(input_text, padding="max_length", max_length=int(maxtokens), return_tensors="pt")
    input_ids=torch.tensor(inputs["input_ids"].numpy())
    attention_mask=torch.tensor(inputs["attention_mask"].numpy())
    loaded_model.eval()
    loadtime = "{:.2f} ms".format((time.time() - start)*1000)
    start = time.time()
    
    #Original
    test_out = loaded_model(input_ids,attention_mask)
    result="{}".format(test_out[0][0].detach().numpy())
    
    #IPEX
    #test_out = loaded_model2(input_ids,attention_mask)
    #result="{}".format(test_out[0][0].detach().numpy())
   
    #Optimum
    #with torch.no_grad():
    #    test_out = model2(**inputs).logits
    #result="{}".format(test_out[0].detach().numpy())

    timerun = "{:.2f} ms".format((time.time() - start)*1000)
    
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "osname": osname,
                "osversion": osversion,
                "type": type,
                "cores": cores,
                "memorysize": memorysize,
                "maxtokens": maxtokens,
                "loadtime": loadtime,
                "timerun": timerun,
                "result": result,
                "functionname": functionname,
                "functionversion": functionversion,
                "awsregion": awsregion
               
            }
        ),
    }

def lambda_handler2(event, context):
     
    body = json.loads(event["body"])

    input_text = body["input_text"]

    if "max_tokens" in body:
        maxtokens = body["max_tokens"]
        if maxtokens=="":
            maxtokens="512"
    else:
        maxtokens="512"
    
    #type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
    type = runprocess("cat /proc/cpuinfo | grep 'model name' | tail -1 | cut -d':' -f2 | xargs")
    #type = runprocess("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
    cores = runprocess("getconf _NPROCESSORS_ONLN")
    osname=runprocess("cat /etc/os-release | grep '^NAME=' | cut -d'=' -f2 | tr -d '\"'")
    osversion=runprocess("cat /etc/os-release | grep 'VERSION=' | cut -d'=' -f2 | tr -d '\"'")
    functionname=os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    functionversion=os.environ.get('AWS_LAMBDA_FUNCTION_VERSION')
    awsregion=os.environ.get('AWS_REGION')
    memorysize=os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
    
    start = time.time()
    
    #input_text="This movie was really horrible and I won't come again!"
    inputs = tokenizer(input_text, padding="max_length", max_length=int(maxtokens), return_tensors="pt")
    input_ids=torch.tensor(inputs["input_ids"].numpy())
    attention_mask=torch.tensor(inputs["attention_mask"].numpy())
    loaded_model.eval()
    #Should be temporary until patch is applied in IPEX
    loaded_model2 = torch.jit.load("traced_bert2.pt")
    loadtime = "{:.2f} ms".format((time.time() - start)*1000)
    start = time.time()
    
    #Original - temp reverting from IPEX
    #test_out = loaded_model(input_ids,attention_mask)
    #result="{}".format(test_out[0][0].detach().numpy())
    
    ##IPEX
    #Workaround until IPEX is fixed
    #https://github.com/intel/intel-extension-for-pytorch/issues/240
    #Moved line back to load time area
    #loaded_model2 = torch.jit.load("traced_bert2.pt")
    test_out = loaded_model2(input_ids,attention_mask)
    result="{}".format(test_out[0][0].detach().numpy())
   
    #Optimum
    #with torch.no_grad():
    #    test_out = model2(**inputs).logits
    #result="{}".format(test_out[0].detach().numpy())

    timerun = "{:.2f} ms".format((time.time() - start)*1000)
    
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "osname": osname,
                "osversion": osversion,
                "type": type,
                "cores": cores,
                "memorysize": memorysize,
                "maxtokens": maxtokens,
                "loadtime": loadtime,
                "timerun": timerun,
                "result": result,
                "functionname": functionname,
                "functionversion": functionversion,
                "awsregion": awsregion
               
            }
        ),
    }

def lambda_handler3(event, context):
     
    body = json.loads(event["body"])

    input_text = body["input_text"]

    if "max_tokens" in body:
        maxtokens = body["max_tokens"]
        if maxtokens=="":
            maxtokens="512"
    else:
        maxtokens="512"
    
    #type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
    type = runprocess("cat /proc/cpuinfo | grep 'model name' | tail -1 | cut -d':' -f2 | xargs")
    #type = runprocess("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
    cores = runprocess("getconf _NPROCESSORS_ONLN")
    osname=runprocess("cat /etc/os-release | grep '^NAME=' | cut -d'=' -f2 | tr -d '\"'")
    osversion=runprocess("cat /etc/os-release | grep 'VERSION=' | cut -d'=' -f2 | tr -d '\"'")
    functionname=os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    functionversion=os.environ.get('AWS_LAMBDA_FUNCTION_VERSION')
    awsregion=os.environ.get('AWS_REGION')
    memorysize=os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
    
    start = time.time()
    
    #input_text="This movie was really horrible and I won't come again!"
    inputs = tokenizer(input_text, padding="max_length", max_length=int(maxtokens), return_tensors="pt")
    input_ids=torch.tensor(inputs["input_ids"].numpy())
    attention_mask=torch.tensor(inputs["attention_mask"].numpy())
    loaded_model.eval()
    loadtime = "{:.2f} ms".format((time.time() - start)*1000)
    start = time.time()
    
    #Original
    #test_out = loaded_model(input_ids,attention_mask)
    #result="{}".format(test_out[0][0].detach().numpy())
    
    #IPEX
    #test_out = loaded_model2(input_ids,attention_mask)
    #result="{}".format(test_out[0][0].detach().numpy())
   
    #Optimum
    with torch.no_grad():
        test_out = model2(**inputs).logits
    result="{}".format(test_out[0].detach().numpy())

    timerun = "{:.2f} ms".format((time.time() - start)*1000)
    
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "osname": osname,
                "osversion": osversion,
                "type": type,
                "cores": cores,
                "memorysize": memorysize,
                "maxtokens": maxtokens,
                "loadtime": loadtime,
                "timerun": timerun,
                "result": result,
                "functionname": functionname,
                "functionversion": functionversion,
                "awsregion": awsregion
               
            }
        ),
    }

def lambda_handler4(event, context):
     
    body = json.loads(event["body"])

    input_text = body["input_text"]

    if "max_tokens" in body:
        maxtokens = body["max_tokens"]
        if maxtokens=="":
            maxtokens="512"
    else:
        maxtokens="512"

    #type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
    type = runprocess("cat /proc/cpuinfo | grep 'model name' | tail -1 | cut -d':' -f2 | xargs")
    #type = runprocess("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
    cores = runprocess("getconf _NPROCESSORS_ONLN")
    osname=runprocess("cat /etc/os-release | grep '^NAME=' | cut -d'=' -f2 | tr -d '\"'")
    osversion=runprocess("cat /etc/os-release | grep 'VERSION=' | cut -d'=' -f2 | tr -d '\"'")
    functionname=os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    functionversion=os.environ.get('AWS_LAMBDA_FUNCTION_VERSION')
    awsregion=os.environ.get('AWS_REGION')
    memorysize=os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
    
    start = time.time()
    
    #input_text="This movie was really horrible and I won't come again!"
    inputs = tokenizer(input_text, padding="max_length", max_length=int(maxtokens), return_tensors="pt")
    input_ids=torch.tensor(inputs["input_ids"].numpy())
    attention_mask=torch.tensor(inputs["attention_mask"].numpy())
    #loaded_model.eval()
    loadtime = "{:.2f} ms".format((time.time() - start)*1000)
    start = time.time()
    
    #Original
    #test_out = loaded_model(input_ids,attention_mask)
    #result="{}".format(test_out[0][0].detach().numpy())
    
    #IPEX
    #test_out = loaded_model2(input_ids,attention_mask)
    #result="{}".format(test_out[0][0].detach().numpy())
   
    #Optimum
    #with torch.no_grad():
    #    test_out = model2(**inputs).logits
    #result="{}".format(test_out[0].detach().numpy())

    #Standard Huggingface
    with torch.no_grad():
        test_out = model3(**inputs).logits
    result="{}".format(test_out[0].detach().numpy())

    timerun = "{:.2f} ms".format((time.time() - start)*1000)
    
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "osname": osname,
                "osversion": osversion,
                "type": type,
                "cores": cores,
                "memorysize": memorysize,
                "maxtokens": maxtokens,
                "loadtime": loadtime,
                "timerun": timerun,
                "result": result,
                "functionname": functionname,
                "functionversion": functionversion,
                "awsregion": awsregion
               
            }
        ),
    }
