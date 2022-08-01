import time, json, os, subprocess
import torch
from transformers import DistilBertTokenizer

def runprocess(command):
    return subprocess.Popen( command, shell=True, stdout=subprocess.PIPE ).communicate()[0].decode('unicode_escape').strip()

#os.environ['TRANSFORMERS_CACHE'] = '/home/app'
name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(name)
loaded_model = torch.jit.load("traced_bert.pt")


def lambda_handler(event, context):
    
    body = json.loads(event['body'])

    input_text = body['input_text']

    #type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
    type = runprocess("cat /proc/cpuinfo | grep 'model name' | tail -1 | cut -d':' -f2 | xargs")
    #type = runprocess("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
    #arm
    #type = runprocess("lscpu | grep 'Arch' | cut -d':' -f2 | xargs")
    cores = runprocess("getconf _NPROCESSORS_ONLN")
    osname=runprocess("cat /etc/os-release | grep '^NAME=' | cut -d'=' -f2 | tr -d '\"'")
    osversion=runprocess("cat /etc/os-release | grep 'VERSION=' | cut -d'=' -f2 | tr -d '\"'")
    functionname=os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    functionversion=os.environ.get('AWS_LAMBDA_FUNCTION_VERSION')
    awsregion=os.environ.get('AWS_REGION')
    memorysize=os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
    
    start = time.time()
    
    #input_text="This movie was really horrible and I won't come again!"
    inputs = tokenizer(input_text, padding="max_length", max_length=512, return_tensors="pt")
    input_ids=torch.tensor(inputs["input_ids"].numpy())
    attention_mask=torch.tensor(inputs["attention_mask"].numpy())
    loaded_model.eval()
    loadtime = "{:.2f} ms".format((time.time() - start)*1000)
    start = time.time()
    test_out = loaded_model(input_ids,attention_mask)
    timerun = "{:.2f} ms".format((time.time() - start)*1000)
    result="{}".format(test_out[0][0].detach().numpy())
    

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "osname": osname,
                "osversion": osversion,
                "type": type,
                "cores": cores,
                "memorysize": memorysize,
                "loadtime": loadtime,
                "timerun": timerun,
                "result": result,
                "functionname": functionname,
                "functionversion": functionversion,
                "awsregion": awsregion
               
            }
        ),
    }
