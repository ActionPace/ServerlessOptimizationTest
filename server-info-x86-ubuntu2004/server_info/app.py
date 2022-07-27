import time, json, os, subprocess

def runprocess(command):
    return subprocess.Popen( command, shell=True, stdout=subprocess.PIPE ).communicate()[0].decode('unicode_escape').strip()

def lambda_handler(event, context):
    
    start = time.time()
    type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
    cores = runprocess("getconf _NPROCESSORS_ONLN")
    osname=runprocess("cat /etc/os-release | grep '^NAME=' | cut -d'=' -f2 | tr -d '\"'")
    osversion=runprocess("cat /etc/os-release | grep 'VERSION=' | cut -d'=' -f2 | tr -d '\"'")
    functionname=os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    functionversion=os.environ.get('AWS_LAMBDA_FUNCTION_VERSION')
    awsregion=os.environ.get('AWS_REGION')
    memorysize=os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE')
    
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
                "timerun": timerun,
                "functionname": functionname,
                "functionversion": functionversion,
                "awsregion": awsregion
                
            }
        ),
    }
