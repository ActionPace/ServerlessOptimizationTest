import time
start = time.time()
import os
import subprocess

#Docker Image amazonlinux:latest
#yum -y import python3
#yum -y import gcc
#pip3 install awslambdaric

def runprocess(command):
    return subprocess.Popen( command, shell=True, stdout=subprocess.PIPE ).communicate()[0].decode('unicode_escape').strip()

type = runprocess("gcc -march=native -Q --help=target|grep march|grep -v 'Known valid'|awk '{print $2}'")
cores = runprocess("getconf _NPROCESSORS_ONLN")
#cores = runprocess("grep 'cpu cores' /proc/cpuinfo | uniq|awk '{print $4}'")

#print("Processor Type: {}".format(type))
#print("Number of Cores: {}".format(cores))
#print("CPU Inference Time= {:.2f} ms".format((time.time() - start)*1000))

message="{} cpu, {} cores, {:.2f}ms runtime".format(type, cores, (time.time() - start)*1000)
print(message)
