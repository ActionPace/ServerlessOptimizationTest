#!/bin/sh
. /envscript/env.sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
    exec /usr/bin/aws-lambda-rie python3 -m awslambdaric $1
else
    exec python3 -m awslambdaric $1
fi