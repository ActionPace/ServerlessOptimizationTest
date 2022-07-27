# Ubuntu:20.04
FROM docker.io/library/ubuntu:focal-20220531

ARG FUNCTION_DIR="/home/app/"

WORKDIR ${FUNCTION_DIR}

COPY requirements.txt ${FUNCTION_DIR}

RUN apt-get update && apt-get install -y \
  gcc \
  python3 \
  python3-pip \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -r requirements.txt -t . && python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY app.py ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-x86_64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]
