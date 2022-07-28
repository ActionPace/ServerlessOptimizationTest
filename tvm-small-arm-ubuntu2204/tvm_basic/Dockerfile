# From local docker image
FROM --platform=linux/arm64/v8 docker.io/arm64v8/ubuntu:jammy-20220531

ARG FUNCTION_DIR="/home/app/"
ENV TRANSFORMERS_CACHE='/home/app/huggingface'

WORKDIR ${FUNCTION_DIR}

COPY requirements.txt ${FUNCTION_DIR}
RUN python3 -m pip install -r requirements.txt -t . && python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY *.py ${FUNCTION_DIR}
RUN python3 prepare.py && rm /home/app/huggingface/* \
&& python3 prepare2.py

COPY app.py ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-x86_64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]
