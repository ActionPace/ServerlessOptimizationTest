# From local docker image
# docker load < ubuntu20.04-x86-tvm-basicv1.tar.gz
FROM ubuntu20.04-x86-tvm-basic:v1

ARG FUNCTION_DIR="/home/app/"
ENV TRANSFORMERS_CACHE="/home/app/huggingface"
ARG ARCH_RIE="x86_" # "arm" or "x86_"

WORKDIR ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
RUN python3 -m pip install pip --upgrade \
&& python3 -m pip install -r requirements.txt -t . \
&& python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY *.py ${FUNCTION_DIR}
RUN python3 prepare.py && rm /home/app/huggingface/* \
&& python3 prepare2.py

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-${ARCH_RIE}64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]
