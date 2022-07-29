# From local docker image
FROM docker.io/library/amazonlinux:2.0.20220606.1 as build

RUN yum -y update && yum -y install curl tar gzip 

ARG UTILS_FILE="installed-python-llvm.tar.gz"
ARG APPS_FILE="progs-installed-Amazon2-x86-basic-ext.tar.gz"
ARG ENV_FILE="env-scripts-all-07252022.tar.gz"

WORKDIR /root
COPY aws_url.txt .

WORKDIR /progs
RUN curl -s "$(cat /root/aws_url.txt)${APPS_FILE}"  | tar zvx -C .
WORKDIR /progs2
RUN curl -s "$(cat /root/aws_url.txt)${UTILS_FILE}" | tar zvx -C .

WORKDIR /envscript
RUN curl -s "$(cat /root/aws_url.txt)${ENV_FILE}" | tar zvx -C .

FROM docker.io/library/amazonlinux:2.0.20220606.1

ARG FOLDER_NAME="installed"
ARG SCRIPT_NAME="env.sh"
ARG FUNCTION_DIR="/home/app/"
ENV TRANSFORMERS_CACHE="/home/app/huggingface"
ARG ARCH_RIE="x86_" # "arm" or "x86_"

WORKDIR /envscript
COPY --from=build /envscript/${SCRIPT_NAME} ./env.sh

WORKDIR /progs/${FOLDER_NAME}
COPY --from=build /progs/${FOLDER_NAME} .

WORKDIR /progs2/${FOLDER_NAME}
COPY --from=build /progs2/${FOLDER_NAME} .

WORKDIR ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& python3 -m pip install -r requirements.txt -t . \
&& python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY *.py ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& python3 prepare.py && rm /home/app/huggingface/* \
&& python3 prepare2.py

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-${ARCH_RIE}64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]
