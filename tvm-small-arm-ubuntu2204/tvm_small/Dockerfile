# From local docker image
FROM --platform=linux/arm64/v8 docker.io/arm64v8/ubuntu:jammy-20220531 as build

RUN yum -y update && yum -y install curl tar gzip 

ARG UTILS_FILE="installedUBARM-python-llvm.tar.gz"
ARG APPS_FILE="installedUBARM-2204-aarch-ext-complete.tar.gz"
ARG ENV_FILE="env-scripts-all-07252022.tar.gz"
ARG SSL_FILE="lib-ssl-crypto-1.0.2k-ARM.tar.gz"
ARG FFI_FILE="libffi.so.7-UB-2004-aarch.tar.gz"

WORKDIR /root
COPY aws_url.txt .

WORKDIR /progs
RUN curl -s "$(cat /root/aws_url.txt)${APPS_FILE}"  | tar zvx -C .
WORKDIR /progs2
RUN curl -s "$(cat /root/aws_url.txt)${UTILS_FILE}" | tar zvx -C .

WORKDIR /envscript
RUN curl -s "$(cat /root/aws_url.txt)${ENV_FILE}" | tar zvx -C .

WORKDIR /missinglibs
RUN curl -s "$(cat /root/aws_url.txt)${SSL_FILE}" | tar zvx -C . \
&& curl -s "$(cat /root/aws_url.txt)${FFI_FILE}" | tar zvx -C . 

FROM --platform=linux/arm64/v8 docker.io/arm64v8/ubuntu:jammy-20220531

ARG FOLDER_NAME="installed"
ARG SCRIPT_NAME="env.sh"
ARG FUNCTION_DIR="/home/app/"
ENV TRANSFORMERS_CACHE="/home/app/huggingface"
ARG FFI_FILE2="libffi.so.7"
ARG ARCH="aarch" # "aarch" or "x86_"
ARG ARCH_RIE="arm" # "arm" or "x86_"

WORKDIR /envscript
COPY --from=build /envscript/${SCRIPT_NAME} ./env.sh

WORKDIR /progs/${FOLDER_NAME}
COPY --from=build /progs/${FOLDER_NAME} .

WORKDIR /progs2/${FOLDER_NAME}
COPY --from=build /progs2/${FOLDER_NAME} .

WORKDIR /usr/lib64
COPY --from=build /missinglibs/* .
COPY --from=build /tmp/${FFI_FILE2}-ARM ./${FFI_FILE2}
RUN ln -s libssl.so.1.0.2k libssl.so.10 \
&& ln -s libcrypto.so.1.0.2k libcrypto.so.10 \
&& ln -s libcrypt.so.2 libcrypt.so.1 

WORKDIR ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& python3 -m pip install -r requirements.txt -t . \
&& python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY *.py ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& python3 prepare.py && rm /home/app/huggingface/* \
&& python3 prepare2.py

COPY app.py ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-arm64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]