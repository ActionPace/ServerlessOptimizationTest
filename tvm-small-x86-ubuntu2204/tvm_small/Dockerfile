# From local docker image
FROM docker.io/library/ubuntu:jammy-20220531 as build

RUN apt -y update && apt-get -y install curl tar gzip

ARG UTILS_FILE="installedUB-python-llvm.tar.gz"
ARG APPS_FILE="installedUB-2204-x86-ext-complete.tar.gz"
ARG ENV_FILE="env-scripts-all-07252022.tar.gz"
ARG SSL_URL="http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/"
ARG SSL_FILE="libssl1.1_1.1.1f-1ubuntu2_amd64.deb"
ARG FFI_FILE="libffi.so.7-UB-2004-x86.tar.gz"

WORKDIR /root
COPY aws_url.txt .

WORKDIR /progs
RUN curl -s "$(cat /root/aws_url.txt)${APPS_FILE}"  | tar zvx -C .
WORKDIR /progs2
RUN curl -s "$(cat /root/aws_url.txt)${UTILS_FILE}" | tar zvx -C .

WORKDIR /envscript
RUN curl -s "$(cat /root/aws_url.txt)${ENV_FILE}" | tar zvx -C .

WORKDIR /tmp
RUN curl "${SSL_URL}${SSL_FILE}" --output ${SSL_FILE} \
&& curl -s "$(cat /root/aws_url.txt)${FFI_FILE}" | tar zvx -C . 

FROM docker.io/library/ubuntu:jammy-20220531

ARG FOLDER_NAME="installedUB"
ARG SCRIPT_NAME="envUB.sh"
ARG FUNCTION_DIR="/home/app/"
ENV TRANSFORMERS_CACHE='/home/app/huggingface'
ARG SSL_FILE="libssl1.1_1.1.1f-1ubuntu2_amd64.deb"
ARG FFI_FILE2="libffi.so.7"
ARG ARCH="x86_" # "aarch" or "x86_"
ARG ARCH_RIE="x86_" # "arm" or "x86_"

WORKDIR /envscript
COPY --from=build /envscript/${SCRIPT_NAME} ./env.sh

WORKDIR /progs/${FOLDER_NAME}
COPY --from=build /progs/${FOLDER_NAME} .

WORKDIR /progs2/${FOLDER_NAME}
COPY --from=build /progs2/${FOLDER_NAME} .

WORKDIR /tmp
COPY --from=build /tmp/${SSL_FILE} .
COPY --from=build /tmp/${FFI_FILE2} .
RUN dpkg -i ${SSL_FILE} \
&& rm ${SSL_FILE} \
&& cp ${FFI_FILE2} /usr/lib/${ARCH}64-linux-gnu \
&& rm ${FFI_FILE2}

WORKDIR /usr/bin
COPY lscpu .

WORKDIR ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& chmod a+x /usr/bin/lscpu \
&& python3 -m pip install -r requirements.txt -t . \
&& python3 -m pip install awslambdaric==2.0.4 --target ${FUNCTION_DIR} \
&& pip3 uninstall torch torchaudio torchvision -y \
&& pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
#&& apt-get -y install util-linux \
&& python3 -m pip install intel_extension_for_pytorch==1.12.100 \
&& pip3 install optimum[onnxruntime]==1.3.0 

COPY *.py ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& python3 prepare.py && rm /home/app/huggingface/* \
&& python3 prepare2.py

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-${ARCH_RIE}64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]