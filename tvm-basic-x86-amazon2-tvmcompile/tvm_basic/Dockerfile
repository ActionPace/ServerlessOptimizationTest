# From local docker image
FROM docker.io/library/amazonlinux:2.0.20220606.1 as build

RUN yum -y update && yum -y install curl tar gzip \
&& yum -y install git llvm llvm-devel make

ARG UTILS_FILE="installed-utils.tar.gz"
#ARG UTILS_FILE="installed-python-llvm.tar.gz"
#ARG APPS_FILE="progs-installed-Amazon2-x86-basic-ext.tar.gz"
ARG ENV_FILE="env-scripts-all-07252022.tar.gz"
ARG SCRIPT_NAME="env.sh"
ARG FOLDER_NAME="installed"

WORKDIR /root
COPY aws_url.txt .

WORKDIR /progs
RUN mkdir -p ${FOLDER_NAME}/Python
#RUN curl -s "$(cat /root/aws_url.txt)${APPS_FILE}"  | tar zvx -C .

WORKDIR /progs2
RUN curl -s "$(cat /root/aws_url.txt)${UTILS_FILE}" | tar zvx -C .

WORKDIR /envscript
RUN curl -s "$(cat /root/aws_url.txt)${ENV_FILE}" | tar zvx -C .

WORKDIR /workdir
RUN . /envscript/${SCRIPT_NAME} \
#&& ln -s /progs2/installed/llvmorg-14.0.6/bin/llvm-config /usr/bin/llvm-config \
&& ln -s /usr/lib64/libz.so.1 /usr/lib64/libz.so \ 
&& git clone --recursive https://github.com/apache/tvm \
&& cd tvm && git checkout 261de53 \
&& mkdir build \
&& cat cmake/config.cmake | grep -v USE_LLVM > build/config.cmake \
&& echo 'set(USE_LLVM "/usr/bin/llvm-config --link-static")' >> build/config.cmake \
&& echo 'set(HIDE_PRIVATE_SYMBOLS ON)' >> build/config.cmake \
&& cd build \
&& cmake .. -D CMAKE_C_COMPILER=/progs2/${FOLDER_NAME}/gcc-11.3.0/bin/gcc -D CMAKE_CXX_COMPILER=/progs2/${FOLDER_NAME}/gcc-11.3.0/bin/g++ \
&& make -j4 
RUN . /envscript/${SCRIPT_NAME} \
&& cd tvm/python && python3 setup.py install --prefix=/progs/${FOLDER_NAME}

FROM docker.io/library/amazonlinux:2.0.20220606.1

ARG FOLDER_NAME="installed"
ARG SCRIPT_NAME="env.sh"
ARG FUNCTION_DIR="/home/app/"
ENV TRANSFORMERS_CACHE="/home/app/huggingface"
ARG ARCH_RIE="x86_" # "arm" or "x86_"

WORKDIR /envscript
COPY ${SCRIPT_NAME} .
#COPY --from=build /envscript/${SCRIPT_NAME} ./env.sh

WORKDIR /progs/${FOLDER_NAME}
COPY --from=build /progs/${FOLDER_NAME}/ .
#RUN ls -l /progs/${FOLDER_NAME}
#drwxr-xr-x 2 root root  6 Aug  1 21:30 Python
#drwxr-xr-x 2 root root 58 Aug  1 22:14 bin
#drwxr-xr-x 3 root root 23 Aug  1 22:13 lib

WORKDIR /progs2/${FOLDER_NAME}
COPY --from=build /progs2/${FOLDER_NAME}/Python-3.8.13/ ./Python-3.8.13/
#RUN ls -l

WORKDIR ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
RUN . /envscript/${SCRIPT_NAME} \
&& python3 -m pip install -r requirements.txt -t . \
&& python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY *.py ${FUNCTION_DIR}
RUN . /envscript/env.sh \
&& python3 -m pip install --upgrade pip \
&& pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
&& pip3 install numpy onnx onnxoptimizer transformers \
&& python3 prepare.py && rm /home/app/huggingface/* \
&& python3 prepare2.py

#ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-${ARCH_RIE}64 /usr/bin/aws-lambda-rie
#COPY entry.sh /
#RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
#ENTRYPOINT [ "/entry.sh" ]

#CMD ["app.lambda_handler"]
