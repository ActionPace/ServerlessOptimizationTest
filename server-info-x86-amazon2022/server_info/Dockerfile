# amazonlinus2022
FROM docker.io/library/amazonlinux:2022.0.20220531.0

ARG FUNCTION_DIR="/home/app/"

WORKDIR ${FUNCTION_DIR}

COPY requirements.txt ${FUNCTION_DIR}

RUN yum -y update && yum -y install \
  gcc \
  python3-pip \
  && yum clean all && rm -rf /var/cache && pip3 install --upgrade pip

RUN  python3 -m pip install -r requirements.txt -t . \
  && rm -Rf ~/.cache/pip


RUN python3 -m pip install awslambdaric --target ${FUNCTION_DIR}

COPY app.py ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.6/aws-lambda-rie-x86_64 /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD ["app.lambda_handler"]