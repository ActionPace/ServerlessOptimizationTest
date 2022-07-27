FROM public.ecr.aws/lambda/python:3.8-arm64

COPY requirements.txt ./

RUN python3.8 -m pip install -r requirements.txt -t .

COPY app.py ./

RUN yum -y install gcc && yum -y clean all  && rm -rf /var/cache

CMD ["app.lambda_handler"]
