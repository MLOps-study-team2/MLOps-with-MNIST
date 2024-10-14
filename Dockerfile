FROM python:3.9-slim

WORKDIR /mlops_with_mnist

COPY . /mlops_with_mnist

CMD ["cd", "mlops_with_mnist"]

RUN pip install -r requirements.txt

ENV GIT_PYTHON_REFRESH=quiet

CMD python model_train.py 
