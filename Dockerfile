FROM python:latest
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install --upgrade pip
RUN pip install rdflib pandas numpy pykeen torch sklearn
COPY main.py /
COPY kg22-carcinogenesis_lps1-train.ttl / .
COPY kg22-carcinogenesis_lps2-test.ttl / .
COPY carcinogenesis/carcinogenesis.owl / .

CMD ["python","main.py"]
