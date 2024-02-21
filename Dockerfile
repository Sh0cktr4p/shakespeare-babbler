FROM pytorch/pytorch

WORKDIR /home/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train_language_model.py", "&&", "python", "shakespeare_babbler.py"]
