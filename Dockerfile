FROM python:3.8-slim
WORKDIR /flaskwebapp
COPY . /flaskwebapp
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./app.py