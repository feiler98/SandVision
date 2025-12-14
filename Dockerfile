FROM fedora:42

COPY . .

RUN yum install -y ffmpeg
RUN yum install -y python3.14
RUN yum install -y pip
RUN yum install -r requirements.txt

CMD ["python3", "script.py"]