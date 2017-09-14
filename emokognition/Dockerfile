FROM jjanzic/docker-python3-opencv

ENV PYTHONUNBUFFERED 1
RUN mkdir /code

ADD cli /code/cli/
ADD emotions /code/emotions/
ADD __init__.py /code/
ADD README.md /code/
ADD requirements.txt /code/
ADD setup.py /code/

WORKDIR /code
RUN pip3 install -e .
RUN python setup.py install
