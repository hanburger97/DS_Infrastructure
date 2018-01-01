FROM python:3.5.3-onbuild

RUN mkdir /data
RUN mkdir /researchs
RUN mkdir /root/.jupyter && mkdir /root/.jupyter/custom

COPY /scripts/content/custom.css /root/.jupyter/custom

COPY ./core/ /researchs/core

VOLUME ["/data","/researchs"]

RUN apt-get install curl

EXPOSE 8888

CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/researchs