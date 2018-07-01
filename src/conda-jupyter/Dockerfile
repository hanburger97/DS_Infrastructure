FROM ubuntu:16.04

RUN apt-get update && apt-get install -y dev-essentials

WORKDIR	/

RUN wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh  && \
		bash Anaconda3-4.3.0-Linux-x86_64.sh -b -p /anaconda

ENV PATH /anaconda/bin:${PATH}

RUN conda update conda

RUN conda install -c anaconda notebook && conda install -c conda-forge jupyterlab

EXPOSE 11235


CMD jupyter lab --ip 0.0.0.0 ---port 11235



