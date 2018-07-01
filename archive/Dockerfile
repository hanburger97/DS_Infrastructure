FROM python:3.5.3-onbuild


RUN mkdir /data
RUN mkdir /researchs

VOLUME ["/data","/researchs"]

RUN mkdir /core
RUN mkdir /root/.jupyter && mkdir /root/.jupyter/custom
RUN mkdir /root/.ipython && mkdir /root/.ipython/profile_default && mkdir /root/.ipython/profile_default/startup

COPY ./scripts/content/custom.css /root/.jupyter/custom
COPY ./scripts/content/logo.png /root/.jupyter/custom
COPY ./scripts/content/startup.ipy /root/.ipython/profile_default/startup
COPY ./core/ /core/

RUN apt-get install curl wget


RUN pip install jupyterthemes



EXPOSE 8888


CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/researchs && jupyter nbextension enable --py --sys-prefix widgetsnbextension
