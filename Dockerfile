FROM python:3.5.3-onbuild


RUN mkdir /data
RUN mkdir /researchs

VOLUME ["/data","/researchs"]

RUN mkdir /core
RUN mkdir /root/.jupyter && mkdir /root/.jupyter/custom
RUN mkdir /root/.ipython && mkdir /root/.ipython/profile_default && mkdir /root/.ipython/profile_default/startup

COPY ./scripts/content/custom.css /root/.jupyter/custom
COPY ./scripts/content/startup.ipy /root/.ipython/profile_default/startup
COPY ./core/ /core/

RUN apt-get install curl wget

RUN wget https://pypi.python.org/packages/32/38/a55150ec018cf6fe11012bf1d988cd737af7f82227e4ac753619f0fb27a4/lesscpy-0.12.0.tar.gz
RUN pip install lesscpy-0.12.0.tar.gz
RUN pip install jupyterthemes


EXPOSE 8888

CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/researchs