FROM continuumio/miniconda

RUN conda config --add channels https://conda.anaconda.org/dranew && conda config --add channels bioconda
RUN conda install remixt
RUN conda install openssl=1.0
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

ENV NAME remixt

ENTRYPOINT ["remixt"]

