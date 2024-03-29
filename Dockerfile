FROM continuumio/miniconda3
ARG app_version

MAINTAINER Andrew McPherson <andrew.mcpherson@gmail.com>

USER root
RUN conda config --add channels https://conda.anaconda.org/dranew && conda config --add channels bioconda
RUN apt update
RUN apt install build-essential -y
RUN apt install libbz2-dev -y
RUN pip install remixt==$app_version
RUN pip install pysam
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN groupadd -r -g 1000 ubuntu && useradd -r -g ubuntu -u 1000 ubuntu
USER ubuntu

ENV NAME remixt

ENTRYPOINT ["remixt"]

