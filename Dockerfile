FROM continuumio/miniconda

RUN conda config --add channels https://conda.anaconda.org/dranew && conda config --add channels bioconda
RUN conda install remixt

ENV NAME remixt

CMD ["remixt"]
