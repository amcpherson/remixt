FROM continuumio/miniconda

RUN conda config --add channels https://conda.anaconda.org/dranew && conda config --add channels bioconda
RUN conda install remixt
RUN mkdir /remixt_ref_data
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
RUN remixt create_ref_data /remixt_ref_data

ENV NAME remixt

CMD ["remixt"]

