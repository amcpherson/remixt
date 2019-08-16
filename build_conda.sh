
yum install gcc-c++ git -y
conda config --set always_yes true
conda config --add channels https://conda.anaconda.org/dranew
conda config --add channels 'bioconda'
conda install conda-build anaconda-client
conda build conda/remixt
anaconda -t $CONDA_UPLOAD_TOKEN upload /usr/local/conda-bld/linux-64/remixt-*.tar.bz2

