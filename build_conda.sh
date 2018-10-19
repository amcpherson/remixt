
yum install gcc-c++ git -y
conda config --set always_yes true
conda config --set anaconda_upload yes
conda config --add channels https://conda.anaconda.org/dranew
conda config --add channels 'bioconda'
conda install conda-build anaconda-client
anaconda login --username dranew
conda build conda/remixt

