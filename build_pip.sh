yum install -y gcc gcc-c++ bzip2-devel git
pip install numpy==1.21.6
python setup.py build --force
python setup.py bdist_wheel
