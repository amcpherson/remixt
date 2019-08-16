docker run -v ${PWD}:/repo -w /repo -e CONDA_UPLOAD_TOKEN=$CONDA_UPLOAD_TOKEN -it conda/miniconda3-centos6 bash -e build_conda.sh
