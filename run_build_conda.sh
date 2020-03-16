docker run -v ${PWD}:/repo -w /repo -e CONDA_UPLOAD_TOKEN=$CONDA_UPLOAD_TOKEN -it continuumio/miniconda3 bash -e build_conda.sh
