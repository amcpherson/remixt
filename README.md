# ReMixT

ReMixT is a tool for joint inference of clone specific segment and breakpoint copy number in whole genome sequencing data.  The input for the tool is a set of segments, a set of breakpoints predicted from the sequencing data, and normal and tumour bam files.  Where multiple tumour samples are available, they can be analyzed jointly for additional benefit.

## How to cite

If you find ReMixT useful, please consider citing our [genome biology article](https://doi.org/10.1186/s13059-017-1267-2).

## Installation

Conda is a prerequisite, install [anaconda python](https://store.continuum.io/cshop/anaconda/) from the continuum website.

### Installing from pip

The recommended method of installation for ReMixT is using `pip`.

    pip install remixt

You will also need to `shapeit` and `samtools` on your path.  They can be installed using conda:

    conda install samtools
    conda install -c dranew shapeit

### Installing from conda

The conda distribution is now out of date.  However, to use conda, add my channel, and the bioconda channel, and install ReMixT as follows.

    conda config --add channels https://conda.anaconda.org/dranew
    conda config --add channels 'bioconda'
    conda install remixt

### Installing from source

#### Clone Source Code

To install the code, first clone from bitbucket.  A recursive clone is preferred to pull in all submodules.

    git clone --recursive git@bitbucket.org:dranew/remixt.git

#### Dependencies

To install from source you will need several dependencies.  A list of dependencies can be found in the `conda` `yaml` file in the repo at `conda/remixt/meta.yaml`.

#### Build executables and install

To build executables and install the ReMixT code as a python package run the following command in the ReMixT repo:

    python setup.py install

## Setup ReMixT

### Reference genome

Download and setup of the reference genome is automated.  The default is hg19.  Select a directory on your system that will contain the reference data, herein referred to as `$ref_data_dir`.  The `$ref_data_dir` directory will be used in many of the subsequent scripts when running destruct.

Download the reference data and build the required indexes:

    remixt create_ref_data $ref_data_dir

### Mappability file

Additionally, ReMixT requires a mappability file to be generated.  We have provided a workflow for generating a mappability file based on `bwa` alignments, for other aligners, you may want to create your own mappability workflow, see `remixt/mappability/bwa/workflow.py` as an example.

To create a mappability file for `bwa`, run:

    remixt mappability_bwa $ref_data_dir

Note that this workflow will take a considerable amount of time and it is recommended you run this part of ReMixT setup on a cluster or multicore machine.

For parallelism options see the section [Parallelism using pypeliner](#markdown-header-parallelism-using-pypeliner).

## Running ReMixT

### Input Data

ReMixT takes multiple bam files as input.  Bam files should be multiple samples from the same patient, with one bam sequenced from a normal sample from that patient.

Additionally, ReMixT takes a list of predicted breakpoints detected by paired end sequencing as an additional input.

#### Breakpoint Prediction Input Format

The predicted breakpoints should be provided in a tab separated file with the following columns:

* `prediction_id`
* `chromosome_1`
* `strand_1`
* `position_1`
* `chromosome_2`
* `strand_2`
* `position_2`

The first line should be the column names, which should be identical to the above list.  Each subsequent line is a breakpoint prediction.  The `prediction_id` should be unique to each breakpoint prediction.  The `chromosome_`, `strand_` and `position_` columns give the position and orientation of each end of the breakpoint.  The values for `strand_` should be either `+` or `-`.  A value of `+` means that sequence to the right of `chromosome_`, `position_` is preserved in the tumour chromosome containing the breakpoint.  Conversely, a value of `-` means that sequence to the left of `chromosome_`, `position_` is preserved in the tumour chromosome containing the breakpoint.

The following table may assist in understanding the strand of a break-end.  Note that an inversion event produces two breakpoints, the strand configurations for both are shown.  Additionally, for inter-chromosomal events, any strand configuration is possible.

| Structural Variation     | Strand of Leftmost Break-End | Strand of Rightmost Break-End |
| ------------------------ | ---------------------------- | ----------------------------- |
| Deletion                 |              +               |              -                |
| Duplication              |              -               |              +                |
| Inversion (Breakpoint A) |              +               |              +                |
| Inversion (Breakpoint B) |              -               |              -                |

### ReMixT Command Line

Running ReMixT involves invoking a single command, `remixt run`.  The result of ReMixT is an [hdf5](https://www.hdfgroup.org) file storing [pandas](http://pandas.pydata.org) tables.

Suppose we have the following list of inputs:

* Normal sample with ID `123N` and bam filename `$normal_bam`
* Tumour sample with ID `123A` and bam filename `$tumour_a_bam`
* Tumour sample with ID `123B` and bam filename `$tumour_b_bam`
* Breakpoint table in TSV format with filename `$breakpoints`

Additionally, ReMixT will generate the following outputs:

* Results as HDF5 file storing pandas tables with filename `$results_h5`
* Temporary files and logs stored in directory `$remixt_tmp_dir` (directory created if it doesnt exist)

Given the above inputs and outputs run ReMixT as follows:

    remixt run $ref_data_dir $raw_data_dir $breakpoints \
        --normal_sample_id 123N \
        --normal_bam_file $normal_bam \
        --tumour_sample_ids 123A 123B \
        --tumour_bam_files $tumour_a_bam $tumour_b_bam \
        --results_files $results_h5
        --tmpdir $remixt_tmp_dir

Note that ReMixT creates multiple jobs and many parts of ReMixT are massively parallelizable, thus it is recommended you run ReMixT on a cluster or multicore machine.  For parallelism options see the section [Parallelism using pypeliner](#markdown-header-parallelism-using-pypeliner).

### Output File Formats

The main output file is an HDF5 store containing pandas dataframes.  These can be extracted in python or viewed using the ReMixT viewer.  Important tables include:

* `stats`: statistics for each restart
* `solutions/solution_{idx}/cn`: segment copy number table for solution `idx`
* `solutions/solution_{idx}/brk_cn`: breakpoint copy number table for solution `idx`
* `solutions/solution_{idx}/h`: haploid depths for solution `idx`

#### Statistics

ReMixT uses optimal restarts and model selection by BIC.  The statistics table contains one row per restart, sorted by BIC.  The table contains the following columns:

* `idx`: the solution index, used to refer to `solutions/solution_{idx}/*` tables.
* `bic`: the bic of this solution
* `log_posterior`: log posterior of the HMM
* `log_posterior_graph`: log posterior of the genome graph model
* `num_clones`: number of clones including normal
* `num_segments`: number of segments
* `h_converged`: whether haploid depths estimation converged
* `h_em_iter`: number of iterations for convergence of h
* `graph_opt_iter`: number of iterations for convergence of genome graph copy number
* `decreased_log_posterior`: whether the genome graph optimization stopped due to a move that decreased the log posterior

#### Segment Copy Number

The segment copy number table adds additional columns to the segment counts table described above, including but not limited to:

* `major_1`
* `minor_1`
* `major_2`
* `minor_2`

The columns refer to the major and minor copy number in tumour clone 1 and 2.

#### Breakpoints Copy Number

The breakpoint copy number table contains the following columns:

* `prediction_id`
* `cn_1`
* `cn_2`

The `prediction_id` column matches the column of the same name in the input breakpoints file, and specifies for which breakpoint prediction the copy number is being provided.  The `cn_1` and `cn_2` columns provide the clone specific copy number for tumour clone 1 and 2 respectively.

#### Haploid Depths

The haploid depths is a vector of `M` depths for each of the `M` clones including the normal.  To recover cell mixture proportions, simply normalize `h`.

### Extracting Tables as TSV files

If preferred, it is possible to extract copy number and metadata in TSV and YaML format.  For results file `$results_h5`, extract segment copy number, breakpoint copy number and meta data to files `$cn_table`, `$brk_cn_table`, `$meta_data` respectively as follows:

    remixt write_results \
        $results_h5 $cn_table $brk_cn_table $meta_data

### ReMixT Viewer

There is an experimental viewer for ReMixT at `tools/remixt_viewer_app.py`.  Bokeh '>0.10.0' is required.  To use the viewer app, organize your patient sample results files as `./patient_*/sample_*.h5`.  From the directory containing patient subdirectories, run the bokeh server:

    bokeh-server --script $REMIXT_DIR/tools/remixt_viewer_app.py

Then navigate to `http://127.0.0.1:5006/remixt`.

## Test Dataset for ReMixT

A test dataset is provided for providing the ability to run a quick analysis of a small dataset to ensure remixt is working correctly.

We will assume that the `REMIXT_DIR` environment variable points to a clone of the ReMixT source code.  Additionally, create a directory, and set the environment variable `WORK_DIR` to the location of that directory.

First use the `remixt create_ref_data` sub-command to create a reference dataset.  Specify a config, and use the example config that restricts to chromosome 15.

    remixt create_ref_data $WORK_DIR/ref_data \
        --config $REMIXT_DIR/examples/chromosome_15_config.yaml

Use `wget` to retrieve a precomputed mappability file.

    wget http://remixttestdata.s3.amazonaws.com/hg19.100.bwa.mappability.h5 --directory-prefix $WORK_DIR/ref_data/

Use `wget` to retrieve the example bam files and their indices for chromosome 15, and the breakpoints file with chromosome 15 breakpoints.

    wget http://remixttestdata.s3.amazonaws.com/HCC1395_chr15.bam --directory-prefix $WORK_DIR/
    wget http://remixttestdata.s3.amazonaws.com/HCC1395_chr15.bam.bai --directory-prefix $WORK_DIR/
    wget http://remixttestdata.s3.amazonaws.com/HCC1395BL_chr15.bam --directory-prefix $WORK_DIR/
    wget http://remixttestdata.s3.amazonaws.com/HCC1395BL_chr15.bam.bai --directory-prefix $WORK_DIR/
    wget http://remixttestdata.s3.amazonaws.com/HCC1395_breakpoints.tsv --directory-prefix $WORK_DIR/

Use the `remixt run` sub-command to run a remixt analysis.

    remixt run $WORK_DIR/ref_data $WORK_DIR/raw_data $WORK_DIR/HCC1395_breakpoints.tsv \
        --config $REMIXT_DIR/examples/chromosome_15_config.yaml \
        --tmpdir $WORK_DIR/tmp_remixt \
        --tumour_sample_ids HCC1395 \
        --tumour_bam_files $WORK_DIR/HCC1395_chr15.bam \
        --normal_sample_id HCC1395BL \
        --normal_bam_file $WORK_DIR/HCC1395BL_chr15.bam \
        --loglevel DEBUG \
        --submit local \
        --results_files $WORK_DIR/HCC1395.h5

Use the `remixt write_results` sub-command to write out tables of results and a yaml file containing inferred parameters and other meta data.

    remixt write_results $WORK_DIR/HCC1395.h5 \
        $WORK_DIR/HCC1395_cn.tsv \
        $WORK_DIR/HCC1395_brk_cn.tsv \
        $WORK_DIR/HCC1395_info.yaml

Finally, create a visualization of the solutions using the `remixt visualize_solutions` sub-command.

    remixt visualize_solutions $WORK_DIR/HCC1395.h5 \
        $WORK_DIR/HCC1395.html

## Parallelism Using Pypeliner

ReMixT uses the pypeliner python library for parallelism.  Several of the scripts described above will complete more quickly on a multi-core machine or on a cluster.

To run a script in multicore mode, using a maximum of 4 cpus, add the following command line option:

    --maxjobs 4

To run a script on a cluster with qsub/qstat, add the following command line option:

    --submit asyncqsub

Often a call to qsub requires specific command line parameters to request the correct queue, and importantly to request the correct amount of memory.  To allow correct calls to qsub, use the `--nativespec` command line option, and use the placeholder `{mem}` which will be replaced by the amount of memory (in gigabytes) required for each job launched with qsub.  For example, to use qsub, and request queue `all.q` and set the `mem_free` to the required memory, add the following command line options:

    --submit asyncqsub --nativespec "-q all.q -l mem_free={mem}G"

# Build

## Docker builds

To build a docker image, for instance version v0.5.13, run the following docker command:

    docker build --build-arg app_version=v0.5.13 -t amcpherson/remixt:v0.5.13 .
    docker push amcpherson/remixt:v0.5.13

## Pip build

To build with pip and distribute to pypi, use the following commands:

    python setup.py build_ext --force sdist
    twine upload --repository pypi dist/*

# License

ReMixT is released under the [MIT License](http://www.opensource.org/licenses/MIT).

