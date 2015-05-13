# Demix

Demix is a tool for joint inference of clone specific segment and breakpoint copy number in whole genome sequencing data.  The input for the tool is a set of segments, a set of breakpoints predicted from the sequencing data, and normal and tumour bam files.  Where multiple tumour samples are available, they can be analyzed jointly for additional benefit.

## Prerequisites

### Python

Demix requires python and the numpy/scipy stack.  The recommended way to install python (also the easiest), is to download and install the latest (Anaconda Python](https://store.continuum.io/cshop/anaconda/) from the Continuum website.

#### Python libraries

If you do no use anaconda, you will be required to install the following python libraries.

* [numpy/scipy](http://www.numpy.org)
* [pandas](http://pandas.pydata.org)
* [matplotlib](http://matplotlib.org)

### Scons

Building the source requires scons, verson 2.3.4 can be installed as follows:

    wget http://prdownloads.sourceforge.net/scons/scons-2.3.4.tar.gz
    tar -xzvf scons-2.3.4.tar.gz
    cd scons-2.3.4
    python setup.py install

### Samtools

[Samtools](http://www.htslib.org) is required and should be on the path.

### Shapeit

[shapeit2](https://mathgen.stats.ox.ac.uk/genetics_software/shapeit/shapeit.html#download) is required and should be on the path.

### BWA

[BWA](http://bio-bwa.sourceforge.net) is used by the `mappability_bwa.py` script for generating a bwa mappability file, for the users whose bam files were created using BWA.

## Install

### Clone Source Code

To install the code, first clone from bitbucket.  A recursive clone is preferred to pull in all submodules.

    git clone --recursive https://bitbucket.org/dranew/demix.git

The following steps will assume you are in the `demix` directory.

    cd demix

### Build Executables

Demix requires compilation of a number of executables using scons.

    cd src
    scons install

### Install Python Libraries

There are two options for installing the python libraries.

#### Option 1:

A temporary solution is to modify the python path to point to the location of the source code on your system.  If you use bash, the following command will correctly modify your environment.

    source pythonpath.sh

#### Option 2:

A more permanent solution is to install the libraries into your python site packages.  Note that if you are using python installed on your system, you may need admin privileges.

To install demix:

    python setup.py install

To install pypeliner, a pipeline management system:

    cd pypeliner
    python setup.py install

## Setup

Download and setup of the reference genome and 1000 genomes dataset is automated.  Select a directory on your system that will contain the reference data, herein referred to as `$ref_data_dir`.  The `$ref_data_dir` directory will be used in many of the subsequent scripts when running demix.

Download the reference data and build the required indexes:

    python setup/create_ref_data.py $ref_data_dir

Demix also requires a mappability file, which can be time consuming to build.  The script `setup/mappability_bwa.py` will build a mappability file compatible with the bwa aligner.  If your bam files are produced using a different aligner, you may want to consider generating a mappability file for your aligner by writing a modified the `setup/mappability_bwa.py` script.

Build the mappability file:

    python setup/mappability_bwa.py $ref_data_dir --tmpdir $tmp_map

where `$tmp_map` is a unique temporary directory.  If you need to stop and restart the script, using the same temporary directory will allow the script to restart where it left off.

The `setup/mappability_bwa.py` script has options for parallelism, see the section [Parallelism using pypeliner](#markdown-header-parallelism-using-pypeliner).

## Run

### Input Data

Demix takes the following as input:

* Normal bam file
* Multiple Tumour bam files from the same individual
* Segmentation of the genome
* Breakpoint predictions in the genome

#### Segmentation Input Format

The segmentation should be provided in a tab separated file with the following columns:

* `chromosome`
* `start`
* `end`

The first line should be the column names, which should be identical to the above list.  Each subsequent line is a segment.

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

### Step 1 - Importing the Data

The first step in the process is to import the relevant concordant read data from each bam file into a demix specific format.  This is accomplished using the `pipeline/extract_seqdata.py` script, which should be applied independently to each input file.  

To extract data from `$normal_bam` and `$tumour_bam` to `$normal_seqdata` and `$tumour_seqdata` respectively:

    python pipeline/extract_seqdata.py $ref_data_dir \
        $normal_bam $normal_seqdata \
        --tmpdir $tmp_seq_1

    python pipeline/extract_seqdata.py $ref_data_dir \
        $tumour_bam $tumour_seqdata \
        --tmpdir $tmp_seq_2

where `$tmp_seq_1` and `$tmp_seq_2` are unique temporary directories.  If you need to stop and restart the script, using the same temporary directory will allow the scripts to restart where it left off.

For parallelism options see the section [Parallelism using pypeliner](#markdown-header-parallelism-using-pypeliner).

### Step 2 - Preparing Segment Read Counts

The second step in the process is to prepare major, minor and total read counts per segment.  An input segmentation is required for this step.  The segmentation boundaries should ideally include ends of breakpoints of interest, though this is not required.  If you are using [destruct](https://bitbucket.org/dranew/destruct) to predict breakpoints, you can use the script `tools/create_segments.py` to create a segmentation file.

Given breakpoint predictions file `$breakpoints`, create `$segments` as follows:

    python tools/create_segments.py $ref_data_dir $breakpoints $segments

Read counts are prepared jointly for multiple tumour samples using the `pipeline/prepare_counts.py` script.  Suppose we have two tumour data files `$tumour_1_seqdata` and `$tumour_2_seqdata` produced from the previous step.  The `pipeline/prepare_counts.py` can analyze them jointly, to output one read count file for each tumour dataset, which we will call `$tumour_1_counts` and `$tumour_2_counts`.  This will provide more accurate allele specific read counts.

To create read counts for segments given in segment file `$segments` for normal data `$normal_seqdata` and tumour data `$tumour_1_seqdata` and `$tumour_2_seqdata`:

    python pipeline/prepare_counts.py $ref_data_dir \
        $segment_data $normal_data \
        --tumour_files $tumour_1_seqdata $tumour_2_seqdata
        --count_files $tumour_1_counts $tumour_2_counts
        --tmpdir $tmp_counts

where `$tmp_counts` is a unique temporary directory.  If you need to stop and restart the script, using the same temporary directory will allow the scripts to restart where it left off.

For parallelism options see the section [Parallelism using pypeliner](#markdown-header-parallelism-using-pypeliner).

### Step 3 - Running Demix

Demix is run on each sample individually, and takes as input the sample specific segment read counts and breakpoints.  The outputs are a segment copy number file (`$cn` below), a breakpoint copy number file (`$brk_cn` below), a segment copy number plot pdf (`$cn_plot` below), and a mixture file (`$mix` below).

To run demix for tumour sample 1 from above with counts file `$tumour_1_counts`, use the `pipeline/run_demix.py` script as follows:

    python pipeline/run_demix.py $tumour_1_counts $breakpoints \
        $cn $brk_cn $cn_plot $mix \
        --tmpdir $tmp_demix

where `$tmp_demix` is a unique temporary directory.  If you need to stop and restart the script, using the same temporary directory will allow the scripts to restart where it left off.

For parallelism options see the section [Parallelism using pypeliner](#markdown-header-parallelism-using-pypeliner).

### Intermediate File Formats

#### Sequence Data Format

The files produced by the `extract_seqdata.py` script are tar files raw data matrices produced for easy and efficient reading by numpy.  They contain information per chromosome about concordant alignment and SNP alleles for each fragment.  They are not intended to be usable except by demix.

#### Segment Counts Format

The segment counts files are tab separated with the first line as the header.  The files should contain at least the following columns (and may contain additional columns):

* `chromosome`
* `start`
* `end`
* `readcount`
* `length`
* `major_readcount`
* `minor_readcount`
* `major_is_allele_a`

The `chromosome`, `start` and `end` columns define the segment.  The `readcount` column is the total reads contained within the segment, and `major_readcount` and `minor_readcount` are the total allele specific reads contained within the segment and only include the reads that can be counted as one allele vs the other using heterozygous SNPs.  The length is the segment length scaled by a factor that takes into account mappability and GC content.  The `major_is_allele_a` column is useful if you have run demix on multiple samples from the same individual.  This column is a binary indicator and will tell you if the major allele from one sample is the same as the major allele from another sample.

### Output File Formats

#### Segment Copy Number

The segment copy number file adds additional columns to the segment counts file described above, including but not limited to:

* `major_1`
* `minor_1`
* `major_2`
* `minor_2`

The columns refer to the major and minor copy number in tumour clone 1 and 2.

#### Breakpoints Copy Number

The breakpoint copy number file is tab separated with the first line as the header and contains the following columns:

* `prediction_id`
* `allele_1`
* `allele_2`
* `cn_1`
* `cn_2`

The `prediction_id` column matches the column of the same name in the input breakpoints file, and specifies for which breakpoint prediction the copy number is being provided.  The `allele_1` and `allele_2` columns refer to which allele at break-end 1 and 2 are part of the tumour chromosome harboring the breakpoints.  The `cn_1` and `cn_2` columns provide the clone specific copy number for clone 1 and 2 respectively.

#### Segment Copy Number Plot

The segment copy number plot shows 5 tracks, each with the chromosomes on the x-axis.  Major and minor copy number are shown in red and blue respectively.  From the top, the plots show the following:

1. raw normalized copy number, normal haploid depth subtracted out and the result divided by tumour haploid depth
2. expected raw copy number given the inferred copy number
3. inferred clone 1 copy number
4. inferred clone 2 copy number
5. inferred clone copy number difference

#### Mixture File

Currently this file contains 3 numbers, tab separated.  These numbers, respectively, are:

1. normal DNA proportion
2. tumour clone 1 DNA proportion
3. tumour clone 2 DNA proportion

## Parallelism Using Pypeliner

Demix uses the pypeliner python library for parallelism.  Several of the scripts described above will complete more quickly on a multi-core machine or on a cluster.

To run a script in multicore mode, using a maximum of 4 cpus, add the following command line option:

    --maxjobs 4

To run a script on a cluster with qsub/qstat, add the following command line option:

    --submit asyncqsub

Often a call to qsub requires specific command line parameters to request the correct queue, and importantly to request the correct amount of memory.  To allow correct calls to qsub, use the `--nativespec` command line option, and use the placeholder `{mem}` which will be replaced by the amount of memory (in gigabytes) required for each job launched with qsub.  For example, to use qsub, and request queue `all.q` and set the `mem_free` to the required memory, add the following command line options:

    --submit asyncqsub --nativespec "-q all.q -l mem_free={mem}G"

