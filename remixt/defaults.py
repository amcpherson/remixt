################################################
# Default configuration for ReMixT
################################################

###
# Reference genome and external datasets
###

# Version of ensembl for gene annotations
ensembl_version                             = '93'

# Associated genome version used by the ensembl version
ensembl_genome_version                      = 'GRCh38'

# Ensemble assemblies to include in the reference genome
ensembl_assemblies                          = ['chromosome.1', 'chromosome.2', 'chromosome.3', 'chromosome.4', 'chromosome.5', 'chromosome.6', 'chromosome.7', 'chromosome.8', 'chromosome.9', 'chromosome.10', 'chromosome.11', 'chromosome.12', 'chromosome.13', 'chromosome.14', 'chromosome.15', 'chromosome.16', 'chromosome.17', 'chromosome.18', 'chromosome.19', 'chromosome.20', 'chromosome.21', 'chromosome.22', 'chromosome.X', 'chromosome.Y', 'chromosome.MT', 'nonchromosomal']

# Base chromosomes
chromosomes                                 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

# Ensemble reference genome chromosome assemblies
ensembl_assembly_url_template               = 'ftp://ftp.ensembl.org/pub/release-{ensembl_version}/fasta/homo_sapiens/dna/Homo_sapiens.{ensembl_genome_version}.dna.{ensembl_assembly}.fa.gz'

# Ucsc genome version (must match ensembl version!)
ucsc_genome_version                         = 'hg19'

# Locally installed reference genome
genome_fasta_template                       = '{ref_data_dir}/Homo_sapiens.{ensembl_genome_version}.{ensembl_version}.dna.chromosomes.fa'
genome_fai_template                         = '{ref_data_dir}/Homo_sapiens.{ensembl_genome_version}.{ensembl_version}.dna.chromosomes.fa.fai'

# Ucsc gap file
gap_url_template                            = 'http://hgdownload.soe.ucsc.edu/goldenPath/{ucsc_genome_version}/database/gap.txt.gz'

# Locally installed gap file
gap_table_template                          = '{ref_data_dir}/{ucsc_genome_version}_gap.txt.gz'

# Segment length for automatically generated segments
segment_length                              = int(5e5)

# Length of simulated reads used to calculate mappability
mappability_length                          = 100

# Mapping quality threshold for filtering mappable reads
map_qual_threshold                          = 1

# Filter reads marked as duplicate
filter_duplicates                           = False

# Locally installed mappability filename produced by mappability setup script
mappability_template                        = '{ref_data_dir}/{ucsc_genome_version}.{mappability_length}.bwa.mappability.h5'

# Thousand genomes dataset
thousand_genomes_impute_url                 = 'http://mathgen.stats.ox.ac.uk/impute/ALL_1000G_phase1integrated_v3_impute.tgz'
thousand_genomes_directory                  = '{ref_data_dir}/ALL_1000G_phase1integrated_v3_impute'
sample_template                             = thousand_genomes_directory+'/ALL_1000G_phase1integrated_v3.sample'
legend_template                             = thousand_genomes_directory+'/ALL_1000G_phase1integrated_v3_chr{chromosome}_impute.legend.gz'
haplotypes_template                         = thousand_genomes_directory+'/ALL_1000G_phase1integrated_v3_chr{chromosome}_impute.hap.gz'
genetic_map_template                        = thousand_genomes_directory+'/genetic_map_chr{chromosome}_combined_b37.txt'
phased_chromosome_x                         = 'X_nonPAR'

# Locally installed snps from thousand genomes
snp_positions_template                      = '{ref_data_dir}/thousand_genomes_snps.tsv'

###
# Algorithm parameters
###

# Male or female for one or two copies of chromosome 'X'
is_female                                   = True

# Maximum inferred fragment length of a read pair classified as concordant
bam_max_fragment_length                     = 1000

# Maximum soft clipped bases before a read is called discordant
bam_max_soft_clipped                        = 8

# Check proper pair flag for identifying concordant pairs,
# disable for irregular fragment length distribution
bam_check_proper_pair                       = True

# Heterozygous snp calling
sequencing_base_call_error                  = 0.01
het_snp_call_threshold                      = 0.9
homozygous_p_value_threshold                = 1e-16

# Shapeit haplotype block resolution
shapeit_num_samples                         = 100
shapeit_confidence_threshold                = 0.95

# Enable correction
do_gc_correction                            = True
do_mappability_correction                   = True

# GC bias correction
sample_gc_num_positions                     = 10000000
gc_position_offset                          = 4

# Method to use for fitting segment/breakpoint copy number model
fit_method                                  = 'hmm_graph'

# Maximum copy number in state space for HMM
max_copy_number                             = 8

# Tumour mixture fractions for initialization of haploid depth
# parameter optimization
tumour_mix_fractions                        = [0.45, 0.3, 0.2, 0.1]

# Maximum and minimum ploidy of initial haploid depth parameters
# Ploidy selection can be performed by setting min and max ploidy to a small range
min_ploidy                                  = 1.5
max_ploidy                                  = 6.0

# Force haploid normal and or tumour to specific values, useful
# for very poor samples for which estimation fails
h_normal                                    = None
h_tumour                                    = None

# Maximum proportion of segments with divergent copy number
# for filtering improbable solutions
max_prop_diverge                            = 0.5

# Table of expected proportion of each genotype for use as prior,
# set to None to use proportion data included in package
cn_proportions_filename                     = None

# Model normal contamination
normal_contamination                        = True

# Minimum length of segments modelled by the likelihood
likelihood_min_segment_length               = 10000

# Minimum proportion genotyped reads for segments modelled by the likelihood
likelihood_min_proportion_genotyped         = 0.01

# Length scaled weights on divergent segments
divergence_weights                          = [1e-6, 1e-7, 1e-8]

# Number of iterations of EM for parameter optimization
num_em_iter                                 = 5

# Number of iterations of Variational Inference per EM iteration
num_update_iter                             = 5

# Disable breakpoints for benchmarking purposes
disable_breakpoints                         = False

# For debug purposes, disable update of the h parameter
do_h_update                                 = True


