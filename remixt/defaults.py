################################################
# Default configuration for ReMixT
################################################

###
# Reference genome and external datasets
###

# Version of ensembl for gene annotations
ensembl_version                             = '70'

# Associated genome version used by the ensembl version
ensembl_genome_version                      = 'GRCh37'

# Ensemble assemblies to include in the reference genome
ensembl_assemblies                          = ['chromosome.1', 'chromosome.2', 'chromosome.3', 'chromosome.4', 'chromosome.5', 'chromosome.6', 'chromosome.7', 'chromosome.8', 'chromosome.9', 'chromosome.10', 'chromosome.11', 'chromosome.12', 'chromosome.13', 'chromosome.14', 'chromosome.15', 'chromosome.16', 'chromosome.17', 'chromosome.18', 'chromosome.19', 'chromosome.20', 'chromosome.21', 'chromosome.22', 'chromosome.X', 'chromosome.Y', 'chromosome.MT', 'nonchromosomal']

# Base chromosomes
chromosomes                                 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

# Ensemble reference genome chromosome assemblies
ensembl_assembly_url_template               = 'ftp://ftp.ensembl.org/pub/release-{ensembl_version}/fasta/homo_sapiens/dna/Homo_sapiens.{ensembl_genome_version}.{ensembl_version}.dna.{ensembl_assembly}.fa.gz'

# Ucsc genome version (must match ensembl version!)
ucsc_genome_version                         = 'hg19'

# Locally installed reference genome
genome_fasta_template                       = '{ref_data_directory}/Homo_sapiens.{ensembl_genome_version}.{ensembl_version}.dna.chromosomes.fa'
genome_fai_template                         = '{ref_data_directory}/Homo_sapiens.{ensembl_genome_version}.{ensembl_version}.dna.chromosomes.fa.fai'

# Segment length for automatically generated segments
segment_length                              = int(3e6)

# Length of simulated reads used to calculate mappability
mappability_length                          = 100

# Mapping quality threshold for filtering mappable reads
map_qual_threshold                          = 1

# Filter reads marked as duplicate
filter_duplicates                           = False

# Locally installed mappability filename produced by mappability setup script
mappability_template                        = '{ref_data_directory}/{ucsc_genome_version}.{mappability_length}.bwa.mappability.h5'

# Thousand genomes dataset
thousand_genomes_impute_url                 = 'http://mathgen.stats.ox.ac.uk/impute/ALL_1000G_phase1integrated_v3_impute.tgz'
thousand_genomes_directory                  = '{ref_data_directory}/ALL_1000G_phase1integrated_v3_impute'
sample_template                             = thousand_genomes_directory+'/ALL_1000G_phase1integrated_v3.sample'
legend_template                             = thousand_genomes_directory+'/ALL_1000G_phase1integrated_v3_chr{chromosome}_impute.legend.gz'
haplotypes_template                         = thousand_genomes_directory+'/ALL_1000G_phase1integrated_v3_chr{chromosome}_impute.hap.gz'
genetic_map_template                        = thousand_genomes_directory+'/genetic_map_chr{chromosome}_combined_b37.txt'
phased_chromosome_x                         = 'X_nonPAR'

# Locally installed snps from thousand genomes
snp_positions_template                      = '{ref_data_directory}/thousand_genomes_snps.tsv'

###
# Algorithm parameters
###

# Maximum inferred fragment length of a read pair classified as concordant
bam_max_fragment_length                     = 1000

# Maximum soft clipped bases before a read is called discordant
bam_max_soft_clipped                        = 8

# Heterozygous snp calling
sequencing_base_call_error                  = 0.01
het_snp_call_threshold                      = 0.9

# Shapeit haplotype block resolution
shapeit_num_samples                         = 100
shapeit_confidence_threshold                = 0.95

# GC bias correction
sample_gc_num_positions                     = 10000000
gc_position_offset                          = 4

