#!/usr/bin/env Rscript

calcLogReadcount <- function(x) {
  x$copy <- x$reads / median(x$reads)
  x$copy[x$copy <= 0] = NA
  x$copy <- log(x$copy, 2)
  return(x)
}

calcReadDepth <- function(tumWig, normWig, 
	genomeStyle = "NCBI", targetedSequence = NULL) {
    
    ### LOAD TUMOUR AND NORMAL FILES ###
    message("Loading tumour file:", tumWig)
    tumour_reads <- wigToRangedData(tumWig)
    message("Loading normal file:", normWig)
    normal_reads <- wigToRangedData(normWig)
    
    ### set the genomeStyle: NCBI or UCSC
    #require(GenomeInfoDb)
    if (seqlevelsStyle(names(tumour_reads)) != genomeStyle){
    	names(tumour_reads) <- mapSeqlevels(names(tumour_reads), genomeStyle)
    }
    if (seqlevelsStyle(names(normal_reads)) != genomeStyle){
    	names(normal_reads) <- mapSeqlevels(names(normal_reads), genomeStyle)
    }
    
    colnames(tumour_reads) <- c("reads")
    colnames(normal_reads) <- c("reads")
    
    ### CALCUATE LOG READ COUNT
    message("Calculating log read count")
    tumour_copy <- calcLogReadcount(tumour_reads)
    normal_copy <- calcLogReadcount(normal_reads)
    
    ### COMPUTE LOG RATIO ###
    message("Normalizing Tumour by Normal")
    tumour_copy$copy <- tumour_copy$copy - normal_copy$copy
    rm(normal_copy)
    
    ### PUTTING TOGETHER THE COLUMNS IN THE OUTPUT ###
    temp <- cbind(chr = as.character(space(tumour_copy)), 
        start = start(tumour_copy), end = end(tumour_copy), 
        logR = tumour_copy$copy)
    temp <- as.data.frame(temp, stringsAsFactors = FALSE)
    mode(temp$start) <- "numeric"
    mode(temp$end) <- "numeric"
    mode(temp$logR) <- "numeric"
    return(temp)
}

run.titan <- function(args)
{
	library(TitanCNA)
	
	data <- loadAlleleCounts(args$counts_file)

	if (args$gc_wig_file != '' & args$map_wig_file != '') {
	    message("GC and mappability correction enabled")
		depth.data <- correctReadDepth(args$tumour_wig_file, args$normal_wig_file, args$gc_wig_file, args$map_wig_file)
	}
	else {
	    message("GC and mappability correction disabled")
		depth.data <- calcReadDepth(args$tumour_wig_file, args$normal_wig_file)
	}
	
	logR <- getPositionOverlap(data$chr, data$posn, depth.data)
	
	data$logR <- log(2^logR)

	rm(logR, depth.data)
	
	if (args$map_wig_file != '') {
		mScore <- as.data.frame(wigToRangedData(args$map_wig_file))
		
		mScore <- getPositionOverlap(data$chr, data$posn, mScore[,-4])
		
		data <- filterData(data, 
						   args$chromosomes, 
						   minDepth=args$min_depth, 
						   maxDepth=args$max_depth,
						   map=mScore, 
						   mapThres=0.9)
	}

	params <- loadDefaultParameters(copyNumber=args$max_copy_number, numberClonalClusters=args$num_clusters)
	
	params$normalParams$n_0 <- args$normal_contamination
	
	params$ploidyParams$phi_0 <- args$ploidy
	
	# adjust the heterozygous to account for noise if symmetric genotypes
	if (args$symmetric){
		params$genotypeParams$rt[c(4, 9)] <- args$het_allelic_ratio
	}
	
	# adjust the prior counts (hyperparameters) for Gaussian variance
	K <- length(params$genotypeParams$rt)
	
	params$genotypeParams$alphaKHyper <- rep(args$variance_hyper_param,K)

	if(args$estimate_normal_contamination){
		normal.estimate.method <- 'map'
 	}
	else{
		normal.estimate.method <- 'fixed'
	}

	converged.params <- runEMclonalCN(data,
									  gParams=params$genotypeParams,
									  nParams=params$normalParams,
									  pParams=params$ploidyParams,
									  sParams=params$cellPrevParams,
									  maxiter=args$max_em_iters,
									  maxiterUpdate=args$max_optimisation_iterations,
									  txnExpLen=args$genotype_transition_rate,
									  txnZstrength=args$clonal_transition_rate,
									  useOutlierState=args$use_outlier_state,
									  normalEstimateMethod=normal.estimate.method, 
									  estimateS=args$estimate_clonal_prevalence,
									  estimatePloidy=args$estimate_ploidy)
							
	optimal.path <- viterbiClonalCN(data, converged.params)
	
	# Save position specific results
	results <- outputTitanResults(data, converged.params, optimal.path, filename=args$cn_file, posteriorProbs=F, subcloneProfiles=T)
	
	# Save all params
	outputModelParameters(converged.params, results, args$param_file)
}

suppressPackageStartupMessages(library("argparse"))

parser <- ArgumentParser(description='CLI to run a TITAN analysis.')

parser$add_argument('counts_file',
					help='Path of file allelic count data.')

parser$add_argument('normal_wig_file',
					help='Path of WIG file with normal coverage data.')			

parser$add_argument('tumour_wig_file',
					help='Path of WIG file with tumour coverage data.')			

parser$add_argument('cn_file',
		            help='Path of filename where output copy number will be written.')
			
parser$add_argument('param_file',
		            help='Path of filename where output parameters will be written.')
			
parser$add_argument('--gc_wig_file', default='',
					help='Path of WIG file with GC content data.')	

parser$add_argument('--map_wig_file', default='',
					help='Path of WIG file with mappability data.')			

parser$add_argument('--chromosomes', default=c(1:22,"X","Y"), nargs='+',
					help='Space delimited list of chromosomes to analyse. Default is to analyse 1-22,X,Y.')			

parser$add_argument('--clonal_transition_rate', default=1e5, type='double',
					help='Controls probability of transition of clonal population state. Lower values will lead to more frequent transitions. Default is 1e5.')			

parser$add_argument('--estimate_clonal_prevalence', default=FALSE, action='store_true',
					help='If set, then the will estimate the prevalence of sub-clonal events.')				
			
parser$add_argument('--estimate_normal_contamination', default=FALSE, action='store_true',
					help='If set, then the proportion of normal cells will be estimated.')				

parser$add_argument('--estimate_ploidy', default=FALSE, action='store_true',
					help='If set a the ploidy of the cancer cells will be estimated.')				
			
parser$add_argument('--genotype_transition_rate', default=1e15, type='double',
					help='Controls probability of transition of genotype state. Lower values will lead to more frequent transitions. Default is 1e15.')				
			
parser$add_argument('--het_allelic_ratio', default=0.58, type='double',
					help='Adjusts baseline heterozygous allelic ratio. Will only be used if --symmetric flag is set. Default is 0.58.')	
					
parser$add_argument('--variance_hyper_param', default=15000, type='double',
					help='Hyperparameter (prior counts) on Gaussian variance for log ratios. Default is 15000.')	

parser$add_argument('--max_copy_number', default=5, type='integer',
					help='Maximum total copy number of a state. Default is 5.')					

parser$add_argument('--max_depth', default=1000, type='integer',
					help='Positions with total allelic counts above this value will be ignored. Default is 1000.')

parser$add_argument('--max_em_iters', default=1000, type='integer',
					help='Maximum number of EM iterations to perform. Default is 1000.')
			
parser$add_argument('--max_optimisation_iterations', default=2000, type='integer',
					help='Maximum number of iterations to use for performing optimisation of parameters at each EM step. Default is 2000.')			
			
parser$add_argument('--min_depth', default=0, type='integer',
					help='Positions with total allelic counts below this value will be ignored. Default is 0.')			

parser$add_argument('--normal_contamination', default=0.0, type='double',
					help='The fraction of normal cells in the sample. If the --estimate_normal_contamination flag is set this will only be used as an initial value. Default is 0.0.')			
			
parser$add_argument('--num_clusters', default=1, type='integer',
					help='Number of sub-clonal population clusters to use. Default is 1.')	

parser$add_argument('--num_cpus', default=1, type='integer',
					help='Number of computer cores to use. Default is 1.')		

parser$add_argument('--ploidy', default=2.0, type='double',
					help='The ploidy of cancer cells. If the --estimate_ploidy flag is set this will be only be used as an initial value. Default is 2.0.')			
			
parser$add_argument('--symmetric', default=TRUE, action='store_true',
					help='If set, then symmetric genotypes are used. This reduces the number of genotype states and can decrease run-time.')	
					
parser$add_argument('--use_outlier_state', default=FALSE, action='store_true',
					help='If set an additional emission state will be added for outliers.')				
			
args <- parser$parse_args()

library(doMC)

registerDoMC(cores=args$num_cpus)

run.titan(args)
