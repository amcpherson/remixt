"""Scan remixt results for chromosomes truncated by the seqdata early stop bug.

Before the fix in this branch, seqdata extraction for a chromosome stopped
silently when a chunk of 10M alignments contained no usable reads (e.g. a
pileup of discordant reads at an artifact locus).  Two signatures remain in
the results /cn table:

  tumour truncated: a long run of segments at the end of a chromosome with
      zero total readcount in that sample (called as 0/0).
  normal truncated: a long run at the end of a chromosome with total reads
      but zero allele (het SNP) reads, in every tumour of the analysis.
      Not checked on X/Y, which lack het SNPs in males.

Usage:
  python check_truncated_seqdata.py results1.h5 [results2.h5 ...] > report.tsv

Prints one row per (file, chromosome) with the trailing spans, flagged when
either span exceeds --min-span (default 5Mb).
"""

import argparse
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def trailing_span(chrom_cn, is_empty):
    """Length (bp) of the run of empty segments at the end of the chromosome."""
    chrom_cn = chrom_cn.sort_values('start')
    nonempty = chrom_cn.loc[~is_empty.loc[chrom_cn.index]]
    chrom_end = chrom_cn['end'].max()
    if len(nonempty) == 0:
        return chrom_end - chrom_cn['start'].min(), chrom_cn['start'].min()
    last_end = nonempty['end'].max()
    return chrom_end - last_end, last_end


def check_file(filename, min_span):
    with pd.HDFStore(filename, 'r') as store:
        cn = store['/cn']
    cn['chromosome'] = cn['chromosome'].astype(str)
    allele_reads = cn['major_readcount'] + cn['minor_readcount']
    no_reads = cn['readcount'] == 0
    no_allele_reads = (allele_reads == 0) & (cn['readcount'] > 0)
    no_allele_reads = no_allele_reads | no_reads

    rows = []
    for chrom, chrom_cn in cn.groupby('chromosome'):
        reads_span, reads_last = trailing_span(chrom_cn, no_reads)
        allele_span, allele_last = trailing_span(chrom_cn, no_allele_reads)
        flag = []
        if reads_span >= min_span:
            flag.append('tumour_truncated')
        elif allele_span >= min_span and chrom not in ('X', 'Y', 'chrX', 'chrY'):
            flag.append('no_allele_reads')
        rows.append({
            'file': filename,
            'chromosome': chrom,
            'chrom_end': chrom_cn['end'].max(),
            'last_pos_with_reads': reads_last,
            'trailing_zero_read_span': reads_span,
            'last_pos_with_allele_reads': allele_last,
            'trailing_zero_allele_span': allele_span,
            'flag': ','.join(flag),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+')
    parser.add_argument('--min-span', type=float, default=5e6)
    parser.add_argument('--all', action='store_true', help='print unflagged chromosomes too')
    args = parser.parse_args()

    out = []
    for filename in args.results:
        try:
            out.extend(check_file(filename, args.min_span))
        except Exception as e:
            out.append({'file': filename, 'flag': f'error: {type(e).__name__}: {e}'})
    out = pd.DataFrame(out)
    if not args.all:
        out = out[out['flag'] != '']
    out.to_csv(sys.stdout, sep='\t', index=False)


if __name__ == '__main__':
    main()
