'''
Extracts a segments from the output of TITAN. Currently only segments from the best ploidy and coarsest segmentation are
considered.

@author Andrew Roth
'''
from collections import OrderedDict

import numpy as np
import pandas as pd

def main(args):
    state_map = get_state_map(args.max_copy_number)

    df = pd.read_csv(args.in_file, sep='\t')
    
    df = df.rename(columns={'Chr' : 'chrom',
                            'Position' : 'coord',
                            'CellularPrevalence' : 'cn_prevalence',
                            'TITANstate' : 'state'})

    df['minor_cn'] = df.state.apply(lambda x: state_map[x]['minor_cn'])
    
    df['major_cn'] = df.state.apply(lambda x: state_map[x]['major_cn'])
    
    df['cn_prevalence'][pd.isnull(df['cn_prevalence'])] = 1.0
    
    minor_cn_indices = df.minor_cn.shift(1) != df.minor_cn
    
    major_cn_indices = df.major_cn.shift(1) != df.major_cn
    
    chrom_indices = df.chrom.shift(1) != df.chrom
    
    cn_prevalence_indices = df.cn_prevalence.shift(1) != df.cn_prevalence
    
    df['segment_id'] = (minor_cn_indices | major_cn_indices | chrom_indices | cn_prevalence_indices).astype(int).cumsum()
    
    group = df.groupby(['chrom', 'minor_cn', 'major_cn', 'cn_prevalence', 'segment_id'], sort=False)
    
    # Find begining and end of segments
    seg_data = group.coord.agg([min, max]).reset_index()
    
    # Rename beg and end columns
    seg_data.rename(columns={'min' : 'beg', 'max' : 'end'}, inplace=True)
    
    # Add total CN columns
    seg_data['total_cn'] = seg_data['minor_cn'] + seg_data['major_cn']
    
    # Sort data
    seg_data = seg_data.sort_values(columns=['chrom', 'beg'])
    
    # Compute dominant genotype
    seg_data['dominant_major_cn'] = seg_data['major_cn']
    
    seg_data['dominant_minor_cn'] = seg_data['minor_cn']
    
    seg_data['dominant_major_cn'][seg_data['cn_prevalence'] < 0.5] = 1
    
    seg_data['dominant_minor_cn'][seg_data['cn_prevalence'] < 0.5] = 1
    
    # Compute alt genotype
    seg_data['alt_major_cn'] = seg_data['major_cn']
    
    seg_data['alt_minor_cn'] = seg_data['minor_cn']
    
    seg_data['alt_major_cn'][seg_data['cn_prevalence'] >= 0.5] = 1
    
    seg_data['alt_minor_cn'][seg_data['cn_prevalence'] >= 0.5] = 1
    
    # Compute alt prevalence
    seg_data['mirror_prevalence'] = 1 - seg_data['cn_prevalence']
    
    seg_data['alt_prevalence'] = seg_data[['cn_prevalence', 'mirror_prevalence']].min(axis=1)
    
    # Update output values
    seg_data['major_cn'] = seg_data['dominant_major_cn']
    
    seg_data['minor_cn'] = seg_data['dominant_minor_cn']
    
    seg_data['total_cn'] = seg_data['major_cn'] + seg_data['minor_cn']
    
    seg_data['alt_total_cn'] = seg_data['alt_major_cn'] + seg_data['alt_minor_cn']

    out_fields = ['chrom', 'beg', 'end', 'minor_cn', 'major_cn', 'total_cn', 'alt_minor_cn', 'alt_major_cn', 'alt_total_cn', 'alt_prevalence']
    
    seg_data = seg_data[out_fields]
    
    seg_data.to_csv(args.out_file, index=False, na_rep='NA', sep='\t')

def get_state_map(max_cn):
    state_map = OrderedDict()
    
    state = 0
    
    for cn in range(max_cn + 1):
        for num_ref_allele in range(cn + 1):
            num_var_allele = cn - num_ref_allele
            
            minor_cn = min(num_ref_allele, num_var_allele)
            
            major_cn = max(num_ref_allele, num_var_allele)
            
            state_map[state] = {'minor_cn' : minor_cn, 'major_cn' : major_cn}
            
            state += 1
    
    state_map[-1] = {'minor_cn' : 'NA', 'major_cn' : 'NA'}
            
    return state_map

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('in_file', help='''Path to position specific results file produced by TITAN.''')
    
    parser.add_argument('out_file', help='''Path where output file will be written in tsv format.''')
    
    parser.add_argument('--max_copy_number', type=int, default=5,
                        help='''Maximum copy number used for the analysis.''')
    
    args = parser.parse_args()
    
    main(args)
