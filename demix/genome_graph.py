import subprocess
import time
import itertools
import uuid
import os
import pickle
import numpy as np
import pandas as pd


blossomv_bin = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'bin', 'blossom5'))

def df_stack(df, cols, suffixes):

    other_columns = list(df.columns.values)
    for suffix in suffixes:
        for col in cols:
            other_columns.remove(col + suffix)

    stacked = list()

    for suffix in suffixes:
        stack_columns = list()
        for col in cols:
            stack_columns.append(col + suffix)
        data = df[stack_columns + other_columns].copy()
        data['suffix'] = suffix
        data = data.rename(columns=dict(zip(stack_columns, cols)))
        stacked.append(data)

    return pd.concat(stacked, ignore_index=True)

def df_merge_product(df, col, values):

    merged = list()

    for value in values:

        data = df.copy()
        data[col] = value

        merged.append(data)

    return pd.concat(merged, ignore_index=True)


vertex_cols = ['n', 'ell', 'side']


edge_cols = ['n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2']


def create_adj_table(adj):

    adj_table = list()

    for ((n_1, ell_1), side_1), ((n_2, ell_2), side_2) in adj:

        end_1 = (n_1, ell_1, side_1)
        end_2 = (n_2, ell_2, side_2)

        # Ensure that vertices are sorted for edge
        end_1, end_2 = sorted([end_1, end_2])

        adj_table.append(end_1 + end_2)

    adj_table = pd.DataFrame(adj_table, columns=edge_cols)

    return adj_table


def f_cn_col(m):
    return 'cn_{0}'.format(m)


class GenomeGraph(object):

    def __init__(self, model, x, l, cn_init, wt_adj, tmr_adj):
        """ Create a GenomeGraph.

        Args:
            model (CopyNumberModel): model used to calculate edge likelihoods
            x (numpy.array): observed read counts
            l (numpy.array): observed lengths of segments
            cn_init (numpy.array): initial copy number matrix
            wt_adj (list of 'breakpoint'): list of wild type adjacencies
            tmr_adj (list of 'breakpoint'): list of tumour adjacencies
            params (dict): parameters of the copy number model

        A 'breakpoint' is represented as the frozenset (['breakend_1', 'breakend_2'])

        A 'breakend' is represented as the tuple (('segment', 'allele'), 'side').

        """

        self.x = x
        self.l = l
        self.cn = cn_init.copy()
        self.model = model

        self.debug_prefix = None

        self.integral_cost_scale = 100.

        # If some tumour adjacencies are also wild type adjacencies, we will
        # get problems with maintenence of the copy balance condition
        assert len(wt_adj.intersection(tmr_adj)) == 0

        self.telomere_cost = 10.
        self.breakpoint_cost = 2.

        self.N = self.cn.shape[0]
        self.M = self.cn.shape[1]

        self.cn_cols = [f_cn_col(m) for m in xrange(self.M)]
        self.tumour_cn_cols = [f_cn_col(m) for m in xrange(1, self.M)]

        # Create reference bond edges
        wt_table = create_adj_table(wt_adj)
        wt_table['is_reference'] = True

        # Create breakpoint bond edges
        tmr_table = create_adj_table(tmr_adj)
        tmr_table['is_breakpoint'] = True

        # Create telomere bond edges
        self.tel_segment_idx = self.N
        tel_table = list()

        v_1_iter = itertools.product(xrange(self.N), xrange(2), xrange(2))
        v_t_iter = itertools.product([self.tel_segment_idx], xrange(2), xrange(2))

        for v_1, v_t in itertools.product(v_1_iter, v_t_iter):
            assert v_1 < v_t
            tel_table.append(v_1 + v_t)

        tel_table = pd.DataFrame(tel_table, columns=edge_cols)
        tel_table['is_telomere'] = True

        # Table of bond copy number
        self.bond_cn = pd.concat([wt_table, tmr_table, tel_table], ignore_index=True)
        self.bond_cn['is_reference'] = self.bond_cn['is_reference'].fillna(False)
        self.bond_cn['is_breakpoint'] = self.bond_cn['is_breakpoint'].fillna(False)
        self.bond_cn['is_telomere'] = self.bond_cn['is_telomere'].fillna(False)

        # Initialize bond copy number to 0
        for m in xrange(self.M):
            self.bond_cn[f_cn_col(m)] = 0

        # Initialize bond reference edge copy number to minimum of adjacent segment copy number
        self.tel_segment_cn = np.zeros((self.M, 2))
        for idx, row in self.bond_cn[self.bond_cn['is_reference']].iterrows():

            for m in xrange(self.M):

                cn_1 = self.cn[row['n_1'],m,row['ell_1']]
                cn_2 = self.cn[row['n_2'],m,row['ell_2']]

                cn_ref = min(cn_1, cn_2)

                self.bond_cn.loc[idx, f_cn_col(m)] = min(cn_1, cn_2)

        # Calculate total bond copy number incident at vertices
        vertex_bond_cn = (
            df_stack(self.bond_cn, vertex_cols, ('_1', '_2'))
            .groupby(vertex_cols)[self.cn_cols+['is_reference']]
            .sum()
        )

        # Initialize telomere edge copy number to remainder from segment and bond
        # Calculate total telomere edge copy number
        for n in xrange(self.N):

            for ell in xrange(2):

                for side in xrange(2):

                    telomere_row = (
                        (self.bond_cn['n_1'] == n) &
                        (self.bond_cn['n_2'] == self.tel_segment_idx) &
                        (self.bond_cn['ell_1'] == ell) &
                        (self.bond_cn['ell_2'] == ell) &
                        (self.bond_cn['side_1'] == side) &
                        (self.bond_cn['side_2'] == side)
                    )

                    if not vertex_bond_cn.loc[(n,ell,side),'is_reference']:

                        self.bond_cn.loc[telomere_row, 'is_telomere'] = False
                        self.bond_cn.loc[telomere_row, 'is_reference'] = True

                    for m in xrange(self.M):

                        cn_seg = self.cn[n,m,ell]
                        cn_bond = vertex_bond_cn.loc[(n,ell,side),f_cn_col(m)]
                        cn_tel = cn_seg - cn_bond

                        self.bond_cn.loc[telomere_row, f_cn_col(m)] = cn_tel

                        self.tel_segment_cn[m,ell] += cn_tel

        # Each copy at a telomere was double counted towards the telomere segment copy number
        self.tel_segment_cn /= 2

        # Sign of modifications
        self.signs = (+1, -1)

        # Create a table of segment edges in modification graph
        self.mod_seg_edges = pd.DataFrame(list(itertools.product(xrange(self.N+1), xrange(2), self.signs)),
                                          columns=['n_1', 'ell_1', 'sign'])
        self.mod_seg_edges['n_2'] = self.mod_seg_edges['n_1']
        self.mod_seg_edges['ell_2'] = self.mod_seg_edges['ell_1']
        self.mod_seg_edges['side_1'] = 0
        self.mod_seg_edges['side_2'] = 1

        self.mod_seg_edges.sort('n_1')

        # List of vertices of matching graph
        v_iter = itertools.product(xrange(self.N+1), xrange(2), xrange(2), self.signs)
        self.matching_vertices = pd.DataFrame(list(v_iter), columns=['n', 'ell', 'side', 'color'])
        self.matching_vertices['vertex_id'] = xrange(len(self.matching_vertices.index))

        # List of vertex edges of matching graph
        self.matching_vertex_edges = self.matching_vertices.set_index(['n', 'ell', 'side', 'color']) \
                                                           .unstack()['vertex_id'] \
                                                           .reset_index(drop=True) \
                                                           .rename(columns={-1:'vertex_id_1', 1:'vertex_id_2'})
        self.matching_vertex_edges['cost'] = -1e-8


    def build_mod_seg_edge_costs(self, delta, h):
        """ Create a table of segment edge costs
        
        Args:
            delta (numpy.array): copy number modification
            h (numpy.array): haploid read depths

        Returns:
            pandas.DataFrame: segment edge modification costs table

        The returned table has the following essential columns:
            'n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2', 'sign', 'cost'

        """

        mod_seg_edge_costs = list()

        log_likelihood = self.model.log_likelihood_cn(self.x, self.l, self.cn, h)

        for sign in self.signs:

            for ell in xrange(2):

                cn_delta = self.cn.copy()
                cn_delta[:,:,ell] += sign * delta

                invalid_cn_delta = np.any(cn_delta < 0, axis=(1, 2))
                cn_delta[invalid_cn_delta] = 1

                log_likelihood_delta = self.model.log_likelihood_cn(self.x, self.l, cn_delta, h)

                log_likelihood_delta[invalid_cn_delta] = -np.inf

                cost = log_likelihood - log_likelihood_delta

                data = pd.DataFrame({'cost':cost})
                for m in xrange(self.M):
                    data[f_cn_col(m)] = cn_delta[:,m,ell]
                data['n_1'] = xrange(self.N)
                data['ell_1'] = ell
                data['sign'] = sign

                mod_seg_edge_costs.append(data)

                tel_cn_delta = self.tel_segment_cn.copy()
                tel_cn_delta[:,ell] += sign * delta

                tel_cost = 0

                if np.any(tel_cn_delta < 0):
                    tel_cost = np.inf

                tel_data = pd.DataFrame([{'cost':tel_cost,
                                          'sign':sign,
                                          'n_1':self.tel_segment_idx,
                                          'n_2':self.tel_segment_idx,
                                          'ell_1':ell,
                                          'ell_2':ell,
                                          'side_1':0,
                                          'side_2':1}])

                for m in xrange(self.M):
                    tel_data[f_cn_col(m)] = tel_cn_delta[m,ell]

                mod_seg_edge_costs.append(tel_data)

        mod_seg_edge_costs = pd.concat(mod_seg_edge_costs, ignore_index=True)
        mod_seg_edge_costs['n_2'] = mod_seg_edge_costs['n_1']
        mod_seg_edge_costs['ell_2'] = mod_seg_edge_costs['ell_1']
        mod_seg_edge_costs['side_1'] = 0
        mod_seg_edge_costs['side_2'] = 1

        return mod_seg_edge_costs


    def build_mod_bond_edge_costs(self, delta, h):
        """ Create a table of bond edge costs
        
        Args:
            delta (numpy.array): copy number modification
            h (numpy.array): haploid read depths

        Returns:
            pandas.DataFrame: bond edge modification costs table

        The returned table has the following essential columns:
            'n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2', 'sign', 'cost'

        """

        # Merge in current copy number
        mod_bond_edge_costs = df_merge_product(self.bond_cn, 'sign', self.signs)

        # Modify copy number
        for m in xrange(self.M):
            mod_bond_edge_costs[f_cn_col(m)] += mod_bond_edge_costs['sign'] * delta[m]

        # Set default edge cost to zero
        mod_bond_edge_costs['cost'] = 0

        # Set cost for telomere edges
        is_telomere = (mod_bond_edge_costs['is_telomere'])
        mod_bond_edge_costs.loc[is_telomere, 'cost'] = (
            self.telomere_cost * mod_bond_edge_costs.loc[is_telomere, 'sign'] * delta.sum())

        # Set cost for breakpoint edges
        is_breakpoint = (mod_bond_edge_costs['is_breakpoint'])
        mod_bond_edge_costs.loc[is_breakpoint, 'cost'] = (
            self.breakpoint_cost * mod_bond_edge_costs.loc[is_breakpoint, 'sign'] * delta.sum())

        # Set cost to infinite for edges with negative copy number
        for m in xrange(self.M):
            mod_bond_edge_costs.loc[(mod_bond_edge_costs[f_cn_col(m)] < 0.), 'cost'] = np.inf

        return mod_bond_edge_costs


    def build_mod_edge_costs(self, delta, h):
        """ Build a table of bond and segment edge costs
        
        Args:
            delta (numpy.array): copy number modification
            h (numpy.array): haploid read depths

        Returns:
            pandas.DataFrame: edge modification costs table

        The returned table has the following essential columns:
            'vertex_id_1', 'vertex_id_2', 'cost'

        """

        mod_seg_edge_costs = self.build_mod_seg_edge_costs(delta, h)
        mod_bond_edge_costs = self.build_mod_bond_edge_costs(delta, h)

        # Label edges as segment or bond
        mod_seg_edge_costs['is_seg'] = True
        mod_bond_edge_costs['is_seg'] = False

        # Label color of edges, invert bond edges
        mod_seg_edge_costs['color'] = mod_seg_edge_costs['sign']
        mod_bond_edge_costs['color'] = -mod_bond_edge_costs['sign']

        # Create a table of all edges
        mod_edge_costs = pd.concat([mod_seg_edge_costs, mod_bond_edge_costs], ignore_index=True)

        # Merge vertex ids
        for a in ('_1', '_2'):
            mod_edge_costs = mod_edge_costs.merge(self.matching_vertices,
                                                  left_on=['n'+a, 'ell'+a, 'side'+a, 'color'],
                                                  right_on=['n', 'ell', 'side', 'color'], how='left')
            mod_edge_costs.drop(['n', 'ell', 'side'], axis=1, inplace=True)
            mod_edge_costs.rename(columns={'vertex_id':'vertex_id'+a}, inplace=True)

        # Drop infinte cost edges
        mod_edge_costs = mod_edge_costs[mod_edge_costs['cost'].replace(np.inf, np.nan).notnull()]

        # Select least cost edge for duplicated edges
        mod_edge_costs = mod_edge_costs.sort('cost')
        mod_edge_costs = mod_edge_costs.groupby(['vertex_id_1', 'vertex_id_2', 'color'],
                                                sort=False) \
                                       .first() \
                                       .reset_index()

        return mod_edge_costs


    def optimize_modification(self, delta, h):
        """ Calculate optimal modification moving by +/- delta.

        Args:
            delta (numpy.array): copy number modification
            h (numpy.array): haploid read depths

        Returns:
            float, list: minimized cost, list of edges in minimum cost modification

        """

        mod_edge_costs = self.build_mod_edge_costs(delta, h)

        blossom_input_filename = str(uuid.uuid4())
        blossom_output_filename = str(uuid.uuid4())

        with open(blossom_input_filename, 'w') as graph_file:
            
            num_edges = len(mod_edge_costs.index) + len(self.matching_vertex_edges.index)
            num_vertices = len(self.matching_vertices.index)
            
            graph_file.write('{0} {1}\n'.format(num_vertices, num_edges))

            # print '# edges ', num_edges
            # print '# vertices ', num_vertices

            edge_cost_cols = ['vertex_id_1', 'vertex_id_2', 'cost']

            edge_costs = [
                self.matching_vertex_edges[edge_cost_cols],
                mod_edge_costs[edge_cost_cols],
            ]
            edge_costs = pd.concat(edge_costs, ignore_index=True)

            edge_costs['cost'] = np.rint(edge_costs['cost'] * self.integral_cost_scale).astype(int)

            edge_costs.to_csv(graph_file, sep='\t', columns=edge_cost_cols, index=False, header=False)

        subprocess.check_call([blossomv_bin,
                               '-e', blossom_input_filename,
                               '-w', blossom_output_filename], stdout=subprocess.PIPE)

        min_cost_edges = list()

        with open(blossom_output_filename, 'r') as graph_file:
            
            first = True
            
            for line in graph_file:
                
                if first:
                    num_vertices, num_edges = [int(a) for a in line.split()]
                    first = False
                    continue
                
                vertex_id_1, vertex_id_2 = [int(a) for a in line.split()]

                min_cost_edges.append((vertex_id_1, vertex_id_2))
                min_cost_edges.append((vertex_id_2, vertex_id_1))

        os.remove(blossom_input_filename)
        os.remove(blossom_output_filename)

        min_cost_edges = pd.DataFrame(min_cost_edges, columns=['vertex_id_1', 'vertex_id_2'])

        min_cost_edges = mod_edge_costs.merge(min_cost_edges)

        # Remove duplicated edges resulting from non-convex cost functions
        min_cost_edges.set_index(edge_cols, inplace=True)
        min_cost_edges['edge_count'] = min_cost_edges.groupby(level=range(len(edge_cols))).size()
        min_cost_edges.reset_index(inplace=True)
        assert not np.any(min_cost_edges['edge_count'] > 2)
        min_cost_edges = min_cost_edges[min_cost_edges['edge_count'] == 1]

        total_cost = min_cost_edges['cost'].sum()

        return total_cost, min_cost_edges


    def apply_modification(self, delta, modification):
        """ Apply the given modification to the segment/bond copy number.

        Args:
            delta (numpy.array): copy number modification
            modification (list): list of edges in minimum cost modification

        """

        seg_mods = modification[modification['is_seg']]
        bond_mods = modification[~modification['is_seg']]

        for n, ell, sign in seg_mods[['n_1', 'ell_1', 'sign']].values:
            if n == self.tel_segment_idx:
                self.tel_segment_cn[:,ell] += sign * delta
            else:
                self.cn[n,:,ell] += sign * delta

        for idx, row in seg_mods[seg_mods.duplicated(edge_cols)].iterrows():
            raise Exception('duplicated edge {0}'.format(row[edge_cols].to_dict()))

        for idx, row in bond_mods[bond_mods.duplicated(edge_cols)].iterrows():
            raise Exception('duplicated edge {0}'.format(row[edge_cols].to_dict()))

        for m in xrange(self.M):

            self.bond_cn = self.bond_cn.merge(bond_mods[edge_cols + [f_cn_col(m)]],
                                              on=edge_cols,
                                              suffixes=('', '_mod'),
                                              how='left')
            mod_cn_col = f_cn_col(m) + '_mod'

            self.bond_cn.loc[self.bond_cn[mod_cn_col].notnull(),
                             f_cn_col(m)] = self.bond_cn[mod_cn_col]

            self.bond_cn = self.bond_cn.drop(mod_cn_col, axis=1)


    def test_circulation(self):
        """ Test if current copy number state represents a valid circulation.

        Raises:
            Exception: raised if not a valid circulation

        """

        vertex_bond_cn = df_stack(self.bond_cn, vertex_cols, ('_1', '_2'))

        vertex_seg_cn = list()

        for n in xrange(self.N):
            
            for ell in xrange(2):

                for side in (0, 1):

                    vertex_seg_cn.append([n, ell, side] + list(-self.cn[n,:,ell]))

        for ell in xrange(2):

            for side in (0, 1):

                vertex_seg_cn.append([self.tel_segment_idx, ell, side] + list(-self.tel_segment_cn[:,ell]))

        vertex_seg_cn = pd.DataFrame(vertex_seg_cn, columns=['n', 'ell', 'side'] + self.cn_cols)

        vertex_cn = pd.concat([vertex_bond_cn, vertex_seg_cn], ignore_index=True)

        vertex_sum = vertex_cn.groupby(vertex_cols).sum()

        for v, v_sum in vertex_sum.iterrows():

            if v_sum.sum() != 0:

                raise Exception('vertex {0} has nonzero sum {1}'.format(v, v_sum.values))


    def calculate_log_posterior(self, h):
        """ Calculate log posterior of segment/bond copy number.

        Args:
            h (numpy.array): haploid read depths

        Returns:
            float: log posterior of segment/bond copy number.

        """

        log_likelihood = self.model.log_likelihood_cn(self.x, self.l, self.cn, h)

        log_likelihood = log_likelihood.sum()

        telomere_copies = self.bond_cn.loc[self.bond_cn['is_telomere'], self.tumour_cn_cols].values.sum()
        breakpoint_copies = self.bond_cn.loc[self.bond_cn['is_breakpoint'], self.tumour_cn_cols].values.sum()

        log_posterior = (log_likelihood - self.telomere_cost * telomere_copies - self.breakpoint_cost * breakpoint_copies)

        return log_posterior


    def optimize(self, h, max_iter=1000):
        """ Calculate optimal segment/bond copy number.

        Args:
            h (numpy.array): haploid read depths

        KwArgs:
            max_iter (int): maximum iterations

        Returns:
            numpy.array: optimized segment copy number

        """

        self.test_circulation()

        deltas = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 1, -1]), np.array([0, 1, 1])]

        log_posterior_prev = self.calculate_log_posterior(h)

        iter = 0

        while True:

            mod_list = list()

            for delta in deltas:

                cost, modification = self.optimize_modification(delta, h)

                mod_list.append((cost, delta, modification))

            mod_list.sort(key=lambda a: a[0])

            best_cost, best_delta, best_modification = mod_list[0]

            if best_cost == 0:
                break

            if self.debug_prefix is not None:
                with open(self.debug_prefix + str(iter) + '.debug', 'w') as debug_file:
                    pickle.dump((best_cost, best_delta, best_modification), debug_file)

            self.apply_modification(best_delta, best_modification)

            self.test_circulation()

            log_posterior = self.calculate_log_posterior(h)

            if log_posterior < log_posterior_prev:
                raise Exception('decreased log posterior from {0} to {1}'.format(log_posterior_prev, log_posterior))

            log_posterior_prev = log_posterior

            iter += 1
            if iter > max_iter:
                break

        return self.cn, log_posterior_prev


    @property
    def breakpoint_copy_number(self):

        brk_cn = dict()

        bond_cn_breakpoints = self.bond_cn[(self.bond_cn['is_breakpoint']) &
                                           (self.bond_cn[self.tumour_cn_cols].sum(axis=1) > 0)]

        for idx, row in bond_cn_breakpoints.iterrows():

            breakpoint = frozenset([((row['n_1'], row['ell_1']), row['side_1']),
                                    ((row['n_2'], row['ell_2']), row['side_2'])])

            brk_cn[breakpoint] = row[self.cn_cols].values

        return brk_cn

