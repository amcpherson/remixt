import collections
import subprocess
import itertools
import uuid
import os
import networkx
import random
import numpy as np
import pandas as pd

import remixt.blossomv

blossomv_bin = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'bin', 'blossom5'))


def min_weight_perfect_matching(edges):
    blossom_input_filename = str(uuid.uuid4())
    blossom_output_filename = str(uuid.uuid4())

    number_of_nodes = np.array(edges.keys()).max() + 1

    with open(blossom_input_filename, 'w') as graph_file:
        
        graph_file.write('{} {}\n'.format(number_of_nodes, len(edges)))

        for (i, j), w in edges.iteritems():
            graph_file.write('{} {} {}\n'.format(i, j, w))

    subprocess.check_call([blossomv_bin,
                           '-e', blossom_input_filename,
                           '-w', blossom_output_filename], stdout=subprocess.PIPE)

    min_cost_edges = set()

    with open(blossom_output_filename, 'r') as graph_file:
        
        first = True
        
        for line in graph_file:
            
            if first:
                num_vertices, num_edges = [int(a) for a in line.split()]
                first = False
                continue
            
            vertex_id_1, vertex_id_2 = [int(a) for a in line.split()]

            min_cost_edges.add((vertex_id_1, vertex_id_2))
            min_cost_edges.add((vertex_id_2, vertex_id_1))

    os.remove(blossom_input_filename)
    os.remove(blossom_output_filename)

    return min_cost_edges


vertex_cols = ['n', 'ell', 'side']


edge_cols = ['n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2']


def f_cn_col(m):
    return 'cn_{0}'.format(m)

N = 3
adjacencies = [(0, 1), (1, 2)]
breakpoints = [((1, 0), (2, 0)), ((0, 0), (0, 1))]

def create_genome_graph(N, adjacencies, breakpoints):
    """ Create a genome graph.

    Args:
        N (int): number of segments
        adjacencies (list of tuple): wild type adjacencies
        breakpoints (list of tuple): breakpoints, pairs of segment, side

    Returns:
        networkx.MultiGraph: genome graph

    A genome graph is defined as a multi-graph on the set of segment
    extremities, plus additional dummy telomere nodes.

    Edges are classified as segment or bond. Bond edges are further
    classified as breakpoint, reference or telomere.  Segment edges
    include dummy edges between dummy telomere nodes.

    Telomere edges are a complete bipartite graph on regular segment
    extremeties and the dummy segment extremeties.  Dummy edges are
    a complete graph on dummy nodes.

    Each edge is given an edge index unique within its edge type.

    """

    G = networkx.MultiGraph()

    num_dummy_nodes = 2

    # Segment node generator
    def segment_nodes():
        return itertools.product(xrange(N), (0, 1))

    # Dummy node generator
    def dummy_nodes():
        return itertools.product(xrange(N, N + num_dummy_nodes), (None,))

    # Dummy edge generator, complete graph on dummy nodes
    def dummy_edges():
        return itertools.combinations(dummy_nodes(), 2)

    # Telomere edge generator, complete bipartite graph on segment and dummy nodes
    def telomere_edges():
        return itertools.product(segment_nodes(), dummy_nodes())

    # Add segment end nodes
    G.add_nodes_from(segment_nodes())

    # Add dummy nodes
    G.add_nodes_from(dummy_nodes())

    # Add segment edges
    for n in xrange(N):
        for allele in xrange(2):
            node_1 = (n, 0)
            node_2 = (n, 1)
            G.add_edge(node_1, node_2, edge_type='segment', edge_idx=n, allele=allele)

    # Add dummy edges as complete graph on dummy nodes
    num_dummy_edges = 0
    for edge_idx, (node_1, node_2) in enumerate(dummy_edges()):
        G.add_edge(node_1, node_2, edge_type='dummy', edge_idx=edge_idx)
        num_dummy_edges += 1

    # Add reference edges
    for edge_idx, (n_1, n_2) in enumerate(adjacencies):
        node_1 = (n_1, 1)
        node_2 = (n_2, 0)
        for node in (node_1, node_2):
            assert node in G
        G.add_edge(node_1, node_2, edge_type='reference', edge_idx=edge_idx)

    # Add breakpoint edges
    for edge_idx, (node_1, node_2) in enumerate(breakpoints):
        for node in (node_1, node_2):
            assert node in G
        G.add_edge(node_1, node_2, edge_type='breakpoint', edge_idx=edge_idx)

    # Add telomere edges
    for edge_idx, (seg_node, tel_node) in enumerate(telomere_edges()):
        G.add_edge(seg_node, tel_node, edge_type='telomere', edge_idx=edge_idx)

    return G


def create_genome_mod_graph(G):
    """ Create a genome modification graph.

    Args:
        G (networkx.MultiGraph): genome graph

    Returns:
        networkx.MultiGraph: genome modification graph

    The genome modification graph is defined as follows:
     - identical node set as genome graph
     - create a +1 and -1 signed edge for each original edge
     - color each edge:
        - segment edges: +1 -> red, -1 -> blue
        - bond edges: +1 -> blue, -1 -> red

    """

    # Create graph with duplicated edges for each sign
    H = networkx.MultiGraph()
    H.add_nodes_from(G)

    # Add two edges to H for each edge in G, give opposite signs
    # Color each edge, +1 for red, -1 for blue
    for node_1, node_2, edge_attr in G.edges_iter(data=True):
        edge_type = edge_attr['edge_type']
        for sign in (-1, 1):
            if edge_type == 'segment':
                color = sign
            elif edge_type == 'dummy':
                color = sign
            elif edge_type == 'breakpoint':
                color = -sign
            elif edge_type == 'reference':
                color = -sign
            elif edge_type == 'telomere':
                color = -sign
            else:
                raise ValueError('unknown edge type {}'.format(edge_type))
            H.add_edge(node_1, node_2, attr_dict=edge_attr, sign=sign, color=color)

    return H


def create_matching_graph(H, transverse_edge_bonus):
    """ Create matching graph.

    Args:
        H (networkx.MultiGraph): genome modification graph
        transverse_edge_bonus: cost of transverse edges

    Returns:
        networkx.Graph: genome modification matching graph

    The genome modification matching graph is defined as follows:
     - duplicate nodes, one set red, one set blue
     - add transverse edges (v_red, v_blue)
     - for each original edge:
       - add (u_red, v_red) for red edges
       - add (u_blue, v_blue) for blue edges
       - replacate edge costs

    """

    M = networkx.Graph()
    for node in H.nodes_iter():
        transverse_edge = []
        for color in (1, -1):
            colored_node = node + (color,)
            M.add_node(colored_node)
            transverse_edge.append(colored_node)
        M.add_edge(*transverse_edge, cost=transverse_edge_bonus)

    for node_1, node_2, edge_attr in H.edges_iter(data=True):
        cost = edge_attr['cost']
        if np.isinf(cost):
            continue
        color = edge_attr['color']
        colored_node_1 = node_1 + (color,)
        colored_node_2 = node_2 + (color,)
        M.add_edge(colored_node_1, colored_node_2, attr_dict=edge_attr, cost=cost)

    return M


class GenomeGraphModel(object):

    def __init__(self, N, M, emission, prior, adjacencies, breakpoints):
        """ Create a GenomeGraph.

        Args:
            N (int): number of segments
            M (int): number of clones including normal
            emission (ReadCountLikelihood): read count likelihood
            prior (CopyNumberPrior): copy number prior
            adjacencies (list of tuple): ordered pairs of segments representing wild type adjacencies
            breakpoints (list of frozenset of tuple): list of pairs of segment/side pairs representing detected breakpoints

        A 'breakpoint' is represented as the frozenset (['breakend_1', 'breakend_2'])

        A 'breakend' is represented as the tuple ('segment', 'side').

        """

        self.N = N
        self.M = M

        # If some tumour adjacencies are also wild type adjacencies, we will
        # get problems with maintenence of the copy balance condition
        for breakpoint in breakpoints:
            (n_1, side_1), (n_2, side_2) = sorted(breakpoint)
            if (n_1, n_2) in adjacencies and side_1 == 1 and side_2 == 0:
                raise ValueError('breakpoint {} equals wild type adjacency {}'.format(repr(breakpoint), repr((n_1, n_2))))

        self.emission = emission
        self.prior = prior

        self.G = create_genome_graph(N, adjacencies, breakpoints)
        self.H = create_genome_mod_graph(self.G)

        # Count edge types
        self.edge_type_count = collections.Counter()
        for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
            self.edge_type_count[edge_attr['edge_type']] += 1

        self.integral_cost_scale = 100.

        self.telomere_penalty = -10.
        self.reference_penalty = 0.
        self.breakpoint_penalty = -.1
        self.transverse_edge_bonus = 1. / self.integral_cost_scale

        # Calculate a per edge penalty for telomeres, allowing for
        # unpenalized telomeres incident to nodes with no incident
        # reference edge
        ref_adj_nodes = set()
        for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
            if edge_attr['edge_type'] == 'reference':
                for node in (node_1, node_2):
                    ref_adj_nodes.add(node)

        is_telomere_penalized = np.zeros(self.edge_type_count['telomere'])
        for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
            if edge_attr['edge_type'] == 'telomere':
                for node in (node_1, node_2):
                    if node in ref_adj_nodes:
                        is_telomere_penalized[edge_attr['edge_idx']] = 1.

        self.telomere_penalty = self.telomere_penalty * is_telomere_penalized

        self.opt_iter = None
        self.decreased_log_prob = None

    def init_copy_number(self, segment_cn, init_breakpoints=True):
        """ Initialize copy number

        Args:
            cn (numpy.array): initial copy number matrix

        KwArgs:
            init_breakpoints (bool): initialize breakpoint copy number
    
        Copy number matrix has dimensions (N, M, L) for N breakpoints,
        and M clones, L alleles.

        """

        if not segment_cn.shape[0] * 2 == self.edge_type_count['segment']:
            raise ValueError('incorrect number segments for segment_cn')

        # Create edge copy number tables
        self.segment_cn = segment_cn.copy()
        self.breakpoint_cn = np.zeros((self.edge_type_count['breakpoint'], self.M))
        self.reference_cn = np.zeros((self.edge_type_count['reference'], self.M))

        # Calculate node copy number from segments
        node_cn = collections.defaultdict(lambda: np.zeros(self.M))
        for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
            if edge_attr['edge_type'] == 'segment':
                allele = edge_attr['allele']
                for node in (node_1, node_2):
                    node_cn[node] += self.segment_cn[edge_attr['edge_idx'], :, allele]

        # Initialize edge copy number as minimum of segment copy number pushed
        # through adjacent nodes, decrease flow pushed through each node. Perform
        # this operation on reference then breakpoint edges, giving breakpoints
        # the remainder of what cannot be explained by reference adjacencies
        # Initializing breakpoint copy number is optional, mainly disabled for
        # testing purposes
        init_edges = [('reference', self.reference_cn)]
        if init_breakpoints:
            init_edges += [('breakpoint', self.breakpoint_cn)]
        for edge_type, edge_cn in init_edges:
            for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
                if edge_attr['edge_type'] != edge_type:
                    continue
                cn = np.minimum(node_cn[node_1], node_cn[node_2])
                for node in (node_1, node_2):
                    node_cn[node] -= cn
                edge_cn[edge_attr['edge_idx']] = cn

        self.init_telomere_copy_number()


    def init_telomere_copy_number(self):
        """ Initialize telomere copy number

        """

        # Create telomere associated copy number tables
        self.dummy_cn = np.zeros((self.edge_type_count['dummy'], self.M))
        self.telomere_cn = np.zeros((self.edge_type_count['telomere'], self.M))

        # Calculate node copy number from segment, reference and breakpoint edges
        node_cn = collections.defaultdict(lambda: np.zeros(self.M))
        for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
            if edge_attr['edge_type'] == 'segment':
                allele = edge_attr['allele']
                for node in (node_1, node_2):
                    node_cn[node] += self.segment_cn[edge_attr['edge_idx'], :, allele]
            elif edge_attr['edge_type'] == 'reference':
                for node in (node_1, node_2):
                    node_cn[node] -= self.reference_cn[edge_attr['edge_idx']]
            elif edge_attr['edge_type'] == 'breakpoint':
                for node in (node_1, node_2):
                    node_cn[node] -= self.breakpoint_cn[edge_attr['edge_idx']]

        # Remove zero nodes and check for negatives
        for node in node_cn.keys():
            assert np.all(node_cn[node] >= 0)
            if np.all(node_cn[node] == 0):
                del node_cn[node]

        # While we still have non-zero nodes, iterate through dummy segment
        # edges, pick a pair of remaining non-zero nodes, increase the dummy
        # segment copy number by the minimum of the pair, and decrease each
        # pair by that same ammount, then delete nodes at zero
        dummy_edges = filter(lambda a: a[2]['edge_type'] == 'dummy', self.G.edges_iter(data=True))

        telomere_edges = filter(lambda a: a[2]['edge_type'] == 'telomere', self.G.edges_iter(data=True))
        telomere_subgraph = networkx.Graph(data=telomere_edges)

        prev_total_node_cn = None
        while len(node_cn) > 0:
            total_node_cn = sum([a.sum() for a in node_cn.values()])
            assert prev_total_node_cn is None or total_node_cn < prev_total_node_cn
            prev_total_node_cn = total_node_cn

            for node_1, node_2, edge_attr in dummy_edges:
                cn_nodes = node_cn.keys()
                random.shuffle(cn_nodes)

                # Find 2 nodes with overlapping positive copy number
                seg_node_1 = cn_nodes.pop()
                seg_node_2 = None
                for seg_node in cn_nodes:
                    cn = np.minimum(node_cn[seg_node_1], node_cn[seg_node])
                    if np.any(cn > 0):
                        seg_node_2 = seg_node
                        break

                # If no pair of nodes have overlapping positive copy number, then
                # the remaining nodes must have even copy number
                if seg_node_2 is None:
                    seg_node_2 = seg_node_1
                    cn = np.round(node_cn[seg_node_1] / 2.)
                    if not np.all(node_cn[seg_node_1] == cn * 2):
                        raise Exception('expected even copy number for all nodes, node_cn=' + repr(node_cn))

                if not np.any(cn > 0):
                    raise Exception('expected overlapping positive copy number, node_cn=' + repr(node_cn))

                dummy_edge_idx = edge_attr['edge_idx']
                self.dummy_cn[dummy_edge_idx] += cn

                for seg_node, dummy_node in zip((seg_node_1, seg_node_2), (node_1, node_2)):
                    telomere_edge_idx = telomere_subgraph[seg_node][dummy_node]['edge_idx']
                    self.telomere_cn[telomere_edge_idx] += cn

                    node_cn[seg_node] -= cn
                    assert np.all(node_cn[seg_node] >= 0)

                    if np.all(node_cn[seg_node] == 0):
                        del node_cn[seg_node]


    def calculate_segment_edge_log_prob(self, cn):
        """ Calculate probability per segment edge

        Args:
            cn (numpy.array): current copy number

        Returns:
            numpy.array: per edge probability

        Copy number matrix has dimensions (N, M, L) for N segments,
        M clones, and L alleles

        """

        # Calculate current log probability for total copy number
        log_prob = (
            self.emission.log_likelihood_total(cn) +
            self.prior.log_prior(cn)
        )

        return log_prob


    def calculate_segment_edge_cost(self, cn, delta, sign, allele):
        """ Calculate cost per segment edge for modifying copy number

        Args:
            cn (numpy.array): current copy number
            delta (numpy.array): copy number change
            sign (int): direction of change
            allele (int): allele to change

        Returns:
            numpy.array: per edge cost

        Copy number matrix has dimensions (N, M, L) for N segments,
        M clones, and L alleles

        """

        # Calculate current log probability for total copy number
        log_prob = self.calculate_segment_edge_log_prob(cn)

        # Calculate delta copy number
        cn_delta = cn.copy()
        cn_delta[:, :, allele] += sign * delta[np.newaxis, :]

        # Calculate log probability delta, checking for negative copy number
        invalid_cn_delta = np.any(cn_delta < 0, axis=(1, 2))
        cn_delta[invalid_cn_delta] = 1
        log_prob_delta = self.calculate_segment_edge_log_prob(cn_delta)
        log_prob_delta[invalid_cn_delta] = -np.inf

        cost = log_prob - log_prob_delta

        return cost


    def calculate_bond_edge_log_prob(self, cn, edge_penalty):
        """ Calculate probability per bond edge

        Args:
            cn (numpy.array): current copy number
            edge_penalty (float): penalty for positive copy number on edge

        Returns:
            numpy.array: per edge cost

        Copy number matrix has dimensions (N, M) for N breakpoints,
        and M clones.

        """

        # Calculate cost based on copy number and event weight
        log_prob = edge_penalty * cn.sum(axis=1)

        return log_prob


    def calculate_bond_edge_cost(self, cn, delta, sign, edge_penalty):
        """ Calculate cost per bond edge for modifying copy number

        Args:
            cn (numpy.array): current copy number
            delta (numpy.array): copy number change
            sign (int): direction of change
            edge_penalty (float): penalty for positive copy number on edge

        Returns:
            numpy.array: per edge cost

        Copy number matrix has dimensions (N, M) for N breakpoints,
        and M clones.

        """

        # Calculate current log probability for total copy number
        log_prob = self.calculate_bond_edge_log_prob(cn, edge_penalty)

        # Calculate delta copy number
        cn_delta = cn + sign * delta[np.newaxis, :]

        # Calculate log probability delta, checking for negative copy number
        log_prob_delta = self.calculate_bond_edge_log_prob(cn_delta, edge_penalty)
        log_prob_delta[np.any(cn_delta < 0, axis=1)] = -np.inf

        cost = log_prob - log_prob_delta

        return cost


    def add_edge_costs(self, H, delta):
        """ Add edge costs to genome modification graph.

        Args:
            H (networkx.MultiGraph): genome modification graph
            delta (numpy.array): clone copy number change

        Adds edge costs to graph.

        """

        segment_edge_costs = {}
        dummy_edge_costs = {}
        breakpoint_edge_costs = {}
        reference_edge_costs = {}
        telomere_edge_costs = {}

        for sign in (-1, 1):
            for allele in xrange(2):
                segment_edge_costs[(sign, allele)] = self.calculate_segment_edge_cost(self.segment_cn, delta, sign, allele)

        for sign in (-1, 1):
            dummy_edge_costs[sign] = self.calculate_bond_edge_cost(self.dummy_cn, delta, sign, 0.)
            breakpoint_edge_costs[sign] = self.calculate_bond_edge_cost(self.breakpoint_cn, delta, sign, self.breakpoint_penalty)
            reference_edge_costs[sign] = self.calculate_bond_edge_cost(self.reference_cn, delta, sign, self.reference_penalty)
            telomere_edge_costs[sign] = self.calculate_bond_edge_cost(self.telomere_cn, delta, sign, self.telomere_penalty)

        # Add edge costs to graph
        for node_1, node_2, edge_attr in H.edges_iter(data=True):
            edge_type = edge_attr['edge_type']
            edge_idx = edge_attr['edge_idx']
            sign = edge_attr['sign']
            if edge_type == 'segment':
                allele = edge_attr['allele']
                edge_attr['cost'] = segment_edge_costs[(sign, allele)][edge_idx]
            elif edge_type == 'dummy':
                edge_attr['cost'] = dummy_edge_costs[sign][edge_idx]
            elif edge_type == 'breakpoint':
                edge_attr['cost'] = breakpoint_edge_costs[sign][edge_idx]
            elif edge_type == 'reference':
                edge_attr['cost'] = reference_edge_costs[sign][edge_idx]
            elif edge_type == 'telomere':
                edge_attr['cost'] = telomere_edge_costs[sign][edge_idx]


    def optimize_modification(self, delta):
        """ Calculate optimal modification moving by +/- delta.

        Args:
            delta (numpy.array): copy number modification

        Returns:
            networkx.Graph: minimum cost modification graph

        """

        self.add_edge_costs(self.H, delta)

        M = create_matching_graph(self.H, self.transverse_edge_bonus)

        # Integer nodes for passing to blossomv
        M1 = networkx.convert_node_labels_to_integers(M, label_attribute='node_tuple')

        # Min cost perfect matching
        edges = networkx.get_edge_attributes(M1, 'cost')
        for edge in edges.keys():
            edges[edge] = int(edges[edge] * self.integral_cost_scale)
        min_cost_edges = remixt.blossomv.min_weight_perfect_matching(edges)

        # Remove unselected edges
        assert set(min_cost_edges).issubset(edges.keys())
        remove_edges = set(edges.keys()).difference(min_cost_edges)
        M2 = M1.copy()
        M2.remove_edges_from(remove_edges)

        # Re-create original graph with matched edges
        M3 = networkx.relabel_nodes(M2, mapping=networkx.get_node_attributes(M2, 'node_tuple'))

        # Create subgraph of H with only selected edges
        H1 = networkx.Graph()
        for node_1, node_2, edge_attr in M3.edges_iter(data=True):
            node_1 = node_1[:-1]
            node_2 = node_2[:-1]
            if node_1 == node_2:
                continue
            if H1.has_edge(node_1, node_2):
                H1.remove_edge(node_1, node_2)
            else:
                H1.add_edge(node_1, node_2, attr_dict=edge_attr)

        return H1


    def calculate_modification_cost(self, modification):
        """ Calculate cost of a modifcation graph.

        Args:
            modification (networkx.Graph): minimum cost modification graph

        Returns:
            float: cost

        """

        # Calculate total cost
        total_cost = sum((a[2]['cost'] for a in modification.edges_iter(data=True)))

        return total_cost


    def apply_modification(self, delta, modification):
        """ Apply the given modification to the segment/bond copy number.

        Args:
            delta (numpy.array): copy number modification
            modification (networkx.Graph): minimum cost modification graph

        """

        for node_1, node_2, edge_attr in modification.edges_iter(data=True):
            edge_type = edge_attr['edge_type']
            edge_idx = edge_attr['edge_idx']
            sign = edge_attr['sign']
            if edge_type == 'segment':
                allele = edge_attr['allele']
                self.segment_cn[edge_idx, :, allele] += sign * delta
            elif edge_type == 'dummy':
                self.dummy_cn[edge_idx] += sign * delta
            elif edge_type == 'breakpoint':
                self.breakpoint_cn[edge_idx] += sign * delta
            elif edge_type == 'reference':
                self.reference_cn[edge_idx] += sign * delta
            elif edge_type == 'telomere':
                self.telomere_cn[edge_idx] += sign * delta


    def test_circulation(self, node_cn=None):
        """ Test if current copy number state represents a valid circulation.

        KwArgs:
            node_cn (dict): dictionary of remainder copy number at each node

        Raises:
            Exception: raised if not a valid circulation

        """

        if node_cn is None:
            node_cn = {}

        for node_1, node_2 in self.G.edges_iter():
            if node_1 == node_2:
                raise Exception('graph should not contain loops')

        for node in self.G.nodes_iter():
            cn = -node_cn.get(node, np.zeros(self.M))
            for node_1, node_2, edge_attr in self.G.edges_iter(node, data=True):
                edge_type = edge_attr['edge_type']
                edge_idx = edge_attr['edge_idx']
                if edge_type == 'segment':
                    allele = edge_attr['allele']
                    cn += self.segment_cn[edge_idx, :, allele]
                elif edge_type == 'dummy':
                    cn += self.dummy_cn[edge_idx]
                elif edge_type == 'breakpoint':
                    cn -= self.breakpoint_cn[edge_idx]
                elif edge_type == 'reference':
                    cn -= self.reference_cn[edge_idx]
                elif edge_type == 'telomere':
                    cn -= self.telomere_cn[edge_idx]
            if cn.sum() != 0:
                raise Exception('node {0} has nonzero sum {1}'.format(repr(node), repr(cn)))


    def calculate_log_prob(self):
        """ Calculate log probability of segment/bond copy number.

        Returns:
            float: log probability of segment/bond copy number.

        """

        log_prob = self.calculate_segment_log_prob() + self.calculate_bond_log_prob()

        return log_prob


    def calculate_segment_log_prob(self):
        """ Calculate log probability of segment/bond copy number.

        Returns:
            float: log probability of segment/bond copy number.

        """

        log_prob = self.calculate_segment_edge_log_prob(self.segment_cn).sum()

        return log_prob


    def calculate_bond_log_prob(self):
        """ Calculate log probability of bond copy number.

        Returns:
            float: log probability of bond copy number.

        """

        log_prob = (
            self.calculate_bond_edge_log_prob(self.breakpoint_cn, self.breakpoint_penalty).sum() +
            self.calculate_bond_edge_log_prob(self.reference_cn, self.reference_penalty).sum() +
            self.calculate_bond_edge_log_prob(self.telomere_cn, self.telomere_penalty).sum()
        )

        return log_prob


    def build_deltas(self, M):
        """ Build list of non-redundant single copy changes

        Args:
            M (int): number of clones

        Returns:
            list: possible non-redundant single copy changes across clones

        """

        return list(np.eye(M-1, M, 1, dtype=int))


    def optimize(self, max_iter=1000, max_shuffle_telomere=10):
        """ Calculate optimal segment/bond copy number.

        KwArgs:
            max_iter (int): maximum iterations
            max_shuffle_telomere (int): maximum number of times to shuffle telomeres

        Returns:
            float: optimized log probability

        """
        
        self.decreased_log_prob = False

        self.test_circulation()

        deltas = self.build_deltas(self.M)

        log_prob_prev = self.calculate_log_prob()

        self.shuffle_telomere_iter = 0
        for self.opt_iter in xrange(max_iter):
            mod_list = list()

            for delta in deltas:
                modification = self.optimize_modification(delta)
                cost = self.calculate_modification_cost(modification)
                mod_list.append((cost, delta, modification))

            mod_list.sort(key=lambda a: a[0])

            best_cost, best_delta, best_modification = mod_list[0]

            if best_cost == 0:
                if self.shuffle_telomere_iter < max_shuffle_telomere:
                    self.init_telomere_copy_number()
                    self.shuffle_telomere_iter += 1
                    continue
                else:
                    self.shuffle_telomere_iter = 0
                    break

            self.apply_modification(best_delta, best_modification)

            self.test_circulation()

            log_prob = self.calculate_log_prob()

            if log_prob < log_prob_prev:
                self.decreased_log_prob = True
                print 'decreased log prob from {0} to {1}'.format(log_prob_prev, log_prob)
                break

            log_prob_prev = log_prob

        return log_prob_prev


    @property
    def breakpoint_copy_number(self):
        """ Table of breakpoint copy number.

        pandas.DataFrame with columns:
            'n_1', 'side_1', 'n_2', 'side_2', 'cn_*'

        """

        brk_cn_table = list()

        for node_1, node_2, edge_attr in self.G.edges_iter(data=True):
            if edge_attr['edge_type'] == 'breakpoint':
                cn = self.breakpoint_cn[edge_attr['edge_idx']]

                row = {
                    'n_1': node_1[0],
                    'side_1': node_1[1],
                    'n_2': node_2[0],
                    'side_2': node_2[1],
                }

                for m in xrange(self.M):
                    row['cn_{}'.format(m)] = cn[m]

                brk_cn_table.append(row)

        brk_cn_table = pd.DataFrame(brk_cn_table)

        return brk_cn_table


    def log_likelihood(self, state):
        return self.emission.log_likelihood(state)


    def optimal_state(self):
        return self.optimize()


