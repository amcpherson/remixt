import networkx
import blossomv.blossomv


def identify_balanced_rearrangements(H):
    
    # Create matching graph
    #  - duplicate nodes, one set red, one set blue
    #  - add transverse edges (v_red, v_blue)
    #  - for each original edge:
    #    - add (u_red, v_red) for red edges
    #    - add (u_blue, v_blue) for blue edges
    #    - replacate edge costs

    transverse_edge_cost = 1.

    M = networkx.Graph()
    for node in H.nodes_iter():
        transverse_edge = []
        for color in (1, -1):
            colored_node = node + (color,)
            M.add_node(colored_node)
            transverse_edge.append(colored_node)
        M.add_edge(*transverse_edge, cost=transverse_edge_cost)

    for edge in H.edges_iter():
        for multi_edge_idx, edge_attr in H[edge[0]][edge[1]].items():
            color = edge_attr['color']
            colored_node_1 = edge[0] + (color,)
            colored_node_2 = edge[1] + (color,)
            M.add_edge(colored_node_1, colored_node_2, attr_dict=edge_attr, cost=0.)

    M1 = networkx.convert_node_labels_to_integers(M, label_attribute='node_tuple')

    # Min cost perfect matching
    edges = networkx.get_edge_attributes(M1, 'cost')
    for edge in edges.keys():
        if edge[0] == edge[1]:
            raise Exception('self loop {}'.format(M1[edge[0]][edge[1]]))
    min_cost_edges = blossomv.blossomv.min_weight_perfect_matching(edges)

    # Remove unselected edges
    assert set(min_cost_edges).issubset(edges.keys())
    remove_edges = set(edges.keys()).difference(min_cost_edges)
    M2 = M1.copy()
    M2.remove_edges_from(remove_edges)

    # Re-create original graph with matched edges
    M3 = networkx.relabel_nodes(M2, mapping=networkx.get_node_attributes(M2, 'node_tuple'))

    # Create subgraph of H with only selected edges
    H1 = networkx.Graph()
    for edge in M3.edges_iter():
        edge_attr = M3[edge[0]][edge[1]]
        node_1 = edge[0][:-1]
        node_2 = edge[1][:-1]
        if node_1 == node_2:
            continue
        if H1.has_edge(node_1, node_2):
            H1.remove_edge(node_1, node_2)
        else:
            H1.add_edge(node_1, node_2, attr_dict=edge_attr)
            
    return H1


def minimize_breakpoint_copies(adjacencies, brk_cn):
    min_brk_cn = dict()
    for brk, cn in brk_cn.items():
        min_brk_cn[brk] = cn.copy()

    num_clones = max([cn.shape[0] for cn in brk_cn.values()])

    while True:
        has_changed = False
        
        for m in range(num_clones):
            H = networkx.MultiGraph()
            
            for brk, cn in min_brk_cn.items():
                if cn[m] > 0:
                    H.add_edge(*brk, color=1)
                    
            for adj in adjacencies:
                for allele in (0, 1):
                    allele_adj = (((adj[0], allele), 1), ((adj[1], allele), 0))
                    H.add_edge(*allele_adj, color=-1)
                
            C = identify_balanced_rearrangements(H)
            
            for edge in C.edges_iter():
                edge = frozenset(list(edge))
                if edge not in min_brk_cn:
                    adj = tuple(sorted([a[0][0] for a in edge]))
                    assert adj in adjacencies or adj[::-1] in adjacencies
                    continue
                assert min_brk_cn[edge][m] > 0
                min_brk_cn[edge][m] -= 1
                has_changed = True
                
        if not has_changed:
            break
    
    return min_brk_cn
