# distutils: language = c++

from libcpp cimport bool

cdef extern from "PerfectMatching.h":
    cdef cppclass CPerfectMatching "PerfectMatching":
        CPerfectMatching(int, int) except +
        int AddEdge(int, int, float)
        void Solve()
        int GetSolution(int)
        cppclass Options:
            bool verbose
        Options options

cdef class PerfectMatching:
    cdef CPerfectMatching *thisptr
    def __cinit__(self, node_num, edge_num_max):
        if node_num % 2 != 0:
            raise ValueError('# of nodes is odd: perfect matching cannot exist')
        self.thisptr = new CPerfectMatching(node_num, edge_num_max)
    def __dealloc__(self):
        del self.thisptr
    def AddEdge(self, i, j, cost):
        return self.thisptr.AddEdge(i, j, cost)
    def Solve(self):
        return self.thisptr.Solve()
    def GetSolution(self, e):
        return self.thisptr.GetSolution(e)

def min_weight_perfect_matching(edges):
    number_of_nodes = max((node for edge in edges.iterkeys() for node in edge)) + 1
    cdef CPerfectMatching *pm = new CPerfectMatching(number_of_nodes, len(edges))
    pm.options.verbose = False
    try:
        for idx, ((i, j), w) in enumerate(edges.iteritems()):
            edge_id = pm.AddEdge(i, j, w)
            assert edge_id == idx
        pm.Solve()
        matching = set()
        for idx, ((i, j), w) in enumerate(edges.iteritems()):
            if pm.GetSolution(idx):
                matching.add((i, j))
        return matching
    finally:
        del pm

