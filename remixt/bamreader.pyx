# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
# cython: language_level = 3

import pandas as pd
import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "BamAlleleReader.h":
    cdef cppclass FragmentData:
        int fragmentID
        int fragmentStart
        int fragmentEnd
        int mappingQuality
        int isDuplicate
    cdef cppclass AlleleData:
        int fragmentID
        int position
        int isAlt
    cdef cppclass CAlleleReader "AlleleReader":
        void CAlleleReader(string bamFilename,
            string snpFilename,
            string chromosome,
            int maxFragmentLength,
            int maxSoftClipped,
            bool checkProperPair) except +
        bool ReadAlignments(int maxAlignments) except +
        vector[FragmentData] mFragmentData
        vector[AlleleData] mAlleleData

def create_fragment_table(nrows):
    return pd.DataFrame(
        data=0,
        index=xrange(nrows),
        dtype=np.int32,
        columns=[
            'fragment_id',
            'start',
            'end',
            'mapping_quality',
            'is_duplicate',
        ],
    )

def create_allele_table(nrows):
    return pd.DataFrame(
        data=0,
        index=xrange(nrows),
        dtype=np.int32,
        columns=[
            'fragment_id',
            'position',
            'is_alt',
        ],
    )

cdef class AlleleReader:
    cdef CAlleleReader *thisptr
    def __cinit__(self, bam_filename, snp_filename, chromosome, max_fragment_length, max_soft_clipped, check_proper_pair):
        self.thisptr = new CAlleleReader(bam_filename, snp_filename, chromosome, max_fragment_length, max_soft_clipped, check_proper_pair)
    def __dealloc__(self):
        del self.thisptr
    def ReadAlignments(self, max_alignments):
        return self.thisptr.ReadAlignments(max_alignments)
    def GetFragmentTable(self):
        data = create_fragment_table(self.thisptr.mFragmentData.size())
        cdef np.ndarray fragmentID = data['fragment_id'].values
        cdef np.ndarray fragmentStart = data['start'].values
        cdef np.ndarray fragmentEnd = data['end'].values
        cdef np.ndarray mappingQuality = data['mapping_quality'].values
        cdef np.ndarray isDuplicate = data['is_duplicate'].values
        cdef int idx
        for idx in range(self.thisptr.mFragmentData.size()):
            fragmentID[idx] = self.thisptr.mFragmentData[idx].fragmentID
            fragmentStart[idx] = self.thisptr.mFragmentData[idx].fragmentStart
            fragmentEnd[idx] = self.thisptr.mFragmentData[idx].fragmentEnd
            mappingQuality[idx] = self.thisptr.mFragmentData[idx].mappingQuality
            isDuplicate[idx] = self.thisptr.mFragmentData[idx].isDuplicate
        return data
    def GetAlleleTable(self):
        data = create_allele_table(self.thisptr.mAlleleData.size())
        cdef np.ndarray fragmentID = data['fragment_id'].values
        cdef np.ndarray position = data['position'].values
        cdef np.ndarray isAlt = data['is_alt'].values
        cdef int idx
        for idx in range(self.thisptr.mAlleleData.size()):
            fragmentID[idx] = self.thisptr.mAlleleData[idx].fragmentID
            position[idx] = self.thisptr.mAlleleData[idx].position
            isAlt[idx] = self.thisptr.mAlleleData[idx].isAlt
        return data

