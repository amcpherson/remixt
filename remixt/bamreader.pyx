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


cdef class AlleleReader:
    cdef CAlleleReader *thisptr
    def __cinit__(self, bam_filename, snp_filename, chromosome, max_fragment_length, max_soft_clipped, check_proper_pair):
        self.thisptr = new CAlleleReader(bam_filename, snp_filename, chromosome, max_fragment_length, max_soft_clipped, check_proper_pair)
    def __dealloc__(self):
        del self.thisptr
    def ReadAlignments(self, max_alignments):
        return self.thisptr.ReadAlignments(max_alignments)
    def GetFragmentTable(self):
        cdef int nrows = self.thisptr.mFragmentData.size()
        cdef np.ndarray[np.int32_t, ndim=1] fragmentID = np.empty(nrows, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] fragmentStart = np.empty(nrows, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] fragmentEnd = np.empty(nrows, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] mappingQuality = np.empty(nrows, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] isDuplicate = np.empty(nrows, dtype=np.int32)
        cdef int idx
        for idx in range(nrows):
            fragmentID[idx] = self.thisptr.mFragmentData[idx].fragmentID
            fragmentStart[idx] = self.thisptr.mFragmentData[idx].fragmentStart
            fragmentEnd[idx] = self.thisptr.mFragmentData[idx].fragmentEnd
            mappingQuality[idx] = self.thisptr.mFragmentData[idx].mappingQuality
            isDuplicate[idx] = self.thisptr.mFragmentData[idx].isDuplicate
        return pd.DataFrame({
            'fragment_id': fragmentID,
            'start': fragmentStart,
            'end': fragmentEnd,
            'mapping_quality': mappingQuality,
            'is_duplicate': isDuplicate,
        }, copy=False)
    def GetAlleleTable(self):
        cdef int nrows = self.thisptr.mAlleleData.size()
        cdef np.ndarray[np.int32_t, ndim=1] fragmentID = np.empty(nrows, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] position = np.empty(nrows, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] isAlt = np.empty(nrows, dtype=np.int32)
        cdef int idx
        for idx in range(nrows):
            fragmentID[idx] = self.thisptr.mAlleleData[idx].fragmentID
            position[idx] = self.thisptr.mAlleleData[idx].position
            isAlt[idx] = self.thisptr.mAlleleData[idx].isAlt
        return pd.DataFrame({
            'fragment_id': fragmentID,
            'position': position,
            'is_alt': isAlt,
        }, copy=False)

