from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "neuralTM.h":
    cdef cppclass c_neuralTM "nplm::neuralTM":
        c_neuralTM()
        void set_normalization(bint)
        void set_map_digits(char)
        void set_log_base(double)
        void read(string filename) except +
        int get_order()
        int lookup_input_word(string)
        int lookup_output_word(string)
        float lookup_ngram(vector[int])
        float lookup_ngram(int *, int)
        void set_cache(int)
        double cache_hit_rate()

cdef class NeuralTM:
    cdef c_neuralTM *thisptr
    cdef int c_lookup_input_word(self, char *s)
    cdef int c_lookup_output_word(self, char *s)
    cdef float c_lookup_ngram(self, int *words, int n)
    cdef readonly int order
    
