from eigency.core cimport *
from libcpp.string cimport string

ctypedef unsigned int uint

cdef extern from "instantAlg_mul-edges.cpp":
	pass

cdef extern from "instantAlg_mul-edges.h" namespace "propagation":
	cdef cppclass Instantgnn:
		Instantgnn() except+
		double initial_operation(string,string,uint,uint,double,double,Map[MatrixXd] &) except +
		void dynamic_operation(uint, uint, double, double, Map[MatrixXd] &) except +
		void snapshot_operation(string, double, double, Map[MatrixXd] &) except +
		void overall_operation(double,double, Map[MatrixXd] &)
