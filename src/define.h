#pragma once

//#define DOUBLE 
namespace nplm {
	#ifdef FLOAT
	#define precision_type float
	#endif 

	#ifdef DOUBLE
	#define precision_type double
	#endif

//bool arg_run_lm;
} //namespace nplm