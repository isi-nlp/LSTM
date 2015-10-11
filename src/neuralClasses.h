#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

//#include <boost/unordered_map.hpp> 
#include <Eigen/Dense>
#include <Eigen/Core>
#include "maybe_omp.h"

#include "util.h"
//#include "graphClasses.h"
#include "USCMatrix.h"

// classes for various kinds of layers
#include "SoftmaxLoss.h"
#include "Activation_function.h"
#define quote(x) #x
//#include <stdlib.h> 

//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_DEFAULT_TO_ROW_MAJOR

using namespace std;
namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::Array;
using Eigen::MatrixBase;
using Eigen::Dynamic;

//typedef boost::unordered_map<int,bool> int_map;

struct Clipper{
  precision_type operator() (precision_type x) const { 
    return (precision_type) std::min(50000000., std::max(double(x),-5000000.));
    //return(x);
  }
};

class Dropout_layer
{
	private:
		Matrix<precision_type, Dynamic, Dynamic> dropout_mask;
		precision_type keep_probability;
		
	public:
		
	Dropout_layer():
		dropout_mask(Matrix<precision_type, Dynamic, Dynamic>()),
		keep_probability(1.) {}
	
	Dropout_layer(int rows, int cols, precision_type keep_probability) :
		dropout_mask(Matrix<precision_type, Dynamic, Dynamic>()),
		keep_probability(keep_probability) {
		resize(rows, cols);
	}
	
	void resize(int rows, int cols){
		dropout_mask.setZero(rows, cols);
	}
	template<typename Engine> 
	void createMask(Engine &eng){
		//Setting up the dropout sampler
		boost::random::uniform_real_distribution<> real_01(0, 1);
		for (int i=0; i<dropout_mask.rows(); i++){
			for (int j=0; j<dropout_mask.cols(); j++){
				dropout_mask(i,j) = 
					( real_01(eng) <1.0 - keep_probability ) ? 0: 1;
			}
		}
	}
	template<typename Derived, typename Engine>
	void fProp(const MatrixBase<Derived> &input,
				Engine &eng) {
		createMask(eng);
		//cerr<<"Created the dropout mask "<<endl;
		//cerr<<"The mask dimensions are "<<this->dropout_mask.rows()<<","<<this->dropout_mask.cols()<<endl;
		//getchar();
		//cerr<<"Before dropout the fProp input is"<<input<<endl;
		dropout(input,
				this->dropout_mask);		
		//cerr<<"After dropout the fProp input is"<<input<<endl;
		//UNCONST(Derived, input, my_input);
		//my_input.array().noalias() *= dropout_mask.array();
		
	}
	
	//Its possible that this might just want to be used a function by itself
	template<typename Derived, typename DropMask>
	static void dropout(const MatrixBase<Derived> &input,
							const MatrixBase<DropMask> &dropout_mask){

		UNCONST(Derived, input, my_input);
		//cerr<<"Dropping out"<<endl;
		//cerr<<"dropout_mask "<<dropout_mask<<endl;
		my_input.array() *= dropout_mask.array();
	}
	
	template<typename DerivedGOut>
	void bProp(const MatrixBase<DerivedGOut> &input) {
		//cerr<<"Before dropout the bProp input is "<<input<<endl;		
		dropout(input,
				this->dropout_mask);
		//cerr<<"After dropout the bProp input is "<<input<<endl;				
				
		//UNCONST(DerivedGOut, input, my_input);
		//my_input.noalias().array() *= dropout_mask.array();
	}
};
class Linear_layer
{
    private: 
        Matrix<precision_type,Dynamic,Dynamic> U;
        Matrix<precision_type,Dynamic,Dynamic> U_gradient;
        Matrix<precision_type,Dynamic,Dynamic> U_velocity;
        Matrix<precision_type,Dynamic,Dynamic> U_running_gradient;
        Matrix<precision_type,Dynamic,Dynamic> U_running_parameter_update;
        // Biases
        Matrix<precision_type,Dynamic,1> b;
        Matrix<precision_type,Dynamic,1> b_velocity;
        Matrix<precision_type,Dynamic,1> b_running_gradient;
        Matrix<precision_type,Dynamic,1> b_running_parameter_update;
        Matrix<precision_type,Dynamic,1> b_gradient;

		friend class model;

    public:
	Linear_layer() { }
        Linear_layer(int rows, int cols) { resize(rows, cols); }

	void resize(int rows, int cols)
	{
	    U.setZero(rows, cols);
      U_gradient.setZero(rows, cols);
      //U_running_gradient.setZero(rows, cols);
      //U_running_parameter_updates.setZero(rows, cols);
      //U_velocity.setZero(rows, cols);
      b.resize(rows);
      b_gradient.setZero(rows);
      //b_running_gradient.resize(rows);
      //b_velocity.resize(rows);
	}

	void read_weights(std::ifstream &U_file) { readMatrix(U_file, U); }
	void write_weights(std::ofstream &U_file) { writeMatrix(U, U_file); }
  void read_biases(std::ifstream &b_file) { readMatrix(b_file, b); }
  void write_biases(std::ofstream &b_file) { writeMatrix(b, b_file); }


	template <typename Engine>
	void initialize(Engine &engine,
      bool init_normal,
      precision_type init_range,
      string &parameter_update,
      precision_type adagrad_epsilon)
	{
      if (parameter_update == "ADA") {
        U_running_gradient = Matrix<precision_type,Dynamic,Dynamic>::Ones(U.rows(),U.cols())*adagrad_epsilon;
        b_running_gradient = Matrix<precision_type,Dynamic,1>::Ones(b.size())*adagrad_epsilon;
      }
      if (parameter_update == "ADAD") {
        U_running_gradient.setZero(U.rows(),U.cols());
        b_running_gradient.setZero(b.size());
        U_running_parameter_update.setZero(U.rows(),U.cols());
        b_running_parameter_update.setZero(b.size());
      }

	    initMatrix(engine, U, init_normal, init_range);
		//std::cerr<<U<<std::endl;
      initBias(engine, b, init_normal, init_range);
	}	  

	int n_inputs () const { return U.cols(); }
	int n_outputs () const { return U.rows(); }

	int rows() const {return U.rows(); }
	int cols() const {return U.cols(); }
	
  template <typename DerivedIn, typename DerivedOut>
	void fProp(const MatrixBase<DerivedIn> &input,
      const MatrixBase<DerivedOut> &output) const
  {
	  /*
	  cerr<<"input"<<endl;
	  cerr<<input<<endl;
	  getchar();
	  cerr<<"output"<<endl;
	  cerr<<output<<endl;
	  getchar();
	  cerr<<"U"<<endl;
	  cerr<<U<<endl;
	  getchar();
	  */
      UNCONST(DerivedOut, output, my_output);
      my_output.leftCols(input.cols()).noalias() = U*input;
	  /*
      int num_examples = input.cols();
      for (int example = 0;example < num_examples;example++) 
      {
          my_output.leftCols(input.cols()).col(example) += b;
      }
	  */
  }

	// Sparse input
  template <typename ScalarIn, typename DerivedOut>
	void fProp(const USCMatrix<ScalarIn> &input,
      const MatrixBase<DerivedOut> &output_const) const
  {
	    UNCONST(DerivedOut, output_const, output);
	    output.setZero();
	    uscgemm(1.0, U, input, output.leftCols(input.cols()));
      // Each column corresponds to a training example. We 
      // parallelize the adding of biases per dimension.
      int num_examples = input.cols();
      for (int example = 0;example < num_examples;example++) 
      {
          output.leftCols(input.cols()).col(example) += b;
      }
  }

  
  void changeRandomParam(precision_type offset, 
						int &rand_row,
						int &rand_col){
  	changeRandomParamInMatrix(U, offset, rand_row, rand_col);
  }  

  precision_type getGradient(int row,
  			 int col) { 
	 		//cerr<<"U gradient is "<<U_gradient<<endl;
			//cerr<<"row is "<<row<<" and col is "<<col<<endl;
 			return U_gradient(row,col);}
				   
  template <typename DerivedGOut, typename DerivedGIn>
	void bProp(const MatrixBase<DerivedGOut> &input,
      MatrixBase<DerivedGIn> &output) const
  {
	    UNCONST(DerivedGIn, output, my_output);
	    my_output.noalias() = U.transpose()*input;
	}

  template <typename DerivedGOut, typename DerivedIn>
  void computeGradient( const MatrixBase<DerivedGOut> &bProp_input, 
     const MatrixBase<DerivedIn> &fProp_input, 
     precision_type learning_rate, precision_type momentum, precision_type L2_reg)
  {
      U_gradient.noalias() = bProp_input*fProp_input.transpose();
      
      // get the bias gradient for all dimensions in parallel
      int size = b.size();
      b_gradient = bProp_input.rowwise().sum();
      // This used to be multithreaded, but there was no measureable difference
      if (L2_reg > 0.0)
      {
          U_gradient -=  2*L2_reg*U;
          b_gradient -= 2*L2_reg*b;
      }
      if (momentum > 0.0)
      {
          U_velocity.noalias() = momentum*U_velocity + U_gradient;
          U.noalias() += learning_rate * U_velocity;
          //b_velocity = momentum*b_velocity + b_gradient;
          //b += learning_rate * b_velocity;
      }
      else
      {
		  /*
          U += learning_rate * U_gradient;
          b += learning_rate * b_gradient;
		  */
           
          //UPDATE CLIPPING
          U += (learning_rate*U_gradient).array().unaryExpr(Clipper()).matrix();
          b += (learning_rate*b_gradient).array().unaryExpr(Clipper()).matrix();
          //GRADIENT CLIPPING
          //U += learning_rate*(U_gradient.array().unaryExpr(Clipper())).matrix();
          //b += learning_rate*(b_gradient.array().unaryExpr(Clipper())).matrix();
          
      }
	}
	
    template <typename DerivedGOut, typename DerivedIn>
    void updateGradient( const MatrixBase<DerivedGOut> &bProp_input, 
       const MatrixBase<DerivedIn> &fProp_input)
    {
		//std::cout<<typeid(*this).name()<<"\t"<< quote(*this) <<"\n";
		//cerr<<"bProp input is "<<bProp_input<<endl;
		//cerr<<"fProp_input is "<<fProp_input<<endl;
        U_gradient += bProp_input*fProp_input.transpose();
		//cerr<<"the U gradient norm is "<<U_gradient.norm()<<endl;
		//cerr<<"current U gradient is "<<U_gradient<<endl;
      
        // get the bias gradient for all dimensions in parallel
        //int size = b.size();
        //b_gradient += bProp_input.rowwise().sum();

  	}
		
    void updateParams(precision_type learning_rate,
					  int current_minibatch_size,
                      precision_type momentum,
					  precision_type L2_reg,
					  bool norm_clipping,
					  //bool update_clipping,
					  precision_type norm_threshold){
						  
      // get the bias gradient for all dimensions in parallel
      int size = b.size();
      // This used to be multithreaded, but there was no measureable difference
      if (L2_reg > 0.0)
      {
          U_gradient -=  2*L2_reg*U;
          //b_gradient -= 2*L2_reg*b;
      }
      if (momentum > 0.0)
      {
          U_velocity = momentum*U_velocity + U_gradient;
          U += learning_rate * U_velocity;
          //b_velocity = momentum*b_velocity + b_gradient;
          //b += learning_rate * b_velocity;
      }
      else
      {
		  //cerr<<"the U gradient norm is "<<U_gradient.norm()<<endl;
		  if (norm_clipping){
			  scaleAndNormClip(U_gradient,
			  				   current_minibatch_size,
			  				   norm_threshold);
			  U += learning_rate * U_gradient;			   
		  } else {
			 U += (learning_rate * U_gradient).unaryExpr(updateClipper());
			 clipParamMatrix(U);
		  }
		  

      } 	
  }
  void resetGradient(){
	  U_gradient.setZero();
	  //b_gradient.setZero();
  }
  
  //Scaling for dropout at test time
  void scale(const precision_type scaling_constant) {
  	  U *= scaling_constant;
  }
  
  template <typename DerivedGOut, typename DerivedIn>
  void computeGradientAdagrad(const MatrixBase<DerivedGOut> &bProp_input, 
      const MatrixBase<DerivedIn> &fProp_input, 
      precision_type learning_rate,
      precision_type L2_reg)
  {
      U_gradient.noalias() = bProp_input*fProp_input.transpose();

      
      // get the bias gradient for all dimensions in parallel
      int size = b.size();
      b_gradient.noalias() = bProp_input.rowwise().sum();

      if (L2_reg != 0)
      {
          U_gradient -=  2*L2_reg*U;
          b_gradient -= 2*L2_reg*b;
      }

      // ignore momentum?
      #pragma omp parallel for
      for (int col=0; col<U.cols(); col++) {
        U_running_gradient.col(col) += U_gradient.col(col).array().square().matrix();
        U.col(col) += learning_rate * (U_gradient.col(col).array() / 
                  U_running_gradient.col(col).array().sqrt()).matrix();
        /*
        //UPDATE CLIPPING
        U.col(col) += (learning_rate * (U_gradient.col(col).array() / U_running_gradient.col(col).array().sqrt())).
              unaryExpr(Clipper()).matrix();
        */
      }
      b_running_gradient += b_gradient.array().square().matrix();
      b += learning_rate * (b_gradient.array()/b_running_gradient.array().sqrt()).matrix();
      /*
      //UPDATE CLIPPING
      b += (learning_rate * (b_gradient.array()/b_running_gradient.array().sqrt())).unaryExpr(Clipper()).matrix();
      */
  }

  template <typename DerivedGOut, typename DerivedIn>
  void computeGradientAdadelta(const MatrixBase<DerivedGOut> &bProp_input, 
      const MatrixBase<DerivedIn> &fProp_input, 
      precision_type learning_rate,
      precision_type L2_reg,
      precision_type conditioning_constant,
      precision_type decay)
  {
      //cerr<<"decay is "<<decay<<" and conditioning constant is "<<conditioning_constant<<endl;
      U_gradient.noalias() = bProp_input*fProp_input.transpose();

      Array<precision_type,Dynamic,1> b_current_parameter_update;
      
      // get the bias gradient for all dimensions in parallel
      int size = b.size();
      b_gradient.noalias() = bProp_input.rowwise().sum();

      if (L2_reg != 0)
      {
          U_gradient -=  2*L2_reg*U;
          b_gradient -= 2*L2_reg*b;
      }

      // ignore momentum?
      #pragma omp parallel for
      //cerr<<"U gradient is "<<U_gradient<<endl;
      for (int col=0; col<U.cols(); col++) {
        Array<precision_type,Dynamic,1> U_current_parameter_update;
        U_running_gradient.col(col) = decay*U_running_gradient.col(col) + 
                            (1-decay)*U_gradient.col(col).array().square().matrix();
        //cerr<<"U running gradient is "<<U_running_gradient.col(col)<<endl;
        //getchar();
        U_current_parameter_update = ((U_running_parameter_update.col(col).array()+conditioning_constant).sqrt()/
                                      (U_running_gradient.col(col).array()+conditioning_constant).sqrt()) *
                                      U_gradient.col(col).array();
        //cerr<<"U current parameter update is "<<U_current_parameter_update<<endl;
        //getchar();
        //update the running parameter update
        U_running_parameter_update.col(col) = decay*U_running_parameter_update.col(col) +
                                          (1.-decay)*U_current_parameter_update.square().matrix();
        U.col(col) += learning_rate*U_current_parameter_update.matrix();  
      }
      b_running_gradient = decay*b_running_gradient + 
                        (1.-decay)*b_gradient.array().square().matrix();
      b_current_parameter_update = ((b_running_parameter_update.array()+conditioning_constant).sqrt()/
                                   (b_running_gradient.array()+conditioning_constant).sqrt()) *
                                  b_gradient.array();
      b_running_parameter_update = decay*(b_running_parameter_update) + 
                                (1.-decay)*b_current_parameter_update.square().matrix();
      b += learning_rate*b_current_parameter_update.matrix();
  }


  template <typename DerivedGOut, typename DerivedIn, typename DerivedGW>
  void computeGradientCheck(const MatrixBase<DerivedGOut> &bProp_input, 
    const MatrixBase<DerivedIn> &fProp_input, 
    const MatrixBase<DerivedGW> &gradient) const
  {
      UNCONST(DerivedGW, gradient, my_gradient);
      my_gradient.noalias() = bProp_input*fProp_input.transpose();
  }
};


class Linear_diagonal_layer
{
    private: 
        Matrix<precision_type,Dynamic,1> U;
        Matrix<precision_type,Dynamic,1> U_gradient;
        Matrix<precision_type,Dynamic,1> U_velocity;
        Matrix<precision_type,Dynamic,1> U_running_gradient;
        Matrix<precision_type,Dynamic,1> U_running_parameter_update;
        // Biases
        Matrix<precision_type,Dynamic,1> b;
        Matrix<precision_type,Dynamic,1> b_velocity;
        Matrix<precision_type,Dynamic,1> b_running_gradient;
        Matrix<precision_type,Dynamic,1> b_running_parameter_update;
        Matrix<precision_type,Dynamic,1> b_gradient;

		friend class model;

    public:
	Linear_diagonal_layer() { }
        Linear_diagonal_layer(int rows) { resize(rows); }

	void resize(int rows)
	{
	    U.setZero(rows);
        U_gradient.setZero(rows);
      //U_running_gradient.setZero(rows, cols);
      //U_running_parameter_updates.setZero(rows, cols);
      //U_velocity.setZero(rows, cols);
       //b.resize(rows);
       //b_gradient.setZero(rows);
      //b_running_gradient.resize(rows);
      //b_velocity.resize(rows);
	}

	void read_weights(std::ifstream &U_file) { readMatrix(U_file, U); }
	void write_weights(std::ofstream &U_file) { writeMatrix(U, U_file); }
  void read_biases(std::ifstream &b_file) { readMatrix(b_file, b); }
  void write_biases(std::ofstream &b_file) { writeMatrix(b, b_file); }


	template <typename Engine>
	void initialize(Engine &engine,
      bool init_normal,
      precision_type init_range,
      string &parameter_update,
      precision_type adagrad_epsilon)
	{
      if (parameter_update == "ADA") {
        U_running_gradient = Matrix<precision_type,Dynamic,1>::Ones(U.size())*adagrad_epsilon;
        //b_running_gradient = Matrix<precision_type,Dynamic,1>::Ones(b.size())*adagrad_epsilon;
      }
      if (parameter_update == "ADAD") {
        U_running_gradient.setZero(U.size());
        //b_running_gradient.setZero(b.size());
        U_running_parameter_update.setZero(U.size());
        //b_running_parameter_update.setZero(b.size());
      }

	    initMatrix(engine, U, init_normal, init_range);
		//std::cerr<<U<<std::endl;
      //initBias(engine, b, init_normal, init_range);
	}	  

	int n_inputs () const { return U.rows(); }
	int n_outputs () const { return U.rows(); }
	
	int rows() const {return U.rows(); }
	int cols() const {return U.cols(); }
  template <typename DerivedIn, typename DerivedOut>
	void fProp(const MatrixBase<DerivedIn> &input,
      const MatrixBase<DerivedOut> &output) const
  {
      UNCONST(DerivedOut, output, my_output);
	  int num_examples = input.cols();
	  //Can this be sped up with broadcasting ?
	  //cerr<<"input to linear diagonal layer is "<<input<<endl; 
	  //cerr<<"U is "<<this->U<<endl;
	  for (int i=0; i<num_examples; i++){
	  	my_output.col(i).noalias() = (U.array()*input.col(i).array()).matrix();
	  }
	  //cerr<<"output to linear diagonal layer is "<<my_output<<endl;
      //my_output.leftCols(input.cols()).noalias() = U.array()*input.array();
	  /*
      int num_examples = input.cols();
      for (int example = 0;example < num_examples;example++) 
      {
          my_output.leftCols(input.cols()).col(example) += b;
      }
	  */
  }
   

  
    template <typename DerivedGOut, typename DerivedGIn>
	void bProp(const MatrixBase<DerivedGOut> &input,
      MatrixBase<DerivedGIn> &output) const
    {
		
	    UNCONST(DerivedGIn, output, my_output);
  	    int num_examples = input.cols();
		//Can this be sped up with broadcasting ? 
  	    for (int i=0; i<num_examples; i++){
		  //cerr<<" i "<<i<<endl;	
  	  	  my_output.col(i).noalias() = (U.array()*input.col(i).array()).matrix();
  	    }
	    //my_output.noalias() = U.array()*input.array();
	}
	
	
    template <typename DerivedGOut, typename DerivedIn>
    void updateGradient( const MatrixBase<DerivedGOut> &bProp_input, 
       const MatrixBase<DerivedIn> &fProp_input)
    {
		int num_examples = bProp_input.cols();
		for (int i=0; i<num_examples; i++){
        	U_gradient += (bProp_input.col(i).array()*fProp_input.col(i).array()).matrix();
		}
      
        // get the bias gradient for all dimensions in parallel
        //int size = b.size();
        //b_gradient += bProp_input.rowwise().sum();

  	}
	
	//
    void changeRandomParam(precision_type offset, 
							int &rand_row,
							int &rand_col){
    	changeRandomParamInMatrix(U, offset, rand_row, rand_col);
    }		

  	precision_type getGradient(int row,
  			 int col) { return U_gradient(row,col);}
				 	
    void updateParams(precision_type learning_rate,
					  int current_minibatch_size,
                      precision_type momentum,
					  precision_type L2_reg,
					  bool norm_clipping,
					  precision_type norm_threshold){
						  
      // get the bias gradient for all dimensions in parallel
      int size = b.size();
      // This used to be multithreaded, but there was no measureable difference
      if (L2_reg > 0.0)
      {
          U_gradient -=  2*L2_reg*U;
          //b_gradient -= 2*L2_reg*b;
      }
      if (momentum > 0.0)
      {
          U_velocity = momentum*U_velocity + U_gradient;
          U += learning_rate * U_velocity;
          //b_velocity = momentum*b_velocity + b_gradient;
          //b += learning_rate * b_velocity;
      }
      else
      {
		  /*
		  U_gradient /= current_minibatch_size;
		  precision_type grad_norm = U_gradient.norm();
		  if (U_gradient.norm() >= 5.) {
			  U_gradient *= 5./grad_norm;
		  }
		  U += learning_rate*U_gradient;	
		  */
		  if (norm_clipping){
			  scaleAndNormClip(U_gradient,
			  				   current_minibatch_size,
			  				   norm_threshold);
			 U += learning_rate * U_gradient;				   
		  }	else {
 			 U += (learning_rate * U_gradient).unaryExpr(updateClipper());
 			 clipParamMatrix(U);		  	
		  }  
          //U.array() += learning_rate * (U_gradient/current_minibatch_size).array().unaryExpr(Clipper());
		  
          //b += learning_rate * b_gradient;
	
	
		  
		  /*
          U += (learning_rate*U_gradient).array().unaryExpr(Clipper()).matrix();
          b += (learning_rate*b_gradient).array().unaryExpr(Clipper()).matrix();
		  */
		  
      } 	
  }
  void resetGradient(){
	  U_gradient.setZero();
	  //b_gradient.setZero();
  }

};

class Output_word_embeddings
{
    private:
        // row-major is better for uscgemm
        //Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W;
        // Having W be a pointer to a matrix allows ease of sharing
        // input and output word embeddings
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W;
        std::vector<precision_type> W_data;
        Matrix<precision_type,Dynamic,1> b;
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W_running_gradient;
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W_gradient;
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W_running_parameter_update;
        Matrix<precision_type,Dynamic,1> b_running_gradient;
        Matrix<precision_type,Dynamic,1> b_gradient;
        Matrix<precision_type,Dynamic,1> b_running_parameter_update;
		int_map update_map;
		Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> shared_noise_embeddings;
    public:
        Output_word_embeddings() { }
        Output_word_embeddings(int rows, int cols) { resize(rows, cols); }

        void resize(int rows, int cols)
        {
          W.setZero(rows, cols);
          b.setZero(rows);
        }
		/*
    void set_W(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> *input_W) {
      W = input_W;
    }
		*/
    void read_weights(std::ifstream &W_file) { readMatrix(W_file, W); }
    void write_weights(std::ofstream &W_file) { writeMatrix(W, W_file); }
    void read_biases(std::ifstream &b_file) { readMatrix(b_file, b); }
    void write_biases(std::ofstream &b_file) { writeMatrix(b, b_file); }
	
    template <typename Engine>
    void initialize(Engine &engine,
        bool init_normal,
        precision_type init_range,
        precision_type init_bias,
        string &parameter_update,
        precision_type adagrad_epsilon)
    {

        W_gradient.setZero(W.rows(),W.cols());
        b_gradient.setZero(b.size());
        if (parameter_update == "ADA") {
          W_running_gradient = Matrix<precision_type,Dynamic,Dynamic>::Ones(W.rows(),W.cols())*adagrad_epsilon;
          b_running_gradient = Matrix<precision_type,Dynamic,1>::Ones(b.size())*adagrad_epsilon;
          //W_gradient.setZero(W.rows(),W.cols());
          //b_gradient.setZero(b.size());
        }
        if (parameter_update == "ADAD") {
          W_running_gradient.setZero(W.rows(),W.cols());
          b_running_gradient.setZero(b.size());
          W_gradient.setZero(W.rows(),W.cols());
          //b_gradient.setZero(b.size());
          //W_running_parameter_update.setZero(W.rows(),W.cols());
          b_running_parameter_update.setZero(b.size());
        }

        initMatrix(engine, W, init_normal, init_range);
		//std::cerr<<*W<<std::endl;e
        b.fill(init_bias);
		//shared_noise_embeddings.setZero()
    }
	
	void resizeNoiseEmbeddings(int num_noise_samples, int minibatch_size){
		shared_noise_embeddings.setZero(num_noise_samples+minibatch_size,W.cols());
	}

    int n_inputs () const { return W.cols(); }
    int n_outputs () const { return W.rows(); }

	int rows() const {return W.rows(); }
	int cols() const {return W.cols(); }
	
    template <typename DerivedIn, typename DerivedOut>
    void fProp(const MatrixBase<DerivedIn> &input,
    const MatrixBase<DerivedOut> &output) const
	  {
        UNCONST(DerivedOut, output, my_output);
        my_output = (W * input).colwise() + b; //No bias for output words
		//my_output = (*W) * input ; //No bias for output words

	  }

	//Scaling is used for dropout
	void scale(precision_type scaling_constant) {
		W *= scaling_constant;
	}
	// Sparse output version
    template <typename DerivedIn, typename DerivedOutI, typename DerivedOutV>
    void fProp(const MatrixBase<DerivedIn> &input,
    const MatrixBase<DerivedOutI> &samples,
    const MatrixBase<DerivedOutV> &output) const
	{
        UNCONST(DerivedOutV, output, my_output);
		//cerr<<"my_output rows and cols"<<my_output.rows()<<" "<<my_output.cols()<<endl;
		//cerr<<"input rows and cols"<<input.rows()<<input.cols()<<endl;
        #pragma omp parallel for
        for (int instance_id = 0; instance_id < samples.cols(); instance_id++)
        {
		  if (samples(0, instance_id) == -1) 
			  continue;
          for (int sample_id = 0; sample_id < samples.rows(); sample_id++)
          {
			//cerr<<"sample is "<<samples(sample_id, instance_id)<<endl;
            my_output(sample_id, instance_id) = b(samples(sample_id, instance_id));
          }
        }
		
        USCMatrix<precision_type> sparse_output(W.rows(), samples, my_output);
        uscgemm_masked(1.0, W, input, sparse_output);
        my_output = sparse_output.values; // too bad, so much copying
	}
	
	/*
	//The NCE fprop version if we are sharing noise samples across the minibatch.
	//If the noise samples are shared, we can densify the gradient computation, at least
	//for the noise sample
    template <typename DerivedIn, typename DerivedOutI, typename DerivedOutV>
    void fPropShared(const MatrixBase<DerivedIn> &input,
    const MatrixBase<DerivedOutI> &samples,
    const MatrixBase<DerivedOutV> &output) const
	{
		//First LOAD in the embeddings into the matrix that is going to store them. 
		//the first set are	
        uscgemm(1.0,
	        W.transpose(), 
	        USCMatrix<precision_type>(W.rows(),samples,Matrix<precision_type,1,Dynamic>::Ones(samples.cols())),
	        my_output.block(ngram*embedding_dimension, 0, embedding_dimension, input.cols()));
		
		
		
        UNCONST(DerivedOutV, output, my_output);
		//cerr<<"my_output rows and cols"<<my_output.rows()<<" "<<my_output.cols()<<endl;
		//cerr<<"input rows and cols"<<input.rows()<<input.cols()<<endl;
        #pragma omp parallel for
        for (int instance_id = 0; instance_id < samples.cols(); instance_id++)
        {
		  if (samples(0, instance_id) == -1) 
			  continue;
          for (int sample_id = 0; sample_id < samples.rows(); sample_id++)
          {
			//cerr<<"sample is "<<samples(sample_id, instance_id)<<endl;
            my_output(sample_id, instance_id) = b(samples(sample_id, instance_id));
          }
        }
		
        USCMatrix<precision_type> sparse_output(W.rows(), samples, my_output);
        uscgemm_masked(1.0, W, input, sparse_output);
        my_output = sparse_output.values; // too bad, so much copying
	}
	*/
    // Return single element of output matrix
    template <typename DerivedIn>
    precision_type fProp(const MatrixBase<DerivedIn> &input, 
           int word,
           int instance) const 
    {
        return W.row(word).dot(input.col(instance)) + b(word);
		//return W.row(word).dot(input.col(instance));// + b(word); //No bias for output words
    }

    // Dense versions (for log-likelihood loss)

    template <typename DerivedGOut, typename DerivedGIn>
    void bProp(const MatrixBase<DerivedGOut> &input_bProp_matrix,
    const MatrixBase<DerivedGIn> &bProp_matrix) const
    {
	    // W is vocab_size x output_embedding_dimension
	    // input_bProp_matrix is vocab_size x minibatch_size
	    // bProp_matrix is output_embedding_dimension x minibatch_size
	    UNCONST(DerivedGIn, bProp_matrix, my_bProp_matrix);
	    my_bProp_matrix.leftCols(input_bProp_matrix.cols()).noalias() =
        W.transpose() * input_bProp_matrix;
	  }

    template <typename DerivedIn, typename DerivedGOut>
          void computeGradient(const MatrixBase<DerivedIn> &predicted_embeddings,
             const MatrixBase<DerivedGOut> &bProp_input,
             precision_type learning_rate,
             precision_type momentum) //not sure if we want 	to use momentum here
    {
        // W is vocab_size x output_embedding_dimension
        // b is vocab_size x 1
        // predicted_embeddings is output_embedding_dimension x minibatch_size
        // bProp_input is vocab_size x minibatch_size
		/*
        W.noalias() += learning_rate * bProp_input * predicted_embeddings.transpose();
        b += learning_rate * bProp_input.rowwise().sum();
		*/
        /*
        //GRADIENT CLIPPING
        W.noalias() += learning_rate * 
          ((bProp_input * predicted_embeddings.transpose()).array().unaryExpr(Clipper())).matrix();
        b += learning_rate * (bProp_input.rowwise().sum().array().unaryExpr(Clipper())).matrix();
		*/
        //UPDATE CLIPPING
        W.noalias() += (learning_rate * 
        (bProp_input * predicted_embeddings.transpose())).array().unaryExpr(Clipper()).matrix();
        b += (learning_rate * (bProp_input.rowwise().sum())).array().unaryExpr(Clipper()).matrix();
        
	  }
	  
	  precision_type getGradient(int row,
	  			 int col) {
					 //cerr<<"W_gradient"<<endl;
					 return W_gradient(row,col);}
				 
template <typename DerivedIn, typename DerivedGOut>
      void updateGradient(const MatrixBase<DerivedIn> &predicted_embeddings,
         const MatrixBase<DerivedGOut> &bProp_input) 
{
    // W is vocab_size x output_embedding_dimension
    // b is vocab_size x 1
    // predicted_embeddings is output_embedding_dimension x minibatch_size
    // bProp_input is vocab_size x minibatch_size
	
    W_gradient += bProp_input * predicted_embeddings.transpose();
    b_gradient += bProp_input.rowwise().sum();
	//cerr<<"the W gradient norm is "<<W_gradient.norm()<<endl;
	//getchar();
	//cerr<<"W_gradient in output layer is "<<W_gradient<<endl
	
    /*
    //GRADIENT CLIPPING
    W.noalias() += learning_rate * 
      ((bProp_input * predicted_embeddings.transpose()).array().unaryExpr(Clipper())).matrix();
    b += learning_rate * (bProp_input.rowwise().sum().array().unaryExpr(Clipper())).matrix();
	*/
	/*
    //UPDATE CLIPPING
    W.noalias() += (learning_rate * 
    (bProp_input * predicted_embeddings.transpose())).array().unaryExpr(Clipper()).matrix();
    b += (learning_rate * (bProp_input.rowwise().sum())).array().unaryExpr(Clipper()).matrix();
    */
  }
  
  void updateParams(precision_type learning_rate,
  		int current_minibatch_size,
  		precision_type momentum,
		precision_type L2_reg,
		bool norm_clipping,
		precision_type norm_threshold){
	  if (L2_reg > 0){
	  	W_gradient -= 2*L2_reg*W;
	  } 
	  //(*W).array() += learning_rate*(W_gradient/current_minibatch_size).array().unaryExpr(Clipper());
	  //(*W).array() += learning_rate*(W_gradient/current_minibatch_size).array();
	  //cerr<<"the W gradient norm is "<<W_gradient.norm()<<endl;
	  if (norm_clipping){
					   
	  }	 
	  if (norm_clipping){
		  scaleAndNormClip(W_gradient,
		  				   current_minibatch_size,
		  				   norm_threshold);
		  scaleAndNormClip(b_gradient,
		  				   current_minibatch_size,
		  				   norm_threshold);	
	 	  W += learning_rate*W_gradient;
	 	  b += learning_rate*b_gradient;						   
			   
	  }	else {
		 W += (learning_rate * W_gradient).unaryExpr(updateClipper());
		 b += (learning_rate * b_gradient).unaryExpr(updateClipper());
		 clipParamMatrix(W);
		 clipParamMatrix(b);		  	
	  } 	   

  }
  
  void resetGradient(){
	  W_gradient.setZero();
	  b_gradient.setZero();
  }
  
  void changeRandomParam(precision_type offset, 
						int &rand_row,
						int &rand_col){
  	changeRandomParamInMatrix(W, offset, rand_row, rand_col);
  } 
  
    template <typename DerivedIn, typename DerivedGOut>
          void computeGradientAdagrad(
             const MatrixBase<DerivedIn> &predicted_embeddings,
             const MatrixBase<DerivedGOut> &bProp_input,
             precision_type learning_rate) //not sure if we want to use momentum here
    {
        // W is vocab_size x output_embedding_dimension
        // b is vocab_size x 1
        // predicted_embeddings is output_embedding_dimension x minibatch_size
        // bProp_input is vocab_size x minibatch_sizea
        W_gradient.setZero(W.rows(), W.cols());
        b_gradient.setZero(b.size());
        W_gradient.noalias() = bProp_input * predicted_embeddings.transpose();
        b_gradient.noalias() = bProp_input.rowwise().sum();
        W_running_gradient += W_gradient.array().square().matrix();
        b_running_gradient += b_gradient.array().square().matrix();
        W.noalias() += learning_rate * (W_gradient.array()/W_running_gradient.array().sqrt()).matrix();
        b += learning_rate * (b_gradient.array()/b_running_gradient.array().sqrt()).matrix();
        /*
        //UPDATE CLIPPING
        *W += (learning_rate * (W_gradient.array()/W_running_gradient.array().sqrt())).unaryExpr(Clipper()).matrix();
        b += (learning_rate * (b_gradient.array()/b_running_gradient.array().sqrt())).unaryExpr(Clipper()).matrix();
        */
	  }

    template <typename DerivedIn, typename DerivedGOut>
          void computeGradientAdadelta(const MatrixBase<DerivedIn> &predicted_embeddings,
             const MatrixBase<DerivedGOut> &bProp_input,
             precision_type learning_rate,
             precision_type conditioning_constant,
             precision_type decay) //not sure if we want to use momentum here
    {
        // W is vocab_size x output_embedding_dimension
        // b is vocab_size x 1
        // predicted_embeddings is output_embedding_dimension x minibatch_size
        // bProp_input is vocab_size x minibatch_size
        Array<precision_type,Dynamic,Dynamic> W_current_parameter_update;
        Array<precision_type,Dynamic,1> b_current_parameter_update;
        W_gradient.setZero(W.rows(), W.cols());
        b_gradient.setZero(b.size());
        W_gradient.noalias() = bProp_input * predicted_embeddings.transpose();
        b_gradient.noalias() = bProp_input.rowwise().sum();
        W_running_gradient = decay*W_running_gradient +
                            (1.-decay)*W_gradient.array().square().matrix();
        b_running_gradient = decay*b_running_gradient+
                            (1.-decay)*b_gradient.array().square().matrix();
        W_current_parameter_update = ((W_running_parameter_update.array()+conditioning_constant).sqrt()/
                                     (W_running_gradient.array()+conditioning_constant).sqrt())*
                                      W_gradient.array();
        b_current_parameter_update = ((b_running_parameter_update.array()+conditioning_constant).sqrt()/
                                     (b_running_gradient.array()+conditioning_constant).sqrt())*
                                     b_gradient.array();
        W_running_parameter_update = decay*W_running_parameter_update + 
                                    (1.-decay)*W_current_parameter_update.square().matrix();
        b_running_parameter_update = decay*b_running_parameter_update +
                                    (1.-decay)*b_current_parameter_update.square().matrix();

        W += learning_rate*W_current_parameter_update.matrix();
        b += learning_rate*b_current_parameter_update.matrix();
	  }

    // Sparse versions

    template <typename DerivedGOutI, typename DerivedGOutV, typename DerivedGIn>
    void bProp(const MatrixBase<DerivedGOutI> &samples,
    const MatrixBase<DerivedGOutV> &weights,
    const MatrixBase<DerivedGIn> &bProp_matrix) const
    {
        UNCONST(DerivedGIn, bProp_matrix, my_bProp_matrix);
        my_bProp_matrix.setZero();
        uscgemm(1.0,
            W.transpose(), 
            USCMatrix<precision_type>(W.rows(), samples, weights),
            my_bProp_matrix.leftCols(samples.cols())); // narrow bProp_matrix for possible short minibatch
    }
	
	template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV>
	    void updateGradient(const MatrixBase<DerivedIn> &predicted_embeddings,
			     const MatrixBase<DerivedGOutI> &samples,
			     const MatrixBase<DerivedGOutV> &weights) 
	{
		
		//ACCUMULATING gradient
		//cerr<<"W gradient is "<<W_gradient<<endl;
		//getchar();
	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(1.0,
	      gradient_output,
	      predicted_embeddings.leftCols(samples.cols()).transpose(),
	      W_gradient);
		  
	    uscgemv(1.0, 
	      gradient_output,
		      Matrix<precision_type,Dynamic,1>::Ones(weights.cols()),
	      b_gradient);
		  
	  //cerr<<"the W gradient norm is "<<W_gradient.norm()<<endl;
	  //getchar();
	  //int_map update_map; //stores all the parameters that have been updated
	  for (int sample_id=0; sample_id<samples.rows(); sample_id++)
	        for (int train_id=0; train_id<samples.cols(); train_id++)
		          update_map[samples(sample_id, train_id)] = 1;

    
	  //cerr<<"Finished gradient"<<endl;
	}

	void updateParamsNCE(precision_type learning_rate,
						int current_minibatch_size,
						precision_type momentum,
						precision_type L2_reg,
						bool norm_clipping,
						precision_type norm_threshold){
					
	    // Convert to std::vector for parallelization
		  //cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
		  //cerr<<"the W gradient norm is "<<W_gradient.norm()<<endl;					
	      std::vector<int> update_items;
	      for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
	      {
	          update_items.push_back(it->first);
	      }
	      int num_items = update_items.size();
		  //#ifdef CLIP_SPARSE
			  if (norm_clipping) {
				//cerr<<"scaling output W"<<endl;
				scaleAndNormClip(W_gradient,
								 update_items,
				  				 current_minibatch_size,
				  				 norm_threshold);
				//cerr<<"Scaling output b"<<endl;
	 			scaleAndNormClip(b_gradient,
	 							 update_items,
	 			  				 current_minibatch_size,
	 			  				 norm_threshold);
			 } 
		 //#endif
		 
		  #pragma omp parallel for
	      for (int item_id=0; item_id<num_items; item_id++)
	      {
	          int update_item = update_items[item_id];
			  #ifdef CLIP_SPARSE
			  if (norm_clipping == 0){
				  W.row(update_item) += (learning_rate * 
					  					W_gradient.row(update_item)).unaryExpr(updateClipper());
				  		  
				  clipParamMatrix(W.row(update_item));
				  precision_type update_value = learning_rate * 
					  					b_gradient(update_item);
				  b(update_item) += std::min(0.01, std::max(double(update_value),-0.01));
				  b(update_item) = std::min(0.5, std::max(double(b(update_item)),-0.5));				  
			  } else {
		          W.row(update_item) += learning_rate*
		              W_gradient.row(update_item);
				  b(update_item) += learning_rate*b_gradient(update_item);
			  }
			  #endif
			  /*
	          W.row(update_item) += learning_rate*
	              W_gradient.row(update_item);
			  b(update_item) += learning_rate*b_gradient(update_item);	
			  */		  
	          //GRADIENT CLIPPING
	          //W.row(update_item) += learning_rate*
	          //    W_gradient.row(update_item).array().unaryExpr(Clipper()).matrix();
	          //SETTING THE GRADIENT TO ZERO
	          W_gradient.row(update_item).setZero();
			      b_gradient(update_item) = 0;
				  
	      }
		//we have to clear the update map
		this->update_map.clear();
	}


	template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV>
        void computeGradient(const MatrixBase<DerivedIn> &predicted_embeddings,
			     const MatrixBase<DerivedGOutI> &samples,
			     const MatrixBase<DerivedGOutV> &weights,
			     precision_type learning_rate, precision_type momentum) //not sure if we want to use momentum here
	{
      //cerr<<"in gradient"<<endl;
		/*
	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(learning_rate,
          gradient_output,
          predicted_embeddings.leftCols(gradient_output.cols()).transpose(),
          *W); // narrow predicted_embeddings for possible short minibatch
	    uscgemv(learning_rate,
          gradient_output,
		      Matrix<precision_type,Dynamic,1>::Ones(gradient_output.cols()),
          b);
		*/
      
      //IN ORDER TO IMPLEMENT CLIPPING, WE HAVE TO COMPUTE THE GRADIENT
      //FIRST
	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(1.0,
          gradient_output,
          predicted_embeddings.leftCols(samples.cols()).transpose(),
          W_gradient);
	    uscgemv(1.0, 
          gradient_output,
		      Matrix<precision_type,Dynamic,1>::Ones(weights.cols()),
          b_gradient);

      int_map update_map; //stores all the parameters that have been updated
      for (int sample_id=0; sample_id<samples.rows(); sample_id++)
	        for (int train_id=0; train_id<samples.cols(); train_id++)
		          update_map[samples(sample_id, train_id)] = 1;

	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
            update_items.push_back(it->first);
        int num_items = update_items.size();

        //#pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            int update_item = update_items[item_id];
            //W.row(update_item) += learning_rate * W_gradient.row(update_item);
            //b(update_item) += learning_rate * b_gradient(update_item);
            //UPDATE CLIPPING
            W.row(update_item) += (learning_rate * W_gradient.row(update_item)).array().unaryExpr(Clipper()).matrix();
            precision_type update = learning_rate * b_gradient(update_item);
            b(update_item) += std::min(0.5, std::max(double(update),-0.5));
            //GRADIENT CLIPPING
            W_gradient.row(update_item).setZero();
            b_gradient(update_item) = 0.;
        }
        
      //cerr<<"Finished gradient"<<endl;
	}

	template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV>
        void computeGradientAdagrad(const MatrixBase<DerivedIn> &predicted_embeddings,
				    const MatrixBase<DerivedGOutI> &samples,
				    const MatrixBase<DerivedGOutV> &weights,
				    precision_type learning_rate) //not sure if we want to use momentum here
        {
	    //W_gradient.setZero(W.rows(), W.cols());
	    //b_gradient.setZero(b.size());
      //FOR CLIPPING, WE DO NOT MULTIPLY THE GRADIENT WITH THE LEARNING RATE
	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(1.0,
          gradient_output,
          predicted_embeddings.leftCols(samples.cols()).transpose(),
          W_gradient);
	    uscgemv(1.0, 
          gradient_output,
		      Matrix<precision_type,Dynamic,1>::Ones(weights.cols()),
          b_gradient);

      int_map update_map; //stores all the parameters that have been updated
      for (int sample_id=0; sample_id<samples.rows(); sample_id++)
	        for (int train_id=0; train_id<samples.cols(); train_id++)
		          update_map[samples(sample_id, train_id)] = 1;

	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
            update_items.push_back(it->first);
        int num_items = update_items.size();

        //#pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            int update_item = update_items[item_id];
            W_running_gradient.row(update_item) += W_gradient.row(update_item).array().square().matrix();
            b_running_gradient(update_item) += b_gradient(update_item) * b_gradient(update_item);
            W.row(update_item) += learning_rate * (W_gradient.row(update_item).array() / W_running_gradient.row(update_item).array().sqrt()).matrix();
            b(update_item) += learning_rate * b_gradient(update_item) / sqrt(b_running_gradient(update_item));
            /*
            //UPDATE CLIPPING
            W.row(update_item) += (learning_rate * (W_gradient.row(update_item).array() / W_running_gradient.row(update_item).array().sqrt())).unaryExpr(Clipper()).matrix();
            precision_type update = learning_rate * b_gradient(update_item) / sqrt(b_running_gradient(update_item));
            b(update_item) += Clipper(update);//std::min(0.5, std::max(update,-0.5));
            */
            W_gradient.row(update_item).setZero();
            b_gradient(update_item) = 0.;
        }
      }

	template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV>
        void computeGradientAdadelta(const MatrixBase<DerivedIn> &predicted_embeddings,
				    const MatrixBase<DerivedGOutI> &samples,
				    const MatrixBase<DerivedGOutV> &weights,
				    precision_type learning_rate,
            precision_type conditioning_constant,
            precision_type decay) //not sure if we want to use momentum here
        {
          //cerr<<"decay is "<<decay<<" and constant is "<<conditioning_constant<<endl;
	    //W_gradient.setZero(W.rows(), W.cols());
	    //b_gradient.setZero(b.size());

	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(1.0,
          gradient_output,
          predicted_embeddings.leftCols(samples.cols()).transpose(),
          W_gradient);
	    uscgemv(1.0, 
          gradient_output,
		      Matrix<precision_type,Dynamic,1>::Ones(weights.cols()),
          b_gradient);

      int_map update_map; //stores all the parameters that have been updated
      for (int sample_id=0; sample_id<samples.rows(); sample_id++)
	        for (int train_id=0; train_id<samples.cols(); train_id++)
		          update_map[samples(sample_id, train_id)] = 1;

	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
            update_items.push_back(it->first);
        int num_items = update_items.size();

        #pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            Array<precision_type,1,Dynamic> W_current_parameter_update;
            precision_type b_current_parameter_update;

            int update_item = update_items[item_id];
            W_running_gradient.row(update_item) = decay*W_running_gradient.row(update_item)+
                                                (1.-decay)*W_gradient.row(update_item).array().square().matrix();
            b_running_gradient(update_item) = decay*b_running_gradient(update_item)+
                                            (1.-decay)*b_gradient(update_item)*b_gradient(update_item);
            //cerr<<"Output: W gradient is "<<W_gradient.row(update_item)<<endl;
            //getchar();

            //cerr<<"Output: W running gradient is "<<W_running_gradient.row(update_item)<<endl;
            //getchar();
            W_current_parameter_update = ((W_running_parameter_update.row(update_item).array()+conditioning_constant).sqrt()/
                                         (W_running_gradient.row(update_item).array()+conditioning_constant).sqrt())*
                                         W_gradient.row(update_item).array();
            b_current_parameter_update = (sqrt(b_running_parameter_update(update_item)+conditioning_constant)/
                                         sqrt(b_running_gradient(update_item)+conditioning_constant))*
                                         b_gradient(update_item);
            //cerr<<"Output: W current parameter update is "<<W_current_parameter_update<<endl;
            //getchar();
            //cerr<<"Output: W running parameter update before is "<<W_running_parameter_update.row(update_item)<<endl;
            //getchar();
            //cerr<<"the second term is "<<(1.-decay)*W_current_parameter_update.square().matrix()<<endl;
            W_running_parameter_update.row(update_item) = decay*W_running_parameter_update.row(update_item)+
                                                         (1.-decay)*(W_current_parameter_update.square().matrix());
            b_running_parameter_update(update_item) = decay*b_running_parameter_update(update_item)+
                                                      (1.-decay)*b_current_parameter_update*b_current_parameter_update;
            //cerr<<"Output: W running parameter update is "<<W_running_parameter_update.row(update_item)<<endl;
            //getchar();
            W.row(update_item) += learning_rate*W_current_parameter_update.matrix();
            b(update_item) += learning_rate*b_current_parameter_update;
            W_gradient.row(update_item).setZero();
            b_gradient(update_item) = 0.;
        }
      }


	template <typename DerivedIn, typename DerivedGOutI, typename DerivedGOutV, typename DerivedGW, typename DerivedGb>
    void computeGradientCheck(const MatrixBase<DerivedIn> &predicted_embeddings,
      const MatrixBase<DerivedGOutI> &samples,
      const MatrixBase<DerivedGOutV> &weights,
      const MatrixBase<DerivedGW> &gradient_W,
      const MatrixBase<DerivedGb> &gradient_b) const
  {
	    UNCONST(DerivedGW, gradient_W, my_gradient_W);
	    UNCONST(DerivedGb, gradient_b, my_gradient_b);
	    my_gradient_W.setZero();
	    my_gradient_b.setZero();
	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(1.0,
          gradient_output,
          predicted_embeddings.leftCols(samples.cols()).transpose(),
          my_gradient_W);
	    uscgemv(1.0, gradient_output,
		    Matrix<precision_type,Dynamic,1>::Ones(weights.cols()), my_gradient_b);
  }
};

class Input_word_embeddings
{
    private:
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W;
        int context_size, vocab_size;
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W_running_gradient;
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W_running_parameter_update;
        Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> W_gradient;
		int_map update_map; //stores all the parameters that have been updated

		//template <class T> friend class model;
		friend class model;

    public:
        Input_word_embeddings() : context_size(0), vocab_size(0) { }
        Input_word_embeddings(int rows, int cols, int context) { resize(rows, cols, context); }
 
      //Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>* get_W() {
      //  return(W);
      //}
      ///void set_W(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> *input_W) {
      //  W = input_W;
      //}
      void resize(int rows, int cols, int context)
      {
        context_size = context;
        vocab_size = rows;
        W.setZero(rows, cols);
      }

        void read(std::ifstream &W_file) { readMatrix(W_file, W); }
        void write(std::ofstream &W_file) { writeMatrix(W, W_file); }

      template <typename Engine>
      void initialize(Engine &engine,
          bool init_normal,
          precision_type init_range,
          string &parameter_update,
          precision_type adagrad_epsilon)
      {
          W_gradient.setZero(W.rows(),W.cols());

          if (parameter_update == "ADA") {
            W_running_gradient =  Matrix<precision_type,Dynamic,Dynamic>::Ones(W.rows(),W.cols())*adagrad_epsilon;
            //W_gradient.setZero(W.rows(),W.cols());
          } 
        if (parameter_update == "ADAD") {
          W_running_gradient.setZero(W.rows(),W.cols());
          //W_gradient.setZero(W.rows(),W.cols());
          W_running_parameter_update.setZero(W.rows(),W.cols());
        }
        initMatrix(engine,
            W,
            init_normal,
            init_range);
		//std::cerr<<*W<<std::endl;
      }

	int n_inputs() const { return -1; }
	int n_outputs() const { return W.cols() * context_size; }

	int rows() const {return W.rows(); }
	int cols() const {return W.cols(); }
	
	// set output_id's embedding to the weighted average of all embeddings
	template <typename Dist>
	void average(const Dist &dist, int output_id)
	{
	    W.row(output_id).setZero();
	    for (int i=0; i < W.rows(); i++)
	        if (i != output_id)
		    W.row(output_id) += dist.prob(i) * W.row(i);
	}

	template <typename DerivedIn, typename DerivedOut>
        void fProp(const MatrixBase<DerivedIn> &input,
		   const MatrixBase<DerivedOut> &output) const
        {
            int embedding_dimension = W.cols();

		    // W      is vocab_size                        x embedding_dimension
		    // input  is ngram_size*vocab_size             x minibatch_size
		    // output is ngram_size*embedding_dimension x minibatch_size

		    /* 
		    // Dense version:
		    for (int ngram=0; ngram<context_size; ngram++)
		        output.middleRows(ngram*embedding_dimension, embedding_dimension) = W.transpose() * input.middleRows(ngram*vocab_size, vocab_size);
		    */

		    UNCONST(DerivedOut, output, my_output);
		    my_output.setZero();
		    for (int ngram=0; ngram<context_size; ngram++)
		    {
		        // input might be narrower than expected due to a short minibatch,
		        // so narrow output to match
				//cerr<<"W is "
				//cerr<<"W rows is "<<W.rows()<<endl;
		        uscgemm(1.0,
	            W.transpose(), 
	            USCMatrix<precision_type>(W.rows(),input.middleRows(ngram, 1),Matrix<precision_type,1,Dynamic>::Ones(input.cols())),
	            my_output.block(ngram*embedding_dimension, 0, embedding_dimension, input.cols()));
		    }
        }

	// When model is premultiplied, this layer doesn't get used,
	// but this method is used to get the input into a sparse matrix.
	// Hopefully this can get eliminated someday
	template <typename DerivedIn, typename ScalarOut>
	void munge(const MatrixBase<DerivedIn> &input, USCMatrix<ScalarOut> &output) const
	{
	  output.resize(vocab_size*context_size, context_size, input.cols());
	  for (int i=0; i < context_size; i++)
	    output.indexes.row(i).array() = input.row(i).array() + i*vocab_size;
	  output.values.fill(1.0);
	}

  template <typename DerivedGOut, typename DerivedIn>
  void computeGradient(const MatrixBase<DerivedGOut> &bProp_input,
     const MatrixBase<DerivedIn> &input_words,
     precision_type learning_rate, precision_type momentum, precision_type L2_reg)
  {
      int embedding_dimension = W.cols();

	    // W           is vocab_size                        x embedding_dimension
	    // input       is ngram_size*vocab_size             x minibatch_size
	    // bProp_input is ngram_size*embedding_dimension x minibatch_size

	    /*
	    // Dense version:
	    for (int ngram=0; ngram<context_size; ngram++)
	        W += learning_rate * input_words.middleRows(ngram*vocab_size, vocab_size) * bProp_input.middleRows(ngram*embedding_dimension, embedding_dimension).transpose()
	    */
	  /*
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
	        uscgemm(learning_rate, 
			USCMatrix<precision_type>(W.rows(), input_words.middleRows(ngram, 1), Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
			bProp_input.block(ngram*embedding_dimension,0,embedding_dimension,input_words.cols()).transpose(),
      	  	*W);
	    }
	  */
      
      //IF WE WANT TO DO GRADIENT CLIPPING, THEN WE FIRST COMPUTE THE GRADIENT AND THEN
      //PERFORM CLIPPING WHILE UPDATING

	    for (int ngram=0; ngram<context_size; ngram++)
	    {
	      uscgemm(1.0, 
          USCMatrix<precision_type>(W.rows(),input_words.middleRows(ngram, 1),Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
          bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
          W_gradient);
	    }
      int_map update_map; //stores all the parameters that have been updated
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
        for (int train_id=0; train_id<input_words.cols(); train_id++)
        {
          update_map[input_words(ngram,train_id)] = 1;
        }
      }

	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        int num_items = update_items.size();

        #pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            int update_item = update_items[item_id];
            //UPDATE CLIPPING
            W.row(update_item) += (learning_rate*
                W_gradient.row(update_item).array().unaryExpr(Clipper())).matrix();
            //GRADIENT CLIPPING
            //W.row(update_item) += learning_rate*
            //    W_gradient.row(update_item).array().unaryExpr(Clipper()).matrix();
            //SETTING THE GRADIENT TO ZERO
            W_gradient.row(update_item).setZero();
        }
		
  }

  template <typename DerivedGOut, typename DerivedIn>
  void updateGradient(const MatrixBase<DerivedGOut> &bProp_input,
     const MatrixBase<DerivedIn> &input_words)
  {
      int embedding_dimension = W.cols();

	    // W           is vocab_size                        x embedding_dimension
	    // input       is ngram_size*vocab_size             x minibatch_size
	    // bProp_input is ngram_size*embedding_dimension x minibatch_size

	    /*
	    // Dense version:
	    for (int ngram=0; ngram<context_size; ngram++)
	        W += learning_rate * input_words.middleRows(ngram*vocab_size, vocab_size) * bProp_input.middleRows(ngram*embedding_dimension, embedding_dimension).transpose()
	    */
	  /*
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
	        uscgemm(learning_rate, 
			USCMatrix<precision_type>(W.rows(), input_words.middleRows(ngram, 1), Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
			bProp_input.block(ngram*embedding_dimension,0,embedding_dimension,input_words.cols()).transpose(),
      	  	*W);
	    }
	  */
      
      //IF WE WANT TO DO GRADIENT CLIPPING, THEN WE FIRST COMPUTE THE GRADIENT AND THEN
      //PERFORM CLIPPING WHILE UPDATING

	    for (int ngram=0; ngram<context_size; ngram++)
	    {
	      uscgemm(1.0, 
          USCMatrix<precision_type>(W.rows(),input_words.middleRows(ngram, 1),Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
          bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
          W_gradient);
	    }
      //int_map update_map; //stores all the parameters that have been updated
	    //for (int ngram=0; ngram<context_size; ngram++)
	    //{
        for (int train_id=0; train_id<input_words.cols(); train_id++)
        {
          this->update_map[input_words(train_id)] = 1;
        }
      //}


		
  }
   
  void updateParams(precision_type learning_rate,
  					int current_minibatch_size,
  					precision_type momentum,
					precision_type L2_reg,
					bool norm_clipping,
					precision_type norm_threshold){
						
	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        int num_items = update_items.size();
		#ifdef CLIP_SPARSE
		if (norm_clipping){
			scaleAndNormClip(W_gradient,
							 update_items,
			  				 current_minibatch_size,
			  				 norm_threshold);
		}
		#endif 
		
        #pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            int update_item = update_items[item_id];

			  #ifdef CLIP_SPARSE
			  if (norm_clipping == 0){
				  W.row(update_item) += (learning_rate * 
					  					W_gradient.row(update_item)).unaryExpr(updateClipper());
			  		  
				  clipParamMatrix(W.row(update_item));
		  
			  } else {
		          W.row(update_item) += learning_rate*
		              W_gradient.row(update_item);
			  }			
			  #endif
	          W.row(update_item) += learning_rate*
	              W_gradient.row(update_item); 
            //GRADIENT CLIPPING
            //W.row(update_item) += learning_rate*
            //    W_gradient.row(update_item).array().unaryExpr(Clipper()).matrix();
            //SETTING THE GRADIENT TO ZERO
            W_gradient.row(update_item).setZero();
        }
		//we have to clear the update map
		this->update_map.clear();
  }
    template <typename DerivedGOut, typename DerivedIn>
    void computeGradientAdagrad(const MatrixBase<DerivedGOut> &bProp_input,
				    const MatrixBase<DerivedIn> &input_words,
				    precision_type learning_rate,
            precision_type L2_reg)
    {
            int embedding_dimension = W.cols();
	    //W_gradient.setZero(W.rows(), W.cols());
      /*
      if (W_running_gradient.rows() != W.rows() || W_running_gradient.cols() != W.cols())
        W_running_gradient = Ones(W.rows(), W.cols())*adagrad_epsilon;
      */
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
	        uscgemm(1.0, 
			USCMatrix<precision_type>(W.rows(),input_words.middleRows(ngram, 1),Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
			bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
      W_gradient);
	    }
      int_map update_map; //stores all the parameters that have been updated
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
        for (int train_id=0; train_id<input_words.cols(); train_id++)
        {
          update_map[input_words(ngram,train_id)] = 1;
        }
      }

	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        int num_items = update_items.size();

        #pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            int update_item = update_items[item_id];
            W_running_gradient.row(update_item) += W_gradient.row(update_item).array().square().matrix();
            W.row(update_item) += learning_rate * 
              (W_gradient.row(update_item).array() / W_running_gradient.row(update_item).array().sqrt()).matrix();
            /*
            //UPDATE CLIPPING
            W.row(update_item) += (learning_rate * 
              (W_gradient.row(update_item).array() / W_running_gradient.row(update_item).array().sqrt()))
                      .unaryExpr(Clipper()).matrix();
            */
            W_gradient.row(update_item).setZero();
        }
    }

    void changeRandomParam(precision_type offset, 
							int &rand_row,
							int &rand_col){
    	changeRandomParamInMatrix(W, offset, rand_row, rand_col);
    }

  	precision_type getGradient(int row,
  			 int col) { return W_gradient(row,col);}	
    template <typename DerivedGOut, typename DerivedIn>
    void computeGradientAdadelta(const MatrixBase<DerivedGOut> &bProp_input,
				    const MatrixBase<DerivedIn> &input_words,
				    precision_type learning_rate,
            precision_type L2_reg,
            precision_type conditioning_constant,
            precision_type decay)
    {
      int embedding_dimension = W.cols();

	    //W_gradient.setZero(W.rows(), W.cols());
      /*
      if (W_running_gradient.rows() != W.rows() || W_running_gradient.cols() != W.cols())
        W_running_gradient = Ones(W.rows(), W.cols())*adagrad_epsilon;
      */
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
	        uscgemm(1.0, 
			USCMatrix<precision_type>(W.rows(),input_words.middleRows(ngram, 1),Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
			bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
      W_gradient);
	    }
      int_map update_map; //stores all the parameters that have been updated
	    for (int ngram=0; ngram<context_size; ngram++)
	    {
        for (int train_id=0; train_id<input_words.cols(); train_id++)
        {
          update_map[input_words(ngram,train_id)] = 1;
        }
      }

	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = update_map.begin(); it != update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        int num_items = update_items.size();

        #pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {

            Array<precision_type,1,Dynamic> W_current_parameter_update;
            int update_item = update_items[item_id];
            W_running_gradient.row(update_item) = decay*W_running_gradient.row(update_item)+
                                                (1.-decay)*W_gradient.row(update_item).array().square().matrix();

            W_current_parameter_update = ((W_running_parameter_update.row(update_item).array()+conditioning_constant).sqrt()/
                                         (W_running_gradient.row(update_item).array()+conditioning_constant).sqrt())*
                                         W_gradient.row(update_item).array();

            //cerr<<"Input: W current parameter update is "<<W_current_parameter_update<<endl;
            //getchar();
            W_running_parameter_update.row(update_item) = decay*W_running_parameter_update.row(update_item)+
                                                         (1.-decay)*W_current_parameter_update.square().matrix();

            W.row(update_item) += learning_rate*W_current_parameter_update.matrix();
            //cerr<<"Input: After update, W is  "<<W.row(update_item)<<endl;
            //getchar();
            W_gradient.row(update_item).setZero();
        }

    }

    template <typename DerivedGOut, typename DerivedIn, typename DerivedGW>
    void computeGradientCheck(const MatrixBase<DerivedGOut> &bProp_input,
      const MatrixBase<DerivedIn> &input_words,
      int x, int minibatch_size,
      const MatrixBase<DerivedGW> &gradient) const //not sure if we want to use momentum here
    {
	    UNCONST(DerivedGW, gradient, my_gradient);
            int embedding_dimension = W.cols();
	    my_gradient.setZero();
	    for (int ngram=0; ngram<context_size; ngram++)
	    uscgemm(1.0, 
			  USCMatrix<precision_type>(W.rows(),input_words.middleRows(ngram, 1),Matrix<precision_type,1,Dynamic>::Ones(input_words.cols())),
			  bProp_input.block(ngram*embedding_dimension, 0, embedding_dimension, input_words.cols()).transpose(),
        my_gradient);
    }
};

class Hidden_layer
{
    private:
        // Biases
        Matrix<precision_type,Dynamic,1> b;
		Matrix<precision_type,Dynamic,1> b_gradient;
		Activation_function hidden_activation;
		int size;
	public:
	Hidden_layer(): hidden_activation(Activation_function()), size(0){}
	
	void resize(int size)
	{ 
		this->size = size;
		hidden_activation.resize(size);
		b.resize(size);
		b_gradient.setZero(size);
	}
	
	void set_activation_function(activation_function_type f) {
		 hidden_activation.set_activation_function(f);
	 }
	int n_outputs(){return size;}
	int n_inputs(){return size;}
	
	int rows() const {return b.rows(); }
	int cols() const {return b.cols(); }	
	
	template <typename Engine>
	void initialize(Engine &engine,
      bool init_normal,
      precision_type init_range,
	  precision_type init_bias,
      string &parameter_update,
      precision_type adagrad_epsilon)
	{
		/*
      	if (parameter_update == "ADA") {
        	//U_running_gradient = Matrix<precision_type,Dynamic,1>::Ones(U.size())*adagrad_epsilon;
        	b_running_gradient = Matrix<precision_type,Dynamic,1>::Ones(b.size())*adagrad_epsilon;
      	}
      	if (parameter_update == "ADAD") {
        	//U_running_gradient.setZero(U.size());
        	b_running_gradient.setZero(b.size());
        	//U_running_parameter_update.setZero(U.size());
        	b_running_parameter_update.setZero(b.size());
      	}
		*/
	    //initMatrix(engine, U, init_normal, init_range);

		b_gradient.setZero();
		if (init_bias == 0.) {
      		initBias(engine, b, init_normal, init_range);
		} else {
			b.fill(init_bias);
		}
		//<<b<<std::endl;
	}	  

   void read_biases(std::ifstream &b_file) { readMatrix(b_file, b); }
   void write_biases(std::ofstream &b_file) { writeMatrix(b, b_file); }
		
   template <typename DerivedIn, typename DerivedOut>
     void fProp(const MatrixBase<DerivedIn> &input,
	   const MatrixBase<DerivedOut> &output) const
     {
		 UNCONST(DerivedIn, input, my_input);
		 //UNCONST(DerivedOut, output, my_output);
		 int num_examples = input.cols();
		 
		 for (int i=0;i<num_examples; i++){
			 my_input.col(i) += b;
		 }
		 
		 //cerr<<"B is "<<b<<endl;
		 hidden_activation.fProp(my_input,output);
	 }	
     template <typename DerivedGOut, typename DerivedGIn, typename DerivedIn, typename DerivedOut>
     void bProp(const MatrixBase<DerivedGOut> &input, 
      MatrixBase<DerivedGIn> &output,
		   const MatrixBase<DerivedIn> &finput,
       const MatrixBase<DerivedOut> &foutput) const
     {
		 hidden_activation.bProp(input,
		 						output,
								finput,
								foutput);		 
	 }
     template <typename DerivedGOut>
     void updateGradient(const MatrixBase<DerivedGOut> &bProp_input)
     {
         b_gradient += bProp_input.rowwise().sum();
	 } 
	 //The accumulated gradient is now added to the parameters
	 void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
	 					precision_type momentum,
						precision_type L2_reg,
						bool norm_clipping,
						precision_type norm_threshold){
		//as of now, only SGD
		//b.array() += learning_rate*(b_gradient/current_minibatch_size).array().unaryExpr(Clipper());	
  		if (norm_clipping){
  			scaleAndNormClip(b_gradient,
  			  				 current_minibatch_size,
  			  				 norm_threshold);
  		}
		//b += learning_rate*b_gradient/current_minibatch_size;
		b += learning_rate*b_gradient;
		//cerr<<"b is "<<b<<endl;				
	}
	void resetGradient(){
		b_gradient.setZero();
	}
	
    void changeRandomParam(precision_type offset, 
							int &rand_row,
							int &rand_col){
    	changeRandomParamInMatrix(b, offset, rand_row, rand_col);
    }
  	precision_type getGradient(int row,
  			 int col) { return b_gradient(row,col);}
};

} // namespace nplm

