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
//#define quote(x) #x
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
	template<typename DerivedIn, typename DerivedOut, typename Engine>
	void fProp(const MatrixBase<DerivedIn> &input,
			   const MatrixBase<DerivedOut> &output,
				Engine &eng) {
		createMask(eng);
		//cerr<<"Created the dropout mask "<<endl;
		//cerr<<"The mask dimensions are "<<this->dropout_mask.rows()<<","<<this->dropout_mask.cols()<<endl;
		//getchar();
		//cerr<<"Before dropout the fProp input is"<<input<<endl;
		dropout(input,
				output,
				this->dropout_mask);		
		//cerr<<"After dropout the fProp input is"<<input<<endl;
		//UNCONST(Derived, input, my_input);
		//my_input.array().noalias() *= dropout_mask.array();
		
	}
	int n_inputs(){ return dropout_mask.rows();}
	int n_outputs() {return dropout_mask.cols();}
	//Its possible that this might just want to be used a function by itself
	template<typename DerivedIn, typename DerivedOut, typename DropMask>
	static void dropout(const MatrixBase<DerivedIn> &input,
						const MatrixBase<DerivedOut> &output,
						const MatrixBase<DropMask> &dropout_mask){

		UNCONST(DerivedOut, output, my_output);
		//cerr<<"Dropping out"<<endl;
		//cerr<<"dropout_mask "<<dropout_mask<<endl;
		my_output.array() = dropout_mask.array()*input.array();
	}
	
	template<typename DerivedGIn, typename DerivedGOut>
	void bProp(const MatrixBase<DerivedGIn> &input,
			   const MatrixBase<DerivedGOut> &output) {
		//cerr<<"Before dropout the bProp input is "<<input<<endl;		
		dropout(input,
				output,
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
		//cerr<<" U "<<endl<<U<<endl;
      initBias(engine, b, init_normal, init_range);
	}	  

	int n_inputs () const { return U.cols(); }
	int n_outputs () const { return U.rows(); }

	int rows() const {return U.rows(); }
	int cols() const {return U.cols(); }
	
  template <typename DerivedIn, typename DerivedOut>
	void fProp(const MatrixBase<DerivedIn> &input,
     	MatrixBase<DerivedOut> const &output) const
  {
	  //cerr<<"input is "<<input<<endl;
      UNCONST(DerivedOut, output, my_output);

      my_output.noalias() = U*input;

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

 			return U_gradient(row,col);
  }
  
  precision_type getGradSqdNorm() { 
	  		//U_gradient /= minibatch_size
 			return U_gradient.squaredNorm();
  }
				   
  template <typename DerivedGOut, typename DerivedGIn>
	void bProp(const MatrixBase<DerivedGOut> &input,
      MatrixBase<DerivedGIn> const &output) const
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
        U_gradient += bProp_input*fProp_input.transpose();
  	}
		
    void updateParams(precision_type learning_rate,
					  int current_minibatch_size,
                      precision_type momentum,
					  precision_type L2_reg,
					  precision_type grad_scale){
						  
      // get the bias gradient for all dimensions in parallel
      //int size = b.size();
      // This used to be multithreaded, but there was no measureable difference

			 U += (learning_rate * U_gradient*grad_scale);
  }
 
  void updateParams(precision_type learning_rate,
				  int current_minibatch_size,
                    precision_type momentum,
				  precision_type L2_reg,
				  bool norm_clipping,
				  //bool update_clipping,
				  precision_type norm_threshold){
					  
    // get the bias gradient for all dimensions in parallel
    //int size = b.size();
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
		//cerr<<"Linear diagonal "<<endl<<U<<endl;
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
	  #pragma omp parallel for firstprivate(num_examples)
	  for (int i=0; i<num_examples; i++){
	  	my_output.col(i).noalias() = (U.array()*input.col(i).array()).matrix();
	  }
	  //cerr<<"output to linear diagonal layer is "<<my_output<<endl;
      //my_output.leftCols(input.cols()).noalias() = U.array()*input.array();

  }
   

  
    template <typename DerivedGOut, typename DerivedGIn>
	void bProp(const MatrixBase<DerivedGOut> &input,
      MatrixBase<DerivedGIn> &output) const
    {
		
	    UNCONST(DerivedGIn, output, my_output);
  	    int num_examples = input.cols();
		//Can this be sped up with broadcasting ? 
		#pragma omp parallel for firstprivate(num_examples)
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
		//uncomment the pragma if you want to go hogwild
		//#pragma omp parallel for firstprivate(num_examples) 
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

	precision_type getGradSqdNorm() {
		return U_gradient.squaredNorm();
	}
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
	
	
      } 	
  }

  void updateParams(precision_type learning_rate,
				  int current_minibatch_size,
                    precision_type momentum,
				  precision_type L2_reg,
				  precision_type grad_scale){
					  
    // get the bias gradient for all dimensions in parallel
    //int size = b.size();
    // This used to be multithreaded, but there was no measureable difference

		 U += (learning_rate * U_gradient*grad_scale);
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
        Output_word_embeddings() :update_map(int_map()){ }
        Output_word_embeddings(int rows, int cols) :update_map(int_map()){ resize(rows, cols); }

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
		//cerr<<"Output W "<<endl<<W<<endl;
		if (init_bias ==0 ){ 
			initBias(engine, b, init_normal, init_range);
		} else {
        	b.fill(init_bias);
		}
		//cerr<<"Output b "<<endl<<b<<endl;
		//shared_noise_embeddings.setZero()
    }
	
	void resizeNoiseEmbeddings(int num_noise_samples, int minibatch_size){
		shared_noise_embeddings.setZero(num_noise_samples+minibatch_size,W.cols());
	}

    int n_inputs () const { return W.cols(); }
    int n_outputs () const { return W.rows(); }

	int rows() const {return W.rows(); }
	int cols() const {return W.cols() + 1; }
	
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
		 if (col == W.cols()) {
		 	return b_gradient(row,0);
		}
		 else{
			 return W_gradient(row,col);
		 }
					 //cerr<<"W_gradient"<<endl;
					 //return W_gradient(row,col);
					 
	 }
				 
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
	
  }
  
  precision_type getGradSqdNorm(){
    //If the udpate map has size 0, then we used NCE and the grad norm
    //comes from the words that are being updated. 
	  if (update_map.size() == 0) {
		  return (W_gradient.squaredNorm()+b_gradient.squaredNorm());
	  } else {
		  std::vector<int> update_items;
		  //Go over all the words to be updated and get their norms
	  	//int num_items = update_items.size();
        for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        int num_items = update_items.size();		
	  	precision_type squared_param_norm = 0.;
	      for (int item_id=0; item_id<num_items; item_id++)
	      {
	          int update_item = update_items[item_id];
	  		//W_gradient.row(update_item) /= current_minibatch_size;
	  		squared_param_norm += W_gradient.row(update_item).squaredNorm()+b_gradient(update_item)*b_gradient(update_item);
			
	  	 }
		 return(squared_param_norm);		  
	  }
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
  
  void updateParams(precision_type learning_rate,
  		int current_minibatch_size,
  		precision_type momentum,
		precision_type L2_reg,
		precision_type grad_scale){

	 	  W += learning_rate*W_gradient*grad_scale;
	 	  b += learning_rate*b_gradient*grad_scale;						   
			   
   }
  
  void resetGradient(){
	  W_gradient.setZero();
	  b_gradient.setZero();
  }
  
  void changeRandomParam(precision_type offset, 
						int &rand_row,
						int &rand_col){
  	//changeRandomParamInMatrix(W, offset, rand_row, rand_col);
	
	 if (rand_col == W.cols()) {
	 	//return b_gradient(row,0);
		 int temp_col = 0;
		changeRandomParamInMatrix(b,offset, rand_row, temp_col);
	}
	 else{
		 changeRandomParamInMatrix(W,offset, rand_row, rand_col);
		 //return W_gradient(row,col);
	 }
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
		//Matrix<precision_type,Dynamic,1> temp_b_gradient;
		//temp_b_gradient.setZero(b.rows(),b.cols());
		//cerr<<"samples are "<<samples<<endl;
	    USCMatrix<precision_type> gradient_output(W.rows(), samples, weights);
	    uscgemm(1.0,
	      gradient_output,
	      predicted_embeddings.leftCols(samples.cols()).transpose(),
	      W_gradient);
		  //cerr<<"Weights are "<<weights<<endl;
		 //cerr<<"b gradient before "<<b_gradient<<endl; 
	    uscgemv(1.0, 
	      gradient_output,
		      Matrix<precision_type,Dynamic,1>::Ones(weights.cols()),
	      b_gradient);
		  /*
  	    uscgemv(1.0, 
  	      gradient_output,
  		      Matrix<precision_type,Dynamic,1>::Ones(weights.cols()),
  	      temp_b_gradient);		  
		  cerr<<"b gradient is "<<b_gradient<<endl;
		  getchar();
		  cerr<<"temp b gradient is "<<temp_b_gradient<<endl;
		  getchar();
		  */
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
		  //cerr<<"W gradient is "<<W_gradient<<endl;
		  //cerr<<"b gradient is "<<b_gradient<<endl;
		  //getchar();
	      std::vector<int> update_items;
	      for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
	      {
	          update_items.push_back(it->first);
	      }
	      int num_items = update_items.size();
		  #ifdef CLIP_SPARSE
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
				  precision_type update_value = learning_rate * 
					  					b_gradient(update_item);
				  b(update_item) += std::min(0.01, std::max(double(update_value),-0.01));
				  b(update_item) = std::min(0.5, std::max(double(b(update_item)),-0.5));				  
			  } else {
		          W.row(update_item) += learning_rate*
		              W_gradient.row(update_item);
				  b(update_item) += learning_rate*b_gradient(update_item);
			  }
			  #else 
		          W.row(update_item) += learning_rate*
		              W_gradient.row(update_item);
				  b(update_item) += learning_rate*b_gradient(update_item);
			  #endif
	          
			  W_gradient.row(update_item).setZero();
			  b_gradient(update_item) = 0;
	
				  
	      }
		//we have to clear the update map
		this->update_map.clear();
	}

	void updateParamsNCE(precision_type learning_rate,
						int current_minibatch_size,
						precision_type momentum,
						precision_type L2_reg,
						precision_type grad_scale){
					
	    // Convert to std::vector for parallelization
		  //cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
		  //cerr<<"the W gradient norm is "<<W_gradient.norm()<<endl;					
	      std::vector<int> update_items;
	      for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
	      {
	          update_items.push_back(it->first);
	      }
	      int num_items = update_items.size();

		  #pragma omp parallel for firstprivate(grad_scale)
	      for (int item_id=0; item_id<num_items; item_id++)
	      {
	          int update_item = update_items[item_id];

	          W.row(update_item) += learning_rate*
	              W_gradient.row(update_item)*grad_scale;
			  b(update_item) += learning_rate*b_gradient(update_item)*grad_scale;


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
		//cerr<<"Input W "<<endl<<W<<endl;
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
					precision_type grad_scale){
						
	    // Convert to std::vector for parallelization
        std::vector<int> update_items;
        for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        int num_items = update_items.size();

        #pragma omp parallel for
        for (int item_id=0; item_id<num_items; item_id++)
        {
            int update_item = update_items[item_id];

			  //Divide the gradient by the grad scale	
	          W.row(update_item) += learning_rate*
	              W_gradient.row(update_item)*grad_scale; 

            W_gradient.row(update_item).setZero();
        }
		//we have to clear the update map
		this->update_map.clear();
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
			  #else 
	          W.row(update_item) += learning_rate*
	              W_gradient.row(update_item);
			 #endif 
            //GRADIENT CLIPPING
            //W.row(update_item) += learning_rate*
            //    W_gradient.row(update_item).array().unaryExpr(Clipper()).matrix();
            //SETTING THE GRADIENT TO ZERO
            W_gradient.row(update_item).setZero();
        }
		//we have to clear the update map
		this->update_map.clear();
  }


    void changeRandomParam(precision_type offset, 
							int &rand_row,
							int &rand_col){
    	changeRandomParamInMatrix(W, offset, rand_row, rand_col);
    }

    precision_type getGradSqdNorm(){
		std::vector<int> update_items;
		//Go over all the words to be updated and get their norms
        for (int_map::iterator it = this->update_map.begin(); it != this->update_map.end(); ++it)
        {
            update_items.push_back(it->first);
        }
        //int num_items = update_items.size();		
	  	int num_items = update_items.size();
	  	precision_type squared_param_norm = 0.;
	    for (int item_id=0; item_id<num_items; item_id++)
	    {
	      int update_item = update_items[item_id];
	  	  //W_gradient.row(update_item) /= current_minibatch_size;
	  	  squared_param_norm += W_gradient.row(update_item).squaredNorm();
	  	}
		return(squared_param_norm);	  
    }
	
  	precision_type getGradient(int row,
  			 int col) { return W_gradient(row,col);}	
			 

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
	

		b_gradient.setZero();
		if (init_bias == 0.) {
      		initBias(engine, b, init_normal, init_range);
		} else {
			b.fill(init_bias);
		}
		//<<b<<std::endl;
		//cerr<<"hidden bias "<<endl<<b<<endl;
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
	 precision_type getGradSqdNorm(){
		 return b_gradient.squaredNorm();
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
	 //The accumulated gradient is now added to the parameters
	 void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
	 					precision_type momentum,
						precision_type L2_reg,
						precision_type grad_scale){
		//as of now, only SGD

		b += learning_rate*b_gradient*grad_scale;
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

