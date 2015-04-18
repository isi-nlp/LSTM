//creating the structure of the nn in a graph that will help in performing backpropagation and forward propagation
#pragma once

#include <cstdlib>
#include "neuralClasses.h"
#include <Eigen/Dense>

namespace nplm
{

template <class X>
class Node {
    public:
        X * param; //what parameter is this
        //vector <void *> children;
        //vector <void *> parents;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> fProp_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> bProp_matrix;
	int minibatch_size;

    public:
        Node() : param(NULL), minibatch_size(0) { }

        Node(X *input_param, int minibatch_size)
	  : param(input_param),
	    minibatch_size(minibatch_size)
        {
	    resize(minibatch_size);
        }

	void resize(int minibatch_size)
	{
	    this->minibatch_size = minibatch_size;
	    if (param->n_outputs() != -1)
	    {
	        fProp_matrix.setZero(param->n_outputs(), minibatch_size);
	    }
        if (param->n_inputs() != -1)
        {
        	bProp_matrix.setZero(param->n_inputs(), minibatch_size);
        }
	}

	void resize() { resize(minibatch_size); }

        /*
        void Fprop(Matrix<double,Dynamic,Dynamic> & input,int n_cols)
        {
            param->fProp(input,fProp_matrix,0,0,n_cols);
        }
        void Fprop(Matrix<double,1,Dynamic> & input,int n_cols)
        {
            param->fProp(input,fProp_matrix,0,0,n_cols);
        }
        */
        //for f prop, just call the fProp node of the particular parameter. 

};

class LSTM_node {
	int minibatch_size;
public:
	//Each LSTM node has a bunch of nodes and temporary data structures
    Node<Input_word_embeddings> input_layer_node;
    Node<Linear_layer> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
	Node<Linear_layer> W_h_to_i_node, W_h_to_f_node, W_h_to_c_node, W_h_to_o_node;
	Node<Linear_diagonal_layer> W_c_to_i_node, W_c_to_f_node, W_c_to_o_node;
    Node<Hidden_layer> i_t_node,f_t_node,o_t_node,tanh_c_prime_t_node;
	Node<Activation_function> tanh_c_t_node;

	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> h_t,c_t;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_h_t,
														d_Err_t_to_n_d_c_t,
														d_Err_t_to_n_d_o_t,
														d_Err_t_to_n_d_f_t,
														d_Err_t_to_n_d_i_t,
														d_Err_t_to_n_d_tanh_c_t,
														d_Err_t_to_n_d_tanh_c_prime_t,
														d_Err_t_to_n_d_x_t;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>	d_Err_t_to_n_d_h_tMinusOne,
														d_Err_t_to_n_d_c_tMinusOne;

	
	LSTM_node(): 
		W_x_to_i_node(),
		W_x_to_f_node(),
		W_x_to_c_node(),
		W_x_to_o_node(),
		W_h_to_i_node(),
		W_h_to_f_node(),
		W_h_to_c_node(),
		W_h_to_o_node(),
		W_c_to_i_node(),
		W_c_to_f_node(),
		W_c_to_o_node(),
		i_t_node(),
		f_t_node(),
		o_t_node(),
		tanh_c_prime_t_node(),
		tanh_c_t_node(),
		input_layer_node() {}

	LSTM_node(model &lstm, int minibatch_size): 
		W_x_to_i_node(&lstm.W_x_to_i, minibatch_size),
		W_x_to_f_node(&lstm.W_x_to_f, minibatch_size),
		W_x_to_c_node(&lstm.W_x_to_c, minibatch_size),
		W_x_to_o_node(&lstm.W_x_to_o, minibatch_size),
		W_h_to_i_node(&lstm.W_h_to_i, minibatch_size),
		W_h_to_f_node(&lstm.W_h_to_f, minibatch_size),
		W_h_to_c_node(&lstm.W_h_to_c, minibatch_size),
		W_h_to_o_node(&lstm.W_h_to_o, minibatch_size),
		W_c_to_i_node(&lstm.W_c_to_i, minibatch_size),
		W_c_to_f_node(&lstm.W_c_to_i, minibatch_size),
		W_c_to_o_node(&lstm.W_c_to_o, minibatch_size),
		i_t_node(&lstm.i_t,minibatch_size),
		f_t_node(&lstm.f_t,minibatch_size),
		o_t_node(&lstm.o_t,minibatch_size),
		tanh_c_prime_t_node(&lstm.tanh_c_prime_t,minibatch_size),
		tanh_c_t_node(&lstm.tanh_c_t,minibatch_size),
		input_layer_node(&lstm.input_layer,minibatch_size)
		 {
			this->minibatch_size = minibatch_size;
		 }
	//Resizing all the parameters
	void resize(int minibatch_size){
		this->minibatch_size = minibatch_size;
		W_x_to_i_node.resize(minibatch_size);
		W_x_to_f_node.resize(minibatch_size);
		W_x_to_c_node.resize(minibatch_size);
		W_x_to_o_node.resize(minibatch_size);
		W_h_to_i_node.resize(minibatch_size);
		W_h_to_f_node.resize(minibatch_size);
		W_h_to_c_node.resize(minibatch_size);
		W_h_to_o_node.resize(minibatch_size);
		W_c_to_i_node.resize(minibatch_size);
		W_c_to_f_node.resize(minibatch_size);
		W_c_to_o_node.resize(minibatch_size);
		i_t_node.resize(minibatch_size);
		f_t_node.resize(minibatch_size);
		o_t_node.resize(minibatch_size);
		tanh_c_prime_t_node.resize(minibatch_size);
		input_layer_node.resize(minibatch_size);
		
		//Resizing all the local node matrices
		h_t.resize(W_h_to_i_node.param->n_inputs(),minibatch_size);
		c_t.resize(W_c_to_i_node.param->n_inputs(),minibatch_size);
		d_Err_t_to_n_d_h_t.resize(W_h_to_i_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_c_t.resize(W_c_to_i_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_o_t.resize(o_t_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_f_t.resize(f_t_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_i_t.resize(i_t_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_tanh_c_t.resize(tanh_c_t_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_tanh_c_prime_t.resize(tanh_c_prime_t_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_x_t.resize(input_layer_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_h_tMinusOne.resize(W_h_to_i_node.param->n_outputs(),minibatch_size);
		d_Err_t_to_n_d_c_tMinusOne.resize(W_c_to_i_node.param->n_outputs(),minibatch_size);
		
	} 
	
	template<typename Derived, typename DerivedIn>
    void fProp(const MatrixBase<Derived> &data,	
		const MatrixBase<DerivedIn> c_t_minus_one,
		// MatrixBase<DerivedOut> const_c_t,
		const MatrixBase<DerivedIn> h_t_minus_one) {
		//const MatrixBase<DerivedOut> const_h_t){
		
		//UNCONST(DerivedOut,const_c_t,c_t);
		//UNCONST(DerivedOut,const_h_t,h_t);
		
        //start_timer(0);
    	input_layer_node.param->fProp(data, input_layer_node.fProp_matrix);
    	//stop_timer(0);
    
		
	    //first_hidden_linear_node.param->fProp(sparse_data,
		//				  first_hidden_linear_node.fProp_matrix);
						  
		//How much to scale the input
		W_x_to_i_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_i_node.fProp_matrix);
		W_h_to_i_node.param->fProp(h_t_minus_one,W_h_to_i_node.fProp_matrix);
		W_c_to_i_node.param->fProp(c_t_minus_one,W_c_to_i_node.fProp_matrix);
		i_t_node.param->fProp(W_x_to_i_node.fProp_matrix + W_h_to_i_node.fProp_matrix + W_c_to_i_node.fProp_matrix,
							i_t_node.fProp_matrix);
		
		//How much to forget
		W_x_to_f_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_f_node.fProp_matrix);
		W_h_to_f_node.param->fProp(h_t_minus_one,W_h_to_f_node.fProp_matrix);
		W_c_to_f_node.param->fProp(c_t_minus_one,W_c_to_f_node.fProp_matrix);
		f_t_node.param->fProp(W_x_to_f_node.fProp_matrix + W_h_to_f_node.fProp_matrix + W_c_to_f_node.fProp_matrix,
							f_t_node.fProp_matrix);
		
		//computing c_prime_t
		W_x_to_c_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_c_node.fProp_matrix);
		W_h_to_c_node.param->fProp(h_t_minus_one,W_h_to_c_node.fProp_matrix);	
		tanh_c_prime_t_node.param->fProp(W_x_to_c_node.fProp_matrix + W_h_to_c_node.fProp_matrix,
										tanh_c_prime_t_node.fProp_matrix);
		
		//Computing the current cell value
		c_t = i_t_node.fProp_matrix.array()*c_t_minus_one.array() + f_t_node.fProp_matrix.array()*tanh_c_prime_t_node.fProp_matrix.array();
		
		//How much to scale the output
		W_x_to_o_node.param->fProp(input_layer_node.fProp_matrix, W_x_to_o_node.fProp_matrix);
		W_h_to_o_node.param->fProp(h_t_minus_one,W_h_to_o_node.fProp_matrix);
		W_c_to_o_node.param->fProp(c_t,W_c_to_i_node.fProp_matrix);
		o_t_node.param->fProp(W_x_to_o_node.fProp_matrix +  W_h_to_o_node.fProp_matrix + W_c_to_o_node.fProp_matrix ,
							o_t_node.fProp_matrix);	
		
		//computing the hidden layer
		tanh_c_t_node.param->fProp(c_t,tanh_c_t_node.fProp_matrix);
		h_t = o_t_node.fProp_matrix.array()*tanh_c_t_node.fProp_matrix.array();		
	}
	
	template<typename DerivedData,typename DerivedIn, typename DerivedDIn, typename DerivedDOut>
	void bProp(const MatrixBase<DerivedData> &data,
			   //const MatrixBase<DerivedIn> c_t,
			   const MatrixBase<DerivedIn> c_t_minus_one,
			   const MatrixBase<DerivedIn> d_Err_t_d_h_t,
			   const MatrixBase<DerivedDIn> d_Err_tPlusOne_to_n_d_c_t,
			   const MatrixBase<DerivedDIn> d_Err_tPlusOne_to_n_d_h_t) {
				   
		Matrix<double,Dynamic,Dynamic> dummy_matrix;
	    //UNCONST(DerivedDOut,const_d_Err_t_to_n_d_c_tMinusOne,d_Err_t_to_n_d_c_tMinusOne);
	    //UNCONST(DerivedDOut,const_d_Err_t_to_n_d_h_tMinusOne,d_Err_t_to_n_d_h_tMinusOne);
		
		//NOTE: d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt h_t. 
		//Similarly, d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt c_t. 
		//This is a slight abuse of notation. In our case, since we're maximizing log likelihood, we're taking derivatives of the negative of the 
		//error function, which is the cross entropy.
		
		//Error derivatives for h_t
		d_Err_t_to_n_d_h_t = d_Err_t_d_h_t + d_Err_tPlusOne_to_n_d_h_t;
		
		//Error derivativs for o_t
		d_Err_t_to_n_d_o_t = d_Err_t_to_n_d_h_t.array()*tanh_c_t_node.fProp_matrix.array();
		o_t_node.param->bProp(d_Err_t_to_n_d_o_t,
						      o_t_node.bProp_matrix,
							  dummy_matrix,
							  o_t_node.fProp_matrix);// the third	 field does not matter. Its a dummy matrix
		/*				  
						  	first_hidden_activation_node.param->bProp(second_hidden_linear_node.bProp_matrix,
						  						  first_hidden_activation_node.bProp_matrix,
						  						  first_hidden_linear_node.fProp_matrix,
						  						  first_hidden_activation_node.fProp_matrix);
		*/
		//Error derivatives for tanh_c_t					   
		d_Err_t_to_n_d_tanh_c_t = d_Err_t_d_h_t.array() * o_t_node.fProp_matrix.array();
		tanh_c_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_t,
							tanh_c_t_node.bProp_matrix,
							dummy_matrix,
							tanh_c_t_node.fProp_matrix);
		
		//Error derivatives for c_t
		W_c_to_o_node.param->bProp(o_t_node.bProp_matrix,
								W_c_to_o_node.bProp_matrix);
		d_Err_t_to_n_d_c_t =  tanh_c_t_node.bProp_matrix + W_c_to_o_node.bProp_matrix + d_Err_tPlusOne_to_n_d_c_t;
		
		//Error derivatives for f_t
		d_Err_t_to_n_d_f_t = d_Err_t_to_n_d_c_t.array()*c_t_minus_one.array();
		f_t_node.param->bProp(d_Err_t_to_n_d_f_t,
						      f_t_node.bProp_matrix,
							  dummy_matrix,
							  f_t_node.fProp_matrix);
		
		//Error derivatives for i_t
		d_Err_t_to_n_d_i_t = d_Err_t_to_n_d_c_t.array()*tanh_c_prime_t_node.fProp_matrix.array();
		i_t_node.param->bProp(d_Err_t_to_n_d_i_t,
						      i_t_node.bProp_matrix,
							  dummy_matrix,
							  i_t_node.fProp_matrix);	
							  	
		//Error derivatives for c_prime_t
		d_Err_t_to_n_d_tanh_c_prime_t = d_Err_t_to_n_d_c_t.array()*i_t_node.fProp_matrix.array();
		
		//Error derivatives for h_t_minus_one
		W_h_to_o_node.param->bProp(o_t_node.bProp_matrix,
						 W_h_to_o_node.bProp_matrix);
 		W_h_to_f_node.param->bProp(f_t_node.bProp_matrix,
 						 W_h_to_f_node.bProp_matrix);
  		W_h_to_i_node.param->bProp(i_t_node.bProp_matrix,
  						 W_h_to_i_node.bProp_matrix);
   		W_h_to_c_node.param->bProp(tanh_c_prime_t_node.bProp_matrix,
   						 W_h_to_c_node.bProp_matrix);
		d_Err_t_to_n_d_h_tMinusOne = W_h_to_o_node.bProp_matrix + 
									W_h_to_f_node.bProp_matrix +
									W_h_to_i_node.bProp_matrix +
									W_h_to_c_node.bProp_matrix;		
		
		//Error derivatives for c_t_minus_one
		W_c_to_f_node.param->bProp(f_t_node.bProp_matrix,
							W_c_to_f_node.bProp_matrix);
		W_c_to_i_node.param->bProp(i_t_node.bProp_matrix,
							W_c_to_i_node.bProp_matrix);			
		d_Err_t_to_n_d_c_tMinusOne = (d_Err_t_to_n_d_c_t.array()*f_t_node.fProp_matrix.array()).matrix()+
									W_c_to_f_node.bProp_matrix +
									W_c_to_i_node.bProp_matrix;
		
		//Error derivatives for the input word embeddings
		W_x_to_c_node.param->bProp(tanh_c_prime_t_node.bProp_matrix,
								W_x_to_c_node.bProp_matrix);
		W_x_to_o_node.param->bProp(o_t_node.bProp_matrix,
								W_x_to_o_node.bProp_matrix);
		W_x_to_f_node.param->bProp(f_t_node.bProp_matrix,
								W_x_to_f_node.bProp_matrix);
		W_x_to_i_node.param->bProp(i_t_node.bProp_matrix,
								W_x_to_i_node.bProp_matrix);
		d_Err_t_to_n_d_x_t = W_x_to_c_node.bProp_matrix + 
							W_x_to_o_node.bProp_matrix +
							W_x_to_f_node.bProp_matrix +
							W_x_to_i_node.bProp_matrix;
		
		//Computing gradients of the paramters
		//Derivative of weights out of h_t
	    W_h_to_o_node.param->updateGradient(o_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);
	    W_h_to_f_node.param->updateGradient(f_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);
	    W_h_to_i_node.param->updateGradient(i_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);		
   		W_h_to_c_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix,
   						 					input_layer_node.fProp_matrix);
											
		//Derivative of weights out of c_t and c_t_minus_one
	    W_c_to_o_node.param->updateGradient(o_t_node.bProp_matrix,
											this->c_t);
	    W_c_to_i_node.param->updateGradient(i_t_node.bProp_matrix,
											c_t_minus_one);
	    W_c_to_f_node.param->updateGradient(f_t_node.bProp_matrix,
											c_t_minus_one);		
 
		//Derivatives of weights out of x_t
		W_x_to_o_node.param->updateGradient(o_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);
		W_x_to_i_node.param->updateGradient(i_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);
		W_x_to_f_node.param->updateGradient(f_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);	
		W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix,
											input_layer_node.fProp_matrix);			
		
		//Derivatives of the input embeddings							
	    input_layer_node.param->updateGradient(o_t_node.bProp_matrix + 
												f_t_node.bProp_matrix + 
												i_t_node.bProp_matrix + 
												tanh_c_prime_t_node.bProp_matrix,
									            data);
	
	}
		  
	/*
	void updateParams(double learning_rate,
	 					double momentum,
						double L2_reg){

		//updating params for weights out of hidden layer 
		W_h_to_o_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
 		W_h_to_f_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
  		W_h_to_i_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
   		W_h_to_c_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);

		//updating params for weights out of cell
		W_c_to_f_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_c_to_i_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_c_to_o_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);				


		//Error derivatives for the input word embeddings
		W_x_to_c_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_x_to_o_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_x_to_f_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_x_to_i_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);


		//Computing gradients of the paramters
		//Derivative of weights out of h_t
	    W_h_to_o_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
	    W_h_to_f_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
	    W_h_to_i_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);		
   		W_h_to_c_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		
		//Derivative of weights out of c_t and c_t_minus_one
	    W_c_to_o_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
	    W_c_to_i_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
	    W_c_to_f_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);		

		//Derivatives of weights out of x_t
		W_x_to_o_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_x_to_i_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		W_x_to_f_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);	
		W_x_to_c_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);			


		o_t_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		f_t_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
		i_t_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);	
		tanh_c_prime_t_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);	
												
		//Derivatives of the input embeddings							
	    input_layer_node.param->updateParams(learning_rate,
											momentum,
											L2_reg);
										
	}
	*/
	
	void resetGradient(){
		
	}	  				  
	
};

} // namespace nplm
