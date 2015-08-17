//creating the structure of the nn in a graph that will help in performing backpropagation and forward propagation
#pragma once

#include <cstdlib>
#include "neuralClasses.h"
#include <Eigen/Dense>
#include <Eigen/Core>

namespace nplm
{

	struct stateClipper{
	  precision_type operator() (precision_type x) const { 
	    return (precision_type) std::min(25., std::max(double(x),-25.));
	    //return(x);
	  }
	};

typedef Matrix<int,Dynamic,Dynamic> IndexMatrix;
typedef Matrix<precision_type,Dynamic,Dynamic> DoubleMatrix;

/*
//Virtual class that specifies the skeleton for the different input nodes
class Input_node{
	Input_node();

	//template <typename Derived>
	virtual void fProp(const IndexMatrix &data) { throw std::logic_error("attempted to use IndexMatrix"); }
	virtual void fProp(const DoubleMatrix &data) { throw std::logic_error("attempted to use DoubleMatrix"); }

	//template<typename DerivedData, typename DerivedDIn>
	virtual void bProp(const IndexMatrix &data,
				const DoubleMatrix &o_t_node_bProp_matrix,
				const DoubleMatrix &i_t_node_bProp_matrix,
				const DoubleMatrix &f_t_node_bProp_matrix,
				const DoubleMatrix &tanh_c_prime_t_node_bProp_matrix) {throw std::logic_error("attempted to use IndexMatrix");}
			
	virtual void bProp(const DoubleMatrix &data,
				const DoubleMatrix &o_t_node_bProp_matrix,
				const DoubleMatrix &i_t_node_bProp_matrix,
				const DoubleMatrix &f_t_node_bProp_matrix,
				const DoubleMatrix &tanh_c_prime_t_node_bProp_matrix) {throw std::logic_error("attempted to use IndexMatrix");}
};

*/


template <class X>
class Node {
    public:
        X * param; //what parameter is this
        //vector <void *> children;
        //vector <void *> parents;
	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> fProp_matrix;
	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> bProp_matrix;
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
        void Fprop(Matrix<precision_type,Dynamic,Dynamic> & input,int n_cols)
        {
            param->fProp(input,fProp_matrix,0,0,n_cols);
        }
        void Fprop(Matrix<precision_type,1,Dynamic> & input,int n_cols)
        {
            param->fProp(input,fProp_matrix,0,0,n_cols);
        }
        */
        //for f prop, just call the fProp node of the particular parameter. 

};

class Output_loss_node {
	int minibatch_size;
public:
	Matrix<precision_type,Dynamic,Dynamic> d_Err_t_d_h_t;
	Output_loss_node() :minibatch_size(0),d_Err_t_d_h_t(Matrix<precision_type,Dynamic,Dynamic>()) {}
	
	void resize(int num_hidden,int minibatch_size) {
		//d_Err_t_d_h_t.resize(num_hidden, minibatch_size);
		//Need to make this smarter
		this->minibatch_size = minibatch_size;
		d_Err_t_d_h_t.setZero(num_hidden, minibatch_size);
	}
	//void resize(int num_hidden,int minibatch_size){
	//	d_Err_t_d_h_t.resize(num_hidden, minibatch_size);
	//}
};

template <class input_node_type>
class LSTM_node {
	int minibatch_size;
public:
	//Each LSTM node has a bunch of nodes and temporary data structures
    Node<Input_word_embeddings> input_layer_node,W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
    //Node<Linear_layer> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
	Node<Linear_layer> W_h_to_i_node, W_h_to_f_node, W_h_to_c_node, W_h_to_o_node;
	Node<Linear_diagonal_layer> W_c_to_i_node, W_c_to_f_node, W_c_to_o_node;
    Node<Hidden_layer> i_t_node,f_t_node,o_t_node,tanh_c_prime_t_node;
	Node<Activation_function> tanh_c_t_node;

	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> h_t,c_t,c_t_minus_one, h_t_minus_one;
	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_h_t,
														d_Err_t_to_n_d_c_t,
														d_Err_t_to_n_d_o_t,
														d_Err_t_to_n_d_f_t,
														d_Err_t_to_n_d_i_t,
														d_Err_t_to_n_d_tanh_c_t,
														d_Err_t_to_n_d_tanh_c_prime_t,
														d_Err_t_to_n_d_x_t,
														i_t_input_matrix,
														f_t_input_matrix,
														o_t_input_matrix,
														tanh_c_prime_t_input_matrix,
														tanh_c_t_input_matrix;
														
														
	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic>	d_Err_t_to_n_d_h_tMinusOne,
														d_Err_t_to_n_d_c_tMinusOne;
										
	input_node_type *input_node;
	
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
		input_layer_node(),
		input_node(NULL) {}

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
		W_c_to_f_node(&lstm.W_c_to_f, minibatch_size),
		W_c_to_o_node(&lstm.W_c_to_o, minibatch_size),
		i_t_node(&lstm.i_t,minibatch_size),
		f_t_node(&lstm.f_t,minibatch_size),
		o_t_node(&lstm.o_t,minibatch_size),
		tanh_c_prime_t_node(&lstm.tanh_c_prime_t,minibatch_size),
		tanh_c_t_node(&lstm.tanh_c_t,minibatch_size),
		input_layer_node(&lstm.input_layer,minibatch_size),
		input_node(NULL)
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
		h_t_minus_one.resize(W_h_to_i_node.param->n_inputs(),minibatch_size);
		c_t_minus_one.resize(W_c_to_i_node.param->n_inputs(),minibatch_size);
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
		i_t_input_matrix.resize(i_t_node.param->n_inputs(),minibatch_size);
		f_t_input_matrix.resize(f_t_node.param->n_inputs(),minibatch_size);
		o_t_input_matrix.resize(o_t_node.param->n_inputs(),minibatch_size);
		tanh_c_prime_t_input_matrix.resize(tanh_c_prime_t_node.param->n_inputs(),minibatch_size);
		
	} 
	
	void set_input_node(input_node_type &input_node){this->input_node = &input_node;}
	
	template<typename Derived> //, typename DerivedCIn, typename DerivedHIn>
    void fProp(const MatrixBase<Derived> &data) { //,	
		//const MatrixBase<DerivedCIn> &c_t_minus_one,
		// MatrixBase<DerivedOut> const_c_t,
		//const MatrixBase<DerivedHIn> &h_t_minus_one) {
		//const MatrixBase<DerivedOut> const_h_t){
		
		//UNCONST(DerivedOut,const_c_t,c_t);
		//UNCONST(DerivedOut,const_h_t,h_t);
		
		//cerr<<"c t -1 is "<<c_t_minus_one<<endl;
		//cerr<<"h t -1 is "<<h_t_minus_one<<endl;
		//getchar();
        //start_timer(0);
		//cerr<<"data is "<<data<<endl;
		input_node->fProp(data);
    	//input_layer_node.param->fProp(data, input_layer_node.fProp_matrix);
		//W_x_to_c_node.param->fProp(data,W_x_to_c_node.fProp_matrix);
		//W_x_to_f_node.param->fProp(data,W_x_to_f_node.fProp_matrix);
		//W_x_to_o_node.param->fProp(data,W_x_to_o_node.fProp_matrix);
		//W_x_to_i_node.param->fProp(data,W_x_to_i_node.fProp_matrix);
		
		//current_minibatch_size = data.cols();
    	//stop_timer(0);
    	//std::cerr<<"input layer fprop matrix is "<<input_layer_node.fProp_matrix<<endl;
		
	    //first_hidden_linear_node.param->fProp(sparse_data,
		//				  first_hidden_linear_node.fProp_matrix);
						  
		//How much to scale the input
		//W_x_to_i_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_i_node.fProp_matrix);
		//std::cerr<<"x to i fprop"<<W_x_to_i_node.fProp_matrix<<std::endl;
		W_h_to_i_node.param->fProp(h_t_minus_one,W_h_to_i_node.fProp_matrix);
		W_c_to_i_node.param->fProp(c_t_minus_one,W_c_to_i_node.fProp_matrix);
		//std::cerr<<"c to i fprop"<<W_c_to_i_node.fProp_matrix<<std::endl;
		//i_t_input_matrix.noalias() = W_x_to_i_node.fProp_matrix + W_h_to_i_node.fProp_matrix + W_c_to_i_node.fProp_matrix;
		i_t_input_matrix.noalias() = input_node->W_x_to_i_node.fProp_matrix + W_h_to_i_node.fProp_matrix + W_c_to_i_node.fProp_matrix;
		//cerr<<"i t input matrix"<<i_t_input_matrix<<endl;
		i_t_node.param->fProp(i_t_input_matrix,
							i_t_node.fProp_matrix);
							
		//std::cerr<<"i_t node fProp value is "<<i_t_node.fProp_matrix<<std::endl;
		
		//How much to forget
		//W_x_to_f_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_f_node.fProp_matrix);
		W_h_to_f_node.param->fProp(h_t_minus_one,W_h_to_f_node.fProp_matrix);
		//std::cerr<<"W_h_to_f_node fprop is "<<W_h_to_f_node.fProp_matrix<<std::endl;
		W_c_to_f_node.param->fProp(c_t_minus_one,W_c_to_f_node.fProp_matrix);
		//std::cerr<<"W_c_to_f_node fprop is "<<W_c_to_f_node.fProp_matrix<<std::endl;
		f_t_input_matrix.noalias() = input_node->W_x_to_f_node.fProp_matrix + W_h_to_f_node.fProp_matrix + W_c_to_f_node.fProp_matrix;
		//std::cerr<<" f t node input matrix is "<<f_t_input_matrix<<std::endl;
		f_t_node.param->fProp(f_t_input_matrix,
							f_t_node.fProp_matrix);
		//std::cerr<<"f_t node fProp value is "<<f_t_node.fProp_matrix<<std::endl;
		//computing c_prime_t
		//W_x_to_c_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_c_node.fProp_matrix);
		W_h_to_c_node.param->fProp(h_t_minus_one,W_h_to_c_node.fProp_matrix);	
		tanh_c_prime_t_input_matrix.noalias() = input_node->W_x_to_c_node.fProp_matrix + W_h_to_c_node.fProp_matrix;
		tanh_c_prime_t_node.param->fProp(tanh_c_prime_t_input_matrix,
										tanh_c_prime_t_node.fProp_matrix);
		
		//std::cerr<<"tanh_c_prime_t_node "<<tanh_c_prime_t_node.fProp_matrix<<std::endl;
		
		//Computing the current cell value
		//cerr<<"c_t_minus_one"<<c_t_minus_one<<endl;
		//cerr<<c_t_minus_one.rows()<<" "<<c_t_minus_one.cols()<<endl;
		c_t.array() = f_t_node.fProp_matrix.array()*c_t_minus_one.array() + 
				i_t_node.fProp_matrix.array()*tanh_c_prime_t_node.fProp_matrix.array();
		//cerr<<"c_t "<<c_t<<endl;
		//How much to scale the output
		//W_x_to_o_node.param->fProp(input_layer_node.fProp_matrix, W_x_to_o_node.fProp_matrix);
		W_h_to_o_node.param->fProp(h_t_minus_one,W_h_to_o_node.fProp_matrix);
		W_c_to_o_node.param->fProp(c_t,W_c_to_o_node.fProp_matrix);
		o_t_input_matrix.noalias() = input_node->W_x_to_o_node.fProp_matrix +  
						   W_h_to_o_node.fProp_matrix + 
						   W_c_to_o_node.fProp_matrix;
		//std::cerr<<"o t input matrix is "<<o_t_input_matrix<<std::endl;
		o_t_node.param->fProp(o_t_input_matrix,
							o_t_node.fProp_matrix);	

		//std::cerr<<"o_t "<<o_t_node.fProp_matrix<<std::endl;
		//computing the hidden layer
		tanh_c_t_node.param->fProp(c_t,tanh_c_t_node.fProp_matrix);
		//<<"tanh_c_t_node.fProp_matrix is "<<tanh_c_t_node.fProp_matrix<<endl;
		h_t.array() = o_t_node.fProp_matrix.array()*tanh_c_t_node.fProp_matrix.array();		
		//std::cerr<<"h_t "<<h_t<<endl;
		//getchar();
	}
	
	template<typename DerivedData, typename DerivedIn, typename DerivedDCIn, typename DerivedDHIn>
	void bProp(const MatrixBase<DerivedData> &data,
			   //const MatrixBase<DerivedIn> c_t,
			   //const MatrixBase<DerivedHIn> &h_t_minus_one,
			   //const MatrixBase<DerivedCIn> &c_t_minus_one,
			   const MatrixBase<DerivedIn> &d_Err_t_d_h_t,
			   const MatrixBase<DerivedDCIn> &d_Err_tPlusOne_to_n_d_c_t,
			   const MatrixBase<DerivedDHIn> &d_Err_tPlusOne_to_n_d_h_t,
			   bool gradient_check,
			   bool norm_clipping) {
				   
		Matrix<precision_type,Dynamic,Dynamic> dummy_matrix;
		int current_minibatch_size = data.cols();
		//cerr<<"h_t_minus_one "<<h_t_minus_one<<endl;
		//cerr<<"c_t_minus_one "<<c_t_minus_one<<endl;
		//cerr<<"d_Err_tPlusOne_to_n_d_c_t "<<d_Err_tPlusOne_to_n_d_c_t<<endl;
		//cerr<<"d_Err_tPlusOne_to_n_d_h_t "<<d_Err_tPlusOne_to_n_d_h_t<<endl;
		//cerr<<"c t -1 is "<<c_t_minus_one<<endl;
	    //UNCONST(DerivedDOut,const_d_Err_t_to_n_d_c_tMinusOne,d_Err_t_to_n_d_c_tMinusOne);
	    //UNCONST(DerivedDOut,const_d_Err_t_to_n_d_h_tMinusOne,d_Err_t_to_n_d_h_tMinusOne);
		
		//NOTE: d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt h_t. 
		//Similarly, d_Err_t_to_n_d_h_t is read as derivative of Error function from time t to n wrt c_t. 
		//This is a slight abuse of notation. In our case, since we're maximizing log likelihood, we're taking derivatives of the negative of the 
		//error function, which is the cross entropy.
		
		//Error derivatives for h_t
		//cerr<<"d_Err_t_d_h_t "<<d_Err_t_d_h_t<<endl;
		//cerr<<"d_Err_tPlusOne_to_n_d_h_t "<<d_Err_tPlusOne_to_n_d_h_t<<endl;
		d_Err_t_to_n_d_h_t = d_Err_t_d_h_t + d_Err_tPlusOne_to_n_d_h_t;
		//cerr<<"d_Err_t_to_n_d_h_t is "<<d_Err_t_to_n_d_h_t<<endl;
		//cerr<<"tanh_c_t_node.fProp_matrix is "<<tanh_c_t_node.fProp_matrix<<endl;
		//Error derivativs for o_t
		d_Err_t_to_n_d_o_t.array() = d_Err_t_to_n_d_h_t.array()*tanh_c_t_node.fProp_matrix.array();
		//cerr<<"d_Err_t_to_n_d_o_t "<<d_Err_t_to_n_d_o_t<<endl;
		//cerr<<"O t node fProp matrix is "<<o_t_node.fProp_matrix<<endl;
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
		//cerr<<"o t node backprop matrix is "<<o_t_node.bProp_matrix<<endl;
		//Error derivatives for tanh_c_t					   
		//d_Err_t_to_n_d_tanh_c_t.array() = d_Err_t_d_h_t.array() * o_t_node.fProp_matrix.array();// THIS WAS THE WRONG GRADIENT!!
		d_Err_t_to_n_d_tanh_c_t.array() = d_Err_t_to_n_d_h_t.array() * o_t_node.fProp_matrix.array();
		//cerr<<"d_Err_t_to_n_d_tanh_c_t "<<d_Err_t_to_n_d_tanh_c_t<<endl;
		tanh_c_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_t,
							tanh_c_t_node.bProp_matrix,
							dummy_matrix,
							tanh_c_t_node.fProp_matrix);
		//cerr<<"tanh_c_t_node.bProp_matrix "<<tanh_c_t_node.bProp_matrix<<endl;
		//Error derivatives for c_t
		//cerr<<"o_t_node.bProp_matrix"<<o_t_node.bProp_matrix<<endl;
		W_c_to_o_node.param->bProp(o_t_node.bProp_matrix,
								W_c_to_o_node.bProp_matrix);
		//cerr<<"W_c_to_o_node.bProp_matrix "<<W_c_to_o_node.bProp_matrix<<endl;
		d_Err_t_to_n_d_c_t =  tanh_c_t_node.bProp_matrix + W_c_to_o_node.bProp_matrix + d_Err_tPlusOne_to_n_d_c_t;
		//cerr<<"d_Err_t_to_n_d_c_t "<<d_Err_t_to_n_d_c_t<<endl;
		
		//Error derivatives for f_t
		d_Err_t_to_n_d_f_t.array() = d_Err_t_to_n_d_c_t.array()*c_t_minus_one.array();
		//cerr<<"d_Err_t_to_n_d_f_t "<<d_Err_t_to_n_d_f_t<<endl;
		f_t_node.param->bProp(d_Err_t_to_n_d_f_t,
						      f_t_node.bProp_matrix,
							  dummy_matrix,
							  f_t_node.fProp_matrix);
		//cerr<<"f_t_node.bProp_matrix "<<f_t_node.bProp_matrix<<endl;
		
		//Error derivatives for i_t
		d_Err_t_to_n_d_i_t.array() = d_Err_t_to_n_d_c_t.array()*tanh_c_prime_t_node.fProp_matrix.array();
		//cerr<<"d_Err_t_to_n_d_i_t "<<d_Err_t_to_n_d_i_t<<endl;
		i_t_node.param->bProp(d_Err_t_to_n_d_i_t,
						      i_t_node.bProp_matrix,
							  dummy_matrix,
							  i_t_node.fProp_matrix);	
		//cerr<<" i_t_node.bProp_matrix "<<i_t_node.bProp_matrix<<endl;
							  	
		//Error derivatives for c_prime_t
		d_Err_t_to_n_d_tanh_c_prime_t.array() = d_Err_t_to_n_d_c_t.array()*i_t_node.fProp_matrix.array();
		//cerr<<" d_Err_t_to_n_d_tanh_c_prime_t "<<d_Err_t_to_n_d_tanh_c_prime_t<<endl;
		//tanh_c_prime_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_prime_t,
		//								tanh_c_prime_t_node.bProp_matrix);
		
		tanh_c_prime_t_node.param->bProp(d_Err_t_to_n_d_tanh_c_prime_t,
						      tanh_c_prime_t_node.bProp_matrix,
							  dummy_matrix,
							  tanh_c_prime_t_node.fProp_matrix);	
		//cerr<<"tanh_c_prime_t_node.bProp_matrix "<<tanh_c_prime_t_node.bProp_matrix<<endl;									
		
		//Error derivatives for h_t_minus_one
		W_h_to_o_node.param->bProp(o_t_node.bProp_matrix,
						 W_h_to_o_node.bProp_matrix);
 		W_h_to_f_node.param->bProp(f_t_node.bProp_matrix,
 						 W_h_to_f_node.bProp_matrix);
  		W_h_to_i_node.param->bProp(i_t_node.bProp_matrix,
  						 W_h_to_i_node.bProp_matrix);
		//cerr<<"tanh_c_prime_t_node.bProp_matrix "<<tanh_c_prime_t_node.bProp_matrix<<endl;
   		W_h_to_c_node.param->bProp(tanh_c_prime_t_node.bProp_matrix,
   						 W_h_to_c_node.bProp_matrix);
		d_Err_t_to_n_d_h_tMinusOne = W_h_to_o_node.bProp_matrix + 
									W_h_to_f_node.bProp_matrix +
									W_h_to_i_node.bProp_matrix +
									W_h_to_c_node.bProp_matrix;		
		//cerr<<"d_Err_t_to_n_d_h_tMinusOne "<<d_Err_t_to_n_d_h_tMinusOne<<endl;
		//Error derivatives for c_t_minus_one
		W_c_to_f_node.param->bProp(f_t_node.bProp_matrix,
							W_c_to_f_node.bProp_matrix);
		W_c_to_i_node.param->bProp(i_t_node.bProp_matrix,
							W_c_to_i_node.bProp_matrix);	
									
		d_Err_t_to_n_d_c_tMinusOne = (d_Err_t_to_n_d_c_t.array()*f_t_node.fProp_matrix.array()).matrix()+
									W_c_to_f_node.bProp_matrix +
									W_c_to_i_node.bProp_matrix;
		//cerr<<"d_Err_t_to_n_d_c_tMinusOne "<<d_Err_t_to_n_d_c_tMinusOne<<endl;
		//Error derivatives for the input word embeddings
		/*
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
		*/
		//For stability, the gradient of the inputs of the loss to the LSTM is clipped, that is before applying the tanh and sigmoid
		//nonlinearities. This is done if there is no norm clipping
		if (!gradient_check && !norm_clipping){
		
			o_t_node.bProp_matrix.leftCols(current_minibatch_size).array() = 
										o_t_node.bProp_matrix.leftCols(current_minibatch_size).array().unaryExpr(gradClipper());
			f_t_node.bProp_matrix.leftCols(current_minibatch_size).array() =
										f_t_node.bProp_matrix.leftCols(current_minibatch_size).array().unaryExpr(gradClipper());
			i_t_node.bProp_matrix.leftCols(current_minibatch_size).array() =
										i_t_node.bProp_matrix.leftCols(current_minibatch_size).array().unaryExpr(gradClipper());		
			tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size).array() =
										tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size).array().unaryExpr(gradClipper());	
			//d_Err_t_to_n_d_x_t.leftCols(current_minibatch_size).array() =
			//							d_Err_t_to_n_d_x_t.leftCols(current_minibatch_size).array().unaryExpr(gradClipper());
		}
		//cerr<<"d_Err_t_to_n_d_x_t "<<d_Err_t_to_n_d_x_t<<endl; 
		//Computing gradients of the paramters
		//Derivative of weights out of h_t
		//cerr<<"W_h_to_o_node"<<endl;
	    W_h_to_o_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size),
											h_t_minus_one.leftCols(current_minibatch_size));
	    //cerr<<"W_h_to_f_node"<<endl;										
	    W_h_to_f_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size),
											h_t_minus_one.leftCols(current_minibatch_size));
		//cerr<<"W_h_to_i_node"<<endl;									
	    W_h_to_i_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size),
											h_t_minus_one.leftCols(current_minibatch_size));		
		//cerr<<"W_h_to_c_node"<<endl;									
   		W_h_to_c_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size),
   						 					h_t_minus_one.leftCols(current_minibatch_size));
											
		//Derivative of weights out of c_t and c_t_minus_one
	    W_c_to_o_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size),
											this->c_t.leftCols(current_minibatch_size));
	    W_c_to_i_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size),
											c_t_minus_one.leftCols(current_minibatch_size));
	    W_c_to_f_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size),
											c_t_minus_one.leftCols(current_minibatch_size));		
 
		//Derivatives of weights out of x_t
		/*
		//cerr<<"input_layer_node.fProp_matrix is "<<input_layer_node.fProp_matrix<<endl;
		//cerr<<"W_x_to_o_node"<<endl;
		W_x_to_o_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size),
											data);
		//cerr<<"W_x_to_i_node"<<endl;									
		W_x_to_i_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size),
											data);
		//cerr<<"W_x_to_f_node"<<endl;									
		W_x_to_f_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size),
											data);	
		//cerr<<"W_x_to_c_node"<<endl;									
		W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size),
											data);	
		*/		
		
		/*
		//Derivatives of the input embeddings. I THINK THIS IS WRONG!							
	    input_layer_node.param->updateGradient(o_t_node.bProp_matrix + 
												f_t_node.bProp_matrix + 
												i_t_node.bProp_matrix + 
												tanh_c_prime_t_node.bProp_matrix,
									            data);
		*/
		// Updating the gradient of the hidden layer biases									
		o_t_node.param->updateGradient(o_t_node.bProp_matrix.leftCols(current_minibatch_size));
		f_t_node.param->updateGradient(f_t_node.bProp_matrix.leftCols(current_minibatch_size));
		i_t_node.param->updateGradient(i_t_node.bProp_matrix.leftCols(current_minibatch_size));
		tanh_c_prime_t_node.param->updateGradient(tanh_c_prime_t_node.bProp_matrix.leftCols(current_minibatch_size));
		
		//updating gradient of input word embeddings input embeddings
		//input_layer_node.param->updateGradient(d_Err_t_to_n_d_x_t.leftCols(current_minibatch_size),
		//										data);		
		input_node->bProp(data,
				o_t_node.bProp_matrix,
				i_t_node.bProp_matrix,
				f_t_node.bProp_matrix,
				tanh_c_prime_t_node.bProp_matrix);							
	
	}
	//This takes the sequence continuation indices, the previous hidden and cell states and creates new ones for this LSTM block
	template <typename DerivedH, typename DerivedC>//, typename DerivedS>
	void copyToHiddenStates(const MatrixBase<DerivedH> &h_t_minus_one,
							const MatrixBase<DerivedC> &c_t_minus_one) {
							//const Eigen::ArrayBase<DerivedS> &sequence_cont_indices){
						//int current_minibatch_size = sequence_cont_indices.cols();	
						int current_minibatch_size = h_t_minus_one.cols();	
						#pragma omp parallel for 
						for (int index=0; index<current_minibatch_size; index++){ 
							//UNCONST(DerivedS,const_sequence_cont_indices,sequence_cont_indices);		
				
							//cerr<<"current minibatch size "<<current_minibatch_size<<endl;
							if (0) { // sequence_cont_indices(index) == 0) {
								this->h_t_minus_one.col(index).setZero(); 			
								this->c_t_minus_one.col(index).setZero();
								//err<<"sequence_cont_indices "<<sequence_cont_indices<<endl;
								//cerr<<"this->h_t_minus_one "<<this->h_t_minus_one<<endl;
								//cerr<<"this->c_t_minus_one "<<this->c_t_minus_one<<endl;
							} else {
								//cerr<<"copying"<<endl;
								this->h_t_minus_one.col(index) = h_t_minus_one.col(index);
								this->c_t_minus_one.col(index) = c_t_minus_one.col(index);
								//this->c_t_minus_one.col(index) = c_t_minus_one.col(index).array().unaryExpr(stateClipper());
							}
						}	
												
						/*
						//UNCONST(DerivedS,const_sequence_cont_indices,sequence_cont_indices);		
						int current_minibatch_size = sequence_cont_indices.cols();
						//cerr<<"current minibatch size "<<current_minibatch_size<<endl;
						this->h_t_minus_one.leftCols(current_minibatch_size).array() = 
							h_t_minus_one.array().leftCols(current_minibatch_size).rowwise()*sequence_cont_indices.template cast<precision_type>();
						this->c_t_minus_one.leftCols(current_minibatch_size).array() = 
							c_t_minus_one.array().leftCols(current_minibatch_size).rowwise()*sequence_cont_indices.template cast<precision_type>();
						//err<<"sequence_cont_indices "<<sequence_cont_indices<<endl;
						//cerr<<"this->h_t_minus_one "<<this->h_t_minus_one<<endl;
						//cerr<<"this->c_t_minus_one "<<this->c_t_minus_one<<endl;
						*/
		
	}

	//This takes the sequence continuation indices, the previous hidden and cell states and creates new ones for this LSTM block
	template <typename DerivedH, typename DerivedC , typename DerivedS>
	static void filterStatesAndErrors(const MatrixBase<DerivedH> &from_h_matrix,
							const MatrixBase<DerivedC> &from_c_matrix,
							const MatrixBase<DerivedH> &const_to_h_matrix,
							const MatrixBase<DerivedC> &const_to_c_matrix,
							const Eigen::ArrayBase<DerivedS> &sequence_cont_indices) {
						int current_minibatch_size = sequence_cont_indices.cols();	
						UNCONST(DerivedC, const_to_c_matrix, to_c_matrix);
						UNCONST(DerivedH, const_to_h_matrix, to_h_matrix);
						//int current_minibatch_size = h_t_minus_one.cols();	
						#pragma omp parallel for 
						for (int index=0; index<current_minibatch_size; index++){ 
							//UNCONST(DerivedS,const_sequence_cont_indices,sequence_cont_indices);		
							//cerr<<"current minibatch size "<<current_minibatch_size<<endl;
							if (sequence_cont_indices(index) == 0) {
								to_h_matrix.col(index).setZero(); 			
								to_c_matrix.col(index).setZero();
								//err<<"sequence_cont_indices "<<sequence_cont_indices<<endl;
								//cerr<<"this->h_t_minus_one "<<this->h_t_minus_one<<endl;
								//cerr<<"this->c_t_minus_one "<<this->c_t_minus_one<<endl;
							} else {
								//cerr<<"copying"<<endl;
								to_h_matrix.col(index) = from_h_matrix.col(index);
								to_c_matrix.col(index) = from_c_matrix.col(index);
								//this->c_t_minus_one.col(index) = c_t_minus_one.col(index).array().unaryExpr(stateClipper());
							}
						}	
												
						/*
						//UNCONST(DerivedS,const_sequence_cont_indices,sequence_cont_indices);		
						int current_minibatch_size = sequence_cont_indices.cols();
						//cerr<<"current minibatch size "<<current_minibatch_size<<endl;
						this->h_t_minus_one.leftCols(current_minibatch_size).array() = 
							h_t_minus_one.array().leftCols(current_minibatch_size).rowwise()*sequence_cont_indices.template cast<precision_type>();
						this->c_t_minus_one.leftCols(current_minibatch_size).array() = 
							c_t_minus_one.array().leftCols(current_minibatch_size).rowwise()*sequence_cont_indices.template cast<precision_type>();
						//err<<"sequence_cont_indices "<<sequence_cont_indices<<endl;
						//cerr<<"this->h_t_minus_one "<<this->h_t_minus_one<<endl;
						//cerr<<"this->c_t_minus_one "<<this->c_t_minus_one<<endl;
						*/
		
	}
		
	//For stability, the gradient of the inputs of the loss to the LSTM is clipped, that is before applying the tanh and sigmoid
	//nonlinearities 
	void clipGradient(){}
	
	void resetGradient(){
		
	}	
	  				  
	
};



class Standard_input_node{
	int minibatch_size;
public:
	//Each LSTM node has a bunch of nodes and temporary data structures
    Node<Input_word_embeddings> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
	
	Standard_input_node():
		minibatch_size(0),
		W_x_to_i_node(),
		W_x_to_f_node(),
		W_x_to_c_node(),
		W_x_to_o_node() {}	
		
	Standard_input_node(standard_input_model &input, int minibatch_size): 
		W_x_to_i_node(&input.W_x_to_i, minibatch_size),
		W_x_to_f_node(&input.W_x_to_f, minibatch_size),
		W_x_to_c_node(&input.W_x_to_c, minibatch_size),
		W_x_to_o_node(&input.W_x_to_o, minibatch_size),
		minibatch_size(minibatch_size) {
			//cerr<<"The input embeddings are"<<*(W_x_to_i_node.param->get_W())<<endl;
		}

	//Resizing all the parameters
	void resize(int minibatch_size){
		//cerr<<"Resizing the input node"<<endl;
		this->minibatch_size = minibatch_size;
		W_x_to_i_node.resize(minibatch_size);
		W_x_to_f_node.resize(minibatch_size);
		W_x_to_c_node.resize(minibatch_size);
		W_x_to_o_node.resize(minibatch_size);
	}
				
	template <typename Derived>
	void fProp(const MatrixBase<Derived> &data){
		//cerr<<"Data is "<<data<<endl;
		W_x_to_c_node.param->fProp(data,W_x_to_c_node.fProp_matrix);
		W_x_to_f_node.param->fProp(data,W_x_to_f_node.fProp_matrix);
		W_x_to_o_node.param->fProp(data,W_x_to_o_node.fProp_matrix);
		W_x_to_i_node.param->fProp(data,W_x_to_i_node.fProp_matrix);			
	}	
	
	template<typename DerivedData, typename DerivedDIn>
	void bProp(const MatrixBase<DerivedData> &data,
				const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix){
		//cerr<<"input_layer_node.fProp_matrix is "<<input_layer_node.fProp_matrix<<endl;
		//cerr<<"W_x_to_o_node"<<endl;
		int current_minibatch_size = data.cols();
		W_x_to_o_node.param->updateGradient(o_t_node_bProp_matrix.leftCols(current_minibatch_size),
											data);
		//cerr<<"W_x_to_i_node"<<endl;									
		W_x_to_i_node.param->updateGradient(i_t_node_bProp_matrix.leftCols(current_minibatch_size),
											data);
		//cerr<<"W_x_to_f_node"<<endl;									
		W_x_to_f_node.param->updateGradient(f_t_node_bProp_matrix.leftCols(current_minibatch_size),
											data);	
		//cerr<<"W_x_to_c_node"<<endl;									
		W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node_bProp_matrix.leftCols(current_minibatch_size),
											data);			
																	
	}
};

class Google_input_node{
	int minibatch_size;
public:
	//Each LSTM node has a bunch of nodes and temporary data structures
    Node<Input_word_embeddings> input_layer_node;
    Node<Linear_layer> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_x_t;
		
	Google_input_node():
		minibatch_size(0),
		input_layer_node(),
		W_x_to_i_node(),
		W_x_to_f_node(),
		W_x_to_c_node(),
		W_x_to_o_node() {}	
		
	Google_input_node(google_input_model &input, int minibatch_size): 
		input_layer_node(&input.input_layer, minibatch_size),
		W_x_to_i_node(&input.W_x_to_i, minibatch_size),
		W_x_to_f_node(&input.W_x_to_f, minibatch_size),
		W_x_to_c_node(&input.W_x_to_c, minibatch_size),
		W_x_to_o_node(&input.W_x_to_o, minibatch_size),
		minibatch_size(minibatch_size) {
			//cerr<<"The input embeddings are"<<*(W_x_to_i_node.param->get_W())<<endl;
		}

	//Resizing all the parameters
	void resize(int minibatch_size){
		//cerr<<"Resizing the input node"<<endl;
		this->minibatch_size = minibatch_size;
		input_layer_node.resize(minibatch_size);
		W_x_to_i_node.resize(minibatch_size);
		W_x_to_f_node.resize(minibatch_size);
		W_x_to_c_node.resize(minibatch_size);
		W_x_to_o_node.resize(minibatch_size);
		d_Err_t_to_n_d_x_t.resize(input_layer_node.param->n_outputs(),minibatch_size);
	}
				
	template <typename Derived>
	void fProp(const MatrixBase<Derived> &data){
		//cerr<<"Data is "<<data<<endl;
		input_layer_node.param->fProp(data, input_layer_node.fProp_matrix);
		W_x_to_c_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_c_node.fProp_matrix);
		W_x_to_f_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_f_node.fProp_matrix);
		W_x_to_o_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_o_node.fProp_matrix);
		W_x_to_i_node.param->fProp(input_layer_node.fProp_matrix,W_x_to_i_node.fProp_matrix);				
			
	}	
	
	template<typename DerivedData, typename DerivedDIn>
	void bProp(const MatrixBase<DerivedData> &data,
				const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix){
		//cerr<<"input_layer_node.fProp_matrix is "<<input_layer_node.fProp_matrix<<endl;
		//cerr<<"W_x_to_o_node"<<endl;
		int current_minibatch_size = data.cols();
		W_x_to_c_node.param->bProp(tanh_c_prime_t_node_bProp_matrix,
								W_x_to_c_node.bProp_matrix);
		W_x_to_o_node.param->bProp(o_t_node_bProp_matrix,
								W_x_to_o_node.bProp_matrix);
		W_x_to_f_node.param->bProp(f_t_node_bProp_matrix,
								W_x_to_f_node.bProp_matrix);
		W_x_to_i_node.param->bProp(i_t_node_bProp_matrix,
								W_x_to_i_node.bProp_matrix);

				
		W_x_to_o_node.param->updateGradient(o_t_node_bProp_matrix.leftCols(current_minibatch_size),
											input_layer_node.fProp_matrix.leftCols(current_minibatch_size));
		//cerr<<"W_x_to_i_node"<<endl;									
		W_x_to_i_node.param->updateGradient(i_t_node_bProp_matrix.leftCols(current_minibatch_size),
											input_layer_node.fProp_matrix.leftCols(current_minibatch_size));
		//cerr<<"W_x_to_f_node"<<endl;									
		W_x_to_f_node.param->updateGradient(f_t_node_bProp_matrix.leftCols(current_minibatch_size),
											input_layer_node.fProp_matrix.leftCols(current_minibatch_size));	
		//cerr<<"W_x_to_c_node"<<endl;									
		W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node_bProp_matrix.leftCols(current_minibatch_size),
											input_layer_node.fProp_matrix.leftCols(current_minibatch_size));		
											
		d_Err_t_to_n_d_x_t = W_x_to_c_node.bProp_matrix + 
							W_x_to_o_node.bProp_matrix +
							W_x_to_f_node.bProp_matrix +
							W_x_to_i_node.bProp_matrix;	
			
		input_layer_node.param->updateGradient(d_Err_t_to_n_d_x_t.leftCols(current_minibatch_size),
									data);																								
	}
};

class Hidden_to_hidden_input_node{
	int minibatch_size;
public:
	//Each LSTM node has a bunch of nodes and temporary data structures
    Node<Linear_layer> W_x_to_i_node, W_x_to_f_node, W_x_to_c_node, W_x_to_o_node;
	Eigen::Matrix<precision_type,Eigen::Dynamic,Eigen::Dynamic> d_Err_t_to_n_d_x_t;
		
	Hidden_to_hidden_input_node():
		minibatch_size(0),
		W_x_to_i_node(),
		W_x_to_f_node(),
		W_x_to_c_node(),
		W_x_to_o_node() {}	
		
	Hidden_to_hidden_input_node(hidden_to_hidden_input_model &input, int minibatch_size): 
		W_x_to_i_node(&input.W_x_to_i, minibatch_size),
		W_x_to_f_node(&input.W_x_to_f, minibatch_size),
		W_x_to_c_node(&input.W_x_to_c, minibatch_size),
		W_x_to_o_node(&input.W_x_to_o, minibatch_size),
		minibatch_size(minibatch_size) {
			//cerr<<"The input embeddings are"<<*(W_x_to_i_node.param->get_W())<<endl;
		}

	//Resizing all the parameters
	void resize(int minibatch_size){
		//cerr<<"Resizing the input node"<<endl;
		this->minibatch_size = minibatch_size;
		W_x_to_i_node.resize(minibatch_size);
		W_x_to_f_node.resize(minibatch_size);
		W_x_to_c_node.resize(minibatch_size);
		W_x_to_o_node.resize(minibatch_size);
		d_Err_t_to_n_d_x_t.resize(W_x_to_o_node.param->n_inputs(),minibatch_size);
	}
				
	template <typename Derived>
	void fProp(const MatrixBase<Derived> &data){
		//cerr<<"Data is "<<data<<endl;
		W_x_to_c_node.param->fProp(data,W_x_to_c_node.fProp_matrix);
		W_x_to_f_node.param->fProp(data,W_x_to_f_node.fProp_matrix);
		W_x_to_o_node.param->fProp(data,W_x_to_o_node.fProp_matrix);
		W_x_to_i_node.param->fProp(data,W_x_to_i_node.fProp_matrix);				
			
	}	
	
	template<typename DerivedData, typename DerivedDIn>
	void bProp(const MatrixBase<DerivedData> &data,
				const MatrixBase<DerivedDIn> &o_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &i_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &f_t_node_bProp_matrix,
				const MatrixBase<DerivedDIn> &tanh_c_prime_t_node_bProp_matrix){
		//cerr<<"input_layer_node.fProp_matrix is "<<input_layer_node.fProp_matrix<<endl;
		//cerr<<"W_x_to_o_node"<<endl;
		int current_minibatch_size = o_t_node_bProp_matrix.cols();
		W_x_to_o_node.param->updateGradient(o_t_node_bProp_matrix,
											data);
		//cerr<<"W_x_to_i_node"<<endl;									
		W_x_to_i_node.param->updateGradient(i_t_node_bProp_matrix,
											data);
		//cerr<<"W_x_to_f_node"<<endl;									
		W_x_to_f_node.param->updateGradient(f_t_node_bProp_matrix,
											data);	
		//cerr<<"W_x_to_c_node"<<endl;									
		W_x_to_c_node.param->updateGradient(tanh_c_prime_t_node_bProp_matrix,
											data);		
											
		d_Err_t_to_n_d_x_t.leftCols(current_minibatch_size) = W_x_to_c_node.bProp_matrix + 
							W_x_to_o_node.bProp_matrix +
							W_x_to_f_node.bProp_matrix +
							W_x_to_i_node.bProp_matrix;	
			
																						
	}
};

} // namespace nplm
