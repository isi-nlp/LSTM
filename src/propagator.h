#ifndef NETWORK_H
#define NETWORK_H

#include "neuralClasses.h"
#include "util.h"
#include "graphClasses.h"

namespace nplm
{
	class propagator {
	public:
	    int minibatch_size;
	    model *plstm;
		vector<LSTM_node> lstm_nodes; //We will allow only 20 positions now. 
		Node<Output_word_embeddings> output_layer_node;
		Matrix<double,Dynamic,Dynamic> d_Err_tPlusOne_to_n_d_c_t,d_Err_tPlusOne_to_n_d_h_t; //Derivatives wrt the future h_t and c_t
		Matrix<double,Dynamic,Dynamic> scores;
		Matrix<double,Dynamic,Dynamic> minibatch_weights;
		Matrix<double,Dynamic,Dynamic> d_Err_t_d_output;
		Matrix<int,Dynamic,Dynamic> minibatch_samples;
		Matrix<double,Dynamic,Dynamic> probs;		

	public:
	    propagator() : minibatch_size(0), plstm(0), lstm_nodes(20,LSTM_node()) { }

	    propagator (model &lstm, int minibatch_size)
	      : plstm(&lstm),
		 	minibatch_size(minibatch_size),
			output_layer_node(&lstm.output_layer,minibatch_size),
			lstm_nodes(vector<LSTM_node>(20,LSTM_node(lstm,minibatch_size))) {}
		    // This must be called if the underlying model is resized.
	    void resize(int minibatch_size) {
	      this->minibatch_size = minibatch_size;
		  output_layer_node.resize(minibatch_size);
		  //Resizing all the lstm nodes
		  for (int i=0; i<lstm_nodes.size(); i++){
			  lstm_nodes[i].resize(minibatch_size);
		  }
		  //I HAVE TO INITIALIZE THE MATRICES
		  d_Err_tPlusOne_to_n_d_c_t.setZero(2,minibatch_size);
		  d_Err_tPlusOne_to_n_d_h_t.setZero(2,minibatch_size);
		  scores.resize(output_layer_node.param->n_outputs(),minibatch_size); 
  		  minibatch_weights.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  minibatch_samples.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  probs.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  d_Err_t_d_output.resize(output_layer_node.param->n_outputs(),minibatch_size);
	    }

	    void resize() { resize(minibatch_size); }
		
		//Both the input and the output sentences are columns. Even ifs a minibatch of sentences, each sentence is a column
	    template <typename Derived>
	    void fProp(const MatrixBase<Derived> &data)
	    {
			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = data.rows();
			Matrix<double,Dynamic,1> c_0,h_0;
			c_0.setZero(2);
			h_0.setZero(2);
			
			for (int i=0; i<sent_len; i++){
				if (i==0) {
					lstm_nodes[i].fProp(data.row(0),	
										c_0,
										h_0);
				} else {
					lstm_nodes[i].fProp(data.row(i),
										lstm_nodes[i-1].c_t,
										lstm_nodes[i-1].h_t);
				}
				//lstm_nodes.fProp();
			}
	    }

	    // Dense version (for standard log-likelihood)
	    template <typename DerivedIn, typename DerivedOut>
	    void bProp(const MatrixBase<DerivedIn> &data,
			 const MatrixBase<DerivedOut> &output) 
	    {	

			int current_minibatch_size = output.cols();
			
			Matrix<double,Dynamic,Dynamic> dummy_zero;
			dummy_zero.setZero(2,current_minibatch_size);
			
			int sent_len = data.rows(); 
			double log_likelihood = 0.;
			
			for (int i=sent_len-1; i>=0; i--) {
				//First doing fProp for the output layer
				output_layer_node.param->fProp(lstm_nodes[i].h_t, scores);
				
				//then compute the log loss of the objective
		        double minibatch_log_likelihood;
		        start_timer(5);
		        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
		                   output.row(i), 
		                   probs, 
		                   minibatch_log_likelihood);
		        stop_timer(5);
		        log_likelihood += minibatch_log_likelihood;

		        ///// Backward propagation
        
		        start_timer(6);
		        //SoftmaxLogLoss().bProp(output.row(i), 
		        //           probs.leftCols(current_minibatch_size), 
		        //           minibatch_weights);
   		        SoftmaxLogLoss().bProp(output.row(i), 
   		                   probs.leftCols(current_minibatch_size), 
   		                   d_Err_t_d_output);
		        stop_timer(6);
				
				//Now computing the derivative of the output layer
				
		        output_layer_node.param->bProp(d_Err_t_d_output.leftCols(current_minibatch_size),
						       output_layer_node.bProp_matrix);
				// Now calling backprop for the LSTM nodes
				if (i == sent_len) {
				    lstm_nodes[i].bProp(data.row(i),
				   			   lstm_nodes[i-1].c_t,
				   			   output_layer_node.bProp_matrix,
				   			   dummy_zero.leftCols(current_minibatch_size), //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
				   			   dummy_zero.leftCols(current_minibatch_size));						
				} else {
				    lstm_nodes[i].bProp(data.row(i),
				   			   lstm_nodes[i-1].c_t,
				   			   output_layer_node.bProp_matrix,
				   			   lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne);								
				}		   
		   
			}
	        // Output embedding layer
		    

	        //start_timer(7);

		 	//stop_timer(7);
			/*
		    start_timer(8);
		  if (parameter_update == "SGD") {
		    output_layer_node.param->computeGradient(second_hidden_activation_node.fProp_matrix,
		               output,
		               learning_rate,
		               momentum);
		  } else if (parameter_update == "ADA") {
		    output_layer_node.param->computeGradientAdagrad(second_hidden_activation_node.fProp_matrix,
		               output,
		               learning_rate);
		  } else if (parameter_update == "ADAD") {
		    //std::cerr<<"Adadelta gradient"<<endl;
		    int current_minibatch_size = second_hidden_activation_node.fProp_matrix.cols();
		    output_layer_node.param->computeGradientAdadelta(second_hidden_activation_node.fProp_matrix,
		               output,
		               1.0/current_minibatch_size,
		               conditioning_constant,
		               decay);
		  } else {
		    std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
		  }
			stop_timer(8);
			
			bPropRest(data, 
		      learning_rate,
		      momentum,
		      L2_reg,
		      parameter_update,
		      conditioning_constant,
		      decay);
			*/
	  }
 };		
} // namespace nplm

#endif
	