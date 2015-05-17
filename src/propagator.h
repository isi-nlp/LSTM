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
		int num_hidden;

	public:
	    propagator() : minibatch_size(0), plstm(0), lstm_nodes(100,LSTM_node()),num_hidden(0) { }

	    propagator (model &lstm, int minibatch_size)
	      : plstm(&lstm),
		 	minibatch_size(minibatch_size),
			output_layer_node(&lstm.output_layer,minibatch_size),
			lstm_nodes(vector<LSTM_node>(100,LSTM_node(lstm,minibatch_size))) {
				resize(minibatch_size);
			}
		    // This must be called if the underlying model is resized.
	    void resize(int minibatch_size) {
	      this->minibatch_size = minibatch_size;
		  output_layer_node.resize(minibatch_size);
		  //Resizing all the lstm nodes
		  for (int i=0; i<lstm_nodes.size(); i++){
			  lstm_nodes[i].resize(minibatch_size);
		  }
		  //cerr<<"minibatch size is propagator is "<<minibatch_size<<endl;
		  //I HAVE TO INITIALIZE THE MATRICES
		  d_Err_tPlusOne_to_n_d_c_t.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  d_Err_tPlusOne_to_n_d_h_t.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  scores.resize(output_layer_node.param->n_outputs(),minibatch_size); 
  		  minibatch_weights.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  minibatch_samples.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  probs.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  d_Err_t_d_output.resize(output_layer_node.param->n_outputs(),minibatch_size);
	    }

	    void resize() { resize(minibatch_size); }
		
		//Both the input and the output sentences are columns. Even ifs a minibatch of sentences, each sentence is a column
	    template <typename Derived, typename DerivedH, typename DerivedC>
	    void fProp(const MatrixBase<Derived> &data,
				const int start_pos,
				const int end_pos,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h)
	    {
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);
			//cerr<<"current_c is "<<current_c<<endl;
			//cerr<<"current_h is "<<current_h<<endl;
			/*
			cerr<<"Data is "<<data<<endl;
			cerr<<"Start pos "<<start_pos<<endl;
			cerr<<"End pos "<<end_pos<<endl;
			cerr<<"In Fprop"<<endl;
			*/
			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = data.rows();
			//Matrix<double,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = data.cols();
			//cerr<<"current minibatch_size is "<<current_minibatch_size<<endl;
			//c_0.setZero(output_layer_node.param->n_inputs(), minibatch_size);
			//h_0.setZero(output_layer_node.param->n_inputs(), minibatch_size);			
			//c_1.setOnes(output_layer_node.param->n_inputs(), minibatch_size);
			//h_1.setOnes(output_layer_node.param->n_inputs(), minibatch_size);
			//cerr<<"c0 is "<<c_0<<endl;
			//cerr<<"h0 is "<<h_0<<endl;
			//getchar();
			for (int i=0; i<=end_pos; i++){
				//cerr<<"i is"<<i<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					lstm_nodes[i].fProp(data.row(i),	
										current_c,
										current_h);
				} else {
					//cerr<<"Data is "<<data.row(i)<<endl;
					//cerr<<"index is "<<i<<endl;
					
					lstm_nodes[i].fProp(data.row(i),
										lstm_nodes[i-1].c_t,
										lstm_nodes[i-1].h_t);
					/*					
					lstm_nodes[i].fProp(data.row(i),
										c_1,
										lstm_nodes[i-1].h_t);	
					*/
				}
				//lstm_nodes.fProp();
			}
			current_c = lstm_nodes[end_pos].c_t;
			current_h = lstm_nodes[end_pos].h_t;
	    }

	    // Dense version (for standard log-likelihood)
	    template <typename DerivedIn, typename DerivedOut, typename DerivedC, typename DerivedH>
	    void bProp(const MatrixBase<DerivedIn> &data,
			 const MatrixBase<DerivedOut> &output,
			 double &log_likelihood,
			 bool gradient_check,
			 bool norm_clipping,
			 const MatrixBase<DerivedC> &init_c,
			 const MatrixBase<DerivedH> &init_h) 
	    {	
			
			//cerr<<"In backprop..."<<endl;
			int current_minibatch_size = output.cols();
			//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
			Matrix<double,Dynamic,Dynamic> dummy_zero,dummy_ones;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
			int sent_len = output.rows(); 
			//double log_likelihood = 0.;
			
			for (int i=sent_len-1; i>=0; i--) {
				//cerr<<"i is "<<i<<endl;
				//First doing fProp for the output layer
				
				//The number of columns in scores will be the current minibatch size
				output_layer_node.param->fProp(lstm_nodes[i].h_t.leftCols(current_minibatch_size), scores);
				//cerr<<"scores.rows "<<scores.rows()<<" scores cols "<<scores.cols()<<endl;
				//then compute the log loss of the objective
				//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
				//cerr<<"Score is"<<endl;
				//cerr<<scores<<endl;
				
		        double minibatch_log_likelihood;
		        start_timer(5);
		        SoftmaxLogLoss().fProp(scores, 
		                   output.row(i), 
		                   probs, 
		                   minibatch_log_likelihood);
				//cerr<<"probs is "<<probs<<endl;
				//cerr<< " minibatch log likelihood is "<<minibatch_log_likelihood<<endl;	
		        stop_timer(5);
		        log_likelihood += minibatch_log_likelihood;
				//getchar();
		        ///// Backward propagation
        
		        start_timer(6);
		        //SoftmaxLogLoss().bProp(output.row(i), 
		        //           probs.leftCols(current_minibatch_size), 
		        //           minibatch_weights);
   		        SoftmaxLogLoss().bProp(output.row(i), 
   		                   probs.leftCols(current_minibatch_size), 
   		                   d_Err_t_d_output);
				//cerr<<"d_Err_t_d_output is "<<d_Err_t_d_output<<endl;
		        stop_timer(6);
				

				//Oh wow, i have not even been updating the gradient of the output embeddings
				//Now computing the derivative of the output layer
				//The number of colums in output_layer_node.bProp_matrix will be the current minibatch size
   		        output_layer_node.param->bProp(d_Err_t_d_output.leftCols(current_minibatch_size),
   						       output_layer_node.bProp_matrix);	
				//cerr<<"ouput layer bprop matrix rows"<<output_layer_node.bProp_matrix.rows()<<" cols"<<output_layer_node.bProp_matrix.cols()<<endl;
				//cerr<<"output_layer_node.bProp_matrix"<<output_layer_node.bProp_matrix<<endl;
				//cerr<<"Dimensions if d_Err_t_d_output "<<d_Err_t_d_output.rows()<<","<<d_Err_t_d_output.cols()<<endl;
				//cerr<<"output_layer_node.bProp_matrix "<<output_layer_node.bProp_matrix<<endl;
   		        output_layer_node.param->updateGradient(lstm_nodes[i].h_t.leftCols(current_minibatch_size),
   						       d_Err_t_d_output.leftCols(current_minibatch_size));									   	 		   
				//cerr<<" i is "<<i<<endl;
				//cerr<<"backprop matrix is "<<output_layer_node.bProp_matrix<<endl;
				//getchar();
				// Now calling backprop for the LSTM nodes
				if (i==0) {
					
				    lstm_nodes[i].bProp(data.row(i),
							   init_h,
				   			   init_c,
				   			   output_layer_node.bProp_matrix,
				   			   lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
							   gradient_check,
							   norm_clipping);	
					
					/*
   				    lstm_nodes[i].bProp(data.row(i),
   							   dummy_zero.leftCols(current_minibatch_size),
   				   			   dummy_zero.leftCols(current_minibatch_size),
   				   			   output_layer_node.bProp_matrix,
   				   			   dummy_zero.leftCols(current_minibatch_size),
   							   lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne);		
					*/
				} else if (i == sent_len-1) {	
					/*	
					const MatrixBase<DerivedData> &data,
								   //const MatrixBase<DerivedIn> c_t,
								   const MatrixBase<DerivedIn> c_t_minus_one,
								   const MatrixBase<DerivedIn> d_Err_t_d_h_t,
								   const MatrixBase<DerivedDIn> d_Err_tPlusOne_to_n_d_c_t,
								   const MatrixBase<DerivedDIn> d_Err_tPlusOne_to_n_d_h_t
					*/
					//cerr<<"previous ct is "<<lstm_nodes[i-1].c_t<<endl;
					
				    lstm_nodes[i].bProp(data.row(i),
							   lstm_nodes[i-1].h_t,
				   			   lstm_nodes[i-1].c_t,
				   			   output_layer_node.bProp_matrix,
				   			   dummy_zero, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
				   			   dummy_zero,
							   gradient_check,
							   norm_clipping);
					/*   
  				    lstm_nodes[i].bProp(data.row(i),
  							   lstm_nodes[i-1].h_t,
  				   			   dummy_ones.leftCols(current_minibatch_size),
  				   			   output_layer_node.bProp_matrix,
  				   			   dummy_zero.leftCols(current_minibatch_size),
  							   dummy_zero.leftCols(current_minibatch_size));	
					*/						
				} else if (i > 0) {
					
				    lstm_nodes[i].bProp(data.row(i),
							   lstm_nodes[i-1].h_t,
				   			   lstm_nodes[i-1].c_t,
				   			   output_layer_node.bProp_matrix,
				   			   lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
							   gradient_check,
							   norm_clipping);		
					/*
  				    lstm_nodes[i].bProp(data.row(i),
  							   lstm_nodes[i-1].h_t,
  				   			   dummy_ones.leftCols(current_minibatch_size),
  				   			   output_layer_node.bProp_matrix,
  				   			   dummy_zero.leftCols(current_minibatch_size),
  							   lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne);	
					*/							   						
				} 		   
		   
			}
			//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
			//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
			//cerr<<"The cross entropy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
			//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;

	  }
	  
	 void updateParams(double learning_rate,
	 					int current_minibatch_size,
				  		double momentum,
						double L2_reg,
						bool norm_clipping,
						double norm_threshold) {
		//cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
		//cerr<<"updating params "<<endl;
		plstm->output_layer.updateParams(learning_rate,
						current_minibatch_size,
	  					momentum,
	  					L2_reg,
						norm_clipping,
						norm_threshold);
		// updating the rest of the parameters
		
		//updating params for weights out of hidden layer 
		//cerr<<"updating params"<<endl;
		plstm->W_h_to_o.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
 		plstm->W_h_to_f.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
  		plstm->W_h_to_i.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
   		plstm->W_h_to_c.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);

		//updating params for weights out of cell
		plstm->W_c_to_f.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->W_c_to_i.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->W_c_to_o.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);				


		//Error derivatives for the input word embeddings
		plstm->W_x_to_c.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->W_x_to_o.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->W_x_to_f.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->W_x_to_i.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);


		plstm->o_t.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->f_t.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		plstm->i_t.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);	
		plstm->tanh_c_prime_t.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);	
		/*						
		//Derivatives of the input embeddings							
	    plstm->input_layer.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);		
		*/
	  }
	  
	  template <typename DerivedOut>
	  void computeProbs(const MatrixBase<DerivedOut> &output,
	  					double &log_likelihood) 
	  {	
			
			//cerr<<"In computeProbs..."<<endl;
			int current_minibatch_size = output.cols();

			Matrix<double,Dynamic,Dynamic> dummy_zero;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

			int sent_len = output.rows(); 
			//double log_likelihood = 0.;

			for (int i=sent_len-1; i>=0; i--) {
				//cerr<<"i is "<<i<<endl;
				//First doing fProp for the output layer
				output_layer_node.param->fProp(lstm_nodes[i].h_t.leftCols(current_minibatch_size), scores);
				//then compute the log loss of the objective
				//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
				//cerr<<"Score is"<<endl;
				//cerr<<scores<<endl;
	
		        double minibatch_log_likelihood;
		        start_timer(5);
		        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
		                   output.row(i), 
		                   probs, 
		                   minibatch_log_likelihood);
				//cerr<<"probs is "<<probs<<endl;
		        stop_timer(5);
		        log_likelihood += minibatch_log_likelihood;		
			}
			//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
			//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
			//cerr<<"The cross entopy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
			//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
			//log_likelihood /= sent_len;
	  }	  
	  
  	void resetGradient(){
		plstm->output_layer.resetGradient();
		// updating the rest of the parameters
		
		//updating params for weights out of hidden layer 
		plstm->W_h_to_o.resetGradient();
 		plstm->W_h_to_f.resetGradient();
  		plstm->W_h_to_i.resetGradient();
   		plstm->W_h_to_c.resetGradient();

		//updating params for weights out of cell
		plstm->W_c_to_f.resetGradient();
		plstm->W_c_to_i.resetGradient();
		plstm->W_c_to_o.resetGradient();				

		
		/*
		//Error derivatives for the input word embeddings
		plstm->W_x_to_c.resetGradient();
		plstm->W_x_to_o.resetGradient();
		plstm->W_x_to_f.resetGradient();
		plstm->W_x_to_i.resetGradient();
		*/

		//Computing gradients of the paramters

		plstm->o_t.resetGradient();
		plstm->f_t.resetGradient();
		plstm->i_t.resetGradient();	
		plstm->tanh_c_prime_t.resetGradient();	
								
		//The gradients of the input layer are being reset in update params sinc the gradient is sparse
		//Derivatives of the input embeddings							
	    //plstm->input_layer.resetGradient();		
  	}	
	
	//Use finite differences to do gradient check
	template <typename DerivedIn, typename DerivedOut, typename DerivedC, typename DerivedH>
    void gradientCheck(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 const MatrixBase<DerivedC> &const_init_c,
			 const MatrixBase<DerivedH> &const_init_h)
    {
		Matrix<double,Dynamic,Dynamic> init_c = const_init_c;
		Matrix<double,Dynamic,Dynamic> init_h = const_init_h;
		//cerr<<"init c is "<<init_c<<endl;
		//cerr<<"init h is "<<init_h<<endl;
		//cerr<<"in gradient check. The size of input is "<<input.rows()<<endl;
		//cerr<<"In gradient check"<<endl;
		/*
		//Checking the gradient of h_t
		lstm_nodes[0].h_t(0,0) += 1e-5;
		fProp(input, 1, input.rows()-1);
		//fProp(input, 1, input.rows()-1);
		
 		double before_log_likelihood = 0;						
 		fProp(input,0, input.rows()-1);	
 		computeProbs(output,
 			  		before_log_likelihood);			

		lstm_nodes[0].h_t(0,0) -= 2e-5;
		fProp(input, 1, input.rows()-1);
		//fProp(input, 1, input.rows()-1);

 		double after_log_likelihood = 0;						
 		fProp(input,0, input.rows()-1);	
 		computeProbs(output,
 			  		after_log_likelihood);
		
		cerr<<"the measured gradient is"<<lstm_nodes[0].d_Err_t_to_n_d_h_t<<endl;
		cerr<<"Gradient diff is "<<	(before_log_likelihood-after_log_likelihood)/2e-5<<endl;
		*/
		//Check every dimension of all the parameters to make sure the gradient is fine
		

		paramGradientCheck(input,output,plstm->output_layer,"output_layer", init_c, init_h);	
		
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_h_to_c,"W_h_to_c",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,plstm->W_h_to_f,"W_h_to_f",init_c, init_h);	
		init_c = const_init_c;
		init_h = const_init_h;										
		paramGradientCheck(input,output,plstm->W_h_to_o,"W_h_to_o",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_h_to_i ,"W_h_to_i",init_c, init_h);	
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_x_to_c,"W_x_to_c",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_x_to_f,"W_x_to_f",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_x_to_o,"W_x_to_o",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_x_to_i,"W_x_to_i",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,plstm->W_c_to_o,"W_c_to_o",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_c_to_f,"W_c_to_f",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->W_c_to_i,"W_c_to_i",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,plstm->o_t,"o_t",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->f_t,"f_t",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->i_t,"i_t",init_c, init_h);
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,plstm->tanh_c_prime_t,"tanh_c_prime_t",init_c, init_h);		
		
		//paramGradientCheck(input,output,plstm->input_layer,"input_layer");
		
		
	}
	template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH>
	void paramGradientCheck(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 testParam &param,
			 const string param_name,
			 const MatrixBase<DerivedC> &init_c,
			 const MatrixBase<DerivedH> &init_h){
		//Going over all dimensions of the parameter
		for(int row=0; row<param.rows(); row++){
			for (int col=0; col<param.cols(); col++){
				getFiniteDiff(input, 
							output, 
							param, 
							param_name, 
							row, 
							col, 
							init_c, 
							init_h);
			}
		}
	}
	
	template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH>
    void getFiniteDiff(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 testParam &param,
			 const string param_name,
			 int row,
			 int col,
			 const MatrixBase<DerivedC> &const_init_c,
			 const MatrixBase<DerivedH> &const_init_h) {
				Matrix<double,Dynamic,Dynamic> init_c; 
				Matrix<double,Dynamic,Dynamic> init_h;
				init_c = const_init_c;
				init_h = const_init_h;
				//cerr<<"Row is :"<<row<<" col is " <<col<<endl;
				int rand_row = row;
		 		int rand_col = col;
		 		//First checking the gradient of the output word embeddings
		 		//cerr<<"Checking the gradient of "<<param_name<<endl;
		 		//rand_row = 0;
				//rand_col= 0;
		 	    param.changeRandomParam(1e-5, 
		 								rand_row,
		 								rand_col);
		 		//then do an fprop
		 		double before_log_likelihood = 0;	
				//cerr<<"input cols is "<<input.cols()<<endl;					
		 		fProp(input, 0, input.rows()-1, init_c, init_h);
		 		computeProbs(output,
		 			  		before_log_likelihood);
		 		//err<<"before log likelihood is "<<
		 	    param.changeRandomParam(-2e-5, 
		 								rand_row,
		 								rand_col);		
				init_c = const_init_c;
				init_h = const_init_h;
		 		double after_log_likelihood = 0;						
		 		fProp(input,0, input.rows()-1, init_c, init_h);	
		 		computeProbs(output,
		 			  		after_log_likelihood);		
		 		//returning the parameter back to its own value
		 	    param.changeRandomParam(1e-5 , 
		 								rand_row,
		 								rand_col);			

				
				//cerr<<"graves "<<pow(10.0, max(0.0, ceil(log10(min(fabs(param.getGradient(rand_row,
		 		//						rand_col)), fabs((before_log_likelihood-after_log_likelihood)/2e-5)))))-6)<<endl;
				double symmetric_finite_diff_grad = (before_log_likelihood-after_log_likelihood)/2e-5;	
				double graves_threshold = pow(10.0, max(0.0, ceil(log10(min(fabs(param.getGradient(rand_row,
		 								rand_col)), fabs(symmetric_finite_diff_grad)))))-6);
				double gradient_diff =  symmetric_finite_diff_grad - param.getGradient(rand_row,
		 								rand_col);
				if (gradient_diff > graves_threshold) {
					cerr<<"!!!GRADIENT CHECKING FAILED!!!"<<endl;
			 		cerr<<"Symmetric finite differences gradient is "<<	symmetric_finite_diff_grad<<endl;
					cerr<<"Algorithmic gradient is "<<param.getGradient(rand_row,rand_col)<<endl;					
		 	    	cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" at row: "<<rand_row
						<<" and col: "<<rand_col<<" is "<<gradient_diff<<endl;	
					exit(1);
				} else {
		 	    	cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" at row: "<<rand_row
						<<" and col: "<<rand_col<<" is "<<gradient_diff<<endl;
				}
		 	
	}	
	
	
	
	template <typename DerivedIn, typename DerivedOut, typename testParam>
    void getFiniteDiff(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 const MatrixBase<testParam> & const_test_param,
			 const string param_name) {
				 
				 UNCONST(testParam,const_test_param,test_param);
		 		int rand_row;
		 		int rand_col;
		 		//First checking the gradient of the output word embeddings
		 		cerr<<"Checking the gradient of "<<param_name<<endl;
		 		rand_row = 0;
				rand_col= 0;
		 	    test_param(rand_row,rand_col) += 1e-5;
		 		//then do an fprop
		 		double before_log_likelihood = 0;	
				cerr<<"input cols is "<<input.cols()<<endl;					
		 		fProp(input, 0, input.rows()-1);
		 		computeProbs(output,
		 			  		before_log_likelihood);
		 		//err<<"before log likelihood is "<<
				/*
		 	    param.changeRandomParam(-2e-5, 
		 								rand_row,
		 								rand_col);		
		
		 		double after_log_likelihood = 0;						
		 		fProp(input,0, input.rows()-1);	
		 		computeProbs(output,
		 			  		after_log_likelihood);		
		
		 		cerr<<"Gradient diff is "<<	(before_log_likelihood-after_log_likelihood)/2e-5<<endl;
				
				cerr<<"graves "<<pow(10.0, max(0.0, ceil(log10(min(fabs(param.getGradient(rand_row,
		 								rand_col)), fabs((before_log_likelihood-after_log_likelihood)/2e-5)))))-6)<<endl;
		 	    cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" is "<<
						(before_log_likelihood-after_log_likelihood)/2e-5 - param.getGradient(rand_row,
		 								rand_col)<<endl;	
		 		//returning the parameter back to its own value
		 	    param.changeRandomParam(1e-5 , 
		 								rand_row,
		 								rand_col);			 
				*/	
	}	
 };		
 
 
} // namespace nplm

#endif
	