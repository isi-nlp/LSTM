#ifndef NETWORK_H
#define NETWORK_H

#include "neuralClasses.h"
#include "util.h"
#include "graphClasses.h"

namespace nplm
{
	template <class input_node_type, class input_model_type>
	class propagator {
	public:
	    int minibatch_size;


	    model *encoder_plstm, *decoder_plstm;
		vector<LSTM_node<input_node_type> > encoder_lstm_nodes; //We will allow only 20 positions now. 
		vector<LSTM_node<input_node_type> > decoder_lstm_nodes; //We will allow only 20 positions now.
		vector<input_node_type > encoder_input_nodes, decoder_input_nodes; 
		Node<Output_word_embeddings> output_layer_node;
		Matrix<double,Dynamic,Dynamic> d_Err_tPlusOne_to_n_d_c_t,d_Err_tPlusOne_to_n_d_h_t; //Derivatives wrt the future h_t and c_t
		Matrix<double,Dynamic,Dynamic> scores;
		Matrix<double,Dynamic,Dynamic> minibatch_weights;
		Matrix<double,Dynamic,Dynamic> d_Err_t_d_output;
		Matrix<int,Dynamic,Dynamic> minibatch_samples;
		Matrix<int,Dynamic,Dynamic> minibatch_samples_no_negative;
		Matrix<double,Dynamic,Dynamic> probs;	
		int num_hidden;
		double fixed_partition_function; 
		//vector<Matrix<double,Dynamic,Dynamic> > losses;
		vector<Output_loss_node> losses;

	public:
	    propagator() : minibatch_size(0), 
					encoder_plstm(0), 
					decoder_plstm(0),
					encoder_lstm_nodes(100,LSTM_node<input_node_type>()),
					decoder_lstm_nodes(100,LSTM_node<input_node_type>()),
					encoder_input_nodes(100,input_node_type()),
					decoder_input_nodes(100,input_node_type()),
					num_hidden(0), 
					fixed_partition_function(0), 
					losses(vector<Output_loss_node>(100,Output_loss_node())){ }

	    propagator (model &encoder_lstm, 
					model &decoder_lstm,
					int minibatch_size)
	      : encoder_plstm(&encoder_lstm),
			decoder_plstm(&decoder_lstm),
		 	minibatch_size(minibatch_size),
			output_layer_node(&decoder_lstm.output_layer,minibatch_size),
			encoder_lstm_nodes(vector<LSTM_node<input_node_type> >(100,LSTM_node<input_node_type>(encoder_lstm,minibatch_size))),
			decoder_lstm_nodes(vector<LSTM_node<input_node_type> >(100,LSTM_node<input_node_type>(decoder_lstm,minibatch_size))),
			encoder_input_nodes(vector<input_node_type >(100,input_node_type (dynamic_cast<input_model_type&>(*(encoder_lstm.input)),minibatch_size))),
			decoder_input_nodes(vector<input_node_type >(100,input_node_type (dynamic_cast<input_model_type&>(*(decoder_lstm.input)),minibatch_size))),
			//losses(vector<Matrix<double,Dynamic,Dynamic> >(100,Matrix<double,Dynamic,Dynamic>()))
			losses(vector<Output_loss_node>(100,Output_loss_node()))
			{
				resize(minibatch_size);
			}
		    // This must be called if the underlying model is resized.
	    void resize(int minibatch_size) {
	      this->minibatch_size = minibatch_size;
		  output_layer_node.resize(minibatch_size);
		  //Resizing all the lstm nodes
		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
			  encoder_lstm_nodes[i].resize(minibatch_size);
			  decoder_lstm_nodes[i].resize(minibatch_size);
			  encoder_input_nodes[i].resize(minibatch_size);
			  decoder_input_nodes[i].resize(minibatch_size);
			  encoder_lstm_nodes[i].set_input_node(encoder_input_nodes[i]);
			  decoder_lstm_nodes[i].set_input_node(decoder_input_nodes[i]);
			  losses[i].resize(output_layer_node.param->n_inputs(),minibatch_size);
			  //losses[i].setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  }
		  //cerr<<"minibatch size is propagator is "<<minibatch_size<<endl;
		  //I HAVE TO INITIALIZE THE MATRICES 
		  
		  //CURRENTLY, THE RESIZING IS WRONG FOR SOME OF THE MINIBATCHES
		  d_Err_tPlusOne_to_n_d_c_t.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  d_Err_tPlusOne_to_n_d_h_t.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  scores.resize(output_layer_node.param->n_outputs(),minibatch_size); 
  		  minibatch_weights.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  minibatch_samples.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  minibatch_samples_no_negative.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  probs.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  d_Err_t_d_output.resize(output_layer_node.param->n_outputs(),minibatch_size);
	    }
		//Resizing some of the NCE mibatch matrices
		void resizeNCE(int num_noise_samples, double fixed_partition_function){
			minibatch_weights.setZero(num_noise_samples+1,minibatch_size);
			minibatch_samples.setZero(num_noise_samples+1,minibatch_size);
			minibatch_samples_no_negative.setZero(num_noise_samples+1,minibatch_size);
			scores.setZero(num_noise_samples+1,minibatch_size);
			probs.setZero(num_noise_samples+1,minibatch_size);
			cerr<<"Size of scores is "<<scores.cols()<<" "<<scores.rows()<<endl;
			this->fixed_partition_function = fixed_partition_function;
		}
	    void resize() { resize(minibatch_size); }
		
		//Both the input and the output sentences are columns. Even ifs a minibatch of sentences, each sentence is a column
	    template <typename DerivedInput, typename DerivedOutput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fProp(const MatrixBase<DerivedInput> &input_data,
					const MatrixBase<DerivedOutput> &output_data,
				const int start_pos,
				const int end_pos,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
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
			int sent_len = input_data.rows();
			int output_sent_len = output_data.rows();
			//Matrix<double,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = input_data.cols();
			//cerr<<"current minibatch_size is "<<current_minibatch_size<<endl;
			//c_0.setZero(output_layer_node.param->n_inputs(), minibatch_size);
			//h_0.setZero(output_layer_node.param->n_inputs(), minibatch_size);			
			//c_1.setOnes(output_layer_node.param->n_inputs(), minibatch_size);
			//h_1.setOnes(output_layer_node.param->n_inputs(), minibatch_size);
			//cerr<<"c0 is "<<c_0<<endl;
			//cerr<<"h0 is "<<h_0<<endl;
			//getchar();
			//Going over the input sentence to generate the hidden states
			for (int i=0; i<=end_pos; i++){
				//cerr<<"i is"<<i<<endl;
				//cerr<<"input is "<<input_data.row(i)<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					encoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);//,sequence_cont_indices.row(i));
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,	
										//current_c,
										//current_h);
				} else {
					//cerr<<"Data is "<<data.row(i)<<endl;
					//cerr<<"index is "<<i<<endl;
					encoder_lstm_nodes[i].copyToHiddenStates(encoder_lstm_nodes[i-1].h_t,encoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,
										//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
										//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
				}
				//encoder_lstm_nodes.fProp();
			}
			//Copying the cell and hidden states if the sequence continuation vectors say so	
			//cerr<<"end pos is"<<end_pos<<endl;
			current_c = encoder_lstm_nodes[end_pos].c_t;
			current_h = encoder_lstm_nodes[end_pos].h_t;
			//cerr<<"current c is "<<current_c<<endl;
			//cerr<<"current h is "<<current_h<<endl;
			//cerr<<"End pos is "<<end_pos<<endl;
			//Going over the output sentence to generate the hidden states
			for (int i=0; i<output_sent_len-1; i++){
				//cerr<<"i is"<<i<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
					decoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);//,sequence_cont_indices.row(i));
					decoder_lstm_nodes[i].fProp(output_data.row(i));//,	
					//cerr<<"output data is "<<output_data.row(i)<<endl;
										//current_c,
										//current_h);
				} else {
					//cerr<<"Data is "<<data.row(i)<<endl;
					//cerr<<"index is "<<i<<endl;
					decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,decoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
					decoder_lstm_nodes[i].fProp(output_data.row(i));//,
										//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
										//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
				}
				//encoder_lstm_nodes.fProp();
			}			

	    }


		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropInput(const MatrixBase<DerivedInput> &input_data,
				const int start_pos,
				const int end_pos,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	    {
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = input_data.rows();
			//int output_sent_len = output_data.rows();
			//Matrix<double,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = input_data.cols();

			//Going over the input sentence to generate the hidden states
			for (int i=0; i<=end_pos; i++){
				//cerr<<"i is"<<i<<endl;
				//cerr<<"input is "<<input_data.row(i)<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					encoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);//,sequence_cont_indices.row(i));
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,	
										//current_c,
										//current_h);
				} else {

					encoder_lstm_nodes[i].copyToHiddenStates(encoder_lstm_nodes[i-1].h_t,encoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,

				}
				//encoder_lstm_nodes.fProp();
			}
			//Copying the cell and hidden states if the sequence continuation vectors say so	
			//cerr<<"end pos is"<<end_pos<<endl;
			current_c = encoder_lstm_nodes[end_pos].c_t;
			current_h = encoder_lstm_nodes[end_pos].h_t;


	    }
		//currently only generate one output at a time
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void generateGreedyOutput(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				vector<int> &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol) {
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(100,1); // I can produce at most 100 output symbols
					predicted_output(0,0) = output_start_symbol;
					UNCONST(DerivedC, const_current_c, current_c);
					UNCONST(DerivedH, const_current_h, current_h);	
				//cerr<<"predicted_output	is "<<predicted_output<<endl;
				for (int i=0; i<99; i++){
					//cerr<<"i is "<<i<<endl;
					//cerr<<"predicted output is "<<predicted_output.row(i);
					if (i==0) {
						//cerr<<"Current c is "<<current_c<<endl;
						//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
						decoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,	
						//cerr<<"output data is "<<output_data.row(i)<<endl;
											//current_c,
											//current_h);
					} else {
						//cerr<<"Data is "<<data.row(i)<<endl;
						//cerr<<"index is "<<i<<endl;
						decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,decoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,
											//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
											//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
					}
					//cerr<<"ht is "<<decoder_lstm_nodes[i].h_t<<endl;
					//cerr<<"ht -1 is "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t, scores);
					//then compute the log loss of the objective
					//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
					//cerr<<"Score is"<<endl;
					//cerr<<scores<<endl;
	
			        double minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().fProp(scores, 
			                   predicted_output.row(i), 
			                   probs, 
			                   minibatch_log_likelihood);	
					int max_index = 0;
					double max_value = -9999999;
					//Matrix<double,1,Dynamic>::Index max_index;
					//probs.maxCoeff(&max_index); 
					//int minibatch_size = 0;
					//THIS HAS TO CHANGE
					for (int index=0; index<probs.rows(); index++){
						//cerr<<"prob is "<<probs(index,0)<<endl;
						if (probs(index,0) > max_value){
							max_value = probs(index,0);
							max_index = index;
						}
						
					}
					//getchar();
			        //Matrix<double,1,Dynamic>::Index max_index;
			        //probs.maxCoeff(&max_index);	
					//if max index equals the end symbol
					predicted_sequence.push_back(max_index);
					if (max_index == output_end_symbol)
						break;
					else{
						predicted_output(i+1,0) = max_index;
						//cerr<<"new predicted output is "<<predicted_output(i+1,0)<<endl;
					}		   				
				}

		}

		
		//Computing losses separately. Makes more sense because some LSTM units might not output units but will be receiving 
		//losses from the next layer
	    template <typename DerivedOut, typename data_type> //, typename DerivedC, typename DerivedH, typename DerivedS>
		void computeLosses(const MatrixBase<DerivedOut> &output,
			 double &log_likelihood,
			 bool gradient_check,
			 bool norm_clipping,
			 loss_function_type loss_function,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss){
	 			int current_minibatch_size = output.cols();
	 			//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
	 			Matrix<double,Dynamic,Dynamic> dummy_zero,dummy_ones;
	 			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
	 			//same dimension in and LSTM. this might not be a good idea
	 			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
	 			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
	 			int sent_len = output.rows(); 
	 			//double log_likelihood = 0.;
			
	 			for (int i=sent_len-1; i>=1; i--) {
	 				//cerr<<"i is "<<i<<endl;
	 				if (loss_function == LogLoss) {
	 					//First doing fProp for the output layer
	 					//The number of columns in scores will be the current minibatch size
						//cerr<<"ht going into loss"<<decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size)<<endl;
	 					output_layer_node.param->fProp(decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size), scores);
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
										losses[i-1].d_Err_t_d_h_t.leftCols(current_minibatch_size));
									   //output_layer_node.bProp_matrix.leftCols(current_minibatch_size));	
	 					//cerr<<"ouput layer bprop matrix rows"<<output_layer_node.bProp_matrix.rows()<<" cols"<<output_layer_node.bProp_matrix.cols()<<endl;
	 					//cerr<<"output_layer_node.bProp_matrix"<<output_layer_node.bProp_matrix<<endl;
	 					//cerr<<"Dimensions if d_Err_t_d_output "<<d_Err_t_d_output.rows()<<","<<d_Err_t_d_output.cols()<<endl;
	 					//cerr<<"output_layer_node.bProp_matrix "<<output_layer_node.bProp_matrix<<endl;
	 	   		        output_layer_node.param->updateGradient(decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size),
	 	   						       d_Err_t_d_output.leftCols(current_minibatch_size));									   	 		   
	 					//cerr<<" i is "<<i<<endl;
	 					//cerr<<"backprop matrix is "<<output_layer_node.bProp_matrix<<endl;
	 				} else if (loss_function == NCELoss){
						cerr<<"NOT IMPLEMENTED"<<endl;
						exit(1);
	 				}

		   
	 			}
	 			//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
	 			//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
	 			//cerr<<"The cross entropy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
	 			//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;		  		 	
		}
		
	    // Dense version (for standard log-likelihood)
	    template <typename DerivedIn, typename DerivedOut> //, typename DerivedC, typename DerivedH, typename DerivedS>
	    void bProp(const MatrixBase<DerivedIn> &input_data,
				const MatrixBase<DerivedOut> &output_data,
			 bool gradient_check,
			 bool norm_clipping)//,
			 //const MatrixBase<DerivedC> &init_c,
			 //const MatrixBase<DerivedH> &init_h,
			 //const Eigen::ArrayBase<DerivedS> &sequence_cont_indices) 
	    {	
		
			//cerr<<"In backprop..."<<endl;
			int current_minibatch_size = input_data.cols();
			//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
			Matrix<double,Dynamic,Dynamic> dummy_zero,dummy_ones;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
			int input_sent_len = input_data.rows();
			int output_sent_len = output_data.rows(); 
			//double log_likelihood = 0.;
			
			//first getting decoder loss
			for (int i=output_sent_len-2; i>=0; i--) {
				//cerr<<"i in output loss is "<<i<<endl;
				//getchar();
				// Now calling backprop for the LSTM nodes
				if (i==0 && output_sent_len-2 > 0) {
					
				    decoder_lstm_nodes[i].bProp(output_data.row(i),
							   //init_h,
				   			   //init_c,
								losses[i].d_Err_t_d_h_t,
							   //output_layer_node.bProp_matrix,
				   			   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
							   gradient_check,
							   norm_clipping);	
				} else if (i == output_sent_len-2) {	

					//cerr<<"previous ht is "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
					//cerr<<"previous ct is "<<decoder_lstm_nodes[i].c_t_minus_one<<endl;
					
				    decoder_lstm_nodes[i].bProp(output_data.row(i),
							   //(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
				   			   //(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
							   losses[i].d_Err_t_d_h_t,
							   //output_layer_node.bProp_matrix,
				   			   dummy_zero, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
				   			   dummy_zero,
							   gradient_check,
							   norm_clipping);
		
				} else if (i > 0) {
					
				    decoder_lstm_nodes[i].bProp(output_data.row(i),
							   //(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
				   			   //(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
							   losses[i].d_Err_t_d_h_t,
							   //output_layer_node.bProp_matrix,
				   			   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
							   gradient_check,
							   norm_clipping);		
					   						
				} 		   
		   
			}

	  }

    // Dense version (for standard log-likelihood)
    template <typename DerivedIn> //, typename DerivedC, typename DerivedH, typename DerivedS>
    void bPropEncoder(const MatrixBase<DerivedIn> &input_data,
		 bool gradient_check,
		 bool norm_clipping)//,
		 //const MatrixBase<DerivedC> &init_c,
		 //const MatrixBase<DerivedH> &init_h,
		 //const Eigen::ArrayBase<DerivedS> &sequence_cont_indices) 
    {	
	
		//cerr<<"In backprop..."<<endl;
		int current_minibatch_size = input_data.cols();
		//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
		Matrix<double,Dynamic,Dynamic> dummy_zero,dummy_ones;
		//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
		//same dimension in and LSTM. this might not be a good idea
		dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		//dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
		
		int input_sent_len = input_data.rows();

		//Now backpropping through the encoder

		//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
		//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
		//cerr<<"The cross entropy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
		//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
		//first getting decoder loss
		//cerr<<"dummy zero is "<<dummy_zero<<endl;
		for (int i=input_sent_len-1; i>=0; i--) {
			//getchar();
			// Now calling backprop for the LSTM nodes
			if (i==0) {
				
			    encoder_lstm_nodes[i].bProp(input_data.row(i),
						   //init_h,
			   			   //init_c,
							dummy_zero,
						   //output_layer_node.bProp_matrix,
			   			   encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
						   encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
						   gradient_check,
						   norm_clipping);	
				

			} else if (i == input_sent_len-1) {	

				//cerr<<"previous ct is "<<encoder_lstm_nodes[i-1].c_t<<endl;
				
			    encoder_lstm_nodes[i].bProp(input_data.row(i),
						   //(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
			   			   //(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
						   dummy_zero,
						   //output_layer_node.bProp_matrix,
			   			   decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
			   			   decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
						   gradient_check,
						   norm_clipping);
	
			} else if (i > 0) {
				
			    encoder_lstm_nodes[i].bProp(input_data.row(i),
						   //(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
			   			   //(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
						   dummy_zero,
						   //output_layer_node.bProp_matrix,
			   			   encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
						   encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
						   gradient_check,
						   norm_clipping);		
				   						
			} 		   
	   
		}
  }
  	  
	 void updateParams(double learning_rate,
	 					int current_minibatch_size,
				  		double momentum,
						double L2_reg,
						bool norm_clipping,
						double norm_threshold,
						loss_function_type loss_function) {
		//cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
		//cerr<<"updating params "<<endl;
		if (loss_function == LogLoss){
			decoder_plstm->output_layer.updateParams(learning_rate,
							current_minibatch_size,
		  					momentum,
		  					L2_reg,
							norm_clipping,
							norm_threshold);			
		} else if (loss_function == NCELoss){
			cerr<<"NOT IMPLEMENTED"<<endl;
			exit(1);
		} else {
			cerr<<loss_function<<" is an invalid loss function type"<<endl;
			exit(0);
		}


		/*						
		//Derivatives of the input embeddings							
	    encoder_plstm->input_layer.updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);		
		*/
	    encoder_plstm->updateParams(learning_rate,
											current_minibatch_size,
											momentum,
											L2_reg,
											norm_clipping,
											norm_threshold);
		decoder_plstm->updateParams(learning_rate,
										current_minibatch_size,
										momentum,
										L2_reg,
										norm_clipping,
										norm_threshold);												
	  }
	  
	  template <typename DerivedOut, typename data_type>
	  void computeProbs(const MatrixBase<DerivedOut> &output,
						multinomial<data_type> &unigram,
						int num_noise_samples,
						boost::random::mt19937 &rng,
						loss_function_type loss_function,
						SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
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

			for (int i=sent_len-1; i>=1; i--) {
				//cerr<<"i in gradient check is "<<i<<endl;
				//First doing fProp for the output layer
				if (loss_function == LogLoss) {
					output_layer_node.param->fProp(decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size), scores);
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
				} else if (loss_function == NCELoss) {
					cerr<<"NOT IMPLEMENTED"<<endl;
					exit(1);
				}
			}
			//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
			//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
			//cerr<<"The cross entopy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
			//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
			//log_likelihood /= sent_len;
	  }	  

	  template <typename DerivedOut>
	  void computeProbsLog(const MatrixBase<DerivedOut> &output,
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

			for (int i=sent_len-1; i>=1; i--) {
				//cerr<<"i is "<<i<<endl;
				//First doing fProp for the output layer
				output_layer_node.param->fProp(decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size), scores);
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
		encoder_plstm->resetGradient();	
		decoder_plstm->resetGradient();	
								
		//The gradients of the input layer are being reset in update params sinc the gradient is sparse
		//Derivatives of the input embeddings							
	    //encoder_plstm->input_layer.resetGradient();		
  	}	
	
	//Use finite differences to do gradient check
	template <typename DerivedIn, typename DerivedOut, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
    void gradientCheck(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 const MatrixBase<DerivedC> &const_init_c,
			 const MatrixBase<DerivedH> &const_init_h,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 loss_function_type loss_function,
			 SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
			 const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
				 
    {
		Matrix<double,Dynamic,Dynamic> init_c = const_init_c;
		Matrix<double,Dynamic,Dynamic> init_h = const_init_h;
		//boost::random::mt19937 init_rng = rng;
		cerr<<"init c is "<<init_c<<endl;
		cerr<<"init h is "<<init_h<<endl;
		//cerr<<"in gradient check. The size of input is "<<input.rows()<<endl;
		//cerr<<"In gradient check"<<endl;
		/*
		//Checking the gradient of h_t
		encoder_lstm_nodes[0].h_t(0,0) += 1e-5;
		fProp(input, 1, input.rows()-1);
		//fProp(input, 1, input.rows()-1);
		
 		double before_log_likelihood = 0;						
 		fProp(input,0, input.rows()-1);	
 		computeProbs(output,
 			  		before_log_likelihood);			

		encoder_lstm_nodes[0].h_t(0,0) -= 2e-5;
		fProp(input, 1, input.rows()-1);
		//fProp(input, 1, input.rows()-1);

 		double after_log_likelihood = 0;						
 		fProp(input,0, input.rows()-1);	
 		computeProbs(output,
 			  		after_log_likelihood);
		
		cerr<<"the measured gradient is"<<encoder_lstm_nodes[0].d_Err_t_to_n_d_h_t<<endl;
		cerr<<"Gradient diff is "<<	(before_log_likelihood-after_log_likelihood)/2e-5<<endl;
		*/
		//Check every dimension of all the parameters to make sure the gradient is fine
		
		
		paramGradientCheck(input,output,decoder_plstm->output_layer,"output_layer", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);	
							 
		//init_rng = rng;					 
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_h_to_c,"W_h_to_c", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;					 
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,encoder_plstm->W_h_to_f,"W_h_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;	
		init_c = const_init_c;
		init_h = const_init_h;										
		paramGradientCheck(input,output,encoder_plstm->W_h_to_o,"W_h_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_h_to_i ,"W_h_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		/*
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_x_to_c,"W_x_to_c", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_x_to_f,"W_x_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_x_to_o,"W_x_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_x_to_i,"W_x_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		*/
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,encoder_plstm->W_c_to_o,"W_c_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_c_to_f,"W_c_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//nit_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_c_to_i,"W_c_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,encoder_plstm->o_t,"o_t",  
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->f_t,"f_t",
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->i_t,"i_t",
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->tanh_c_prime_t,"tanh_c_prime_t", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);		
		//Doing gradient check for the input nodes
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_i,"Standard_input_node: W_x_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 sequence_cont_indices);		
  		init_c = const_init_c;
  		init_h = const_init_h;
  		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_f,"Standard_input_node: W_x_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 sequence_cont_indices);		
					   		init_c = const_init_c;
					   		init_h = const_init_h;
   		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_c,"Standard_input_node: W_x_to_c", 
  							 init_c,
  							 init_h,
  							 unigram,
  							 num_noise_samples,
  				   			 rng,
  				   			 loss_function,
  							 softmax_nce_loss,
  							 sequence_cont_indices);
		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_o,"Standard_input_node: W_x_to_o", 
						 init_c,
						 init_h,
						 unigram,
						 num_noise_samples,
			   			 rng,
			   			 loss_function,
						 softmax_nce_loss,
						 sequence_cont_indices);								 									 							 	

		//paramGradientCheck(input,output,encoder_plstm->input_layer,"input_layer");
		
		
	}
	template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
	void paramGradientCheck(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 testParam &param,
			 const string param_name,
			 const MatrixBase<DerivedC> &init_c,
			 const MatrixBase<DerivedH> &init_h, 
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 loss_function_type loss_function,			 
			 SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
			 const Eigen::ArrayBase<DerivedS> &sequence_cont_indices){
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
							init_h,
				   			unigram,
				   			num_noise_samples,
				   			rng,
				   			loss_function,
							softmax_nce_loss,
							sequence_cont_indices);
			}
		}
	}
	
	template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
    void getFiniteDiff(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &output,
			 testParam &param,
			 const string param_name,
			 int row,
			 int col,
			 const MatrixBase<DerivedC> &const_init_c,
			 const MatrixBase<DerivedH> &const_init_h,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 loss_function_type loss_function,
			 SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
			 const Eigen::ArrayBase<DerivedS> &sequence_cont_indices) {
				Matrix<double,Dynamic,Dynamic> init_c; 
				Matrix<double,Dynamic,Dynamic> init_h;
				boost::random::mt19937 init_rng = rng;
				init_c = const_init_c;
				init_h = const_init_h;
				//cerr<<"Row is :"<<row<<" col is " <<col<<endl;
				int rand_row = row;
		 		int rand_col = col;
		 		//First checking the gradient of the output word embeddings
		 		//cerr<<"Checking the gradient of "<<param_name<<endl;
		 		//rand_row = 0;
				//rand_col= 0;
				double perturbation = 1e-5;
		 	    param.changeRandomParam(perturbation, 
		 								rand_row,
		 								rand_col);
		 		//then do an fprop
		 		double before_log_likelihood = 0;	
				//cerr<<"input cols is "<<input.cols()<<endl;					
		 		fProp(input, output, 0, input.rows()-1, init_c, init_h, sequence_cont_indices);
		 		computeProbs(output,
				   			 unigram,
				   			 num_noise_samples,
				   			 init_rng,
				   			 loss_function,	
							 softmax_nce_loss,
		 			  		 before_log_likelihood);
		 		//err<<"before log likelihood is "<<
		 	    param.changeRandomParam(-2*perturbation, 
		 								rand_row,
		 								rand_col);		
				init_c = const_init_c;
				init_h = const_init_h;
				init_rng = rng;
		 		double after_log_likelihood = 0;						
		 		fProp(input,output, 0, input.rows()-1, init_c, init_h, sequence_cont_indices);	
		 		computeProbs(output,
				   			 unigram,
				   			 num_noise_samples,
				   			 init_rng,
				   			 loss_function,	
							 softmax_nce_loss,
		 			  		 after_log_likelihood);		
		 		//returning the parameter back to its own value
		 	    param.changeRandomParam(perturbation , 
		 								rand_row,
		 								rand_col);			

				double threshold = 1e-02;
				//cerr<<"graves "<<pow(10.0, max(0.0, ceil(log10(min(fabs(param.getGradient(rand_row,
		 		//						rand_col)), fabs((before_log_likelihood-after_log_likelihood)/2e-5)))))-6)<<endl;
				double symmetric_finite_diff_grad = (before_log_likelihood-after_log_likelihood)/(2*perturbation);	
				double graves_threshold = pow(10.0, (double) max(0.0, (double) ceil(log10(min(fabs(param.getGradient(rand_row,
		 								rand_col)), fabs(symmetric_finite_diff_grad)))))-6);
				double gradient_diff =  symmetric_finite_diff_grad - param.getGradient(rand_row,
		 								rand_col);
				double relative_error = fabs(param.getGradient(rand_row,rand_col)-symmetric_finite_diff_grad)/
					(fabs(param.getGradient(rand_row,rand_col)) + fabs(symmetric_finite_diff_grad));
				if (gradient_diff > threshold|| relative_error > threshold) {
					cerr<<"!!!GRADIENT CHECKING FAILED!!!"<<endl;
			 		cerr<<"Symmetric finite differences gradient is "<<	symmetric_finite_diff_grad<<endl;
					cerr<<"Algorithmic gradient is "<<param.getGradient(rand_row,rand_col)<<endl;					
		 	    	cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" at row: "<<rand_row
						<<" and col: "<<rand_col<<" is "<<gradient_diff<<endl;	
					cerr<<"The likelihoods before and after perturbation are "<< before_log_likelihood<<" "<<
									after_log_likelihood<<endl;
					cerr<<"Graves threshold is "<<graves_threshold<<endl;
					cerr<<"Relative error is "<<relative_error<<endl;
					exit(1);
				} else {
		 	    	cerr<<"The difference between computed gradient and symbolic gradient for "<<param_name<<" at row: "<<rand_row
						<<" and col: "<<rand_col<<" is "<<gradient_diff<<" and relative error is "<<relative_error<<endl;
					cerr<<"Symmetric finite differences gradient is "<<	symmetric_finite_diff_grad<<endl;
					cerr<<"Algorithmic gradient is "<<param.getGradient(rand_row,rand_col)<<endl;
					//cerr<<"Relative error is "<<relative error<<endl
				}
		 	
	}	
	
	
	

 };		
 
 
} // namespace nplm

#endif
	