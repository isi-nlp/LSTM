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
		Matrix<precision_type,Dynamic,Dynamic> d_Err_tPlusOne_to_n_d_c_t,d_Err_tPlusOne_to_n_d_h_t; //Derivatives wrt the future h_t and c_t
		Matrix<precision_type,Dynamic,Dynamic> scores;
		Matrix<precision_type,Dynamic,Dynamic> minibatch_weights;
		Matrix<precision_type,Dynamic,Dynamic> d_Err_t_d_output;
		Matrix<int,Dynamic,Dynamic> minibatch_samples;
		Matrix<int,Dynamic,Dynamic> minibatch_samples_no_negative;
		Matrix<precision_type,Dynamic,Dynamic> probs;	
		int num_hidden;
		precision_type fixed_partition_function; 
		boost::random::uniform_real_distribution<> unif_real;
		//vector<Matrix<precision_type,Dynamic,Dynamic> > losses;
		vector<Output_loss_node> losses;

	public:
	    propagator() : minibatch_size(0), 
					encoder_plstm(0), 
					decoder_plstm(0),
					encoder_lstm_nodes(105,LSTM_node<input_node_type>()),
					decoder_lstm_nodes(105,LSTM_node<input_node_type>()),
					encoder_input_nodes(105,input_node_type()),
					decoder_input_nodes(105,input_node_type()),
					num_hidden(0), 
					fixed_partition_function(0), 
					losses(vector<Output_loss_node>(105,Output_loss_node())),
					unif_real(0.0,1.0){ }

	    propagator (model &encoder_lstm, 
					model &decoder_lstm,
					int minibatch_size)
	      : encoder_plstm(&encoder_lstm),
			decoder_plstm(&decoder_lstm),
		 	minibatch_size(minibatch_size),
			output_layer_node(&decoder_lstm.output_layer,minibatch_size),
			encoder_lstm_nodes(vector<LSTM_node<input_node_type> >(105,LSTM_node<input_node_type>(encoder_lstm,minibatch_size))),
			decoder_lstm_nodes(vector<LSTM_node<input_node_type> >(105,LSTM_node<input_node_type>(decoder_lstm,minibatch_size))),
			encoder_input_nodes(vector<input_node_type >(105,input_node_type (dynamic_cast<input_model_type&>(*(encoder_lstm.input)),minibatch_size))),
			decoder_input_nodes(vector<input_node_type >(105,input_node_type (dynamic_cast<input_model_type&>(*(decoder_lstm.input)),minibatch_size))),
			//losses(vector<Matrix<precision_type,Dynamic,Dynamic> >(100,Matrix<precision_type,Dynamic,Dynamic>()))
			losses(vector<Output_loss_node>(105,Output_loss_node())),
			unif_real(0.0,1.0)
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
		void resizeNCE(int num_noise_samples, precision_type fixed_partition_function){
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
			//Matrix<precision_type,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = input_data.cols();
			//cerr<<"current minibatch_size is "<<current_minibatch_size<<endl;
			//Going over the input sentence to generate the hidden states
			for (int i=0; i<=end_pos; i++){
				//cerr<<"i is"<<i<<endl;
				//cerr<<"input is "<<input_data.row(i)<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					//encoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);//,sequence_cont_indices.row(i));
					encoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);
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

		//Both the input and the output sentences are columns. Even ifs a minibatch of sentences, each sentence is a column
	    template <typename DerivedOutput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropDecoder(const MatrixBase<DerivedOutput> &output_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	    {
			//UNCONST(DerivedC, const_current_c, current_c);
			//UNCONST(DerivedH, const_current_h, current_h);
			//cerr<<"current_c in fPropDecoder is "<<const_current_c<<endl;
			//cerr<<"current_h is fPropDecoder is "<<const_current_h<<endl;
			/*
			cerr<<"Data is "<<data<<endl;
			cerr<<"Start pos "<<start_pos<<endl;
			cerr<<"End pos "<<end_pos<<endl;
			cerr<<"In Fprop"<<endl;
			*/
			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			//int sent_len = input_data.rows();
			int output_sent_len = output_data.rows();
			//Matrix<precision_type,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = output_data.cols();
			//cerr<<"current minibatch_size is "<<current_minibatch_size<<endl;
			//Copying the cell and hidden states if the sequence continuation vectors say so	
			//cerr<<"end pos is"<<end_pos<<endl;
			//current_c = encoder_lstm_nodes[end_pos].c_t;
			//current_h = encoder_lstm_nodes[end_pos].h_t;
			//cerr<<"current c is "<<current_c<<endl;
			//cerr<<"current h is "<<current_h<<endl;
			//cerr<<"End pos is "<<end_pos<<endl;
			//Going over the output sentence to generate the hidden states
			for (int i=0; i<output_sent_len-1; i++){
				//cerr<<"i is"<<i<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
					//decoder_lstm_nodes[i].copyToHiddenStates(const_current_h,const_current_c);//,sequence_cont_indices.row(i));
					
					decoder_lstm_nodes[i].filterStatesAndErrors(const_current_h,
																const_current_c,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));			
																
					decoder_lstm_nodes[i].fProp(output_data.row(i));//,	
					//cerr<<"output data is "<<output_data.row(i)<<endl;
										//current_c,
										//current_h);
				} else {
					//cerr<<"Data is "<<data.row(i)<<endl;
					//cerr<<"index is "<<i<<endl;
					//decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,decoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
					
					decoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[i-1].h_t,
																decoder_lstm_nodes[i-1].c_t,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));						
																
					decoder_lstm_nodes[i].fProp(output_data.row(i));//,
					
										//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
										//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
				}
				//encoder_lstm_nodes.fProp();
				//cerr<<"decoder_lstm_nodes[i].h_t_minus_one "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
				//cerr<<"decoder_lstm_nodes[i].c_t_minus_one "<<decoder_lstm_nodes[i].c_t_minus_one<<endl;
				//cerr<<"decoder_lstm_nodes["<<i<<"].h_t_minus_one "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
				//cerr<<"decoder_lstm_nodes["<<i<<"].h_t "<<decoder_lstm_nodes[i].h_t<<endl;
			}			

	    }
		
		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropEncoder(const MatrixBase<DerivedInput> &input_data,
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
			//Matrix<precision_type,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = input_data.cols();
			
			//cerr<<"input data is "<<input_data<<endl;
			//Going over the input sentence to generate the hidden states
			//Matrix<DerivedInput> dummy_ones;
			//dummy_ones.setOnes(1,current_minibatch_size);
			for (int i=0; i<=end_pos; i++){
				//cerr<<"sequence cont indices are "<<sequence_cont_indices.row(i)<<endl;
				//cerr<<"i is"<<i<<endl;
				//cerr<<"input is "<<input_data.row(i)<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					//encoder_lstm_nodes[i].copyToHiddenStates(current_h,current_c);//,sequence_cont_indices.row(i));
					
				
					encoder_lstm_nodes[i].filterStatesAndErrors(current_h,
																current_c,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																//dummy_ones); //this might just be a patch for now
																sequence_cont_indices.row(i));	
					//cerr<<"encoder_lstm_nodes["<<i<<"].h_t_minus_one "<<encoder_lstm_nodes[i].h_t_minus_one<<endl;											
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,	
										//current_c,
										//current_h);
				} else {

					//encoder_lstm_nodes[i].copyToHiddenStates(encoder_lstm_nodes[i-1].h_t,encoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
					
					encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i-1].h_t,
																encoder_lstm_nodes[i-1].c_t,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i-1)); //THIS IS JUST A TEMPORARY FIX THAT ASSUMES THE INITIAL HIDDEN STATES ARE 0
					//cerr<<"encoder_lstm_nodes["<<i<<"].h_t_minus_one "<<encoder_lstm_nodes[i].h_t_minus_one<<endl;																			
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,

				}
				//encoder_lstm_nodes.fProp();
				//cerr<<"encoder_lstm_nodes["<<i<<"].h_t "<<encoder_lstm_nodes[i].h_t<<endl;
				
			}
			//Copying the cell and hidden states if the sequence continuation vectors say so	
			//cerr<<"end pos is"<<end_pos<<endl;
			current_c = encoder_lstm_nodes[end_pos].c_t;
			current_h = encoder_lstm_nodes[end_pos].h_t;


	    }
		
		template <typename DerivedH> 
		void getHiddenStates(const MatrixBase<DerivedH> &const_hidden_states,
							const int max_sent_len,
							const bool for_encoder,
							const int sent_index) const { //if for_encoder is 1, then you will print the 
													//hidden states for the encoder. else decoder
			UNCONST(DerivedH, const_hidden_states, hidden_states);
			//cerr<<"Sent index is "<<sent_index<<endl;
			int num_hidden = decoder_lstm_nodes[0].h_t.rows();
			//cerr<<"num hidden is "<<num_hidden<<endl;
			//cerr<<"hidden states is "<<hidden_states<<endl;
			//hidden_states.resize(num_hidden,max_sent_len);
			//hidden_states.setZero();
			for(int i=0; i<max_sent_len; i++){
				//cerr<<"looking for hidden states in node "<<i<<"with index "<<index<<endl;
				//cerr<<"encoder_lstm_nodes[i].h_t.col(sent_index)"<<encoder_lstm_nodes[i].h_t.col(sent_index)<<endl;
				if (for_encoder == 1) {
					hidden_states.col(i) = encoder_lstm_nodes[i].h_t.col(sent_index);
					//cerr<<"encoder_lstm_nodes[i].h_t.col(sent_index)"<<encoder_lstm_nodes[i].h_t.col(sent_index)<<endl;
				} else {
					hidden_states.col(i) = decoder_lstm_nodes[i].h_t.col(sent_index);
				}
				
				//cerr<<"hidden states col is "<<	hidden_states.col(i)<<endl;					
			}
		}
		
	    template<typename DerivedO, 
					typename DerivedF, 
					typename DerivedI, 
					typename DerivedH, 
					typename DerivedC>		
	    void getInternals(const MatrixBase<DerivedH> &const_get_h_t,
						const MatrixBase<DerivedC>   &const_get_c_t,
						const MatrixBase<DerivedF>   &const_get_f_t,
						const MatrixBase<DerivedI>   &const_get_i_t,
						const MatrixBase<DerivedO>   &const_get_o_t,
						const int max_sent_len,
						const bool for_encoder,
						const int sent_index){
			UNCONST(DerivedH, const_get_h_t, get_h_t);
			UNCONST(DerivedH, const_get_c_t, get_c_t);
			UNCONST(DerivedH, const_get_f_t, get_f_t);
			UNCONST(DerivedH, const_get_i_t, get_i_t);
			UNCONST(DerivedH, const_get_o_t, get_o_t);			
			for (int i=0; i<max_sent_len; i++){
				if (for_encoder == 1){
					encoder_lstm_nodes[i].getInternals(get_h_t.col(i),
													get_c_t.col(i),
													get_f_t.col(i),
													get_i_t.col(i),
													get_o_t.col(i),
													sent_index);
					//cerr<<"propagator const_get_h_t.col("<<i<<") "<<get_h_t.col(i)<<endl;
				} else {
					decoder_lstm_nodes[i].getInternals(get_h_t.col(i),
													get_c_t.col(i),
													get_f_t.col(i),
													get_i_t.col(i),
													get_o_t.col(i),
													sent_index);
				}
			}

	    }
			
		//currently only generate one output at a time
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void generateGreedyOutput(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				vector<vector<int> > &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol) {
					int current_minibatch_size = input_data.cols();
					//cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(101,current_minibatch_size); // I can produce at most 100 output symbols
					//predicted_output.resize(1,current_minibatch_size);
					//predicted_output.fill(output_start_symbol);
					predicted_output.row(0).fill(output_start_symbol);
					//predicted_output(0,0) = output_start_symbol;
					UNCONST(DerivedC, const_current_c, current_c);
					UNCONST(DerivedH, const_current_h, current_h);	
					//vector<int> current_words(output_start_symbol,current_minibatch_size);
					vector<int> live_words (current_minibatch_size,1);
					int remaining_live_words = current_minibatch_size;
					//cerr<<"live words are"<<live_words<<endl;
					//predicted_output = resize(1,current_minibatch_size);
					
					//predicted_output
				//cerr<<"predicted_output	is "<<predicted_output<<endl;
				for (int i=0; i<100 && remaining_live_words > 0; i++){
					//cerr<<"Predicted output is "<<predicted_output.row(i)<<endl;
					//current_minibatch_size = current_words.size();
					//predicted_output = Map< Matrix<int,Dynamic,Dynamic> >(current_words.data(), 1, current_minibatch_size);
					//cerr<<"i is "<<i<<endl;
					//cerr<<"predicted output is "<<predicted_output.row(i);
					if (i==0) {
						//cerr<<"Current c is "<<current_c<<endl;
						//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
						decoder_lstm_nodes[i].copyToHiddenStates(current_h,
											current_c);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,	
						//cerr<<"output data is "<<output_data.row(i)<<endl;
											//current_c,
											//current_h);
					} else {
						//cerr<<"Data is "<<data.row(i)<<endl;
						//cerr<<"index is "<<i<<endl;
						decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,
											decoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,
											//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
											//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
					}
					//cerr<<"ht is "<<decoder_lstm_nodes[i].h_t<<endl;
					//cerr<<"ht -1 is "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));
					//then compute the log loss of the objective
					//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
					//cerr<<"Score is"<<endl;
					//cerr<<scores<<endl;
	
			        precision_type minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
			                   predicted_output.row(i), 
			                   probs, 
			                   minibatch_log_likelihood);	
					//int max_index = 0;
					//precision_type max_value = -9999999;
					//Matrix<precision_type,1,Dynamic>::Index max_index;
					//probs.col(maxCoeff(&max_index); 
					//int minibatch_size = 0;
					//THIS HAS TO CHANGE
					/*
					for (int index=0; index<probs.rows(); index++){
						//cerr<<"prob is "<<probs(index,0)<<endl;
						if (probs(index,0) > max_value){
							max_value = probs(index,0);
							max_index = index;
						}
						
					}
					*/
					//getchar();
			        //Matrix<precision_type,1,Dynamic>::Index max_index;
			        //probs.maxCoeff(&max_index);	
					//if max index equals the end symbol
					//current_minibatch_size = 0;
					//nt live_index=0;
					//current_words.clear();
					for (int index=0; index<live_words.size(); index++){
						if (live_words[index] == 1){
							//predicted_sequence[index].push_back()
							Matrix<precision_type,1,Dynamic>::Index max_index;
							probs.col(index).maxCoeff(&max_index);
							//cerr<<"max index is "<<max_index<<endl;
							if (max_index == output_end_symbol){
								live_words[index] = -1;
								remaining_live_words--;
							} //else {
							//	current_words.push_back(max_index);
								//current_minibatch_size++;
							//}
							predicted_sequence[index].push_back(max_index);
							predicted_output(i+1,index) = max_index;
						} else {
							predicted_output(i+1,index) = output_end_symbol;
						}
						//cerr<<"remaining live words are"<<remaining_live_words<<endl;
					}
					/*
					predicted_sequence.push_back(max_index);
					if (max_index == output_end_symbol)
						break;
					else{
						predicted_output(i+1,0) = max_index;
						//cerr<<"new predicted output is "<<predicted_output(i+1,0)<<endl;
					}
					*/		   				
				}

		}

		//currently only generate one output at a time
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void beamDecoding(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				vector<vector<int> > &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol) {
					int current_minibatch_size = input_data.cols();
					cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(101,current_minibatch_size); // I can produce at most 100 output symbols
					//predicted_output.resize(1,current_minibatch_size);
					//predicted_output.fill(output_start_symbol);
					predicted_output.row(0).fill(output_start_symbol);
					//predicted_output(0,0) = output_start_symbol;
					UNCONST(DerivedC, const_current_c, current_c);
					UNCONST(DerivedH, const_current_h, current_h);	
					//vector<int> current_words(output_start_symbol,current_minibatch_size);
					vector<int> live_words (current_minibatch_size,1);
					int remaining_live_words = current_minibatch_size;
					//cerr<<"live words are"<<live_words<<endl;
					//predicted_output = resize(1,current_minibatch_size);
					
					//predicted_output
				//cerr<<"predicted_output	is "<<predicted_output<<endl;
				for (int i=0; i<100 && remaining_live_words > 0; i++){
					//cerr<<"Predicted output is "<<predicted_output.row(i)<<endl;
					//current_minibatch_size = current_words.size();
					//predicted_output = Map< Matrix<int,Dynamic,Dynamic> >(current_words.data(), 1, current_minibatch_size);
					//cerr<<"i is "<<i<<endl;
					//cerr<<"predicted output is "<<predicted_output.row(i);
					if (i==0) {
						//cerr<<"Current c is "<<current_c<<endl;
						//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
						decoder_lstm_nodes[i].copyToHiddenStates(current_h,
											current_c);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,	
						//cerr<<"output data is "<<output_data.row(i)<<endl;
											//current_c,
											//current_h);
					} else {
						//cerr<<"Data is "<<data.row(i)<<endl;
						//cerr<<"index is "<<i<<endl;
						decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,
											decoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,
											//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
											//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
					}
					//cerr<<"ht is "<<decoder_lstm_nodes[i].h_t<<endl;
					//cerr<<"ht -1 is "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));
					//then compute the log loss of the objective
					//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
					//cerr<<"Score is"<<endl;
					//cerr<<scores<<endl;
	
			        precision_type minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
			                   predicted_output.row(i), 
			                   probs, 
			                   minibatch_log_likelihood);	
					//int max_index = 0;
					//precision_type max_value = -9999999;
					//Matrix<precision_type,1,Dynamic>::Index max_index;
					//probs.col(maxCoeff(&max_index); 
					//int minibatch_size = 0;
					//THIS HAS TO CHANGE
					/*
					for (int index=0; index<probs.rows(); index++){
						//cerr<<"prob is "<<probs(index,0)<<endl;
						if (probs(index,0) > max_value){
							max_value = probs(index,0);
							max_index = index;
						}
						
					}
					*/
					//getchar();
			        //Matrix<precision_type,1,Dynamic>::Index max_index;
			        //probs.maxCoeff(&max_index);	
					//if max index equals the end symbol
					//current_minibatch_size = 0;
					//nt live_index=0;
					//current_words.clear();
					for (int index=0; index<live_words.size(); index++){
						if (live_words[index] == 1){
							//predicted_sequence[index].push_back()
							Matrix<precision_type,1,Dynamic>::Index max_index;
							probs.col(index).maxCoeff(&max_index);
							//cerr<<"max index is "<<max_index<<endl;
							if (max_index == output_end_symbol){
								live_words[index] = -1;
								remaining_live_words--;
							} //else {
							//	current_words.push_back(max_index);
								//current_minibatch_size++;
							//}
							predicted_sequence[index].push_back(max_index);
							predicted_output(i+1,index) = max_index;
						} else {
							predicted_output(i+1,index) = output_end_symbol;
						}
						//cerr<<"remaining live words are"<<remaining_live_words<<endl;
					}
					/*
					predicted_sequence.push_back(max_index);
					if (max_index == output_end_symbol)
						break;
					else{
						predicted_output(i+1,0) = max_index;
						//cerr<<"new predicted output is "<<predicted_output(i+1,0)<<endl;
					}
					*/		   				
				}

		}

		//currently only generate one output at a time
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void generateStochasticOutput(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				vector<vector<int> > &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol,
				boost::random::mt19937 &eng) {
					int current_minibatch_size = input_data.cols();
					cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(101,current_minibatch_size); // I can produce at most 100 output symbols
					//predicted_output.resize(1,current_minibatch_size);
					//predicted_output.fill(output_start_symbol);
					predicted_output.row(0).fill(output_start_symbol);
					//predicted_output(0,0) = output_start_symbol;
					UNCONST(DerivedC, const_current_c, current_c);
					UNCONST(DerivedH, const_current_h, current_h);	
					//vector<int> current_words(output_start_symbol,current_minibatch_size);
					vector<int> live_words (current_minibatch_size,1);
					int remaining_live_words = current_minibatch_size;
					//cerr<<"live words are"<<live_words<<endl;
					//predicted_output = resize(1,current_minibatch_size);
					
					//predicted_output
				//cerr<<"predicted_output	is "<<predicted_output<<endl;
				for (int i=0; i<101 && remaining_live_words > 0; i++){

					if (i==0) {
						//cerr<<"Current c is "<<current_c<<endl;
						//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
						decoder_lstm_nodes[i].copyToHiddenStates(current_h,
											current_c);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,	
						//cerr<<"output data is "<<output_data.row(i)<<endl;
											//current_c,
											//current_h);
					} else {
						//cerr<<"Data is "<<data.row(i)<<endl;
						//cerr<<"index is "<<i<<endl;
						decoder_lstm_nodes[i].copyToHiddenStates(decoder_lstm_nodes[i-1].h_t,
											decoder_lstm_nodes[i-1].c_t);//,sequence_cont_indices.row(i));
						decoder_lstm_nodes[i].fProp(predicted_output.row(i));//,
											//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
											//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());
					}
					//cerr<<"ht is "<<decoder_lstm_nodes[i].h_t<<endl;
					//cerr<<"ht -1 is "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));

			        precision_type minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
			                   predicted_output.row(i), 
			                   probs, 
			                   minibatch_log_likelihood);	

					for (int index=0; index<live_words.size(); index++){
						if (live_words[index] == 1){
							//predicted_sequence[index].push_back()
							//Matrix<precision_type,1,Dynamic>::Index max_index;
							//probs.col(index).maxCoeff(&max_index);
							//cerr<<"max index is "<<max_index<<endl;
							precision_type rand_value = unif_real(eng);
							//cerr<<"rand value is"<<rand_value<<endl;
							int rand_index =0;
							precision_type cumul =0;
							for (int vocab_index=0; vocab_index<probs.col(index).rows(); vocab_index++){
								cumul += exp(probs(vocab_index,index));
								//cerr<<"cumul is "<<cumul<<endl;
								if (cumul >= rand_value){
									rand_index = vocab_index;
									break;
								}
								
							}
							if (rand_index == output_end_symbol){
								live_words[index] = -1;
								remaining_live_words--;
							} //else {
							//	current_words.push_back(max_index);
								//current_minibatch_size++;
							//}
							predicted_sequence[index].push_back(rand_index);
							predicted_output(i+1,index) = rand_index;
						} else {
							predicted_output(i+1,index) = output_end_symbol;
						}
						//cerr<<"remaining live words are"<<remaining_live_words<<endl;
					}
	   				
				}

		}
		
		//Computing losses separately. Makes more sense because some LSTM units might not output units but will be receiving 
		//losses from the next layer
	    template <typename DerivedOut, typename data_type> //, typename DerivedC, typename DerivedH, typename DerivedS>
		void computeLosses(const MatrixBase<DerivedOut> &output,
			 precision_type &log_likelihood,
			 bool gradient_check,
			 bool norm_clipping,
			 loss_function_type loss_function,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss){
	 			int current_minibatch_size = output.cols();
	 			//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
	 			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
	 			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
	 			//same dimension in and LSTM. this might not be a good idea
	 			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
	 			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
	 			int sent_len = output.rows(); 
	 			//precision_type log_likelihood = 0.;
			
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
				
	 			        precision_type minibatch_log_likelihood;
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
	    void bPropDecoder(const MatrixBase<DerivedIn> &input_data,
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
			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
			int input_sent_len = input_data.rows();
			int output_sent_len = output_data.rows(); 
			//precision_type log_likelihood = 0.;
			
			//first getting decoder loss
			for (int i=output_sent_len-2; i>=0; i--) {
				//cerr<<"i in decoder bprop is "<<i<<endl;
				//getchar();
				// Now calling backprop for the LSTM nodes
				//cerr<<"losses[i].d_Err_t_d_h_t "<<losses[i].d_Err_t_d_h_t<<endl;
				//getchar();
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
				//cerr<<" decoder_lstm_nodes[i].d_Err_t_to_n_d_h_tMinusOne "<<decoder_lstm_nodes[i].d_Err_t_to_n_d_h_tMinusOne<<endl;
				//cerr<<" decoder_lstm_nodes[i].d_Err_t_to_n_d_c_tMinusOne "<<decoder_lstm_nodes[i].d_Err_t_to_n_d_c_tMinusOne<<endl;
				//getchar();
			}

	  }


	  
  // Dense version (for standard log-likelihood)
  template <typename DerivedIn, typename DerivedS> //, typename DerivedC, typename DerivedH, typename DerivedS>
  void bPropEncoder(const MatrixBase<DerivedIn> &input_data,
	 bool gradient_check,
	 bool norm_clipping,
	 const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	 //const MatrixBase<DerivedC> &init_c,
	 //const MatrixBase<DerivedH> &init_h,
	 //const Eigen::ArrayBase<DerivedS> &sequence_cont_indices) 
  {	

	//cerr<<"In backprop..."<<endl;
	int current_minibatch_size = input_data.cols();
	//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
	Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
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
		if (i == input_sent_len-1) {	
			
			//cerr<<"previous ct is "<<encoder_lstm_nodes[i-1].c_t<<endl;
			encoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
														decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
														decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
														decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne,
														sequence_cont_indices.row(i));			
			//cerr<<"decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne is "<<decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne<<endl;
			//cerr<<"decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne is "<<decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne<<endl;												
		    encoder_lstm_nodes[i].bProp(input_data.row(i),
					   //(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
		   			   //(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i)).matrix(),
					   dummy_zero,
					   //output_layer_node.bProp_matrix,
		   			   decoder_lstm_nodes[0].d_Err_t_to_n_d_c_tMinusOne, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
		   			   decoder_lstm_nodes[0].d_Err_t_to_n_d_h_tMinusOne,
					   gradient_check,
					   norm_clipping);

		} else{
			encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
														encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
														encoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
														encoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
														sequence_cont_indices.row(i));		
																	
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
	 void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
				  		precision_type momentum,
						precision_type L2_reg,
						bool norm_clipping,
						precision_type norm_threshold,
						loss_function_type loss_function,
						bool arg_run_lm) {
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
		if (arg_run_lm == 0) {
		    encoder_plstm->updateParams(learning_rate,
												current_minibatch_size,
												momentum,
												L2_reg,
												norm_clipping,
												norm_threshold);
		}
		
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
	  					precision_type &log_likelihood) 
	  {	
			
			//cerr<<"In computeProbs..."<<endl;
			int current_minibatch_size = output.cols();

			Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

			int sent_len = output.rows(); 
			//precision_type log_likelihood = 0.;

			for (int i=sent_len-1; i>=1; i--) {
				//cerr<<"i in gradient check is "<<i<<endl;
				//First doing fProp for the output layer
				if (loss_function == LogLoss) {
					output_layer_node.param->fProp(decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));
					//then compute the log loss of the objective
					//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
					//cerr<<"Score is"<<endl;
					//cerr<<scores<<endl;
	
			        precision_type minibatch_log_likelihood;
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
	  					precision_type &log_likelihood) 
	  {	
			
			//cerr<<"In computeProbs..."<<endl;
			int current_minibatch_size = output.cols();

			Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

			int sent_len = output.rows(); 
			//precision_type log_likelihood = 0.;

			for (int i=sent_len-1; i>=1; i--) {
				//cerr<<"i is "<<i<<endl;
				//First doing fProp for the output layer
				output_layer_node.param->fProp(decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size), scores);
				//then compute the log loss of the objective
				//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
				//cerr<<"Score is"<<endl;
				//cerr<<scores<<endl;

		        precision_type minibatch_log_likelihood;
		        start_timer(5);
		        SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
		                   output.row(i), 
		                   probs, 
		                   minibatch_log_likelihood);
				//cerr<<"probs is "<<probs<<endl;
		        stop_timer(5);
		        log_likelihood += minibatch_log_likelihood;		
			}

	  }	  
	  
	  //void LogProbs

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
			 const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
			 const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
			 bool arg_run_lm)
				 
    {
		Matrix<precision_type,Dynamic,Dynamic> init_c = const_init_c;
		Matrix<precision_type,Dynamic,Dynamic> init_h = const_init_h;
		//boost::random::mt19937 init_rng = rng;
		cerr<<"init c is "<<init_c<<endl;
		cerr<<"init h is "<<init_h<<endl;
		//cerr<<"in gradient check. The size of input is "<<input.rows()<<endl;
		//cerr<<"In gradient check"<<endl;

		//Check every dimension of all the parameters to make sure the gradient is fine
		

		paramGradientCheck(input,output,decoder_plstm->output_layer,"output_layer", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);	
					 		//init_rng = rng;					 
					 		init_c = const_init_c;
					 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->W_h_to_c,"Decoder: W_h_to_c", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;					 
 		init_c = const_init_c;
 		init_h = const_init_h;		
 		paramGradientCheck(input,output,decoder_plstm->W_h_to_f,"Decoder: W_h_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;	
 		init_c = const_init_c;
 		init_h = const_init_h;										
 		paramGradientCheck(input,output,decoder_plstm->W_h_to_o,"Decoder: W_h_to_o", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->W_h_to_i ,"Decoder: W_h_to_i", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);

 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;		
 		paramGradientCheck(input,output,decoder_plstm->W_c_to_o,"Decoder: W_c_to_o", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->W_c_to_f,"Decoder: W_c_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//nit_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->W_c_to_i,"Decoder: W_c_to_i", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;		
 		paramGradientCheck(input,output,decoder_plstm->o_t,"Decoder: o_t",  
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->f_t,"Decoder: f_t",
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->i_t,"Decoder: i_t",
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,decoder_plstm->tanh_c_prime_t,"Decoder: tanh_c_prime_t", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);		
 		//Doing gradient check for the input nodes
  		//init_rng = rng;
  		init_c = const_init_c;
  		init_h = const_init_h;
  		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_i,"Decoder: Standard_input_node: W_x_to_i", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);		
   		init_c = const_init_c;
   		init_h = const_init_h;
   		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_f,"Decoder: Standard_input_node: W_x_to_f", 
  							 init_c,
  							 init_h,
  							 unigram,
  							 num_noise_samples,
  				   			 rng,
  				   			 loss_function,
  							 softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices);
		 		
 					   		init_c = const_init_c;
 					   		init_h = const_init_h;
		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_c,"Decoder: Standard_input_node: W_x_to_c", 
						 init_c,
						 init_h,
						 unigram,
						 num_noise_samples,
			   			 rng,
			   			 loss_function,
						 softmax_nce_loss,
						 input_sequence_cont_indices,
						 output_sequence_cont_indices);
		 
 		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_o,"Decoder: Standard_input_node: W_x_to_o", 
 						 init_c,
 						 init_h,
 						 unigram,
 						 num_noise_samples,
 			   			 rng,
 			   			 loss_function,
 						 softmax_nce_loss,
 						 input_sequence_cont_indices,
 						 output_sequence_cont_indices);	
		
		
		//Encoder params				
							 					 							 
		//init_rng = rng;					 
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_h_to_c,"Encoder: W_h_to_c", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;					 
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,encoder_plstm->W_h_to_f,"Encoder: W_h_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;	
		init_c = const_init_c;
		init_h = const_init_h;										
		paramGradientCheck(input,output,encoder_plstm->W_h_to_o,"Encoder: W_h_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_h_to_i ,"Encoder: W_h_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);

		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,encoder_plstm->W_c_to_o,"Encoder: W_c_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_c_to_f,"Encoder: W_c_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//nit_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->W_c_to_i,"Encoder: W_c_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,output,encoder_plstm->o_t,"Encoder: o_t",  
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->f_t,"Encoder: f_t",
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->i_t,"Encoder: i_t",
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,output,encoder_plstm->tanh_c_prime_t,"Encoder: tanh_c_prime_t", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);		
		//Doing gradient check for the input nodes
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_i,"Encoder: Standard_input_node: W_x_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);		
  		init_c = const_init_c;
  		init_h = const_init_h;
  		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_f,"Encoder: Standard_input_node: W_x_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
							 		
					   		init_c = const_init_c;
					   		init_h = const_init_h;
   		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_c,"Encoder: Standard_input_node: W_x_to_c", 
  							 init_c,
  							 init_h,
  							 unigram,
  							 num_noise_samples,
  				   			 rng,
  				   			 loss_function,
  							 softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices);
							 
		paramGradientCheck(input,output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_o,"Encoder: Standard_input_node: W_x_to_o", 
						 init_c,
						 init_h,
						 unigram,
						 num_noise_samples,
			   			 rng,
			   			 loss_function,
						 softmax_nce_loss,
						 input_sequence_cont_indices,
						 output_sequence_cont_indices);								 									 							 	

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
			 const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
			 const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices){
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
							input_sequence_cont_indices,
							output_sequence_cont_indices);
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
			 const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
			 const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices) {
				Matrix<precision_type,Dynamic,Dynamic> init_c; 
				Matrix<precision_type,Dynamic,Dynamic> init_h;
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
				precision_type perturbation = 1e-3;
		 	    param.changeRandomParam(perturbation, 
		 								rand_row,
		 								rand_col);
		 		//then do an fprop
		 		precision_type before_log_likelihood = 0;	
				//cerr<<"input cols is "<<input.cols()<<endl;					
		 		//fProp(input, output, 0, input.rows()-1, init_c, init_h, sequence_cont_indices);
				//cerr<<"const init c is "<<const_init_c<<endl;
				//cerr<<"const init h is "<<const_init_h<<endl;
				fPropEncoder(input,
							0,
							input.rows()-1,
							init_c,
							init_h,
							input_sequence_cont_indices);	
				//cerr<<"just before passing const init c is "<<const_init_c<<endl;
				//cerr<<"just before passing const init h is "<<const_init_h<<endl;							
			    fPropDecoder(output,
						init_c,
						init_h,
						output_sequence_cont_indices);					
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
		 		precision_type after_log_likelihood = 0;						
		 		//fProp(input,output, 0, input.rows()-1, init_c, init_h, input_sequence_cont_indices);	
				fPropEncoder(input,
							0,
							input.rows()-1,
							init_c,
							init_h,
							input_sequence_cont_indices);	
			    fPropDecoder(output,
						init_c,
						init_h,
						output_sequence_cont_indices);							
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

				precision_type threshold = 1e-03;
				//cerr<<"graves "<<pow(10.0, max(0.0, ceil(log10(min(fabs(param.getGradient(rand_row,
		 		//						rand_col)), fabs((before_log_likelihood-after_log_likelihood)/2e-5)))))-6)<<endl;
				precision_type symmetric_finite_diff_grad = (before_log_likelihood-after_log_likelihood)/(2*perturbation);	
				precision_type graves_threshold = pow(10.0, (double) max(0.0, (double) ceil(log10(min(fabs(param.getGradient(rand_row,
		 								rand_col)), fabs(symmetric_finite_diff_grad)))))-6);
				precision_type gradient_diff =  symmetric_finite_diff_grad - param.getGradient(rand_row,
		 								rand_col);
				precision_type relative_error = fabs(param.getGradient(rand_row,rand_col)-symmetric_finite_diff_grad)/
					(fabs(param.getGradient(rand_row,rand_col)) + fabs(symmetric_finite_diff_grad));
				cerr<<std::setprecision(15);
				if (gradient_diff > threshold || relative_error > threshold) {
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
	