#ifndef NETWORK_H
#define NETWORK_H


#include "neuralClasses.h"
#include "util.h"
#include "graphClasses.h"
#include "SoftmaxLoss.h"

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
		//Matrix<precision_type,Dynamic,Dynamic> d_Err_tPlusOne_to_n_d_c_t,d_Err_tPlusOne_to_n_d_h_t; //Derivatives wrt the future h_t and c_t
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
		vector<Dropout_layer> output_dropout_layers;
		SoftmaxNCELoss nce_loss;

	public:
	    propagator() : minibatch_size(0), 
					encoder_plstm(0), 
					decoder_plstm(0),
					encoder_lstm_nodes(251,LSTM_node<input_node_type>()),
					decoder_lstm_nodes(251,LSTM_node<input_node_type>()),
					encoder_input_nodes(251,input_node_type()),
					decoder_input_nodes(251,input_node_type()),
					num_hidden(0), 
					fixed_partition_function(0), 
					losses(vector<Output_loss_node>(251,Output_loss_node())),
					unif_real(0.0,1.0),
					output_dropout_layers(vector<Dropout_layer>()),
					nce_loss(){ }

	    propagator (model &encoder_lstm, 
					model &decoder_lstm,
					int minibatch_size)
	      : encoder_plstm(&encoder_lstm),
			decoder_plstm(&decoder_lstm),
		 	minibatch_size(minibatch_size),
			output_layer_node(&decoder_lstm.output_layer,minibatch_size),
			encoder_lstm_nodes(vector<LSTM_node<input_node_type> >(251,LSTM_node<input_node_type>(encoder_lstm,minibatch_size))),
			decoder_lstm_nodes(vector<LSTM_node<input_node_type> >(251,LSTM_node<input_node_type>(decoder_lstm,minibatch_size))),
			encoder_input_nodes(vector<input_node_type >(251,input_node_type (dynamic_cast<input_model_type&>(*(encoder_lstm.input)),minibatch_size))),
			decoder_input_nodes(vector<input_node_type >(251,input_node_type (dynamic_cast<input_model_type&>(*(decoder_lstm.input)),minibatch_size))),
			//losses(vector<Matrix<precision_type,Dynamic,Dynamic> >(100,Matrix<precision_type,Dynamic,Dynamic>()))
			losses(vector<Output_loss_node>(251,Output_loss_node())),
			unif_real(0.0,1.0),
			output_dropout_layers(vector<Dropout_layer>()),
			nce_loss()
			{
				resize(minibatch_size);
			}
		
		void resizeOutput(int minibatch_size){
		  output_layer_node.resize(minibatch_size);			
		}
		void resizeRest(int minibatch_size) {
	      //this->minibatch_size = minibatch_size;
		  //CURRENTLY, THE RESIZING IS WRONG FOR SOME OF THE MINIBATCHES
		  //d_Err_tPlusOne_to_n_d_c_t.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  //d_Err_tPlusOne_to_n_d_h_t.setZero(output_layer_node.param->n_inputs(),minibatch_size);
		  scores.resize(output_layer_node.param->n_outputs(),minibatch_size); 
  		  minibatch_weights.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  minibatch_samples.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  //minibatch_samples_no_negative.resize(output_layer_node.param->n_outputs(),minibatch_size);
  		  probs.resize(output_layer_node.param->n_outputs(),minibatch_size);
		  //cerr<<"probs.rows() "<<probs.rows()<<" probs.cols() "<<probs.cols()<<endl;
		  d_Err_t_d_output.resize(output_layer_node.param->n_outputs(),minibatch_size);		  			
		}
		
		void resizeEncoderNodes(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  encoder_lstm_nodes[i].resize(minibatch_size);
  			  //losses[i].setZero(output_layer_node.param->n_inputs(),minibatch_size);
  		  }			
		}
		void resizeDecoderNodes(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  decoder_lstm_nodes[i].resize(minibatch_size);
  		  }			
		}
		
		void resizeLosses(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  losses[i].resize(output_layer_node.param->n_inputs(),minibatch_size);
  			  //losses[i].setZero(output_layer_node.param->n_inputs(),minibatch_size);
  		  }			
		}
		void resizeEncoderInputs(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  encoder_input_nodes[i].resize(minibatch_size);
  			  encoder_lstm_nodes[i].set_input_node(encoder_input_nodes[i]);

  		  }			
		}
		void resizeDecoderInputs(int minibatch_size){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  decoder_input_nodes[i].resize(minibatch_size);
  			  decoder_lstm_nodes[i].set_input_node(decoder_input_nodes[i]);
  		  }			
		}

		void resizeEncoderInputsDropout(int minibatch_size, precision_type dropout_probability){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  encoder_input_nodes[i].resizeDropout(minibatch_size, dropout_probability);
  			  encoder_lstm_nodes[i].set_input_node(encoder_input_nodes[i]);

  		  }			
		}
		void resizeDecoderInputsDropout(int minibatch_size, precision_type dropout_probability){
  		  for (int i=0; i<encoder_lstm_nodes.size(); i++){
  			  decoder_input_nodes[i].resizeDropout(minibatch_size, dropout_probability);
  			  decoder_lstm_nodes[i].set_input_node(decoder_input_nodes[i]);
  		  }			
		}
				
	    void resize(int minibatch_size) {
			this->minibatch_size = minibatch_size;
			resizeOutput(minibatch_size);
			resizeEncoderNodes(minibatch_size);
			resizeDecoderNodes(minibatch_size);
			resizeEncoderInputs(minibatch_size);
			resizeDecoderInputs(minibatch_size);
			resizeLosses(minibatch_size);
			resizeRest(minibatch_size);
		}
		
		void resizeOutputDropoutLayers(int minibatch_size, precision_type dropout_probability) {
			this->output_dropout_layers = vector<Dropout_layer>(251,Dropout_layer(output_layer_node.param->n_inputs(),
																		minibatch_size,
																		1-dropout_probability));
			/*
			for (int i=0; i<this->output_dropout_layers.size(); i++){
				this->output_dropout_layers[i].resize(minibatch_size, 1-dropout_probability);
			}
			*/
		}
		
	    void resizeDropout(int minibatch_size, precision_type dropout_probability) {
			this->minibatch_size = minibatch_size;
			resizeOutput(minibatch_size);
			resizeEncoderNodes(minibatch_size);
			resizeDecoderNodes(minibatch_size);
			resizeEncoderInputsDropout(minibatch_size, dropout_probability);
			resizeDecoderInputsDropout(minibatch_size, dropout_probability);
			resizeLosses(minibatch_size);
			resizeRest(minibatch_size);
			resizeOutputDropoutLayers(minibatch_size, dropout_probability);
		}		

		
		//Resizing some of the NCE mibatch matrices
		//template <typename X> 
		//I should template multinomial<data_size_t>
		void resizeNCE(int num_noise_samples, precision_type fixed_partition_function, multinomial<data_size_t> &unigram){
			minibatch_weights.setZero(num_noise_samples+1,minibatch_size);
			minibatch_samples.setZero(num_noise_samples+1,minibatch_size);
			minibatch_samples_no_negative.setZero(num_noise_samples+1,minibatch_size);
			scores.setZero(num_noise_samples+1,minibatch_size);
			probs.setZero(num_noise_samples+1,minibatch_size);
			//cerr<<"Size of scores is "<<scores.cols()<<" "<<scores.rows()<<endl;
			this->fixed_partition_function = fixed_partition_function;
			this->nce_loss.set_unigram(&unigram);
			d_Err_t_d_output.resize(num_noise_samples+1,minibatch_size);	
			//this->nce_loss = SoftmaxNCELoss(unigram);
		}
	    void resize() { resize(minibatch_size); }
		


		//Both the input and the output sentences are columns. Even ifs a minibatch of sentences, each sentence is a column
	    template <typename DerivedOutput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropDecoder(const MatrixBase<DerivedOutput> &output_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	    {

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
			for (int i=0; i<output_sent_len; i++){
				//cerr<<"i is"<<i<<endl;
				if (i==0) {
					//cerr<<"Current c is "<<current_c<<endl;
					//NEED TO CHECK THIS!! YOU SHOULD JUST TAKE THE HIDDEN STATE FROM THE LAST POSITION
					//decoder_lstm_nodes[i].copyToHiddenStates(const_current_h,const_current_c);//,sequence_cont_indices.row(i));
					//cerr<<"const_current_c"<<const_current_c<<endl;
					//cerr<<"const_current_h"<<const_current_h<<endl;
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
				/*
				//encoder_lstm_nodes.fProp();
				cerr<<"decoder_lstm_nodes[i].h_t_minus_one "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
				cerr<<"decoder_lstm_nodes[i].c_t_minus_one "<<decoder_lstm_nodes[i].c_t_minus_one<<endl;
				cerr<<"decoder_lstm_nodes["<<i<<"].h_t_minus_one "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
				cerr<<"decoder_lstm_nodes["<<i<<"].h_t "<<decoder_lstm_nodes[i].h_t<<endl;
				*/
			}			

	    }
		
		//Both the input and the output sentences are columns. Even ifs a minibatch of sentences, each sentence is a column
	    template <typename DerivedOutput, typename DerivedH, typename DerivedC, typename DerivedS, typename Engine>
	    void fPropDecoderDropout(const MatrixBase<DerivedOutput> &output_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices,
				Engine &eng)
	    {

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			//int sent_len = input_data.rows();
			int output_sent_len = output_data.rows();
			//Matrix<precision_type,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = output_data.cols();

			//Going over the output sentence to generate the hidden states
			for (int i=0; i<output_sent_len; i++){
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
																
					decoder_lstm_nodes[i].fPropDropout(output_data.row(i), eng);//,	
					//cerr<<"output data is "<<output_data.row(i)<<endl;
										//current_c,
										//current_h);
				} else {

					
					decoder_lstm_nodes[i].filterStatesAndErrors(decoder_lstm_nodes[i-1].h_t,
																decoder_lstm_nodes[i-1].c_t,
																decoder_lstm_nodes[i].h_t_minus_one,
																decoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i));						
																
					decoder_lstm_nodes[i].fPropDropout(output_data.row(i), eng);//,
					

				}

			}			

	    }
				
		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS>
	    void fPropEncoder(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
	    {
			//cerr<<"input_data.rows() "<<input_data.rows()<<endl;
			//cerr<<"input_data "<<input_data<<endl;
			UNCONST(DerivedC, const_current_c, current_c);
			UNCONST(DerivedH, const_current_h, current_h);

			//The data is just an eigen matrix. Now I have to go over each column and do fProp
			int sent_len = input_data.rows();
			//int output_sent_len = output_data.rows();
			//Matrix<precision_type,Dynamic,Dynamic> c_0,h_0,c_1,h_1;
			int current_minibatch_size = input_data.cols();

			for (int i=0; i<sent_len; i++){
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
					//cerr<<"encoder_lstm_nodes["<<i<<"].h_t_minus_one 					"<<encoder_lstm_nodes[i].h_t_minus_one<<endl;											
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,	
										//current_c,
										//current_h);
				} else {

					//encoder_lstm_nodes[i].copyToHiddenStates(encoder_lstm_nodes[i-1].h_t,encoder_lstm_nodes[i-1].c_t);//,sequence_cont_ind					ices.row(i));
					
					encoder_lstm_nodes[i].filterStatesAndErrors(encoder_lstm_nodes[i-1].h_t,
																encoder_lstm_nodes[i-1].c_t,
																encoder_lstm_nodes[i].h_t_minus_one,
																encoder_lstm_nodes[i].c_t_minus_one,
																sequence_cont_indices.row(i-1)); //THIS IS JUST A TEMPORARY FIX THAT ASSUMES THE INITIAL HIDDEN STATES ARE 0
					//cerr<<"encoder_lstm_nodes["<<i<<"].h_t_minus_one 					"<<encoder_lstm_nodes[i].h_t_minus_one<<endl;																			
					encoder_lstm_nodes[i].fProp(input_data.row(i));//,

				}
				//encoder_lstm_nodes.fProp();
				//cerr<<"encoder_lstm_nodes["<<i<<"].h_t "<<encoder_lstm_nodes[i].h_t<<endl;
				
			}
			//Copying the cell and hidden states if the sequence continuation vectors say so	
			//cerr<<"end pos is"<<end_pos<<endl;
			current_c = encoder_lstm_nodes[sent_len-1].c_t;
			current_h = encoder_lstm_nodes[sent_len-1].h_t;


	    }

		template <typename DerivedInput, typename DerivedH, typename DerivedC, typename DerivedS, typename Engine>
	    void fPropEncoderDropout(const MatrixBase<DerivedInput> &input_data,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				const Eigen::ArrayBase<DerivedS> &sequence_cont_indices,
				Engine &eng)
	    {
			//cerr<<"In fprop encoder dropout"<<endl;
			//cerr<<"input_data.rows() "<<input_data.rows()<<endl;
			//cerr<<"input_data "<<input_data<<endl;
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
			for (int i=0; i<sent_len; i++){
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
					encoder_lstm_nodes[i].fPropDropout(input_data.row(i), eng);//,	
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
					encoder_lstm_nodes[i].fPropDropout(input_data.row(i), eng);//,

				}
				//encoder_lstm_nodes.fProp();
				//cerr<<"encoder_lstm_nodes["<<i<<"].h_t "<<encoder_lstm_nodes[i].h_t<<endl;
				
			}
			//Copying the cell and hidden states if the sequence continuation vectors say so	
			//cerr<<"end pos is"<<end_pos<<endl;
			current_c = encoder_lstm_nodes[sent_len-1].c_t;
			current_h = encoder_lstm_nodes[sent_len-1].h_t;


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
				vocabulary &decoder_input_vocab,
				vocabulary &decoder_output_vocab,
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
							//Because the decoder output vocabulary is different from
							//the decoder input vocabulary, the max index is from the 
							//decoder output vocabulary. However, when we populate 
							//predicted_output, we have to use the decoder input vocabulary.
							predicted_sequence[index].push_back(max_index);
							predicted_output(i+1,index) = decoder_input_vocab.lookup_word(decoder_output_vocab.get_word(max_index));
						} else {
							//predicted_output(i+1,index) = output_end_symbol;// This does not matter. It could be anything. 
																			// I guess its best to make it 0
							predicted_output(i+1,index) = 0;
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
				vocabulary &decoder_input_vocab,
				vocabulary &decoder_output_vocab,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				vector<k_best_seq_item> &final_k_best_seq_list,
				const int output_start_symbol,
				const int output_end_symbol,
				const int beam_size) {
					int k = beam_size;
					int current_beam_size, previous_beam_size;
					current_beam_size = previous_beam_size = k;
					//int current_minibatch_size = input_data.cols();
					int current_minibatch_size = 1;
					//cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
					Matrix<int,Dynamic,Dynamic> predicted_output;
					predicted_output.resize(1,1); //For now, I'm processing one sentence at a time. 
												  //In the beginning, there is only one symbol,<s>. 
												  //After that, we will have k items in predicted output
					//predicted_output.resize(1,current_minibatch_size);
					//predicted_output.fill(output_start_symbol);
					predicted_output.row(0).fill(output_start_symbol);
					//predicted_output(0,0) = output_start_symbol;
					//UNCONST(DerivedC, const_current_c, current_c);
					//UNCONST(DerivedH, const_current_h, current_h);	
					//vector<int> current_words(output_start_symbol,current_minibatch_size);
					//predicted_output = resize(1,current_minibatch_size);
					
					//predicted_output
				//cerr<<"predicted_output	is "<<predicted_output<<endl;
				//First generate k items to populate the beam
				decoder_lstm_nodes[0].copyToHiddenStates(const_current_h.leftCols(1),
									const_current_c.leftCols(1));
				decoder_lstm_nodes[0].fProp(predicted_output.row(0));
				//Get the k-best items from the list first
				//Note that if vocab size < k, then the initial k-best list will have less than
				//k items.
				vector<beam_item> initial_k_best_list;
				output_layer_node.param->fProp(decoder_lstm_nodes[0].h_t.leftCols(1), 
									scores.leftCols(1));
		        SoftmaxLogLoss().computeProbs(scores.leftCols(1), 
		                   predicted_output.row(0), 
		                   probs);					
				vector<k_best_seq_item> k_best_seq_list;		   					
				getKBest(probs.leftCols(1), initial_k_best_list, k_best_seq_list, k);
				//Initial k-best list might have less than k items.
				current_beam_size = initial_k_best_list.size();
				//cerr<<"probs.leftCols(1) "<<probs.leftCols(1)<<endl;
				//Now populate the k-best sequences with the initial k best list
				
				assert(initial_k_best_list.size() <= k);
				for (int i=0; i<initial_k_best_list.size(); i++)	{
					k_best_seq_item seq_item;
					seq_item.seq.push_back(initial_k_best_list.at(i).row); //Row indicates the word index in the probability matrix
					//cerr<<"initial_k_best_list.at(i).row "<<initial_k_best_list.at(i).row<<endl;
					seq_item.value = initial_k_best_list.at(i).value;
					//cerr<<"seq_item.value "<<seq_item.value<<endl;					
					k_best_seq_list.push_back(seq_item);
				}
				//vector<k_best_seq_item> final_k_best_seq_list;
				//Now create a new predicted output from the k_best list and also transfer the hidden states
				//For the 1st position, all the hidden states will come from the same hidden state of just seeing 
				//<s>
				vector<int> k_best_state_copy_indices = vector<int>(initial_k_best_list.size(),0);
				//Resizing predicted output to now contain 12 items in the k-best list
				predicted_output.resize(101,k); // I can produce at most 100 output symbols
				for (int i=0; i<initial_k_best_list.size();  i++){
					predicted_output(1,i) = initial_k_best_list.at(i).row;
					//cerr<<"initial_k_best_list.at(i).row "<<initial_k_best_list.at(i).row<<endl;
					string string_output_word = decoder_output_vocab.get_word(initial_k_best_list.at(i).row);
					//cerr<<"string_output_word "<<string_output_word<<endl;
					predicted_output(1,i) = decoder_input_vocab.lookup_word(string_output_word);
					//cerr<<"predicted_output(1,i) "<<predicted_output(1,i)<<endl;
				}
				//getchar();
				/*
				decoder_lstm_nodes[1].copyKBestHiddenStates(decoder_lstm_nodes[0].h_t,
									  decoder_lstm_nodes[0].c_t,
									  decoder_lstm_nodes[1].h_t,
									  decoder_lstm_nodes[1].c_t,
									  k_best_state_copy_indices);
				*/

				for (int i=1; i<100 ; i++){
					//previous_beam_size = current_beam_size;
					//cerr<<"Predicted output is "<<predicted_output.row(i)<<endl;
					//current_minibatch_size = current_words.size();
					//predicted_output = Map< Matrix<int,Dynamic,Dynamic> >(current_words.data(), 1, current_minibatch_size);
					//cerr<<"i is "<<i<<endl;
					//cerr<<"predicted output is "<<predicted_output.row(i);
					//Copying the decoder hidden states according to the k best copy indices
					/*
					for (int copy_index=0; copy_index<k_best_state_copy_indices.size();copy_index++){
						cerr<<"k_best_state_copy_indices.at(copy_index)" <<k_best_state_copy_indices.at(copy_index)<<endl;
					}
					*/
					decoder_lstm_nodes[i].copyKBestHiddenStates(decoder_lstm_nodes[i-1].h_t,
										  decoder_lstm_nodes[i-1].c_t,
										  decoder_lstm_nodes[i].h_t_minus_one,
										  decoder_lstm_nodes[i].c_t_minus_one,
										  k_best_state_copy_indices);	
	  				//cerr<<"decoder_lstm_nodes[i].h_t "<<decoder_lstm_nodes[i].h_t<<endl;
	  				//cerr<<"decoder_lstm_nodes[i].c_t "<<decoder_lstm_nodes[i].c_t<<endl;	
					//cerr<<"predicted_output.row(i) "<<cerr<<predicted_output.row(i)<<endl;
					decoder_lstm_nodes[i].fProp(predicted_output.row(i).leftCols(current_beam_size));//,
					//cerr<<"predicted_output.row(i)"
					//	<<predicted_output.row(i)<<endl;
										//(encoder_lstm_nodes[i-1].c_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix(),
										//	(encoder_lstm_nodes[i-1].h_t.array().rowwise()*sequence_cont_indices.row(i-1)).matrix());

					//cerr<<"ht is   "<<decoder_lstm_nodes[i].h_t<<endl;
					//cerr<<"ht -1 is "<<decoder_lstm_nodes[i].h_t_minus_one<<endl;
					//cerr<<"Current beam size is "<<current_beam_size<<endl;
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_beam_size), 
										scores.leftCols(current_beam_size));
					//then compute the log loss of the objective
					//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
					//cerr<<"Score is"<<endl;
					//cerr<<scores<<endl;
	
			        //precision_type minibatch_log_likelihood;
			        start_timer(5);
			        SoftmaxLogLoss().computeProbs(scores.leftCols(current_beam_size), 
			                   predicted_output.row(i), 
			                   probs);	
					//cerr<<"probs is "<<probs<<endl;
					vector<beam_item> k_best_list;
					//Extracting at most 2k best items because we could have k
					//items with end symbols
					//Now that we have the k-best list, we add to the k_best_seq_list and remove 
					//the previous items.					
					getKBest(probs.leftCols(current_beam_size), k_best_list, k_best_seq_list, 2*k);
					//cerr<<"The k best list size was "<<k_best_list.size()<<endl;
					assert(k_best_list.size() <= 2*k);
					//cerr<<"probs.leftCols(k) "<<probs.leftCols(k)<<endl;
					int new_k = 0;
					k_best_state_copy_indices.clear();
					//cerr<<"K best seq list size is "<<k_best_seq_list.size()<<endl;
					//cerr<<"K best list size is "<<k_best_list.size()<<endl;
					for (int item_index=0; new_k<k && item_index < k_best_list.size() ; item_index++){
						//cerr<<"item_index"<<item_index<<endl;
						//cerr<<"k_best_list.at(item_index).value "<<k_best_list.at(item_index).value<<endl;
						k_best_seq_item seq_item;
						int prev_k_best_seq_item_index = k_best_list.at(item_index).col;
						int word_index = k_best_list.at(item_index).row;
						
						//cerr<<"Word index in k_best_item "<<item_index<<" is "<<word_index<<endl;
						//cerr<<"prev_k_best_seq_item_index "<<prev_k_best_seq_item_index<<endl;
						//cerr<<"k_best_seq_list.at(prev_k_best_seq_item_index).value "<<
						//		k_best_seq_list.at(prev_k_best_seq_item_index).value<<endl;
						seq_item.seq = k_best_seq_list.at(prev_k_best_seq_item_index).seq;
						
						seq_item.seq.push_back(word_index);
						//seq_item.value = k_best_seq_list.at(prev_k_best_seq_item_index).value +
						//				k_best_list.at(item_index).value;
						seq_item.value = k_best_list.at(item_index).value;
						/*
						for (int seq_item_index=0; seq_item_index<seq_item.seq.size(); seq_item_index++){
							cerr<<"seq_item.seq["<<seq_item_index<<"] "<<seq_item.seq.at(seq_item_index)<<endl;
						}
						cerr<<"seq_item.value "<<seq_item.value<<endl;
						getchar();
						*/
						if (word_index != output_end_symbol){
							k_best_seq_list.push_back(seq_item);
							//The hidden state to be transmitted to the next LSTM block is 
							//the one with the index of the previous k_best seq item 
							k_best_state_copy_indices.push_back(prev_k_best_seq_item_index);
							
							//cerr<<"predicted output rows "<<predicted_output.rows()<<" predicted_output cols: "<<predicted_output.cols()<<endl;
							//cerr<<"i is "<<i<<endl;
							//cerr<<"new k is "<<new_k<<endl;
							//We need to convert the output word index into the input word index
							string string_output_word = decoder_output_vocab.get_word(word_index);
							//cerr<<"string_output_word "<<string_output_word<<endl;
							predicted_output(i+1,new_k) = decoder_input_vocab.lookup_word(string_output_word);							
							//predicted_output(i+1,new_k) = word_index;
							
							//cerr<<" predicted_output(i+1,new_k) is "<<predicted_output(i+1,new_k)<<endl;
							new_k++;
						} else {
							final_k_best_seq_list.push_back(seq_item);
							/*
							cerr<<"just inserted "<<endl;
							for (int seq_item_index=0; 
								seq_item_index<final_k_best_seq_list.at(final_k_best_seq_list.size()-1).seq.size(); 
									seq_item_index++){
								cerr<<"seq_item.seq["<<seq_item_index<<"] "<<
									final_k_best_seq_list.at(final_k_best_seq_list.size()-1).seq.at(seq_item_index)<<endl;
							}		
							getchar();	
							*/				
							//Should I make the k-best list smaller now ? Not sure. 
							//new_k--;
						}
					}
					//k = new_k;
					//The k_best_seq_list has expanded now and the previou previous_beam_size-items should be deleted
					previous_beam_size = current_beam_size;
					current_beam_size = new_k;
					assert (k_best_seq_list.size() <= 2*k);
					k_best_seq_list.erase(k_best_seq_list.begin(), k_best_seq_list.begin()+previous_beam_size);
					//cerr<<"After erasing the k best seq list size is "<<k_best_seq_list.size()<<endl;
					//assert(k_best_list.size() == k);
				}
				//cerr<<"k best seq list size is "<<k_best_seq_list.size()<<endl;
				//Now adding the k_best_seq list to the final list
				for (int item_index=0; item_index<k_best_seq_list.size(); item_index++){
					final_k_best_seq_list.push_back(k_best_seq_list.at(item_index));
				}
				//cerr<<"Final k best seq list size is "<<final_k_best_seq_list.size()<<endl;
				//First soring the k_best_list
				//Getting the average symbol probability
				std::make_heap(final_k_best_seq_list.begin(), final_k_best_seq_list.end(), comparator<k_best_seq_item>());
				std::sort_heap(final_k_best_seq_list.begin(), final_k_best_seq_list.end(), comparator<k_best_seq_item>());
				//Now populating the predicted sequence

		}

		//currently only generate one output at a time
		template <typename DerivedInput,typename DerivedH, typename DerivedC>
		void generateStochasticOutput(const MatrixBase<DerivedInput> &input_data,
				vocabulary &decoder_input_vocab,
				vocabulary &decoder_output_vocab,
				const MatrixBase<DerivedC> &const_current_c,
				const MatrixBase<DerivedH> &const_current_h,
				vector<vector<int> > &predicted_sequence,
				int output_start_symbol,
				int output_end_symbol,
				boost::random::mt19937 &eng) {
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
							//predicted_output(i+1,index) = rand_index;
							predicted_output(i+1,index) = decoder_input_vocab.lookup_word(decoder_output_vocab.get_word(rand_index));
						} else {
							//predicted_output(i+1,index) = output_end_symbol;
							predicted_output(i+1,index) = 0;
						}
						//cerr<<"remaining live words are"<<remaining_live_words<<endl;
					}
	   				
				}

		}
		
		template<typename Derived>
		void printHiddenStates(const MatrixBase<Derived> &matrix, const string &type){
			for (int i=0; i<matrix.cols(); i++){
				cerr<<"state "<<type<<":"<<i<<" has norm "<<matrix.col(i).norm()<<endl;
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
			 boost::random::mt19937 &rng) {
			 //SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss){
	 			int current_minibatch_size = output.cols();
	 			//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
	 			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
	 			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
	 			//same dimension in and LSTM. this might not be a good idea
	 			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
	 			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
				//cerr<<"output is "<<output<<endl;
	 			int sent_len = output.rows(); 
	 			//precision_type log_likelihood = 0.;
				cerr<<"Sent len is "<<sent_len<<endl;
	 			for (int i=sent_len-1; i>=0; i--) {
	 				//cerr<<"i is "<<i<<endl;
					precision_type minibatch_log_likelihood;
					/*
					string state_type = "h_t";
					printHiddenStates(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), state_type);
					state_type = "c_t";
					printHiddenStates(decoder_lstm_nodes[i].c_t.leftCols(current_minibatch_size), state_type);
					//getchar();
					*/
	 				if (loss_function == LogLoss) {
	 					//First doing fProp for the output layer
	 					//The number of columns in scores will be the current minibatch size
						//cerr<<"ht going into loss"<<decoder_lstm_nodes[i-1].h_t.leftCols(current_minibatch_size)<<endl;
	 					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), scores);
	 					//cerr<<"scores.rows "<<scores.rows()<<" scores cols "<<scores.cols()<<endl;
	 					//then compute the log loss of the objective
	 					//cerr<<"probs dimension is "<<probs.rows()<<" "<<probs.cols()<<endl;
	 					//cerr<<"Score is"<<endl;
	 					//cerr<<scores<<endl;
						//cerr<<"output.row(i) "<<output.row(i)<<endl;
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
						//cerr<<"d_Err_t_d_output.leftCols(current_minibatch_size)"<<d_Err_t_d_output.leftCols(current_minibatch_size)<<endl;
	 			        stop_timer(6);
				

	 					//Oh wow, i have not even been updating the gradient of the output embeddings
	 					//Now computing the derivative of the output layer
	 					//The number of colums in output_layer_node.bProp_matrix will be the current minibatch size
	 	   		        output_layer_node.param->bProp(d_Err_t_d_output.leftCols(current_minibatch_size),
										losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
									   //output_layer_node.bProp_matrix.leftCols(current_minibatch_size));	
	 					//cerr<<"ouput layer bprop matrix rows"<<output_layer_node.bProp_matrix.rows()<<" cols"<<output_layer_node.bProp_matrix.cols()<<endl;
	 					//cerr<<"output_layer_node.bProp_matrix"<<output_layer_node.bProp_matrix<<endl;
	 					//cerr<<"Dimensions if d_Err_t_d_output "<<d_Err_t_d_output.rows()<<","<<d_Err_t_d_output.cols()<<endl;
	 					//cerr<<"output_layer_node.bProp_matrix "<<output_layer_node.bProp_matrix<<endl;
	 	   		        output_layer_node.param->updateGradient(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size),
	 	   						       d_Err_t_d_output.leftCols(current_minibatch_size));									   	 		   
	 					//cerr<<" i is "<<i<<endl;
	 					//cerr<<"backprop matrix is "<<output_layer_node.bProp_matrix<<endl;		   	 					
	 				} else if (loss_function == NCELoss){
						//cerr<<"NOT IMPLEMENTED"<<endl;
						//exit(1);
						
						generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
						//cerr<<"minibatch_samples.rows() "<<minibatch_samples.rows()<<" minibatch_samples.cols() "<<minibatch_samples.cols()<<endl;
						//cerr<<"output "<<output<<endl;
						//cerr<<" output.row(0)" <<output.row(0)<<endl;
						//minibatch_samples.row(0) = output.row(0); //The first item is the minbiatch instance
						//cerr<<"minibatch_samples "<<minibatch_samples<<endl;
						//cerr<<"output.row(0) "<<output.row(0)<<endl;
						//getchar();
						minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);
						//cerr<<"minibatch_samples "<<minibatch_samples<<endl;
						//getchar();
						//preparing the minbatch with no zeros for fprop nce
						minibatch_samples_no_negative = minibatch_samples;
						for (int col=0; col<current_minibatch_size; col++){ 
							if(minibatch_samples_no_negative(0,col) == -1){
								minibatch_samples_no_negative(0,col) = 0;
							}
						}
						/*
						int num_noise_samples = minibatch_samples.rows()-1;
						//std::cerr<<"num noise samples are "<<num_noise_samples<<std::endl;
						precision_type log_num_noise_samples = std::log(num_noise_samples);	
						for (int row=0; row<minibatch_samples_no_negative.rows(); row++) {
							for (int col =0; col<minibatch_samples_no_negative.row(row).cols(); col++){
								int sample = minibatch_samples_no_negative(row,col);
								cerr<<" minibatch sample "<<sample<<" \nand prob is "<<
									log_num_noise_samples + unigram->logprob(sample)<<endl;
							}
						}
						*/
						//cerr<<"minibatch_samples_no_negative "<<minibatch_samples_no_negative<<endl;
						//getchar();
						//cerr<<"Score is "<<scores<<endl;
						scores.setZero();
						output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
														minibatch_samples_no_negative.leftCols(current_minibatch_size),
														scores);
						//cerr<<"this->fixed_partition_function "<<this->fixed_partition_function<<endl;
						//cerr<<"minibatch samples "<<minibatch_samples<<endl;
						//getchar();
						nce_loss.fProp(scores, 
	       			 				  minibatch_samples,
	       						   	  probs, 
		   						  	  minibatch_log_likelihood,
									  this->fixed_partition_function);
						log_likelihood += minibatch_log_likelihood;
						//cerr<<"probs.leftCols(current_minibatch_size) "<<probs.leftCols(current_minibatch_size)<<endl;
							
						nce_loss.bProp(probs.leftCols(current_minibatch_size),
										d_Err_t_d_output);
						//cerr<<"d_Err_t_d_output "<<d_Err_t_d_output<<endl;
			 	   		output_layer_node.param->bProp(minibatch_samples_no_negative.leftCols(current_minibatch_size),
			 	   									d_Err_t_d_output.leftCols(current_minibatch_size),
													losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
						//cerr<<"d_Err_t_d_output.leftCols(current_minibatch_size) "<<
						//		d_Err_t_d_output.leftCols(current_minibatch_size)<<endl;
						//cerr<<"losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size) "<<losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size)<<endl;
						//getchar();
			 	   		output_layer_node.param->updateGradient(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size),
													     minibatch_samples_no_negative.leftCols(current_minibatch_size),
													     d_Err_t_d_output.leftCols(current_minibatch_size));
	 				}
					

	 			}
	 			//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
	 			//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
	 			//cerr<<"The cross entropy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
	 			//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;		  		 	
		}
		
		template <typename Derived, typename Engine> 
		void generateSamples(MatrixBase<Derived> const &minibatch, multinomial<data_size_t> &unigram, Engine &eng){	
			
			UNCONST(Derived, minibatch, my_minibatch);
			#ifdef SHARE_SAMPLES
			
				for (int row=0; row<my_minibatch.rows(); row++){
					int sample = unigram.sample(eng);
					my_minibatch.row(row).fill(sample);
				}	
				//cerr<<"my_minibatch samples"<<my_minibatch<<endl;
				//getchar();
			#else 		
				for (int row=0; row<my_minibatch.rows(); row++){
					for (int col=0; col<my_minibatch.cols(); col++){
						my_minibatch(row,col) = unigram.sample(eng);
					}
				}
			#endif	
			//cerr<<"samples are "<<my_minibatch<<endl;
			//getchar();
		}
		//Computing losses separately. Makes more sense because some LSTM units might not output units but will be receiving 
		//losses from the next layer
	    template <typename DerivedOut, typename data_type> //, typename DerivedC, typename DerivedH, typename DerivedS>
		void computeLossesDropout(const MatrixBase<DerivedOut> &output,
			 precision_type &log_likelihood,
			 bool gradient_check,
			 bool norm_clipping,
			 loss_function_type loss_function,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng) {
			 //SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss){
	 			int current_minibatch_size = output.cols();
	 			//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
	 			Matrix<precision_type,Dynamic,Dynamic> dummy_zero,dummy_ones;
	 			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
	 			//same dimension in and LSTM. this might not be a good idea
	 			dummy_zero.setZero(output_layer_node.param->n_inputs(),minibatch_size);
	 			dummy_ones.setOnes(output_layer_node.param->n_inputs(),minibatch_size);
			
	 			int sent_len = output.rows(); 
	 			//precision_type log_likelihood = 0.;
			
	 			for (int i=sent_len-1; i>=0; i--) {
	 				//cerr<<"i in losses is "<<i<<endl;
					precision_type minibatch_log_likelihood;
	 				if (loss_function == LogLoss) {
						//Applying dropout to the output layer
						output_dropout_layers[i].fProp(decoder_lstm_nodes[i].h_t,rng);
	 					//First doing fProp for the output layer
	 					//The number of columns in scores will be the current minibatch size
						
	 					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), scores);

				
	 			        
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
						//cerr<<"d_Err_t_d_output.leftCols(current_minibatch_size)"<<d_Err_t_d_output.leftCols(current_minibatch_size)<<endl;
	 			        stop_timer(6);
				


	 					//Now computing the derivative of the output layer
	 					//The number of colums in output_layer_node.bProp_matrix will be the current minibatch size
	 	   		        output_layer_node.param->bProp(d_Err_t_d_output.leftCols(current_minibatch_size),
										losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
									   //output_layer_node.bProp_matrix.leftCols(current_minibatch_size));	

	 	   		        output_layer_node.param->updateGradient(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size),
	 	   						       d_Err_t_d_output.leftCols(current_minibatch_size));	
														   	 		   
						//Applying dropout to the backpropagated loss
						//output_dropout_layers[i].bProp(losses[i].d_Err_t_d_h_t);
	 				} else if (loss_function == NCELoss){
						//cerr<<"NOT IMPLEMENTED"<<endl;
						//exit(1);
						output_dropout_layers[i].fProp(decoder_lstm_nodes[i].h_t,rng);
						generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
						//cerr<<"minibatch_samples.rows() "<<minibatch_samples.rows()<<" minibatch_samples.cols() "<<minibatch_samples.cols()<<endl;
						//cerr<<"output "<<output<<endl;
						//cerr<<" output.row(0)" <<output.row(0)<<endl;
						//minibatch_samples.row(0) = output.row(0); //The first item is the minbiatch instance
						//cerr<<"minibatch_samples "<<minibatch_samples<<endl;
						//cerr<<"output.row(0) "<<output.row(0)<<endl;
						//getchar();
						minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);
						//cerr<<"minibatch_samples "<<minibatch_samples<<endl;
						//getchar();
						//preparing the minbatch with no zeros for fprop nce
						minibatch_samples_no_negative = minibatch_samples;
						for (int col=0; col<current_minibatch_size; col++){ 
							if(minibatch_samples_no_negative(0,col) == -1){
								minibatch_samples_no_negative(0,col) = 0;
							}
						}
						//cerr<<"minibatch_samples_no_negative "<<minibatch_samples_no_negative<<endl;
						//getchar();
						//cerr<<"Score is "<<scores<<endl;
						scores.setZero();
						output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
														minibatch_samples_no_negative.leftCols(current_minibatch_size),
														scores);
						//cerr<<"this->fixed_partition_function "<<this->fixed_partition_function<<endl;
						nce_loss.fProp(scores, 
	       			 				  minibatch_samples,
	       						   	  probs, 
		   						  	  minibatch_log_likelihood,
									  this->fixed_partition_function);
						log_likelihood += minibatch_log_likelihood;
						//cerr<<"probs.leftCols(current_minibatch_size) "<<probs.leftCols(current_minibatch_size)<<endl;
							
						nce_loss.bProp(probs.leftCols(current_minibatch_size),
										d_Err_t_d_output);
						//cerr<<"d_Err_t_d_output "<<d_Err_t_d_output<<endl;
			 	   		output_layer_node.param->bProp(minibatch_samples_no_negative.leftCols(current_minibatch_size),
			 	   									d_Err_t_d_output.leftCols(current_minibatch_size),
													losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size));
						//cerr<<"d_Err_t_d_output.leftCols(current_minibatch_size) "<<d_Err_t_d_output.leftCols(current_minibatch_size)<<endl;
						//cerr<<"losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size) "<<losses[i].d_Err_t_d_h_t.leftCols(current_minibatch_size)<<endl;
						//getchar();
			 	   		output_layer_node.param->updateGradient(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size),
													     minibatch_samples_no_negative.leftCols(current_minibatch_size),
													     d_Err_t_d_output.leftCols(current_minibatch_size));	
						//output_dropout_layers[i].bProp(losses[i].d_Err_t_d_h_t);								 					
	 				}
					//Applying dropout
					output_dropout_layers[i].bProp(losses[i].d_Err_t_d_h_t);		
		   
	 			}
 	
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
			for (int i=output_sent_len-1; i>=0; i--) {
				//cerr<<"i in decoder bprop is "<<i<<endl;
				//getchar();
				// Now calling backprop for the LSTM nodes
				//cerr<<"losses[i].d_Err_t_d_h_t "<<losses[i].d_Err_t_d_h_t<<endl;
				//getchar();
				if (i==0 && output_sent_len-1 > 0) {
				    decoder_lstm_nodes[i].bProp(output_data.row(i),
							   //init_h,
				   			   //init_c,
								losses[i].d_Err_t_d_h_t,
							   //output_layer_node.bProp_matrix,
				   			   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
							   gradient_check,
							   norm_clipping);	
				} else if (i == output_sent_len-1) {	

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
	    template <typename DerivedIn, typename DerivedOut> //, typename DerivedC, typename DerivedH, typename DerivedS>
	    void bPropDecoderDropout(const MatrixBase<DerivedIn> &input_data,
				const MatrixBase<DerivedOut> &output_data,
			 bool gradient_check,
			 bool norm_clipping)
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
			for (int i=output_sent_len-1; i>=0; i--) {
				//cerr<<"i in backprop decoder dropout is "<<i<<endl;
				if (i==0 && output_sent_len-1 > 0) {
					
				    decoder_lstm_nodes[i].bPropDropout(output_data.row(i),
							   //init_h,
				   			   //init_c,
								losses[i].d_Err_t_d_h_t,
							   //output_layer_node.bProp_matrix,
				   			   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_c_tMinusOne,
							   decoder_lstm_nodes[i+1].d_Err_t_to_n_d_h_tMinusOne,
							   gradient_check,
							   norm_clipping);	
				} else if (i == output_sent_len-1) {	


				    decoder_lstm_nodes[i].bPropDropout(output_data.row(i),
							   losses[i].d_Err_t_d_h_t,
							   //output_layer_node.bProp_matrix,
				   			   dummy_zero, //for the last lstm node, I just need to supply a bunch of zeros as the gradient of the future
				   			   dummy_zero,
							   gradient_check,
							   norm_clipping);
		
				} else if (i > 0) {
					
				    decoder_lstm_nodes[i].bPropDropout(output_data.row(i),

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

  // Dense version (for standard log-likelihood)
  template <typename DerivedIn, typename DerivedS> //, typename DerivedC, typename DerivedH, typename DerivedS>
  void bPropEncoderDropout(const MatrixBase<DerivedIn> &input_data,
	 bool gradient_check,
	 bool norm_clipping,
	 const Eigen::ArrayBase<DerivedS> &sequence_cont_indices)
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
		    encoder_lstm_nodes[i].bPropDropout(input_data.row(i),
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
																	
		    encoder_lstm_nodes[i].bPropDropout(input_data.row(i),
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
	 precision_type getGradSqdNorm(precision_type &grad_norm,
	 								loss_function_type loss_function,
	 								bool arg_run_lm){
 		//cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
 		//cerr<<"updating params "<<endl;
 		//First compute the norm of the gradients for norm scaling
 		//precision_type grad_norm = 0;
										
 		grad_norm += decoder_plstm->output_layer.getGradSqdNorm();
		//cerr<<"Output grad squared norm is "<<grad_norm<<endl;
 		if (arg_run_lm == 0) {
 		    grad_norm += encoder_plstm->getGradSqdNorm();
 		}
		
 		grad_norm += decoder_plstm->getGradSqdNorm();
		return(grad_norm);
	 
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
		//First compute the norm of the gradients for norm scaling
		if (loss_function == LogLoss){
			//cerr<<"updating log output layer"<<endl;
			decoder_plstm->output_layer.updateParams(learning_rate,
							current_minibatch_size,
		  					momentum,
		  					L2_reg,
							norm_clipping,
							norm_threshold);			
		} else if (loss_function == NCELoss){
			//cerr<<"NOT IMPLEMENTED"<<endl;
			//exit(1);
			//cerr<<"updating nce output layer"<<endl;
			decoder_plstm->output_layer.updateParamsNCE(learning_rate,
								current_minibatch_size,
								momentum,
								L2_reg,
								norm_clipping,
								norm_threshold);	
		} else {
			cerr<<loss_function<<" is an invalid loss function type"<<endl;
			exit(0);
		}



		if (arg_run_lm == 0) {
			//cerr<<"updating encoder params"<<endl;
		    encoder_plstm->updateParams(learning_rate,
												current_minibatch_size,
												momentum,
												L2_reg,
												norm_clipping,
												norm_threshold);
		}
		
		//cerr<<"updating decoder params"<<endl;
		decoder_plstm->updateParams(learning_rate,
										current_minibatch_size,
										momentum,
										L2_reg,
										norm_clipping,
										norm_threshold);												
	  }

 	 void updateParams(precision_type learning_rate,
 	 					int current_minibatch_size,
 				  		precision_type momentum,
 						precision_type L2_reg,
 						precision_type grad_scale,
 						loss_function_type loss_function,
 						bool arg_run_lm) {
 		//cerr<<"current minibatch size is "<<current_minibatch_size<<endl;
 		//cerr<<"updating params "<<endl;
 		//First compute the norm of the gradients for norm scaling
 		if (loss_function == LogLoss){
 			//cerr<<"updating log output layer"<<endl;
 			decoder_plstm->output_layer.updateParams(learning_rate,
 							current_minibatch_size,
 		  					momentum,
 		  					L2_reg,
 							grad_scale);			
 		} else if (loss_function == NCELoss){
 			//cerr<<"NOT IMPLEMENTED"<<endl;
 			//exit(1);
 			//cerr<<"updating nce output layer"<<endl;
 			decoder_plstm->output_layer.updateParamsNCE(learning_rate,
 								current_minibatch_size,
 								momentum,
 								L2_reg,
 								grad_scale);	
 		} else {
 			cerr<<loss_function<<" is an invalid loss function type"<<endl;
 			exit(0);
 		}



 		if (arg_run_lm == 0) {
 			//cerr<<"updating encoder params"<<endl;
 		    encoder_plstm->updateParams(learning_rate,
 												current_minibatch_size,
 												momentum,
 												L2_reg,
 												grad_scale);
 		}
		
 		//cerr<<"updating decoder params"<<endl;
 		decoder_plstm->updateParams(learning_rate,
 										current_minibatch_size,
 										momentum,
 										L2_reg,
 										grad_scale);												
 	}
	  	  
	  template <typename DerivedOut, typename data_type>
	  void computeProbs(const MatrixBase<DerivedOut> &output,
	  				  	multinomial<data_type> &unigram,
						int num_noise_samples,
						boost::random::mt19937 &rng,
						//multinomial<data_type> &unigram,
						loss_function_type loss_function,
						//SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
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

			for (int i=sent_len-1; i>=0; i--) {
				//cerr<<"i in gradient check is "<<i<<endl;
				//First doing fProp for the output layer
				if (loss_function == LogLoss) {
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
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
					//cerr<<"NOT IMPLEMENTED"<<endl;
					//exit(1);
					precision_type minibatch_log_likelihood;
					generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
					//cerr<<"minibatch_samples.rows() "<<minibatch_samples.rows()<<" minibatch_samples.cols() "<<minibatch_samples.cols()<<endl;
					//cerr<<"output "<<output<<endl;
					//cerr<<" output.row(0)" <<output.row(0)<<endl;
					//minibatch_samples.row(0) = output.row(0); //The first item is the minbiatch instance
					minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);
					
					//cerr<<"minibatch_samples "<<minibatch_samples<<endl;
					//getchar();
					//preparing the minbatch with no zeros for fprop nce
					minibatch_samples_no_negative = minibatch_samples;
					for (int col=0; col<current_minibatch_size; col++){ 
						if(minibatch_samples_no_negative(0,col) == -1){
							minibatch_samples_no_negative(0,col) = 0;
						}
					}
					//cerr<<"minibatch_samples_no_negative "<<minibatch_samples_no_negative<<endl;
					//getchar();
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
													minibatch_samples_no_negative.leftCols(current_minibatch_size),
													scores);

					nce_loss.fProp(scores, 
       			 				  minibatch_samples,
       						   	  probs, 
	   						  	  minibatch_log_likelihood,
								  this->fixed_partition_function);
					log_likelihood += minibatch_log_likelihood;		
				}
			}
			//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
			//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
			//cerr<<"The cross entopy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
			//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
			//log_likelihood /= sent_len;
	  }	  

	  template <typename DerivedOut>
	  void computeProbsDropout(const MatrixBase<DerivedOut> &output,
	  					multinomial<data_size_t> &unigram,
						int num_noise_samples,
						boost::random::mt19937 &rng,
						loss_function_type loss_function,
						//SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
	  					precision_type &log_likelihood) 
	  {	
			
			//cerr<<"In computeProbs..."<<endl;
			int current_minibatch_size = output.cols();

			Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			//dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

			int sent_len = output.rows(); 
			//precision_type log_likelihood = 0.;

			for (int i=sent_len-1; i>=0; i--) {
				//cerr<<"i in compute probs dropout is "<<i<<endl;
				//First doing fProp for the output layer
				if (loss_function == LogLoss) {
					output_dropout_layers[i].fProp(decoder_lstm_nodes[i].h_t,rng);					
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
										scores.leftCols(current_minibatch_size));

	
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
					//cerr<<"NOT IMPLEMENTED"<<endl;
					//exit(1);
					precision_type minibatch_log_likelihood;
					output_dropout_layers[i].fProp(decoder_lstm_nodes[i].h_t,rng);
					generateSamples(minibatch_samples.block(1,0, num_noise_samples,current_minibatch_size), unigram, rng);
					//cerr<<"minibatch_samples.rows() "<<minibatch_samples.rows()<<" minibatch_samples.cols() "<<minibatch_samples.cols()<<endl;
					//cerr<<"output "<<output<<endl;
					//cerr<<" output.row(0)" <<output.row(0)<<endl;
					//minibatch_samples.row(0) = output.row(0); //The first item is the minbiatch instance
					minibatch_samples.block(0, 0, 1, current_minibatch_size) = output.row(i);
					
					//cerr<<"minibatch_samples "<<minibatch_samples<<endl;
					//getchar();
					//preparing the minbatch with no zeros for fprop nce
					minibatch_samples_no_negative = minibatch_samples;
					for (int col=0; col<current_minibatch_size; col++){ 
						if(minibatch_samples_no_negative(0,col) == -1){
							minibatch_samples_no_negative(0,col) = 0;
						}
					}
					//cerr<<"minibatch_samples_no_negative "<<minibatch_samples_no_negative<<endl;
					//getchar();
					output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), 
													minibatch_samples_no_negative.leftCols(current_minibatch_size),
													scores);

					nce_loss.fProp(scores, 
       			 				  minibatch_samples,
       						   	  probs, 
	   						  	  minibatch_log_likelihood,
								  this->fixed_partition_function);
					log_likelihood += minibatch_log_likelihood;							
				}
			}
	  }	  
	  
	  template <typename DerivedOut>
	  void computeProbsLog(const MatrixBase<DerivedOut> &output,
	  					precision_type &log_likelihood,
						precision_type validation_correct_labels) 
	  {	
		  	//cerr<<"output is "<<output<<endl;
			//cerr<<"In computeProbs..."<<endl;
			int current_minibatch_size = output.cols();

			Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

			int sent_len = output.rows(); 
			//precision_type log_likelihood = 0.;

			for (int i=sent_len-1; i>=0; i--) {
				//cerr<<"i is "<<i<<endl;
				//First doing fProp for the output layer
				output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), scores);
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
	            for (int minibatch_instance=0; minibatch_instance < current_minibatch_size; minibatch_instance++){
	                Matrix<double,1,Dynamic>::Index max_index;
	                probs.col(minibatch_instance).maxCoeff(&max_index);
	                //validation_argmaxes.push_back(max_index);
	                if (max_index ==
	                    output(i,minibatch_instance) && 
							output(i,minibatch_instance) != -1){
	                        validation_correct_labels += 1.;
	                    }
	            }				
			}

	  }	  

	  template <typename DerivedOut>
	  void computeProbsLog(const MatrixBase<DerivedOut> &output,
	  					precision_type &log_likelihood,
						vector<precision_type> &sentence_probabilities) 
	  {	
			
			//cerr<<"In computeProbs..."<<endl;
			int current_minibatch_size = output.cols();
			//cerr<<"output is "<<output<<endl;
			Matrix<precision_type,Dynamic,Dynamic> dummy_zero;
			//Right now, I'm setting the dimension of dummy zero to the output embedding dimension becase everything has the 
			//same dimension in and LSTM. this might not be a good idea
			dummy_zero.setZero(output_layer_node.param->n_inputs(),current_minibatch_size);

			int sent_len = output.rows(); 
			//precision_type log_likelihood = 0.;
			//first initializing the sentence log probabilities to 0
			sentence_probabilities = vector<precision_type> (current_minibatch_size,0.);
			for (int i=sent_len-1; i>=0; i--) {
				//cerr<<"i is "<<i<<endl;
				//First doing fProp for the output layer
				output_layer_node.param->fProp(decoder_lstm_nodes[i].h_t.leftCols(current_minibatch_size), scores);
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
				//Now adding the sentence probabilities
				#pragma omp parallel for 
				for (int sent_index=0; sent_index<current_minibatch_size; sent_index++){
					int output_word_index = output(i,sent_index);
					//If the output word is not -1, then add to sentence log probability
					sentence_probabilities[sent_index] += (output_word_index >= 0) ? probs(output_word_index,sent_index) : 0.;
				}
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
			 const MatrixBase<DerivedOut> &decoder_input,
			 const MatrixBase<DerivedOut> &decoder_output,
			 const MatrixBase<DerivedC> &const_init_c,
			 const MatrixBase<DerivedH> &const_init_h,
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 loss_function_type loss_function,
			 const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
			 const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
			 bool arg_run_lm,
			 precision_type dropout_probability)
				 
    {
		Matrix<precision_type,Dynamic,Dynamic> init_c = const_init_c;
		Matrix<precision_type,Dynamic,Dynamic> init_h = const_init_h;
		//boost::random::mt19937 init_rng = rng;
		//cerr<<"init c is "<<init_c<<endl;
		//cerr<<"init h is "<<init_h<<endl;
		//cerr<<"in gradient check. The size of input is "<<input.rows()<<endl;
		cerr<<"In gradient check"<<endl;
		cerr<<"Decoder input is "<<decoder_input<<endl;
		cerr<<"Decoder output is "<<decoder_output<<endl;
		//Check every dimension of all the parameters to make sure the gradient is fine
		
		//cerr<<"Arg run lm is "<<arg_run_lm<<endl;
		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 decoder_plstm->output_layer,
							 "output_layer", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);	
 		//init_rng = rng;					 
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 decoder_plstm->W_h_to_c,
							 "Decoder: W_h_to_c", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;					 
 		init_c = const_init_c;
 		init_h = const_init_h;		
 		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 decoder_plstm->W_h_to_f,
							 "Decoder: W_h_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;	
 		init_c = const_init_c;
 		init_h = const_init_h;										
 		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 decoder_plstm->W_h_to_o,
							 "Decoder: W_h_to_o", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->W_h_to_i ,
							 "Decoder: W_h_to_i", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
	#ifdef PEEP						 
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;		
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->W_c_to_o,
							 "Decoder: W_c_to_o", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->W_c_to_f,
							 "Decoder: W_c_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//nit_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 decoder_plstm->W_c_to_i,
							 "Decoder: W_c_to_i", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
	#endif
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;		
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->o_t,
							 "Decoder: o_t",  
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->f_t,
							 "Decoder: f_t",
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->i_t,
							 "Decoder: i_t",
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 decoder_plstm->tanh_c_prime_t,
							 "Decoder: tanh_c_prime_t", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);		
 		//Doing gradient check for the input nodes
  		//init_rng = rng;
  		init_c = const_init_c;
  		init_h = const_init_h;
  		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 (dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_i,"Decoder: Standard_input_node: W_x_to_i", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);		
   		init_c = const_init_c;
   		init_h = const_init_h;
   		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 (dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_f,
							 "Decoder: Standard_input_node: W_x_to_f", 
  							 init_c,
  							 init_h,
  							 unigram,
  							 num_noise_samples,
  				   			 rng,
  				   			 loss_function,
  							 //softmax_nce_loss,
 							 input_sequence_cont_indices,
 							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		 		
   		init_c = const_init_c;
   		init_h = const_init_h;
		paramGradientCheck(input,
						 decoder_input, 
						 decoder_output,
						 (dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_c,
						 "Decoder: Standard_input_node: W_x_to_c", 
						 init_c,
						 init_h,
						 unigram,
						 num_noise_samples,
			   			 rng,
			   			 loss_function,
						 //softmax_nce_loss,
						 input_sequence_cont_indices,
						 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		init_c = const_init_c;
		init_h = const_init_h;		 
 		paramGradientCheck(input,decoder_input, 
						 decoder_output,
						 (dynamic_cast<input_model_type*>(decoder_plstm->input))->W_x_to_o,
						 "Decoder: Standard_input_node: W_x_to_o", 
 						 init_c,
 						 init_h,
 						 unigram,
 						 num_noise_samples,
 			   			 rng,
 			   			 loss_function,
 						 //softmax_nce_loss,
 						 input_sequence_cont_indices,
 						 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
 		init_c = const_init_c;
 		init_h = const_init_h;		
		//cerr<<"decoder_input "<<decoder_input<<endl;
		//cerr<<"decoder output "<<decoder_output<<endl;
		//getchar();
  		paramGradientCheck(input,
						 decoder_input, 
 						 decoder_output,
 						 (dynamic_cast<input_model_type*>(decoder_plstm->input))->input_layer,
 						 "Decoder: input_layer", 
  						 init_c,
  						 init_h,
  						 unigram,
  						 num_noise_samples,
  			   			 rng,
  			   			 loss_function,
  						 //softmax_nce_loss,
  						 input_sequence_cont_indices,
  						 output_sequence_cont_indices,
 							 dropout_probability,
							 arg_run_lm);							 
		
							
		if (arg_run_lm == 0) {	
							 					 							 
		//init_rng = rng;					 
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 encoder_plstm->W_h_to_c,
							 "Encoder: W_h_to_c", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;					 
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->W_h_to_f,
							 "Encoder: W_h_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;	
		init_c = const_init_c;
		init_h = const_init_h;										
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->W_h_to_o,
							 "Encoder: W_h_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->W_h_to_i ,
							 "Encoder: W_h_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
	#ifdef PEEP
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->W_c_to_o,
							 "Encoder: W_c_to_o", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->W_c_to_f,
							 "Encoder: W_c_to_f", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//nit_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 encoder_plstm->W_c_to_i,
							 "Encoder: W_c_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
	#endif
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;		
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->o_t,
							 "Encoder: o_t",  
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,
							 decoder_input, 
							 decoder_output,
							 encoder_plstm->f_t,
							 "Encoder: f_t",
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 encoder_plstm->i_t,
							 "Encoder: i_t",
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
		//init_rng = rng;
		init_c = const_init_c;
		init_h = const_init_h;
		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 encoder_plstm->tanh_c_prime_t,
							 "Encoder: tanh_c_prime_t", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);		
		//Doing gradient check for the input nodes
 		//init_rng = rng;
 		init_c = const_init_c;
 		init_h = const_init_h;
 		paramGradientCheck(input,decoder_input, 
							 decoder_output,(dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_i,
							 "Encoder: Standard_input_node: W_x_to_i", 
							 init_c,
							 init_h,
							 unigram,
							 num_noise_samples,
				   			 rng,
				   			 loss_function,
							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);		
  		init_c = const_init_c;
  		init_h = const_init_h;
  		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 (dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_f,
							 "Encoder: Standard_input_node: W_x_to_f", 
 							 init_c,
 							 init_h,
 							 unigram,
 							 num_noise_samples,
 				   			 rng,
 				   			 loss_function,
 							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
							 		
   		init_c = const_init_c;
   		init_h = const_init_h;
   		paramGradientCheck(input,decoder_input, 
							 decoder_output,
							 (dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_c,
							 "Encoder: Standard_input_node: W_x_to_c", 
  							 init_c,
  							 init_h,
  							 unigram,
  							 num_noise_samples,
  				   			 rng,
  				   			 loss_function,
  							 //softmax_nce_loss,
							 input_sequence_cont_indices,
							 output_sequence_cont_indices,
							 dropout_probability,
							 arg_run_lm);
  		init_c = const_init_c;
  		init_h = const_init_h;					 
		paramGradientCheck(input,decoder_input, 
						 decoder_output,
						 (dynamic_cast<input_model_type*>(encoder_plstm->input))->W_x_to_o,
						 "Encoder: Standard_input_node: W_x_to_o", 
						 init_c,
						 init_h,
						 unigram,
						 num_noise_samples,
			   			 rng,
			   			 loss_function,
						 //softmax_nce_loss,
						 input_sequence_cont_indices,
						 output_sequence_cont_indices,
						 dropout_probability,
						 arg_run_lm);								 									 							 	
		}		
		//paramGradientCheck(input,output,encoder_plstm->input_layer,"input_layer");
		
						 
	}
	template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
	void paramGradientCheck(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &decoder_input,
			 const MatrixBase<DerivedOut> &decoder_output,
			 testParam &param,
			 const string param_name,
			 const MatrixBase<DerivedC> &init_c,
			 const MatrixBase<DerivedH> &init_h, 
			 multinomial<data_type> &unigram,
			 int num_noise_samples,
			 boost::random::mt19937 &rng,
			 loss_function_type loss_function,			 
			 //SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
			 const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
			 const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
			 precision_type dropout_probability,
			 bool arg_run_lm) {
		//cerr<<"going over all dimensions"<<endl;
		//Going over all dimensions of the parameter
		//cerr<<"param.rows()"<<param.rows()<<endl;
		//cerr<<"param.cols()"<<param.cols()<<endl;
		for(int row=0; row<param.rows(); row++){
			for (int col=0; col<param.cols(); col++){		
				getFiniteDiff(input,
							decoder_input,
							decoder_output, 
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
							//softmax_nce_loss,
							input_sequence_cont_indices,
							output_sequence_cont_indices,
							dropout_probability,
							arg_run_lm);
			}
		}
	}
	
	template <typename DerivedIn, typename DerivedOut, typename testParam, typename DerivedC, typename DerivedH, typename DerivedS, typename data_type>
    void getFiniteDiff(const MatrixBase<DerivedIn> &input,
			 const MatrixBase<DerivedOut> &decoder_input,
			 const MatrixBase<DerivedOut> &decoder_output,
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
			 //SoftmaxNCELoss<multinomial<data_type> > &softmax_nce_loss,
			 const Eigen::ArrayBase<DerivedS> &input_sequence_cont_indices,
			 const Eigen::ArrayBase<DerivedS> &output_sequence_cont_indices,
			 precision_type dropout_probability,
			 bool arg_run_lm) {
				//cerr<<"Arg run lm is "<<arg_run_lm<<endl;
				//cerr<<"inside finite diff"<<endl;
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
				precision_type perturbation = 6*1e-3;
				//cerr<<"param name "<<param_name<<endl;
				//cerr<<"rand row "<<rand_row<<endl;
				//cerr<<"rand col "<<rand_col<<endl;
				//getchar();
		 	    param.changeRandomParam(perturbation, 
		 								rand_row,
		 								rand_col);
		 		//then do an fprop
		 		precision_type before_log_likelihood = 0;	
				//cerr<<"input cols is "<<input.cols()<<endl;					
		 		//fProp(input, output, 0, input.rows()-1, init_c, init_h, sequence_cont_indices);
				//cerr<<"const init c is "<<const_init_c<<endl;
				//cerr<<"const init h is "<<const_init_h<<endl;
				//cerr<<"first perturbing"<<endl;
				if (dropout_probability > 0) {
					if (arg_run_lm == 0 ) {
						fPropEncoderDropout(input,
									init_c,
									init_h,
									input_sequence_cont_indices,
									init_rng);	
					}
					//cerr<<"init_c"<<init_c<<endl;
					//cerr<<"init_h"<<init_h<<endl;
				    fPropDecoderDropout(decoder_input,
							init_c,
							init_h,
							output_sequence_cont_indices,
							init_rng);		
			 		computeProbsDropout(decoder_output,
								 unigram,
					   			 num_noise_samples,
					   			 init_rng,
					   			 loss_function,	
								 //softmax_nce_loss,
			 			  		 before_log_likelihood);																					
				} else {	
					if (arg_run_lm == 0) {			
					fPropEncoder(input,
								init_c,
								init_h,
								input_sequence_cont_indices);
					}
				    fPropDecoder(decoder_input,
							init_c,
							init_h,
							output_sequence_cont_indices);	
			 		computeProbs(decoder_output,
								 unigram,
					   			 num_noise_samples,
					   			 init_rng,
					   			 loss_function,	
								 //softmax_nce_loss,
			 			  		 before_log_likelihood);																
				}
				//cerr<<"just before passing const init c is "<<const_init_c<<endl;
				//cerr<<"just before passing const init h is "<<const_init_h<<endl;			

		 		//err<<"before log likelihood is "<<
		 	    param.changeRandomParam(-2*perturbation, 
		 								rand_row,
		 								rand_col);		
				init_c = const_init_c;
				init_h = const_init_h;
				init_rng = rng;
		 		precision_type after_log_likelihood = 0;	
				//cerr<<"second perturbing"<<endl;					
		 		//fProp(input,output, 0, input.rows()-1, init_c, init_h, input_sequence_cont_indices);	
				if (dropout_probability > 0) {
					if (arg_run_lm == 0){
					fPropEncoderDropout(input,
								init_c,
								init_h,
								input_sequence_cont_indices,
								init_rng);	
					}
				    fPropDecoderDropout(decoder_input,
							init_c,
							init_h,
							output_sequence_cont_indices,
							init_rng);				
			 		computeProbsDropout(decoder_output,
								 unigram,
					   			 num_noise_samples,
					   			 init_rng,
					   			 loss_function,	
								 //softmax_nce_loss,
			 			  		 after_log_likelihood);	
					 												
				} else {		
					if (arg_run_lm == 0){
					fPropEncoder(input,
								init_c,
								init_h,
								input_sequence_cont_indices);	
					}
				    fPropDecoder(decoder_input,
							init_c,
							init_h,
							output_sequence_cont_indices);	
 			 		computeProbs(decoder_output,
								 unigram,
 					   			 num_noise_samples,
 					   			 init_rng,
 					   			 loss_function,	
 								 //softmax_nce_loss,
 			 			  		 after_log_likelihood);																		
				}
	
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
				precision_type abs_diff = fabs(param.getGradient(rand_row,rand_col)-symmetric_finite_diff_grad);
				precision_type abs_max = max(fabs(param.getGradient(rand_row,rand_col)),fabs(symmetric_finite_diff_grad));
				//if (abs_diff > 
				//	perturbation + perturbation*abs_max) {
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
					assert(!(gradient_diff > threshold || relative_error > threshold));
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
	