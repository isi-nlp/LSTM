#ifndef NETWORK_H
#define NETWORK_H

#include "neuralClasses.h"
#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

class propagator {
    int minibatch_size;
    model *pnn;

public:
    InputWordEmbeddingsNode input_layer_node, rhs_rule_input_layer_node, Td_input_layer_node;
    LinearNode first_hidden_linear_node;
    HiddenNode first_hidden_activation_node;
    LinearNode second_hidden_linear_node;
    HiddenNode second_hidden_activation_node;
    OutputWordEmbeddingsNode output_layer_node;
	HiddenNode Td1_hidden_activation_node,
			   d1_hidden_activation_node;
	OutputWordEmbeddingsNode rhs_rule_output_layer_node,
							 Td_output_layer_node,
							 d_output_layer_node;
	LinearNode rhs_rule_linear_node,
			   Td_linear_node;

public:
    propagator () : minibatch_size(0), pnn(0) { }

    propagator (model &nn, int minibatch_size)
      :
      pnn(&nn),
    input_layer_node(&nn.input_layer, minibatch_size),
	first_hidden_linear_node(&nn.first_hidden_linear, minibatch_size),
	first_hidden_activation_node(&nn.first_hidden_activation, minibatch_size),
    second_hidden_linear_node(&nn.second_hidden_linear, minibatch_size),
	second_hidden_activation_node(&nn.second_hidden_activation, minibatch_size),
	output_layer_node(&nn.output_layer, minibatch_size),
	rhs_rule_output_layer_node(&nn.rhs_rule_output_layer,minibatch_size),
	Td1_hidden_activation_node(&nn.Td1_hidden_activation, minibatch_size),
	d1_hidden_activation_node(&nn.d1_hidden_activation, minibatch_size),
	Td_output_layer_node(&nn.Td_output_layer, minibatch_size),
	d_output_layer_node(&nn.d_output_layer, minibatch_size),
	rhs_rule_linear_node(&nn.rhs_rule_linear_layer,minibatch_size),
	Td_linear_node(&nn.Td_linear_layer,minibatch_size),
	minibatch_size(minibatch_size),
	Td_input_layer_node(&nn.Td_input_layer, minibatch_size),
	rhs_rule_input_layer_node(&nn.rhs_rule_input_layer, minibatch_size)
    {
		
		//Adding the children to the hidden activation nodes
		Td1_hidden_activation_node.addChild(&first_hidden_linear_node);
		Td1_hidden_activation_node.addChild(&rhs_rule_linear_node);
		
		d1_hidden_activation_node.addChild(&first_hidden_linear_node);
		d1_hidden_activation_node.addChild(&rhs_rule_linear_node);
		d1_hidden_activation_node.addChild(&Td_linear_node);
		
		//Adding the parents to the linear nodes
		first_hidden_linear_node.addParent(&first_hidden_activation_node);
		first_hidden_linear_node.addParent(&Td1_hidden_activation_node);
		first_hidden_linear_node.addParent(&d1_hidden_activation_node);
		
		rhs_rule_linear_node.addParent(&Td1_hidden_activation_node);
		rhs_rule_linear_node.addParent(&d1_hidden_activation_node);
		
		//Td_linear_node.addParent(&d1_hidden_activation_node);
    }

    // This must be called if the underlying model is resized.
    void resize(int minibatch_size) {
      this->minibatch_size = minibatch_size;
      input_layer_node.resize(minibatch_size);
      first_hidden_linear_node.resize(minibatch_size);
      first_hidden_activation_node.resize(minibatch_size);
      second_hidden_linear_node.resize(minibatch_size);
      second_hidden_activation_node.resize(minibatch_size);
      output_layer_node.resize(minibatch_size);
	  rhs_rule_output_layer_node.resize(minibatch_size);
	  Td1_hidden_activation_node.resize(minibatch_size);
	  d1_hidden_activation_node.resize(minibatch_size);
	  Td_output_layer_node.resize(minibatch_size);
	  d_output_layer_node.resize(minibatch_size);
	  rhs_rule_linear_node.resize(minibatch_size);
	  rhs_rule_input_layer_node.resize(minibatch_size);
	  Td_linear_node.resize(minibatch_size);
	  Td_input_layer_node.resize(minibatch_size);
    }

    void resize() { resize(minibatch_size); }

    template <typename Derived>
    void fProp(const MatrixBase<Derived> &data,
				int rhs_rule_index,
				int Td_index)
    {
		//cerr<<"rhs rule index is"<<rhs_rule_index<<endl;
	    if (!pnn->premultiplied)
		{
	        start_timer(0);
		    input_layer_node.param->fProp(data.topRows(rhs_rule_index-1), input_layer_node.fProp_matrix);
		    stop_timer(0);
	    
		    start_timer(1);
		    first_hidden_linear_node.param->fProp(input_layer_node.fProp_matrix, 
							  first_hidden_linear_node.fProp_matrix);

		} 
		else
		{
		    int n_inputs = first_hidden_linear_node.param->n_inputs();
		    USCMatrix<double> sparse_data;
		    input_layer_node.param->munge(data.topRows(rhs_rule_index-1), sparse_data);

		    start_timer(1);
		    first_hidden_linear_node.param->fProp(sparse_data,
							  first_hidden_linear_node.fProp_matrix);
		}
		first_hidden_activation_node.param->fProp(first_hidden_linear_node.fProp_matrix,
							  first_hidden_activation_node.fProp_matrix);
	    //std::cerr<<"in fprop first hidden activation node fprop is "<<first_hidden_activation_node.fProp_matrix<<std::endl;
	    //std::getchar();

    
		//I ALSO HAVE TO RUN THE PROPAGATORS FOR THE INPUT WORDS. 
		//DOING FPROP FOR THE RHS_RULE AND Td INPUT LAYERS
		rhs_rule_input_layer_node.param->fProp(data.row(rhs_rule_index-1), rhs_rule_input_layer_node.fProp_matrix);
		Td_input_layer_node.param->fProp(data.row(Td_index-1), Td_input_layer_node.fProp_matrix);
		stop_timer(1);
		
		/*
		start_timer(2);
		second_hidden_linear_node.param->fProp(first_hidden_activation_node.fProp_matrix,
						       second_hidden_linear_node.fProp_matrix);
		second_hidden_activation_node.param->fProp(second_hidden_linear_node.fProp_matrix,
							   second_hidden_activation_node.fProp_matrix);
		stop_timer(2);
		*/
		
		//Propagating for Td
		start_timer(3);
		rhs_rule_linear_node.param->fProp(rhs_rule_input_layer_node.fProp_matrix,
								rhs_rule_linear_node.fProp_matrix);
							
		start_timer(4);
		Td1_hidden_activation_node.accumulateInput();
		Td1_hidden_activation_node.param->fProp(Td1_hidden_activation_node.cumul_input_matrix,
										Td1_hidden_activation_node.fProp_matrix);
		stop_timer(4);
	
		//Propagating for d
		start_timer(5);
		Td_linear_node.param->fProp(Td_input_layer_node.fProp_matrix,
								Td_linear_node.fProp_matrix);
		stop_timer(5);
	
		start_timer(6);
		d1_hidden_activation_node.accumulateInput();
		d1_hidden_activation_node.param->fProp(d1_hidden_activation_node.cumul_input_matrix,
								d1_hidden_activation_node.fProp_matrix);
		stop_timer(6);

	// The propagation stops here because the last layers are very expensive.
    }
	
	template <typename DerivedOut>
	void bPropRhsRuleAndTd(const MatrixBase<DerivedOut> &rhs_rule_output,
		   const MatrixBase<DerivedOut> &Td_output,
	       double learning_rate,
           double momentum,
           double L2_reg,
           std::string &parameter_update,
           double conditioning_constant,
           double decay){
           //start_timer(7);
           rhs_rule_output_layer_node.param->bProp(rhs_rule_output,	
   				       rhs_rule_output_layer_node.bProp_matrix);
           Td_output_layer_node.param->bProp(Td_output,
   				       Td_output_layer_node.bProp_matrix);
   		    //stop_timer(7);

   		   //start_timer(8);
		   //ONLY ALLOWING SGD RIGHD NOW
   		   //if (parameter_update == "SGD") {
   		    rhs_rule_output_layer_node.param->computeGradient(second_hidden_activation_node.fProp_matrix,
   		               rhs_rule_output,
   		               learning_rate,
   		               momentum);
  		    Td_output_layer_node.param->computeGradient(Td1_hidden_activation_node.fProp_matrix,
  		               Td_output,
  		               learning_rate,
  		               momentum);					   
			/*		   
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
		   */
   		   //stop_timer(8);			   
		
	}
    // Dense version (for standard log-likelihood)
    template <typename DerivedIn, typename DerivedOut>
    void bProp(const MatrixBase<DerivedIn> &data,
	       const MatrixBase<DerivedOut> &d_output,
		   const MatrixBase<DerivedOut> &rhs_rule_output,
		   const MatrixBase<DerivedOut> &Td_output,
	       double learning_rate,
           double momentum,
           double L2_reg,
           std::string &parameter_update,
           double conditioning_constant,
           double decay,
		   int rhs_rule_index,
		   int Td_index) 
    {
        // Output embedding layer

        start_timer(7);
        d_output_layer_node.param->bProp(d_output,
				       d_output_layer_node.bProp_matrix);
		stop_timer(7);
	
		start_timer(8);
		  if (parameter_update == "SGD") {
			  /*
		    output_layer_node.param->computeGradient(second_hidden_activation_node.fProp_matrix,
		               output,
		               learning_rate,
		               momentum);
			 */
   		    d_output_layer_node.param->computeGradient(d1_hidden_activation_node.fProp_matrix,
   		               d_output,
   		               learning_rate,
   		               momentum);
		  } else if (parameter_update == "ADA") {
			cerr<< " ADAGRDAD NOT IMPLEMENTED";
			exit(1);
		    d_output_layer_node.param->computeGradientAdagrad(d1_hidden_activation_node.fProp_matrix,
		               d_output,
		               learning_rate);
		  } else if (parameter_update == "ADAD") {
  			cerr<< " ADADELTA NOT IMPLEMENTED";
  			exit(1);
		    //std::cerr<<"Adadelta gradient"<<endl;
		    int current_minibatch_size = second_hidden_activation_node.fProp_matrix.cols();
		    d_output_layer_node.param->computeGradientAdadelta(d1_hidden_activation_node.fProp_matrix,
		               d_output,
		               1.0/current_minibatch_size,
		               conditioning_constant,
		               decay);
		  } else {
		    std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
		  }
		stop_timer(8);
		//Calling backprop for Td output layer nodes and the rhs rule output layer nodes. 
		bPropRhsRuleAndTd(rhs_rule_output,
				   Td_output,
			       learning_rate,
		           momentum,
		           L2_reg,
		           parameter_update,
		           conditioning_constant,
		           decay);
				   
		bPropRest(data, 
	      learning_rate,
	      momentum,
	      L2_reg,
	      parameter_update,
	      conditioning_constant,
	      decay,
		  rhs_rule_index,
		  Td_index);
    }

    // Sparse version (for NCE log-likelihood)
    template <typename DerivedIn, typename DerivedOutI, typename DerivedOutV, typename DerivedOutB>
    void bProp(const MatrixBase<DerivedIn> &data,
	     const MatrixBase<DerivedOutI> &d_samples,
         const MatrixBase<DerivedOutV> &d_weights,
		 const MatrixBase<DerivedOutB> &rhs_rule_output,
		 const MatrixBase<DerivedOutB> &Td_output,
	     double learning_rate,
         double momentum,
         double L2_reg,
         std::string &parameter_update,
         double conditioning_constant,
         double decay,
		 int rhs_rule_index,
		 int Td_index) 
    {

        // Output embedding layer

        start_timer(7);
        d_output_layer_node.param->bProp(d_samples,
            d_weights, 
			d_output_layer_node.bProp_matrix);
		stop_timer(7);
	

		start_timer(8);
	  if (parameter_update == "SGD") {
		  /*
	    output_layer_node.param->computeGradient(second_hidden_activation_node.fProp_matrix,
	               samples,
	               weights,
	               learning_rate,
	               momentum);
		  */
	   d_output_layer_node.param->computeGradient(d1_hidden_activation_node.fProp_matrix,
	              d_samples,
	              d_weights,
	              learning_rate,
	              momentum);
	  } else if (parameter_update == "ADA") {
		cerr<< " ADAGRDAD NOT IMPLEMENTED";
		exit(1);
	    d_output_layer_node.param->computeGradientAdagrad(d1_hidden_activation_node.fProp_matrix,
	               d_samples,
	               d_weights,
	               learning_rate);
	  } else if (parameter_update == "ADAD") {
		cerr<< " ADADELTA NOT IMPLEMENTED";
		exit(1);
	    int current_minibatch_size = d1_hidden_activation_node.fProp_matrix.cols();
	    //std::cerr<<"Adadelta gradient"<<endl;
	    d_output_layer_node.param->computeGradientAdadelta(d1_hidden_activation_node.fProp_matrix,
	               d_samples,
	               d_weights,
	               1.0/current_minibatch_size,
	               conditioning_constant,
	               decay);
	  } else {
	    std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
	  }

		stop_timer(8);
	
		//Calling backprop for Td output layer nodes and the rhs rule output layer nodes. 
		bPropRhsRuleAndTd(rhs_rule_output,
				   Td_output,
			       learning_rate,
		           momentum,
		           L2_reg,
		           parameter_update,
		           conditioning_constant,
		           decay);
		bPropRest(data,
	      learning_rate,
	      momentum,
	      L2_reg,
	      parameter_update,
	      conditioning_constant,
	      decay,
		  rhs_rule_index,
		  Td_index);
    }

private:
    template <typename DerivedIn>
    void bPropRest(const MatrixBase<DerivedIn> &data,
		   double learning_rate, double momentum, double L2_reg,
       std::string &parameter_update,
       double conditioning_constant,
       double decay,
	   int rhs_rule_index,
	   int Td_index) 
    {
	// Second hidden layer


  
	  // All the compute gradient functions are together and the backprop
	  // functions are together
	  ////////BACKPROP////////////
	   d1_hidden_activation_node.param->bProp(d_output_layer_node.bProp_matrix,
	   										d1_hidden_activation_node.bProp_matrix,
											d1_hidden_activation_node.cumul_input_matrix,
											d1_hidden_activation_node.fProp_matrix);

	   //BACKPROP AND GRADIENT COMPUTATION FOR Td_linear_node
	   Td_linear_node.param->bProp(d1_hidden_activation_node.bProp_matrix,
	   								Td_linear_node.bProp_matrix);
	   Td_linear_node.param->computeGradient(d1_hidden_activation_node.bProp_matrix,
	                Td_input_layer_node.fProp_matrix,
	                learning_rate, 
					momentum, 
					L2_reg);
		/*
		cerr<<"Td index is "<<Td_index<<endl;
		cerr<<"Data is "<<data<<endl;
		cerr<<" data row is"<<data.row(Td_index-1)<<endl;
		cerr<<" bprop matrix is "<<Td_linear_node.bProp_matrix<<endl;
		*/
	   //TODO: UPDATE THE PARAMETERS FOR THE EMBEDDINGS OF Td
		Td_input_layer_node.param->computeGradient(Td_linear_node.bProp_matrix,
		          data.row(Td_index-1),
		          learning_rate, 
				  momentum, 
				  L2_reg);
	   //The input layer of the model
	   //Td_input_layer_node.param->bProp()
	   /*
	   Td_input_layer_node.param->computeGradient(Td_linear_node.bProp_matrix,
	                 data,
	                 learning_rate, momentum, L2_reg);	   
		*/		
		Td1_hidden_activation_node.param->bProp(Td_output_layer_node.bProp_matrix,
										Td1_hidden_activation_node.bProp_matrix,
									    Td1_hidden_activation_node.cumul_input_matrix,
										Td1_hidden_activation_node.fProp_matrix); 
   
	   //BACKPROP AND GRADIENT COMPUTATION FOR LINEAR NODE
	   rhs_rule_linear_node.accumulateGrad();
	   rhs_rule_linear_node.param->bProp(rhs_rule_linear_node.cumul_grad_matrix,
	   									rhs_rule_linear_node.bProp_matrix);
	   rhs_rule_linear_node.param->computeGradient(rhs_rule_linear_node.cumul_grad_matrix,
	                rhs_rule_input_layer_node.fProp_matrix,
	                learning_rate, 
					momentum, 
					L2_reg);
				
	   //TODO: UPDATE THE PARAMETERS FOR THE EMBEDDINGS OF Td		
		rhs_rule_input_layer_node.param->computeGradient(rhs_rule_linear_node.bProp_matrix,
		          data.row(rhs_rule_index-1),
		          learning_rate, 
				  momentum, 
				  L2_reg);						
	   start_timer(9);
	   /*
	   second_hidden_activation_node.param->bProp(output_layer_node.bProp_matrix,
	                                           second_hidden_activation_node.bProp_matrix,
	                                           second_hidden_linear_node.fProp_matrix,
	                                           second_hidden_activation_node.fProp_matrix);

										   
		second_hidden_linear_node.param->bProp(second_hidden_activation_node.bProp_matrix,
						       second_hidden_linear_node.bProp_matrix);
		stop_timer(9);
	    */
		start_timer(11);
		first_hidden_activation_node.param->bProp(rhs_rule_output_layer_node.bProp_matrix,
							  first_hidden_activation_node.bProp_matrix,
							  first_hidden_linear_node.fProp_matrix,
							  first_hidden_activation_node.fProp_matrix);
						  
	  	//ACCUMULATE GRADIENT
	  	first_hidden_linear_node.accumulateGrad();
	    first_hidden_linear_node.param->bProp(first_hidden_activation_node.bProp_matrix,
						      first_hidden_linear_node.bProp_matrix);
						  
		stop_timer(11);
	  //std::cerr<<"First hidden layer node backprop matrix is"<<first_hidden_linear_node.bProp_matrix<<std::endl;
	  //std::getchar();
	  ////COMPUTE GRADIENT/////////
	  if (parameter_update == "SGD") {
		  /*
	    start_timer(10);
	    second_hidden_linear_node.param->computeGradient(second_hidden_activation_node.bProp_matrix,
	                 first_hidden_activation_node.fProp_matrix,
	                 learning_rate,
	                 momentum,
	                 L2_reg);
	    stop_timer(10);
		 */
		  
	    // First hidden layer

    
	    start_timer(12);

	    first_hidden_linear_node.param->computeGradient(first_hidden_linear_node.cumul_grad_matrix,
	                input_layer_node.fProp_matrix,
	                learning_rate, 
					momentum, 
					L2_reg);
	    stop_timer(12);

	    // Input word embeddings
    
	    start_timer(13);
	    input_layer_node.param->computeGradient(first_hidden_linear_node.bProp_matrix,
	              data.topRows(rhs_rule_index-1),
	              learning_rate, 
				  momentum, 
				  L2_reg);
	    stop_timer(13);
	  } else if (parameter_update == "ADA") {
		  cerr<<"ADAGRAD UPDATE METHOD NOT SUPPORTED!"<<endl;
		  exit(1);
		  /*
	    start_timer(10);
	    second_hidden_linear_node.param->computeGradientAdagrad(second_hidden_activation_node.bProp_matrix,
	                 first_hidden_activation_node.fProp_matrix,
	                 learning_rate,
	                 L2_reg);
	    stop_timer(10);
		  */
		  
	    // First hidden layer 
	    start_timer(12);
	    first_hidden_linear_node.param->computeGradientAdagrad(first_hidden_activation_node.bProp_matrix,
	                input_layer_node.fProp_matrix,
	                learning_rate,
	                L2_reg);
	    stop_timer(12);

	    // Input word embeddings
     
	    start_timer(13);
	    input_layer_node.param->computeGradientAdagrad(first_hidden_linear_node.bProp_matrix,
	              data.topRows(rhs_rule_index-1),
	              learning_rate, 
	              L2_reg);
	    stop_timer(13);
	  } else if (parameter_update == "ADAD") {
		  cerr<<"ADADELTA UPDATE METHOD NOT SUPPORTED!"<<endl;
		  exit(1);
	    int current_minibatch_size = first_hidden_activation_node.fProp_matrix.cols();
	    //std::cerr<<"Adadelta gradient"<<endl;
		/*
	    start_timer(10);
	    second_hidden_linear_node.param->computeGradientAdadelta(second_hidden_activation_node.bProp_matrix,
	                 first_hidden_activation_node.fProp_matrix,
	                 1.0/current_minibatch_size,
	                 L2_reg,
	                 conditioning_constant,
	                 decay);
	    stop_timer(10);
		*/
		
	    //std::cerr<<"Finished gradient for second hidden linear layer"<<std::endl;

	    // First hidden layer

    
	    start_timer(12);
	    first_hidden_linear_node.param->computeGradientAdadelta(first_hidden_activation_node.bProp_matrix,
	                input_layer_node.fProp_matrix,
	                1.0/current_minibatch_size,
	                L2_reg,
	                conditioning_constant,
	                decay);
	    stop_timer(12);

	    //std::cerr<<"Finished gradient for first hidden linear layer"<<std::endl;
	    // Input word embeddings
     
	    start_timer(13);
	    input_layer_node.param->computeGradientAdadelta(first_hidden_linear_node.bProp_matrix,
	              data.topRows(rhs_rule_index-1),
	              1.0/current_minibatch_size, 
	              L2_reg,
	              conditioning_constant,
	              decay);
	    stop_timer(13);
  
	    //std::cerr<<"Finished gradient for first input layer"<<std::endl;
	  } else {
	    std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
	  }

    }
};

} // namespace nplm

#endif

