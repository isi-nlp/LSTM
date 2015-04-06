//creating the structure of the nn in a graph that will help in performing backpropagation and forward propagation
#pragma once

#include <cstdlib>
#include "neuralClasses.h"
#include <Eigen/Dense>
#include <vector>
#include "activation_function.h"

using namespace std;
namespace nplm
{
	/*
template <class X>
class Node {
    public:
        X * param; //what parameter is this
        //vector <Node *> children;
        //vector <void *> parents;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> fProp_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> bProp_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cumul_input_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cumul_grad_matrix;
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
			//cumul_fProp_matrix.setZero(param->n_inputs(),minibatch_size);
	    }
            if (param->n_inputs() != -1)
            {
	        bProp_matrix.setZero(param->n_inputs(), minibatch_size);
            }
	}

	void resize() { resize(minibatch_size); }
	
	 
	//void addChild(Node *child){
	//	children.push_back(child);
};
*/
	
class Node {
    public:
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> fProp_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> bProp_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cumul_input_matrix;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cumul_grad_matrix;
	int minibatch_size;
	vector<Node *> children;
	vector<Node *> parents;
    public:
        Node() :minibatch_size(0),
			 	children(vector<Node *>()),
				parents(vector<Node *>()){ }
				
		template <class X>		
        Node(int minibatch_size, X *param)
	  	  : minibatch_size(minibatch_size),
	 		children(vector<Node *>()),
			parents(vector<Node *>())
        	{
	    		resize(minibatch_size, param);
        	}

	template <class X>
	void resize(int minibatch_size, X *param)
	{
	    this->minibatch_size = minibatch_size;
	    if (param->n_outputs() != -1)
	    {
	        fProp_matrix.setZero(param->n_outputs(), minibatch_size);
			cumul_grad_matrix.setZero(param->n_outputs(), minibatch_size);
			//cumul_fProp_matrix.setZero(param->n_inputs(),minibatch_size);
	    }
        if (param->n_inputs() != -1)
        {
        	bProp_matrix.setZero(param->n_inputs(), minibatch_size);
			cumul_input_matrix.setZero(param->n_inputs(), minibatch_size);
        }
		//cerr<<"Resizing the node"<<endl;
	}
	 	
	template<class X> 
	void resize(X *param) { resize(minibatch_size,param); }
	
	void addChild(Node *child){
		this->children.push_back(child);
	}
	
	void addParent(Node *parent){
		this->parents.push_back(parent);
	}	
	void accumulateInput() {
		this->cumul_input_matrix.setZero();
		for (int i=0; i<children.size(); i++){
			this->cumul_input_matrix +=  children[i]->fProp_matrix;
		}
	}
	
	void accumulateGrad() {
		this->cumul_grad_matrix.setZero();
		for (int i=0; i<parents.size(); i++){
			this->cumul_grad_matrix +=  parents[i]->bProp_matrix;
		}
	}	 
	
};


class HiddenNode : public Node {
	public:
		Activation_function *param;
	public:
	HiddenNode():
		param(NULL),
		Node(){}
		
	HiddenNode(Activation_function *param, int minibatch_size):
		param(param),
		Node(minibatch_size, param){}	
		
	void resize(){ Node::resize(param);}
	void resize(int minibatch_size){
		Node::resize(minibatch_size,param);
	}
};

class InputWordEmbeddingsNode : public Node{
	public:
		class Input_word_embeddings *param;
	public:
	InputWordEmbeddingsNode():
		param(NULL),
		Node(){}
		
	InputWordEmbeddingsNode(Input_word_embeddings *param, int minibatch_size):
		param(param),
		Node(minibatch_size, param){}	
		
	void resize(){ Node::resize(param);}
	void resize(int minibatch_size){
		cerr<<"Calling resize"<<endl;
		Node::resize(minibatch_size,param);
	}
};

class LinearNode : public Node{
	public:
		class Linear_layer *param;
	public:
	LinearNode():
		param(NULL),
		Node(){}
		
	LinearNode(Linear_layer *param, int minibatch_size):
		param(param),
		Node(minibatch_size, param){}
	
	void resize(int minibatch_size){
		Node::resize(minibatch_size,param);
	}
	void resize(){ Node::resize(param);}
};

class OutputWordEmbeddingsNode : public Node{
	public:
		class Output_word_embeddings *param;
	public:
	OutputWordEmbeddingsNode():
		param(NULL),
		Node(){}
		
	OutputWordEmbeddingsNode(Output_word_embeddings *param, int minibatch_size):
		param(param),
		Node(minibatch_size, param){}	
	
	void resize(){ Node::resize(param);}
	void resize(int minibatch_size){
		Node::resize(minibatch_size, param);
	}
};

} // namespace nplm
