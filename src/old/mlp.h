#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "param.h"

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DEFAULT_TO_ROW_MAJOR

using namespace std;
using namespace Eigen;
using namespace boost::random;


typedef double Real;
typedef std::vector<Real> Reals;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>               VectorReal;

class hiddenLayer{
    public:
        MatrixReal W;
        VectorReal b;
        MatrixReal logit; //this will hold the minibatch of activations
        MatrixReal activation;  //this will hold the outputs, which are the sigmoids or the tanh's
        MatrixReal minibatch_b; //having a matrix b of minibatch size helps the forward prop
        
    hiddenLayer(int n_input,int n_output,int minibatch_size)
    {
        W.setZero(n_output,n_input);
        b.setZero(n_output);
        logit.setZero(n_output,minibatch_size);
        activation.setZero(n_output,minibatch_size);
        //initialize bais and W
        unsigned seed = std::time(0);
        clock_t t;
        mt19937 eng_W (t+rand());  // mt19937 is a standard mersenne_twister_engine
        mt19937 eng_b (t+rand());  // mt19937 is a standard mersenne_twister_engine
        cout<<"W rows is "<<W.rows()<<" and W cols is "<<W.cols()<<endl;


        cout<<"bias rows is "<<b.rows()<<" and v_bias cols is "<<b.cols()<<endl;


        //void * distribution ;
        /*
        if (myParam.init_normal == 0)
        {
            uniform_real_distribution<> unif_real(-0.01, 0.01); 
            //initializing W
            for (int i =0;i<W.rows();i++)
            {
                //cout<<"i is "<<i<<endl;
                for (int j =0;j<W.cols();j++)
                {
                    //cout<<"j is "<<j<<endl;
                    W(i,j) = unif_real(eng_W);    
                }
            }
            //initializing bias
            for (int i =0;i<b.cols();i++)
            {
                //cout<<"i is "<<i<<endl;
                b(i) = unif_real(eng_b);
            }

        }
        else //initialize with gaussian distribution with mean 0 and stdev 0.01
        {
        */
            normal_distribution<double> unif_normal(0.,0.01);
            //initializing W
            for (int i =0;i<W.rows();i++)
            {
                //cout<<"i is "<<i<<endl;
                for (int j =0;j<W.cols();j++)
                {
                    //cout<<"j is "<<j<<endl;
                    W(i,j) = unif_normal(eng_W);    
                }
            }
            //initializing bias
            for (int i =0;i<b.cols();i++)
            {
                //cout<<"i is "<<i<<endl;
                b(i) = unif_normal(eng_b);
            }

        //}
        
        //copying b into each column of minibatch_b
        for (int col =0;col<minibatch_b.cols();col++)
        {
            minibatch_b.col(col) = b;
        }
    }
    void forwardProp(MatrixReal minibatch_input,param myParam)
    {
        logit =  W*minibatch_input + minibatch_b;
        if (myParam.activation == 't') //tanh activation
        {
            activation = (2/(1+(-2*logit.array()).exp())).matrix();
        }
        else //sigmoid
        {
            activation = (1/(1+(-1*logit.array()).exp())).matrix();
        }
        
    }
};
