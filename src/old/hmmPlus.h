#include <boost/shared_ptr.hpp>
#include <iostream>
#include <fstream>
//#include "util.h"

#include <Eigen/Dense>
#include "param.h"


typedef double Real;
typedef std::vector<Real> Reals;
typedef std::vector<int> Ints;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;

class hmmPlusModel {
public:
    MatrixReal emission_matrix;
    vector<MatrixReal > transition_matrix;
    VectorReal pi_vector;
    Ints transition_words_list;
    


public:
    hmmPlusModel(const hmmPlusModel& model)
    {
        cout<<"in lame constructor"<<endl;   
    }
    hmmPlusModel(param & myParam)
    {
        initializeParametersRange(myParam); 
        readTransitionWordsList(myParam);
    }
    void initializeParametersUniform(param & myParam)
    {
        //cout<<"initializing the parameters "<<endl;
        //initialzing the parameters
        emission_matrix.setZero(myParam.num_tag_types,myParam.num_words);
        for (int i=0;i<myParam.num_words;i++)
        {
            MatrixReal temp;
            temp.setZero(myParam.num_tag_types,myParam.num_tag_types);
            transition_matrix.push_back(temp);
        }   
        //cout<<"initialized the transition_matrix"<<endl;
        pi_vector.setZero(myParam.num_tag_types);

        //initialize parameters uniformly
        for (int row = 0;row<emission_matrix.rows();row++)
        {
            for (int col = 0;col<emission_matrix.cols();col++)
            {
                emission_matrix(row,col) = 1.0/myParam.num_words;  
            }
        }
        for (int i =0;i<myParam.num_words;i++)
        {
            for (int row = 0;row<transition_matrix[i].rows();row++)
            {
                for (int col = 0;col<transition_matrix[i].cols();col++)
                {
                    transition_matrix[i](row,col) = 1.0/myParam.num_tag_types;  
                }
            }
        }
        for (int col = 0;col<pi_vector.rows();col++)
        {
            pi_vector(col) = 1./myParam.num_tag_types;      
        }

        //cout<<"initialized the matrices"<<endl;

        //cout<<"pi "<<pi_vector;
        //getchar();
        //cout<<"emission"<<emission_matrix;

    }
    void initializeParametersRange(param & myParam)
    {
        emission_matrix.setZero(myParam.num_tag_types,myParam.num_words);
        for (int i=0;i<myParam.num_words;i++)
        {
            MatrixReal temp;
            temp.setZero(myParam.num_tag_types,myParam.num_tag_types);
            transition_matrix.push_back(temp);
        }   
        //cout<<"initialized the transition_matrix"<<endl;
        pi_vector.setZero(myParam.num_tag_types);

        //initialize parameters in a range
        for (int row = 0;row<emission_matrix.rows();row++)
        {
            Real denom = emission_matrix.cols()*(emission_matrix.cols()+1)/2.;
            for (int col = 0;col<emission_matrix.cols();col++)
            {
                emission_matrix(row,col) = (col+1.0)/denom;
            }
        }
        for (int i =0;i<myParam.num_words;i++)
        {
            for (int row = 0;row<transition_matrix[i].rows();row++)
            {
                Real denom = transition_matrix[i].cols()*(transition_matrix[i].cols()+1)/2.;
                for (int col = 0;col<transition_matrix[i].cols();col++)
                {
                    transition_matrix[i](row,col) = (col+1.0)/denom;
                }
            }
        }
        //initializing pi vector uniformly
        for (int col = 0;col<pi_vector.rows();col++)
        {
            pi_vector(col) = 1./myParam.num_tag_types;      
        }
    }
    void print()
    {
        cout<<"just printing"<<endl;

    }
    void readTransitionWordsList(param &myParam)
    {
        cout << "Reading transition words from : " << myParam.transition_words_file<< endl;


        ifstream TRAININ;
        TRAININ.open(myParam.transition_words_file.c_str());
        if (! TRAININ)
        {
          cerr << "Error : can't read training data from file " << myParam.transition_words_file<< endl;
          exit(-1);
        }

        while (! TRAININ.eof())
        {
          string line;
          vector<int> ngram; 
          getline(TRAININ, line);
          if (line == "")
          {
            continue;
          }

          //ngram = splitBySpace(line);

          //transition_words_list.push_back((int)atoi(ngram[0]));
          transition_words_list.push_back((int)atoi(line.c_str()));
        }
        TRAININ.close();
        
    }
    void updateTransitionFromEM(param &myParam,vector<MatrixReal > &expec_transition_matrix)
    {
        for (int i = 0;i<transition_words_list.size();i++)
        {
            for (int tag =0;tag<myParam.num_tag_types;tag++)
            {
                transition_matrix[transition_words_list[i]].row(tag) = 
                (expec_transition_matrix[transition_words_list[i]].row(tag).array()+myParam.smoothing) /
                (expec_transition_matrix[transition_words_list[i]].row(tag).sum() + myParam.smoothing*myParam.num_tag_types);
            }
        }
    }
    void updateEmissionFromEM(param &myParam,MatrixReal &expec_emission_matrix)
    {
        for (int tag =0;tag<myParam.num_tag_types;tag++)
        {
            emission_matrix.row(tag) = 
            (expec_emission_matrix.row(tag).array()+myParam.smoothing) /
            (expec_emission_matrix.row(tag).sum() + myParam.smoothing*myParam.num_words);
            //cout<<"the epec emission matrix is "<<expec_emission_matrix.row(tag)<<endl;
            //getchar();
            //cout<<"the emission matrix is "<<endl;
            //cout<<emission_matrix.row(tag)<<endl;
            //getchar();

        }
    }
    void updatePIFromEM(param &myParam,VectorReal &expec_pi_vector)
    {
        //cout<<"expec pi vector is "<<endl;
        //cout<<expec_pi_vector<<endl;
        pi_vector = (expec_pi_vector.array()+myParam.smoothing)/(expec_pi_vector.sum()+myParam.smoothing*myParam.num_tag_types);
        //cout<<"pi vector is "<<pi_vector<<endl;
        //getchar();
    }


};
