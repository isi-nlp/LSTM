//to run all the forward backward iterations
#include <Eigen/Dense>
#include "param.h"
#include <iomanip>

using namespace std;
using namespace Eigen;
using namespace boost::random;

typedef double Real;
typedef std::vector<Real> Reals;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>               VectorReal;


void forwardBackward(VectorReal &pi_vector,vector<MatrixReal >& transition_matrix,MatrixReal & emission_matrix,param & myParam,
                    vector<vector <int> > &training_data,MatrixReal &expec_emission_matrix,vector<MatrixReal >& expec_transition_matrix,
                    VectorReal &expec_pi_vector)
{

    Real total_corpus_prob = 0.;
    int num_tag_types = myParam.num_tag_types;
    for (int counter = 0;counter < training_data.size();counter++)
    {
        if (counter%100 ==0)
        {
            cout<<". ";
        }
        int sent_len = training_data[counter].size();
        MatrixReal alpha_t,alpha_t_tilde,alpha_t_hat;
        VectorReal scaling_factor_alpha;
        alpha_t.setZero(sent_len,num_tag_types);
        alpha_t_tilde = alpha_t;
        alpha_t_hat = alpha_t;
        scaling_factor_alpha.setZero(sent_len);
        //computing \alpha_0
        alpha_t_tilde.row(0) = pi_vector.array() * emission_matrix.col(training_data[counter][0]).array();
        //computign scaling factor
        //cout<<"here 0"<<endl;
        scaling_factor_alpha(0) = 1./alpha_t_tilde.row(0).sum();
        alpha_t_hat.row(0) = scaling_factor_alpha(0)*alpha_t_tilde.row(0) ;
        //cout<<"here1"<<endl;

        for (int t = 0;t<sent_len-1;t++)
        {
            //cout<<"finished dot "<<endl;
            int word = training_data[counter][t+1];
            for (int current_tag =0;current_tag < num_tag_types;current_tag++)
            {
                alpha_t_tilde(t+1,current_tag) = alpha_t_hat.row(t).dot(transition_matrix[training_data[counter][t]].col(current_tag));
                //cout<<alpha_t_tilde(t+1,current_tag)<<endl;
                //getchar();
            }
            //cout<<"finished dot"<<endl;
            //cout<<"the size of rhs is "<<emission_matrix.col(word).cols()<<endl;
            //cout<<"the size of lhs is "<<alpha_t_tilde.row(t+1).cols()<<endl;
            alpha_t_tilde.row(t+1) = (alpha_t_tilde.row(t+1).transpose().array()*emission_matrix.col(word).array()).matrix();
            //cout<<alpha_t_tilde.row(t+1)<<endl;
            //compute the scaling factor
            scaling_factor_alpha(t+1) = 1./alpha_t_tilde.row(t+1).sum();
            //cout<<"scaling factor was "<<scaling_factor_alpha(t+1)<<endl;
            alpha_t_hat.row(t+1) = alpha_t_tilde.row(t+1)*scaling_factor_alpha(t+1);
            //cout<<"alpha t hat is now "<<alpha_t_hat.row(t+1);
            //getchar();
            //getchar();
            //cout<<"we are here "<<endl;
        }
        total_corpus_prob += -((scaling_factor_alpha.array()).log()).sum();
        //computing the betas
        MatrixReal beta_t_tilde,beta_t_hat;
        beta_t_hat.setZero(sent_len,num_tag_types);
        beta_t_tilde.setZero(sent_len,num_tag_types);
        
        //initializing the betas
        beta_t_tilde.row(sent_len-1).setOnes(); 
        beta_t_hat.row(sent_len-1).setOnes();
        //#scaling the betas by c_{T-1}
        beta_t_hat.row(sent_len-1) *= scaling_factor_alpha(sent_len-1);
        
        for (int t = 0;t<sent_len-1;t++)
        {
            int word = training_data[counter][(sent_len-1)-(t+1)];
            //compute the beta
            for (int current_tag =0;current_tag < num_tag_types;current_tag++)
            {
                //cout<<"in the loop"<<endl;
                beta_t_tilde((sent_len-1)-(t+1),current_tag) = beta_t_hat.row((sent_len-1)-t).dot((transition_matrix[word].row(current_tag).array()*emission_matrix.col(training_data[counter][(sent_len-1)-t]).transpose().array()).matrix());
                //cout<<"beta t tilde is "<< beta_t_tilde((sent_len-1)-(t+1),current_tag)<<endl;
                //getchar();
            }
            //scale the betas
            beta_t_hat.row((sent_len-1)-(t+1)) = beta_t_tilde.row((sent_len-1)-(t+1)) * scaling_factor_alpha((sent_len-1)-(t+1));
            //cout<<"beta t hat is "<<endl;
            //cout<<beta_t_hat.row((sent_len-1)-(t+1))<<endl;
            //getchar();
            //beta_t_tilde[(sent_len-1)-(t+1),:] = [(beta_t_hat[(sent_len-1)-t,:]*transition_matrix[word,current_tag,:]*emission_matrix[:,sent[(sent_len-1)-t]]).sum()            for current_tag in range(num_tag_types)]

            //#scale the betas
            //beta_t_hat[(sent_len-1)-(t+1),:] = beta_t_tilde[(sent_len-1)-(t+1),:] * scaling_factor_alpha[(sent_len-1)-(t+1)]

            
        }
        //now that we've computed the alphas and the betas, its time to compute the expected counts for the transition parameters
        //computing the probability of a path passing through a particular pair i,j at position t (gamma_ij_t) and the path passing through i at position t
        for (int t=0;t<sent_len-1;t++)
        {
            int word = training_data[counter][t];
            MatrixReal gamma_ij_t;
            VectorReal gamma_i_t;
            gamma_i_t.setZero(num_tag_types,1);
            gamma_ij_t.setZero(num_tag_types,num_tag_types);
            for (int current_tag =0;current_tag < num_tag_types;current_tag++)
            {
                gamma_ij_t.row(current_tag) = alpha_t_hat(t,current_tag)*
                                          (transition_matrix[word].row(current_tag).array()*emission_matrix.col(training_data[counter][t+1]).transpose().array()*
                                          beta_t_hat.row(t+1).array()).matrix();
                gamma_i_t(current_tag) = gamma_ij_t.row(current_tag).sum();

            }
            //cout<<"gamma ij t is "<<endl;
            //cout<<gamma_ij_t<<endl;
            //cout<<"gamma i t is"<<endl;
            //cout<<gamma_i_t;
            //getchar();
            if (t ==0)
            {
                expec_pi_vector += gamma_i_t;  
            }
            expec_transition_matrix[word] += gamma_ij_t;
            //cout<<expec_transition_matrix[word]<<endl;
            expec_emission_matrix.col(word) += gamma_i_t;
            //cout<<expec_emission_matrix.col(word)<<endl;
            //getchar();
        }
        //adding the expected counts for the emission probs from the last cell
        VectorReal gamma_i_last;
        gamma_i_last.setZero(num_tag_types,1);
        gamma_i_last = (alpha_t_hat.row(sent_len-1).array()*beta_t_hat.row(sent_len-1).array()).matrix() / scaling_factor_alpha(sent_len-1);
        expec_emission_matrix.col(training_data[counter][sent_len-1]) += gamma_i_last;
        //cout<<expec_emission_matrix.col(training_data[counter][sent_len-1])<<endl;
        //getchar();

        /*
        for t,word in enumerate(sent[0:-1]):

        gamma_ij_t = [alpha_t_hat[t][current_tag]*transition_matrix[word,current_tag,:]*emission_matrix[:,sent[t+1]]*beta_t_hat[t+1,:] for current_tag in range(num_tag_types)] --here
        gamma_i_t= [gamma_ij_t[current_tag].sum() for current_tag in range(num_tag_types)]

                    if t == 0:
                expec_pi_vector += gamma_i_t
           #adding the expected counts
            #print 'the word is',word
            expec_transition_matrix[word,:,:] += gamma_ij_t
            expec_emission_matrix[:,word] += gamma_i_t

          #computing the probability of a a path passing through a particular tag i at position t. This will give us exp
        gamma_i_last = alpha_t_hat[sent_len-1,:]*beta_t_hat[sent_len-1,:] / scaling_factor_alpha[sent_len-1]
        posterior_decoding.append(argmax(gamma_i_last))
        expec_emission_matrix[:,sent[sent_len-1]] += gamma_i_last


        */

        //for t,word in enumerate(sent[0:sent_len-1][::-1]) 
    }

    cout<<endl;
    setprecision(16);
    cout<<"the total corpus prob was "<<setprecision(16)<<total_corpus_prob<<endl;
}
/*
def forwardBackward(training_data,transition_matrix,emission_matrix,expec_transition_matrix,expec_emission_matrix,num_tag_types,pi_vector,expec_pi_vector):
    #\alpha_t(i) is p(o_0,..,0_{t-1},x_t=i). basically the sum of probabilities all data completions up to position t ending in i and producing o_t as well. 
    #emission_matrix_transpose = emission_matrix.transpose()
    #path_prob_matrix = numpy.zeros((num_tag_types,num_tag_types),dtype=theano.config.floatX)
    total_corpus_prob = 0.

    posterior_decoding = []
    for counter,sent in enumerate(training_data) :
        print 'starting a sentence '  
        #raw_input()
        sent_len = len(sent)
        alpha_t = numpy.zeros((sent_len,num_tag_types),dtype=theano.config.floatX)
        alpha_t_tilde = numpy.zeros((sent_len,num_tag_types),dtype=theano.config.floatX)
        alpha_t_hat = numpy.zeros((sent_len,num_tag_types),dtype=theano.config.floatX)
        scaling_factor_alpha = numpy.zeros(sent_len,theano.config.floatX)
        
        #computing \alpha_0
        alpha_t_tilde[0,:] = pi_vector * emission_matrix[:,sent[0]]
        #computign scaling factord
        scaling_factor_alpha[0] = 1./alpha_t_tilde[0].sum()
        alpha_t_hat[0,:] = alpha_t_tilde[0,:] * scaling_factor_alpha[0]

        #computing the alphas
        for t,word in enumerate(sent[1:]):
            #compute the alphas
            #for current_tag in range(num_tag_types):
            #    alpha_t_tilde[t+1,current_tag] = (alpha_t_hat[t,:]*transition_matrix[sent[t],:,current_tag]).sum()
            alpha_t_tilde[t+1,:] = [(alpha_t_hat  [t,:]*transition_matrix[sent[t],:,current_tag]).sum() for current_tag in range(num_tag_types)]--

            print 'the word is ',word
            print alpha_t_tilde[t+1,:]
            print 'the transition matrix is'
            print transition_matrix[sent[t],:,:]
            if numpy.isnan(alpha_t_tilde[t+1,:]).sum() >= 1:
                print 'we got a nan in position ',t
                print alpha_t_tilde[t+1,:]
                sys.exit()
            raw_input()
            #raw_input()
            alpha_t_tilde[t+1,:] *= emission_matrix[:,word] --

            #compute the scaling factor
            scaling_factor_alpha[t+1] = 1/alpha_t_tilde[t+1].sum()

            #print 'the scaling factor is ',scaling_factor_alpha[t+1]
            #scale the alphas
            alpha_t_hat[t+1,:] = alpha_t_tilde[t+1,:]*scaling_factor_alpha[t+1]--

            #print 'alpha t hat is '
            #print alpha_t_hat
            #raw_input()
        #print 'the corpus prob is ',1./scaling_factor_alpha.prod()
        total_corpus_prob += -numpy.log(scaling_factor_alpha).sum()


        #computing the betas
        beta_t_tilde = numpy.zeros((sent_len,num_tag_types),dtype=theano.config.floatX)
        beta_t_hat = numpy.zeros((sent_len,num_tag_types),dtype=theano.config.floatX) --here

        beta_t_tilde[sent_len-1,:] = [1.]*num_tag_types 
        beta_t_hat[sent_len-1,:] = [1.]*num_tag_types 
        #scaling the betas by c_{T-1}
        beta_t_hat[sent_len-1,:] *= scaling_factor_alpha[sent_len-1] --here
        
        for t,word in enumerate(sent[0:sent_len-1][::-1]) : --here
            #print 'the position is ',(sent_len-1)-(t+1),' and the word is ',word
            #compute the betas
            '''
            for current_tag in range(num_tag_types) :
                print 'next word is ',sent[(sent_len-1)-t]
                print 'beta '
                print beta_t_hat[(sent_len-1)-t,:]
                print 'emission matrix'
                print emission_matrix[:,sent[(sent_len-1)-t]]
                print 'beta tilde'
                print (beta_t_hat[(sent_len-1)-t,:]*transition_matrix[word,current_tag,:]*emission_matrix[:,sent[(sent_len-1)-t]]).sum()
                #raw_input()
            '''
            beta_t_tilde[(sent_len-1)-(t+1),:] = [(beta_t_hat[(sent_len-1)-t,:]*transition_matrix[word,current_tag,:]*emission_matrix[:,sent[(sent_len-1)-t]]).sum() for current_tag in range(num_tag_types)]

            #scale the betas
            beta_t_hat[(sent_len-1)-(t+1),:] = beta_t_tilde[(sent_len-1)-(t+1),:] * scaling_factor_alpha[(sent_len-1)-(t+1)] --here
            '''
            print 'beta t hat is '
            print beta_t_hat[(sent_len-1)-(t+1),:]
            raw_input()
            '''
            --- here

        #now that we've computed the alphas and the betas, its time to compute the expected counts for the transition parameters
        #computing the probability of a path passing through a particular pair i,j at position t (gamma_ij_t) and the path passing through i at position t
        for t,word in enumerate(sent[0:-1]):
            '''
            print 'alpha t hat is '
            print alpha_t_hat[t,:]
            print 'transition matrix'
            print transition_matrix[word,current_tag,:]
            print 'emission matrix'
            print emission_matrix[:,sent[t+1]]
            print 'beta t hat'
            print beta_t_hat[t+1,:]
            for current_tag in range(num_tag_types) :
                print 'the gamma ij vector for i being',current_tag
                print alpha_t_hat[t,current_tag]*transition_matrix[word,current_tag,:]*emission_matrix[:,sent[t+1]]*beta_t_hat[t+1,:]
                raw_input()
            '''
            gamma_ij_t = [alpha_t_hat[t][current_tag]*transition_matrix[word,current_tag,:]*emission_matrix[:,sent[t+1]]*beta_t_hat[t+1,:] for current_tag in range(num_tag_types)] --here
            #print 'gamma_ij_t is'
            #print gamma_ij_t

            gamma_i_t= [gamma_ij_t[current_tag].sum() for current_tag in range(num_tag_types)] --here
            posterior_decoding.append(argmax(gamma_i_t))

            '''
            gamma_i_t_temp = alpha_t_hat[t]*beta_t_hat[t]/scaling_factor_alpha[t]

            print 'gamma i t is '
            print gamma_i_t
            print 'gamma i t temp is'
            print gamma_i_t_temp
            #raw_input()
            '''
            #getting the expected counts for the start probabilities -here
            if t == 0:
                expec_pi_vector += gamma_i_t
           #adding the expected counts
            #print 'the word is',word
            expec_transition_matrix[word,:,:] += gamma_ij_t
            expec_emission_matrix[:,word] += gamma_i_t --here

          #computing the probability of a a path passing through a particular tag i at position t. This will give us exp
        gamma_i_last = alpha_t_hat[sent_len-1,:]*beta_t_hat[sent_len-1,:] / scaling_factor_alpha[sent_len-1]
        posterior_decoding.append(argmax(gamma_i_last))
        expec_emission_matrix[:,sent[sent_len-1]] += gamma_i_last --here
        
        if counter % 100 ==0:
            print 'finished processing sentence',counter        
    '''
    print 'printing expected transition counts'
    print expec_transition_matrix
    print 'printing expected emission counts'
    print expec_emission_matrix
    '''
    print 'total corpus prob is ',total_corpus_prob 
    return(numpy.asarray(posterior_decoding))
    #raw_input()
*/
