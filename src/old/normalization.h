//this only has static fucntions that will be used by other classes
class c_h_methods
{
    public:
    static void updateChAdagrad(boost::unordered_map<std::vector<int>, double> & c_h,std::vector<boost::unordered_map<std::vector<int>, double> > & threads_c_h_gradient
                        ,boost::unordered_map<std::vector<int>, double> & c_h_running_gradient,int n_threads,double learning_rate)
    {
        boost::unordered_map<std::vector<int>, double> current_c_h_gradient;
        for (int thread_id=0;thread_id<n_threads;thread_id++)
        {
            boost::unordered_map<std::vector<int>, double> ::iterator it;
            for (it = threads_c_h_gradient[thread_id].begin();it != threads_c_h_gradient[thread_id].end();it++)
            {
                double gradient = (*it).second;
                if (current_c_h_gradient.find((*it).first) == current_c_h_gradient.end())
                {
                    current_c_h_gradient[(*it).first] = gradient;
                }
                else
                {
                    current_c_h_gradient[(*it).first] += gradient;
                }
            }
        }
        boost::unordered_map<std::vector<int>, double>::iterator it;
        //now go over the c_h gradients and update the c_h paramter
        for (it = current_c_h_gradient.begin();it != current_c_h_gradient.end();it++)
        {
            double square = (*it).second*(*it).second;
            if (c_h_running_gradient.find((*it).first) == c_h_running_gradient.end())
            {
                c_h_running_gradient[(*it).first] = square;
            }
            else
            {
                c_h_running_gradient[(*it).first] += square;
            }
            //adagrad update
            c_h[(*it).first] += learning_rate*current_c_h_gradient[(*it).first]/sqrt(c_h_running_gradient[(*it).first]);
        }

    
    }
    static void updateChAdagrad(boost::unordered_map<std::vector<int>, std::vector<double> > & c_h,std::vector<boost::unordered_map<std::vector<int>, double> > & threads_c_h_gradient
                        ,boost::unordered_map<std::vector<int>, double> & c_h_running_gradient,int n_threads,double learning_rate)
    {
        boost::unordered_map<std::vector<int>, double> current_c_h_gradient;
        for (int thread_id=0;thread_id<n_threads;thread_id++)
        {
            boost::unordered_map<std::vector<int>, double> ::iterator it;
            for (it = threads_c_h_gradient[thread_id].begin();it != threads_c_h_gradient[thread_id].end();it++)
            {
                double gradient = (*it).second;
                if (current_c_h_gradient.find((*it).first) == current_c_h_gradient.end())
                {
                    current_c_h_gradient[(*it).first] = gradient;
                }
                else
                {
                    current_c_h_gradient[(*it).first] += gradient;
                }
            }
        }
        boost::unordered_map<std::vector<int>, double>::iterator it;
        //now go over the c_h gradients and update the c_h paramter
        for (it = current_c_h_gradient.begin();it != current_c_h_gradient.end();it++)
        {
            double square = (*it).second*(*it).second;
            if (c_h_running_gradient.find((*it).first) == c_h_running_gradient.end())
            {
                c_h_running_gradient[(*it).first] = square;
            }
            else
            {
                c_h_running_gradient[(*it).first] += square;
            }
            //adagrad update
            c_h[(*it).first][0] += learning_rate*current_c_h_gradient[(*it).first]/sqrt(c_h_running_gradient[(*it).first]);
            c_h[(*it).first][1] = exp(c_h[(*it).first][0]);
        }
    }

};
