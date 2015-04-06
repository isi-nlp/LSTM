#include <Eigen/Dense>
#include "myMutex.h"

//#include "util.h"

//all this struct is doing is allowing thread safe write access into a parameter.There will be one of these for each row in the matrix
class threadAdd 
{
private:
    MutexType lock;
public:
    void add(double learning_rate,double weight,Matrix<double, Dynamic, Dynamic>::RowXpr& lhs_row,Matrix<double, Dynamic, Dynamic>::ColXpr& gradient_col)
    {
        ScopedLock lck(lock); // locks the mutex
        lhs_row  += learning_rate*weight*gradient_col.transpose();   
    } // automatically releases the lock when lck goes out of scope.
};

