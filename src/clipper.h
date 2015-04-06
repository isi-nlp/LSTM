#ifndef CLIPPER_H
#define CLIPPER_H

namespace nplm {
  struct Clipper{
  double operator() (double x) const { 
    return std::min(0.5, std::max(x,-0.5));
    //return(x);
  }
};

}

#endif


