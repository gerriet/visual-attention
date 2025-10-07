#ifndef __NF__
#define __NF__

#include <string>
#include <sstream>
#include "adt/jadt.h"
#include "adt/jmatrix.h"
#include "adt/image/pic2d.h"
#include "pic3d.h"
#include <stdlib.h>
#include <stdio.h>

namespace gtools {

template<class nf_t> class NeuralField {
 public:
  NeuralField() : alpha(0.5), restingvalue(nf_t(0.25)), 
    globalmultiplier(nf_t(0)), beta(30), sigmoid_active(false), debug(1), 
    change_thresh(0.0), inputmultiplier (nf_t(1)), sigtype(2) {};
  ~NeuralField() {};

  void setparameters(float alphaValue, nf_t globalMultiplier, 
		     nf_t rValue=-0.25, float Beta=30.0,
		     nf_t inputMult=1.0);
  void setverbose(int v) {
    debug = v;
  }
  bool active() const {
    return sigmoid_active;
  };
  
  void initialize();
  void updateSigmoid();

  nf_t sigmoid(const nf_t in) const {
    if (sigtype == 0)
      return(in > 0);
    else if (sigtype == 1)
      return(1/(1+exp(-beta*in)));
    else if (sigtype == 2) // still to do
      return(approx_sig(in));
    else
      return(0);
  };

  // Approximation of sigmoid curve by five linear pieces
  nf_t approx_sig(const nf_t in) const  {
    nf_t abs_in = fabs(in);
    if (abs_in > 5 / beta) // 
      return(in>0?1:0);
    else if (abs_in < 2.5 / beta)
      return(0.5+0.17*beta*in);
    else
      if (in > 0)
	return(.8575+in*beta*0.027);
      else
	return(.1425+in*beta*0.027);
  };

 protected:
  float alpha;
  nf_t restingvalue;
  nf_t globalmultiplier;
  float beta;
  bool sigmoid_active;
  int debug;
  float change_thresh;
  nf_t inputmultiplier;
  unsigned char sigtype;
};

template<class nf_t>
void NeuralField<nf_t>::setparameters(float alphaValue, 
				      nf_t globalMultiplier, 
				      nf_t rValue, float Beta,
				      nf_t inputMult) {
  alpha = alphaValue;
  globalmultiplier = globalMultiplier;
  restingvalue = rValue;
  beta = Beta;
  inputmultiplier = inputMult;
}

}

#endif
