#ifndef __ATTENTION_ONSET__
#define __ATTENTION_ONSET__

#include <string>
#include <iostream>
#include <time.h>
#include "adt/image/pic2d.h"
#include <stdlib.h>
#include <stdio.h>
#include "pic3d.h"
#include "attention.h"
#include "esab2_config.h"

using jt::pic2d;

namespace gtools {

class OnsetFeature : public AttentionFeature {
 public:
  OnsetFeature(unsigned int n);
  // main computation routine: 
  // expects that input is set via setinput and delivers results in
  // result, saliencydistribution, ...
  void compute();
  void setparameters() {
  };
  void setsavedir(string dir) {
    savedir = dir;
  };

  // Problem with resizing is NOT solved
  void setinput(pic2d<unsigned char>* input) {

  };

  void dump(ostream &os) {
    os << "OnsetFeature properties : " << featurenum << " " << savedir << endl
  };

 private:
  string savedir;
  pic2d<unsigned char>* inputframe;
  pic2d<unsigned char>* lastframe;
  unsigned int xsize, ysize;
};

}

#endif
