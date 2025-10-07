#ifndef __ATTENTION__
#define __ATTENTION__

#include <string>
#include <sstream>
#include <time.h>
#include "adt/jadt.h"
#include "adt/image/pic2d.h"
#include "imagefilter.h"
#include "unixspec/jthreads.h"
#include "pthread.h"
#include "pictureop.h"
#include <stdlib.h>
#include <stdio.h>
#include "pic3d.h"

namespace gtools {

  int const nf_size = 64;

  using jt::pic2d;


  class AttentionFeature {
  public:
    typedef enum {
      color,
      stereo,
      stereomulti,
      symmetry,
      symmetrymulti,
      eccentricity,
    } featuretype;
    
    virtual ~AttentionFeature() {};

    void setinput(pic2d<unsigned char> *inputimage) {
      input = inputimage;
    }; 

    virtual void compute() = 0;

    void setsize(unsigned int csize, unsigned int rsize) {
      use_own_thread = false;
      size = csize;
      resultsize = rsize;
    };

    void set_verbose(int verb=1) {
      verbose = verb;
    };

    pic2d<unsigned char> current_result() {return (*result);};

    void wait_on();
    
    int featurenum;
    pic2d<unsigned char>* oldresult;
    pic2d<unsigned char>* result;
    pic2d<unsigned char>* input;
    unsigned int size, resultsize;
    bool use_own_thread;
    featuretype ftype;
    int verbose;
  };

}

#endif
