#ifndef __SymmetryFeature__
#define __SymmetryFeature__

#include "adt/image/pic2d.h"
#include "pic3d.h"
#include "attention.h"
#include "unixspec/jtimer.h"
#include <vector>

namespace gtools {

class SymmetryFeature : public AttentionFeature {
 public:
  SymmetryFeature() {
    SymmetryFeature(0);
  };

  SymmetryFeature(unsigned int n) {
    ftype = symmetry;
    featurenum = n;
    offset = 4;
    result = NULL;
    oldresult = NULL;
    verbose = 1;
  };

  void compute(); 

  void initialize(int const fsize=256, int const off=-1, int const stp=-1, 
		  int const a_bands=5, int const orient=12, 
		  float const k0=0.75, float const r=0.5);

  void savestate();

  pic3d<int> getbands(int rsize=-1);

  //   void wait_on();

  int a_num_bands;

 private:
  void symmetry_intern(unsigned char const orientation_filter[], 
		       int const alpha);
  int step;
  int orientations;
  int offset;
  int deltaalpha;
  gaborfilter<float>* gabor12;
  //  pic3d<int>* symmetry_bands;
  int* symmetrybands;
  // jt::thread *threadp;
};



class SymmetryMultiFeature : public AttentionFeature {
  /** Multiscale version of the SymmetreFeature
      - need a clever parametrization mechanism!
  */
 public:
  SymmetryMultiFeature(unsigned int n) {
    featurenum = n;
    ftype = symmetrymulti;
    result = NULL;
    oldresult = NULL;
    verbose=1;
  };
  
  void compute();

  void savestate();

  void initialize(int const fsize=256, int const numscales=3, int const off=0, 
		  int const stp=4, int const a_bands=5, int const orient=12, 
		  float const k0=0.25, float const r=0.5) {
    scales = numscales;
    size = fsize;
  };

  void initialize_scale(int const which=0, int const fsize=-1, int const off=4, 
			int const stp=4, int const a_bands=4, 
			int const orient=12, float const k0=0.25, 
			float const r=0.5) {

    if (which < scales) {
      int use_size = fsize;

      if (fsize < 0) {
	int sz = 1 << which;
	use_size = fsize < 0 ? int(size / sz) : fsize;
      }

      feat.push_back(new SymmetryFeature(which));
      feat[which]->set_verbose(2);
      feat[which]->setsize(use_size, use_size);
      feat[which]->initialize(use_size, off, stp, a_bands, orient, k0, r);

    } else
      cerr << "SymmMultiFeature initialize_scale - scale not existent: " 
	   << which << " >= " << scales << endl;
  };
 private:
  vector<SymmetryFeature*> feat;
  int scales;
};


}

#endif
