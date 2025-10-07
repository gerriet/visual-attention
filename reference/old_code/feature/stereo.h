#ifndef __ATTENTION_STEREO__
#define __ATTENTION_STEREO__

#include <string>
#include <iostream>
#include <time.h>
#include "adt/image/pic2d.h"
#include <stdlib.h>
#include <stdio.h>
#include "pic3d.h"
#include <vector>
#include "pictureop.h"
#include "attention.h"

using jt::pic2d;

namespace gtools {

class StereoFeature : public AttentionFeature {
 public:
  StereoFeature(unsigned int n, unsigned int depth_range,
		int x_size, int y_size, int numo, int anzahl,
		int min_disp, int max_disp, int corr_method=0, int filtern=1,
		int xfield=64, int yfield=64);

  StereoFeature() {
    StereoFeature(0,11, 256, 256, 3, 3, -25, 0, 1);
  };

  /** Main Feature computation routine: 
      expects that input is set via setinput and delivers results in
      result, saliencydistribution, ...
  */
  void compute();

  void setparameters(int dlength, float korr_sw, float sig_w, int m_d,
		     int sig_w_2 = -1, float excl=1.0);

  void setsavedir(string dir) { savedir = dir; };

  // Problem with resizing is NOT solved
  void setinput(pic2d<unsigned char>* leftinput,
		pic2d<unsigned char>* rightinput);

  pic3d<unsigned char> getsaliencydistribution(int, int);

  void dump(ostream &os);

  void savestate();

 protected:
  void compute_stereo_feature(int const stereosize=128, 
			      int const internalsize=64, 
			      int const fieldsize=64);
  void computedisp();
  void lkfdisp(int b, int y, float* leftrow, float* rightrow);
  void lkwindisp(int b, int y, float* leftrow, float* rightrow);
  void complex_disp(int b, int y, float* leftreal, float* rightreal,
		    float* leftimag, float* rightimag);
  void signal_stat(float* input, float* out_mw, float* out_var,
		   int length, int dl, int hdl);
  void sort_in_row(int b, int y);

  pic2d<unsigned char>* dispwinner;
  pic3d<unsigned char>* saliencydistribution;
  pic3d<float>* confidence;
  pic3d<unsigned char>* alldisp;
  gaborfilter<float>* gabor_stereo;


  string savedir;
  pic2d<unsigned char>* stereoinput;
  float weight[9];
  float* win;
  float* row_corr;
  unsigned int depth, xsize, ysize;
  pic3d<float>* lgabor;
  pic3d<float>* rgabor;
  pic3d<unsigned char>* sigl;
  pic3d<unsigned char>* sigr;
  pic2d<int>* winner;
  pic3d<unsigned char>* completedisp;
  pic2d<unsigned char>* dispbest;
  unsigned int numorient;
  int anz, dl, md, mindisp, maxdisp;
  int filternum, correlation_method;
  float korrsw, sigw, sigw2, exclusivity;
};



class StereoMultiFeature : public StereoFeature {
  /** Multiscale version of the StereoFeature   */
 public:
  StereoMultiFeature(unsigned int n, unsigned int depth_range,
		     int x_size, int y_size, int numo, int anzahl,
		     int min_disp, int max_disp, int corr_method=0, 
		     int filtern=1, int xfield=64, int yfield=64);
  void initialize(int const anzdisp, int const numscales, 
		  int const sc_method, unsigned int orientations[]);
  void compute();
  void savestate();

 protected:
  void computedisp();
  void lkfdisp(int b, int y, int scale, bool first_scale, int dl,
	       float* leftrow, float* rightrow);
  void lkwindisp(int b, int y, int scale, bool first_scale, int dl,
		 float* leftrow, float* rightrow);
  void disp_borders(int anz, int size, int scale, int x,
		    int &from, int &to) const;
  void sort_in_row(int b, int y);
  int scales;
  int scale_method;
  int anz_disp;
  bool initialized;
  int scale_step;
  float exclusivity;
  vector< vector < vector<float> > > row_corr;
  vector< gaborfilter<float>* > gaborf_list;
  vector< pic3d<float>* > lgabor_m;
  vector< pic3d<float>* > rgabor_m;
  vector< pic3d<float>* > scale_conf;
  unsigned int scale_factor[];
};

}

#endif
