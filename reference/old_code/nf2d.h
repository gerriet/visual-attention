#ifndef __NF2D__
#define __NF2D__

#include <string>
#include <sstream>
#include "adt/jadt.h"
#include "adt/jmatrix.h"
#include "adt/image/pic2d.h"
#include "gdisplayer.h"
#include "pixelop.h"
#include "pic3d.h"
#include <stdlib.h>
#include <stdio.h>
#include "nf.h"

namespace gtools {

  /** classical 2D Neural Field implementation 
   */

template<class nf_t>
class NeuralField2D : public NeuralField<nf_t> {
 public:
  NeuralField2D(unsigned int x, unsigned int y,	bool use_memory=false,
		int memx=256, int memy=256, int resolution=1, 
		int debug_level=1);

  ~NeuralField2D();

  void initialize();

  template<class map_t> void update(unsigned int nCycles, 
				    pic2d<map_t> const &input,
				    float const multiplier=1.0,
				    int framenum=0);
  void updateSigmoid();

  void getActivityValues(pic2d<nf_t> &activity) const;
  void getActivityValues(pic3d<nf_t> &activity) const;
  void getSigmoidValues(pic2d<unsigned char> &sigmoid_data) const;
  void getSigmoidValues(pic3d<unsigned char> &sigmoid_data) const;

  void setparameters(float alphaValue, float globalMultiplier, nf_t rValue, 
		     float Beta, float inputMult=1.0);
  void set_thresh(float change) {
    change_thresh = change;
  };

  void setkernels(unsigned int horsize, unsigned int versize, 
		  float s=5, float k=0.06);
  void set_DoG_kernels(unsigned int horsize, unsigned int versize, 
		       float s = 5, float k = 0.06, 
		       float s2 = 12, float k2 = 0.02);
  void set_G_kernels(unsigned int horsize, unsigned int versize, 
		     float s = 5, float k = 0.06);
  void displacefield(unsigned int x_move, unsigned int y_move, 
		     bool usememory=false);

  void addnfdisplayer(gdisplayer< pic2d<nf_t> > *display,
		      bool do_display=false, bool do_save=false) {
    displayer_exist=true;
    display_do = do_display;
    save_do = do_save;
    nfdisplayer = display;
  };
  
  void addsigdisplayer(gdisplayer< pic3d<float> > *display) {
    sigdisp_exist=true;
    sigdisplayer = display;
  };
  
  /* get information about NF */
  
  void activeborders(ostream &os, nf_t ulimit);
  void dumpkernel(ostream &os);

 private:
  pic2d<nf_t>* activityvalues;
  pic2d<nf_t>* sigmoidvalues;
  pic2d<nf_t>* memoryvalues;
  jt::matrix<float>* kernel2d;
  unsigned int xsize, ysize;

  bool save_do, display_do, displayer_exist, sigdisp_exist;
  gdisplayer< pic2d<nf_t> >* nfdisplayer;
  gdisplayer< pic2d<float> >* sigdisplayer;
};

template<class nf_t> 
NeuralField2D<nf_t>::NeuralField2D(unsigned int x, unsigned int y,	
				   bool use_memory,
				   int memx, int memy, int resolution, 
				   int debug_level) : xsize(x), ysize(y) {
  activityvalues = new(pic2d<float>) (x, y);
  sigmoidvalues = new(pic2d<float>) (x, y);
  sigdisp_exist = false;
  displayer_exist =false;
  display_do = false;
  change_thresh = (xsize * ysize) * 0.01; 
  debug = debug_level;
  /*  if (use_memory) {
    memoryvalues = new(pic2d<float>) (memx, memy);
    mem_x=memx;
    mem_y=memy;
    mem_resolution=resolution;
  } 
  else
    mem_resolution=0;
  */
  // memory (outside visible area) has to be implemented

};


template<class nf_t> NeuralField2D<nf_t>::~NeuralField2D() {
  delete activityvalues;
  delete sigmoidvalues;
  if (kernel2d)
    delete kernel2d;
};


template<class nf_t> template<class map_t> 
void NeuralField2D<nf_t>::update(unsigned int nCycles, 
				 pic2d<map_t> const &input,
				 float const multiplier,
				 int framenum) {

  /** Zentrale Routine zur Aktualisierung der Aktivation im NF */

 
  if (debug > 0)
    cout << "Start NF update" << endl;
  sigmoid_active = false;
  updateSigmoid();

  // quick hack for suppressing wrong border activation (missing inhibition) 
  pic2d<nf_t> top_down(xsize, ysize);
  int x2 = xsize/2;
  int y2 = ysize/2;
  int rand = 9;
  top_down.clear(nf_t(0.0));
  for (int x=0; x < int(xsize); x++) 
    for (int y=0; y < int(ysize); y++) {
      int dist = (x2-abs(x-x2)) <? (y2-abs(y-y2));
      if (dist<rand)
	top_down.pset(x, y, -0.5/(dist+2));
    }

  if (debug > 0) cout << "M:" << input.max() << endl;

  float newvalue=0.0, oldvalue, change;
  change = change_thresh+1;
  for (unsigned int cycle=0; (((cycle < nCycles) && (change > change_thresh)) || 
			      (cycle < 3)); cycle++) {
    float sigmean = meanvalue(*sigmoidvalues);
    pic2d<nf_t> tmp((*sigmoidvalues) * (*kernel2d));
    change = 0;
    
    float globaleffect = restingvalue - sigmean * globalmultiplier;
    for (unsigned int x=0; x < xsize; x++)
      for (unsigned int y=0; y < ysize; y++) {
	oldvalue = activityvalues->get(x, y);
	newvalue =
	  alpha * (globaleffect 
		   + tmp.get(x,y) 
		   + top_down.get(x,y)
		   + inputmultiplier * input.get(x, y)) +
	  (1 - alpha) * oldvalue;
	activityvalues->pset(x, y, newvalue);
	if (newvalue > 0)
	  sigmoid_active = true;
	change += fabs(newvalue-oldvalue);
      }
    if (debug > 0)
      cout << "NFchange abs - rel (cycle) mean: " << change << " - " 
	   << change/(xsize*ysize) << " (" << cycle << ") " 
	   << meanvalue(*activityvalues)	<< endl;
    if (displayer_exist && display_do) {
      pic2d<float> tmp(*activityvalues);
      addConstValue(tmp,float(3.0));
      multConstValue(tmp,float(50));
      nfdisplayer->display(*activityvalues,"NeuralField");
    }
    updateSigmoid();
    if (sigdisp_exist && display_do) 
      sigdisplayer->display(*sigmoidvalues,"Sigmoid");
  }
}


template<class nf_t> 
void NeuralField2D<nf_t>::getActivityValues(pic2d<nf_t> &activity) const {
  activity = *activityvalues;
};

template<class nf_t> 
void NeuralField2D<nf_t>::getActivityValues(pic3d<nf_t> &activity) const {
  activity.setpic2d(0, *activityvalues);
};

template<class nf_t>  
void NeuralField2D<nf_t>::getSigmoidValues(pic2d<unsigned char> &sigmoid_data) 
     const {
  for (unsigned int x = 0; x < xsize; x++)
    for (unsigned int y = 0; y < ysize; y++)
      sigmoid_data.pset(x, y, (unsigned char)
			sigmoid(activityvalues->get(x,y)));
};

template<class nf_t>  
void NeuralField2D<nf_t>::getSigmoidValues(pic3d<unsigned char> &sigmoid_data) 
     const {
  for (unsigned int x = 0; x < xsize; x++)
    for (unsigned int y = 0; y < ysize; y++)
      sigmoid_data.pset(x, y, 0, (unsigned char)
			sigmoid(activityvalues->get(x,y)));
};

template<class nf_t>
void NeuralField2D<nf_t>::setparameters(float alphaValue, float globalMultiplier, 
					nf_t rValue, float Beta,
					float inputMult) {
  alpha = alphaValue;
  globalmultiplier = globalMultiplier;
  restingvalue = rValue;
  beta = Beta;
  inputmultiplier = inputMult;
}


template<class nf_t> void NeuralField2D<nf_t>::initialize() {
  if (debug > 0)
    cout << "NF init" << endl;
  activityvalues->clear(restingvalue);
  updateSigmoid();
}

template<class nf_t> void NeuralField2D<nf_t>::updateSigmoid() {
  /** Updates the stored values of sigmoid activity
   *  should be called after every change of the activity values
   */

  for (unsigned int x = 0; x < xsize; x++)
    for (unsigned int y = 0; y < ysize; y++)
      sigmoidvalues->pset(x, y, sigmoid(activityvalues->get(x,y)));
}

template<class nf_t> void 
NeuralField2D<nf_t>::setkernels(unsigned int horsize, unsigned int versize, 
			        float s, float k) {
  kernel2d = new(jt::matrix<float>)(horsize,versize);
  for (unsigned int i = 0; i < horsize; i++) 
    for (unsigned int j = 0; j < versize; j++) {
      float dist = ((i-int(horsize-1)/2) * (i-int(horsize-1)/2) +
		    (j-int(versize-1)/2) * (j-int(versize-1)/2))/(s*s) ;
      if (false) 
	// Braumann Version - k=2, s=5
	(*kernel2d)[i][j] = k*exp(-3*dist/2)-exp(-dist);
      else 
	// Backer Version k=0.06, s=5
	(*kernel2d)[i][j] = k*exp(-dist)-k/3*exp(-dist/10);
      
    }
}

template<class nf_t> void 
NeuralField2D<nf_t>::set_DoG_kernels(unsigned int horsize, unsigned int versize, 
				     float s, float k, 
				     float s2, float k2 ) {
  kernel2d = new(jt::matrix<float>)(horsize,versize);
  for (unsigned int i = 0; i < horsize; i++) 
    for (unsigned int j = 0; j < versize; j++) {
      float dist = (i-int(horsize-1)/2) * (i-int(horsize-1)/2) +
	(j-int(versize-1)/2) * (j-int(versize-1)/2) ;
      (*kernel2d)[i][j] = k*exp(-dist/(s*s))-k2*exp(-dist/(s2*s2));
      
    }
}

template<class nf_t> void 
NeuralField2D<nf_t>::set_G_kernels(unsigned int horsize, unsigned int versize, 
				   float s, float k) {
  kernel2d = new(jt::matrix<float>)(horsize,versize);
  for (unsigned int i = 0; i < horsize; i++) 
    for (unsigned int j = 0; j < versize; j++) {
      float dist = (i-int(horsize-1)/2) * (i-int(horsize-1)/2) +
	(j-int(versize-1)/2) * (j-int(versize-1)/2) ;
      (*kernel2d)[i][j] = k*exp(-dist/(s*s));      
    }
}

template<class nf_t> void NeuralField2D<nf_t>::activeborders(ostream &os,
							     nf_t ulimit) {
  int maxx=0, maxy=0;
  int minx=activityvalues->w(), miny=activityvalues->h();
  int count = 0;
  bool activity = false;
  for (int x = 0; x < activityvalues->w(); x++)
    for (int y = 0; y < activityvalues->h(); y++)
      if (activityvalues->get(x,y) > ulimit) {
	count++;
	activity = true;
	maxx = maxx >? x;
	maxy = maxy >? y;
	minx = minx <? x;
	miny = miny <? y;
      }
  if (activity)
    os << "(" << minx << "/" << miny << ") - ("
       << maxx << "/" << maxy << ") : " << count << endl;
  else
    os << "0 activity" << endl;
}

template<class nf_t>  void NeuralField2D<nf_t>::dumpkernel(ostream &os) {
  os << *kernel2d;
};

}

#endif
