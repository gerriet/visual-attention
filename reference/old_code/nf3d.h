#ifndef __NF3D__
#define __NF3D__

#include <string>
#include <sstream>
#include <list>
#include "adt/jadt.h"
#include "adt/jmatrix.h"
#include "adt/image/pic2d.h"
#include "gdisplayer.h"
#include "pic3d.h"
#include <cstdlib>
#include <cstdio>
#include "nf.h"

namespace gtools {

typedef float activity_type;

/**
 * Neural Field 3d implementation 
 */

template<class nf_t> class NeuralField3D : public NeuralField<nf_t> {
 public:
  NeuralField3D(unsigned int x, unsigned int y, unsigned int z,
		bool use_memory=false, int memx=256, int memy=256,
		int resolution=1, int debug_level=1);
   ~NeuralField3D();

  template<class map_t> void update(unsigned int nCycles, 
				    pic3d<map_t> const &input,
				    float const multiplier=1.0,
				    int framenum=0);
  void updateSigmoid();
  void initialize();

  void getActivityValues(pic3d<nf_t> &activity) const;
  void getSigmoidValues(pic3d<unsigned char> &sigmoid) const;

  void setkernels(unsigned int horsize, float* const horKernel, 
		  unsigned int versize, float* const verKernel, 
		  unsigned int depthsize, float* depthKernel);
  void setkernels(unsigned int horsize, unsigned int versize, 
		  unsigned int depthsize, float s=5, float k=0.06);
  void set_thresh(float change) {
    change_thresh = change;
  };

  void addnfdisplayer(gdisplayer< pic3d<nf_t> > *display, 
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

  void activeborders(ostream &os, nf_t ulimit);

  void displacefield(unsigned int x_move, unsigned int y_move, 
		     unsigned int z_move, bool usememory=false);

 private:
  pic3d<nf_t>* activityvalues;
  pic3d<nf_t>* sigmoidvalues;
  pic3d<nf_t>* memoryvalues;
  jt::matrix<float>* kernel2d;
  unsigned int xsize, ysize, zsize;
  float* horkernel;
  float* verkernel;
  float* depthkernel;
  int offset_x, offset_y;
  int mem_x, mem_y, mem_resolution;

  bool save_do, display_do, displayer_exist, sigdisp_exist;
  gdisplayer< pic3d<nf_t> >* nfdisplayer;
  gdisplayer< pic3d<float> >* sigdisplayer;
};

template<class nf_t>
NeuralField3D<nf_t>::NeuralField3D(unsigned int x, unsigned int y, unsigned int z, 
				   bool use_memory, int memx, 
				   int memy, int resolution, 
				   int debug_level) {
  
  /** 3D Neural field implementation following DynPerz2000 
   *  corresponding workspace: depth_current
   *  Modifications: no convolution across the depth
   *                 using true 2D kernel
   *                 moving methods into generic nf-class
   * (21.11.)        optional display during field-update
   */

  xsize = x;
  ysize = y;
  zsize = z;
  activityvalues = new(pic3d<float>) (x, y, z);
  activityvalues->setbandimgproperties(5,1,255,40,2.0);
  sigmoidvalues = new(pic3d<float>) (x, y, z);
  sigdisp_exist = false;
  displayer_exist =false;
  display_do = false;
  change_thresh = (xsize * ysize * zsize) * 0.01; 
  debug = debug_level;
  if (use_memory) {
    memoryvalues = new(pic3d<float>) (memx, memy, z);
    mem_x=memx;
    mem_y=memy;
    mem_resolution=resolution;
  } 
  else
    mem_resolution=0;
}

template<class nf_t> NeuralField3D<nf_t>::~NeuralField3D() {
  delete activityvalues;
  delete sigmoidvalues;
  if (kernel2d) 
    delete kernel2d;
}


template<class nf_t> 
void NeuralField3D<nf_t>::getActivityValues(pic3d<nf_t> &activity) const {
  for (unsigned int z = 0; z < zsize; z++)
    activity.setpic2d(z,activityvalues->getpic2d(z));
}


template<class nf_t>  
void NeuralField3D<nf_t>::getSigmoidValues(pic3d<unsigned char> &sigmoid_data) const {
  for (unsigned int x = 0; x < xsize; x++)
    for (unsigned int y = 0; y < ysize; y++)
      for (unsigned int z = 0; z < zsize; z++)
	sigmoid_data.pset(x, y, z, (unsigned char)
			  sigmoid(activityvalues->get(x,y,z)));
}


template<class nf_t> template<class map_t> 
void NeuralField3D<nf_t>::update(unsigned int nCycles, 
				 pic3d<map_t> const &input,
				 float const multiplier,
				 int framenum) {
  /** Main routine for updating the neural field the given number of cycles
   *  with the given input
   */
  
  pic3d<float> tmp(xsize, ysize, zsize);
  pic2d<float> depth_accu(xsize, ysize);
  pic2d<float> sigplane(xsize,ysize);
  float tmpvalue, newvalue, globaleffect, planeeffect, change;
  float sigmoids[zsize];
  // stop updating when change is smaller than 1 percent !
  if (debug > 0)
    cout << "Start NF update" << endl;
  sigmoid_active = false;
  updateSigmoid();
  change = change_thresh+1;
  // Problem: bei cycle = 0 und change
  for (unsigned int cycle=0; (((cycle < nCycles) 
			       && (change > change_thresh)) || 
			      (cycle < 2)); cycle++) {
    depth_accu.clear(0.0);
    float sigmean = 0.0;
    for (unsigned int plane = 0; plane < zsize; plane ++) {
      pic2d<float> sigplane(sigmoidvalues->getpic2d(plane));
      sigmoids[plane] = meanvalue(sigplane);
      sigmean += sigmoids[plane];
      for (unsigned int x=0; x < xsize; x++)
	for (unsigned int y=0; y < ysize; y++) {
	  tmpvalue = sigplane.get(x, y);
	  depth_accu.pset(x, y, 0, tmpvalue + depth_accu.get(x,y));
	}
      tmp.setpic2d(plane, sigplane * (*kernel2d));
    }   
    globaleffect = restingvalue - sigmean * globalmultiplier;
    float oldvalue;
    change = 0;
    for (unsigned int plane=0; plane < zsize; plane ++) {
      planeeffect = - globalmultiplier * sigmoids[plane] * 5; // * 3
      for (unsigned int x=0; x < xsize; x++)
	 for (unsigned int y=0; y < ysize; y++) {
	  oldvalue = activityvalues->get(x,y,plane);
	  newvalue =
	    alpha * (globaleffect 
		     + tmp.get(x,y,plane) 
		     - depth_accu.get(x,y) / zsize 
		     + planeeffect 
		     + inputmultiplier * input.get(x,y,plane)) +
	    (1 - alpha) * oldvalue;
	  activityvalues->pset(x, y, plane, nf_t(newvalue));
	  if (newvalue > 0)
	    sigmoid_active = true;
	  change += fabs(newvalue-oldvalue);
	}
    }
    if (debug > 0)
      cout << "NFchange abs - rel (cycle) mean: " << change << " - " 
	   << change/(xsize*ysize*zsize) << " (" << cycle << ") " 
	   << meanvalue(activityvalues->getpic2d(4)) << endl;
    if (displayer_exist && display_do)
      nfdisplayer->display(*activityvalues, "NeuralField");
    updateSigmoid();
    if (sigdisp_exist && display_do) 
      sigdisplayer->display(*sigmoidvalues, "Sigmoid");
  }
};


template<class nf_t> void NeuralField3D<nf_t>::updateSigmoid() {
  /** Updates the stored values of sigmoid activity
   *  should be called after every change of the activity values
   */

  for (unsigned int x = 0; x < xsize; x++)
    for (unsigned int y = 0; y < ysize; y++)
      for (unsigned int z = 0; z < zsize; z++) 
	sigmoidvalues->pset(x, y, z,
			    sigmoid(activityvalues->get(x,y,z)));
}


template<class nf_t> void NeuralField3D<nf_t>::initialize() {
  if (debug > 0)
    cout << "NF init" << endl;
  for (unsigned int x = 0; x < xsize; x++)
    for (unsigned int y = 0; y < ysize; y++)
      for (unsigned int z = 0; z < zsize; z++) {
	// add random value
	activityvalues->pset(x,y,z,restingvalue);
      }
  updateSigmoid();
}

template<class nf_t> void 
NeuralField3D<nf_t>::setkernels(unsigned int horsize, float* const horKernel, 
				unsigned int versize, float* const verKernel, 
				unsigned int depthsize, float* depthKernel) {
  cerr << "this setkernels is depricated " << endl;  
}

template<class nf_t> void 
NeuralField3D<nf_t>::setkernels(unsigned int horsize, unsigned int versize, 
			       unsigned int depthsize, float s, float k) {
  kernel2d = new(jt::matrix<float>)(horsize,versize);
  for (unsigned int i = 0; i < horsize; i++) 
    for (unsigned int j = 0; j < versize; j++) {
      if (false) {
	// Braumann Version - k=2, s=5
	float dist = ((i-int(horsize-1)/2) * (i-int(horsize-1)/2) +
		      (j-int(versize-1)/2) * (j-int(versize-1)/2))/(s*s) ;
	(*kernel2d)[i][j] = k*exp(-3*dist/2)-exp(-dist);
      } else {
	// Backer Version k=0.06, s=5
	float dist = ((i-int(horsize-1)/2) * (i-int(horsize-1)/2) +
		      (j-int(versize-1)/2) * (j-int(versize-1)/2))/(s*s) ;
	(*kernel2d)[i][j] = k*exp(-dist)-k/3*exp(-dist/10);
      }
    }
}

template<class nf_t> void 
NeuralField3D<nf_t>::displacefield(unsigned int x_move, unsigned int y_move, 
				  unsigned int z_move, bool usememory) {
  int tx, ty;
  pic2d<float> singlemap (xsize, xsize,1);
  for (unsigned int z=0; z < zsize; z++) {
    singlemap = activityvalues->getpic2d(z);
    for (unsigned int x=0; x < xsize; x++)
      for (unsigned int y=0; y < ysize; y++) {
	tx = x + x_move;
	ty = y + y_move;
	if ((tx > 0) && (ty > 0) && (tx < int(xsize)) && (ty < int(ysize)))
	  activityvalues->pset(x, y, z, singlemap.get(x+ x_move,y + y_move));
	else
	  activityvalues->pset(x, y, z, restingvalue); 
      }
  }
  updateSigmoid();
}

template<class nf_t> void NeuralField3D<nf_t>::activeborders(ostream &os,
							     nf_t ulimit) {
  int maxx=0, maxy=0, maxz=0;
  int minx=activityvalues->w(), miny=activityvalues->h(),
    minz= activityvalues->d();
  int count = 0;
  bool activity = false;
  for (int x = 0; x < activityvalues->w(); x++)
    for (int y = 0; y < activityvalues->h(); y++)
      for (int z = 0; z < activityvalues->d(); z++)
	if (activityvalues->get(x,y,z) > ulimit) {
	  count++;
	  activity = true;
	  maxx = maxx >? x;
	  maxy = maxy >? y;
	  maxz = maxz >? z;
	  minx = minx <? x;
	  miny = miny <? y;
	  minz = minz <? z;
	}
  if (activity)
    os << "(" << minx << "/" << miny << "/" << minz << ") - ("
       << maxx << "/" << maxy << "/" << maxz << ") : " << count << endl;
  else
    os << "0 activity" << endl;
}

}

#endif
