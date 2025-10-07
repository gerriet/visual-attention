#ifndef __COLOR_FEATURE__
#define __COLOR_FEATURE__

#include "adt/jadt.h"
#include "adt/image/pic2d.h"
#include "imagefilter.h"
#include <stdlib.h>
#include <stdio.h>
#include "pic3d.h"
#include "attention.h"
#include "eccentricity.h" 

namespace gtools {

using jt::pic2d;

/** ColorSegment ist erweitert aus der Segmentklasse des EccentricityFeatures
 */
 class ColorSegment : public Segment_Properties {
 public:
   ColorSegment(unsigned int thislabel);
 private:
   list<ColorSegment*> listNeighbours;
   float mean[3];
   float sigma_2[3];
   float mf_gradient;
   float pg_x, pg_y;
   int perimeter;
 };


 struct neighbour_list {
   int label;
   int border_length;
   struct region *reg;
   struct neighbour_list *next;
 };
 
 struct region {
   int label;
   int n_of_points;
   double mean[3];
   double sigma_2[3];
   double mf_gradient;
   double pg_x;
   double pg_y;
   int perimeter;
   struct neighbour_list *neighbours;
   struct region *next;
   struct region *prev;
 };

 /* Merkmalsberechnung Farbe 
    - eine fast identische Kopie der Khoros-Sourcen */
  class ColorFeature : public AttentionFeature {
  public:

    ColorFeature(unsigned int n) {
      featurenum = n;
      ftype = color;
      result = NULL;
      oldresult = NULL;
      output_data = 0;
      attr_values = 0;
      orients = 0;
      munsell = NULL;
    };

    void compute();

    pic2d<unsigned char> compute_color_features(string const inputfile, 
						int const internalsize=128);

    void evalColor(pic3d<float> const &munsell, pic2d<unsigned char> &result);

    void initialize(int fsize, float thres=8, float t_par=5, int attr_t=10,
		    float min_seg=0.0005, float max_seg=0.15, float msal=32,
		    float prior=0.0);

    float saliency(float in) { // [0,1] -> [0,1]
      return(1/(1+exp(-4*(in*2-1))));
    };

    template<class map_t> void rgb_to_munsell(pic2d<map_t> const &source, 
					      pic3d<float> &dest,
					      bool hvc=true);

    void savestate() const;

  private:
    float threshold, thres_par, max_sal, priorize_exclusivity;
    int attr_thres, min_seg_size, max_seg_size;
    int* output_data;
    int* orients;
    int quantity[12];  /* number of labels in each of the color intervals */
    float* attr_values;
    pic3d<float>* munsell;
  };
 
  /** Konvertiert das Eingabebild von RGB in den Munsell-Farbraum
   */
  template<class map_t> void 
    ColorFeature::rgb_to_munsell(pic2d<map_t> const &source, 
				 pic3d<float> &dest, bool hvc) {
    if (source.channels() != 3)
      cerr << "rgb_to_munsell works only for 3-channel images!" 
	   << source.channels() << endl;
    float out_vect[3];
    float X, Y, Z, H1, H2, H3, M1, M2, Hbar, S1, S2, H, V, C;
    for (int y=0; y < source.h(); y++)
      for (int x=0; x < source.w(); x++) {
	// FCC RGB to CIE XYZ transform. 
	X = 0.607 * float(source.get(x,y,0)) 
	  + 0.174 * float(source.get(x,y,1)) 
	  + 0.201 * float(source.get(x,y,2));
	Y = 0.299 * float(source.get(x,y,0)) 
	  + 0.587 * float(source.get(x,y,1)) 
	  + 0.114 * float(source.get(x,y,2));
	Z = 0.066 * float(source.get(x,y,1))
	  + 1.117 * float(source.get(x,y,2));
	X = cbrt(100*X/250.410);
	Y = cbrt(100*Y/255.000);
	Z = cbrt(100*Z/301.665);
	
	H1=11.6*(X-Y);
	H2=11.6*(Z-Y);
	H3=11.6*Y-1.6; // H3=25.0*Y-16
	
	M1=H1;
	M2=0.4*H2;
	
	Hbar=atan2(M2, M1);
	S1 = (8.880+0.966*cos(Hbar)) * M1;
	S2 = (8.025+2.558*sin(Hbar)) * M2;
	
	if (hvc) {             // hvc
	  H=(atan2(S2, S1))*57.29578;// degrees
	  if (H<0)
	    H+=360.0;
	  V=H3;
	  C=sqrt(S1*S1+S2*S2);
	  out_vect[0]= H;
	  out_vect[1]= V;
	  out_vect[2]= C;
	} else {               // ls1s2
	  out_vect[0]= H3;
	  out_vect[1]= S1;
	  out_vect[2]= S2;
	}
	dest.pset(x,y,0,out_vect[0]);
	dest.pset(x,y,1,out_vect[1]);
	dest.pset(x,y,2,out_vect[2]);      
      }
  }  

  int add_pix_to_reg (double data_vect[3], struct region *reg_ptr, int x, 
		      int y, int i);
  struct region *search_reg_label(struct region *start_region, 
				  int search_label);
  double euclid_metric(double vect_1[3], double vect_2[3]);
  double euclid_metric_ptr(double *vect_1[3], double *vect_2[3]);
  int min_ind(double vect[4]);
  double updated_mean(double inp_vect[3], struct region *reg_ptr, int k);
  int make_neighbours(struct region *reg_ptr_1, struct region *reg_ptr_2);
  int new_neighbour(struct region *reg_ptr_1, struct region *reg_ptr_2);
  struct neighbour_list 
    *search_label_in_n_list(int label, struct region *reg_ptr);
  double win_sigma(pic3d<float> const &input, int pos, int w, int h);
  double min_el(double inp_vect[3]);
  double max_el(double inp_vect[3]);
  double updated_sigma_2(double inp_vect[3], struct region *reg_ptr, int k);
}

#endif
