#ifndef __ATTENTION_ECCENTRICITY__
#define __ATTENTION_ECCENTRICITY__

#include "adt/image/pic2d.h"
#include "pic3d.h"
#include "adt/container/jassoc.h"
#include "math/moments2.h"

namespace gtools {

  /** Einfache Segmentstruktur für Merkmalsberechnung Exzentrizität
   */
  class Segment_Properties {
    friend class Eccentricity_Feature;
  public:
    Segment_Properties(unsigned int thislabel);
    void addpoint(int const x, int const y, int const value);
    void delpoint(int const x, int const y, int const value);
    void merge_with_segment(Segment_Properties const &other);

    template<class T1, class T2> void copy_segment(pic2d<T1> const &source, 
						   pic2d<T2> &dest) const;
    template<class T1, class T2> void copy_segment(pic2d<T1> const &source, 
						   pic3d<T2> &dest,
						   unsigned int const which) 
      const;

    void setvalue(float value) {
      store = value;
    };

    float getvalue() const {
      return(store);
    };

    float angle() const;

    float exc() const;

    float meangray() const {
      if (size > 0)
	return(sum_gray/size);
      else
	return(0);
    };

    float variance() const {
      if (size > 0)
	return(gray_square/size - sum_gray*sum_gray/(size*size));
      else
	return(0);
    };

    friend ostream& operator<< (ostream &o, Segment_Properties &s);
    
  private:
    unsigned int size, label, diameter, sum_gray, gray_square;
    jt::moments2 m;
    int in_x, in_y;
    float store;
  };
  


  /** Merkmalsberechnung Exzentrizität
   */
  class Eccentricity_Feature : public AttentionFeature {
  public:
    Eccentricity_Feature(unsigned int n=0) {
      ftype = eccentricity;
      featurenum = n;
      result = NULL;
      oldresult = NULL;
      verbose = 1;   
      exclusivity = 0.0;
      initialized = false;
    };
    void compute();
    void setparameters(float excl) {
      exclusivity = excl;
    };
    void initialize(int const fsize, float thr=0.75, 
		    unsigned int ave_thresh=20, int dilat=4,
		    float minarea=0.05, float maxarea=30,
		    float var_thresh=1.12, int connect=4,
		    float sal_offset=0.2);
    void setsavedir(string dir) {
      savedir = dir;
    };
    void savestate();

  private:
    void sobel_gradient(pic2d<unsigned char> const &input, bool do_x=true, 
			bool do_y=true, unsigned int size=3);
    void initial_segment(pic2d<unsigned char> const &input);

    // Ist dieser Punkt zum Wachstum zugelassen?
    bool is_legal(unsigned int const x, unsigned int const y) const {
      return(sobel->get(x,y) < thresh);
    };

    // Ist dieser Punkt ein zulässiger Startpunkt?
    bool start_point(unsigned int const x, unsigned int const y) const {
      return(sobel->get(x,y) < thresh);
    };

    void grow_region(pic2d<unsigned char> const &input,
		     unsigned int x, unsigned int y, unsigned int label,
		     pic2d<bool> &processed);
    void dilation(pic2d<unsigned char> const &input, pic2d<int> &dilated,
		  unsigned int count=1); 
    unsigned int look_around(unsigned int x, unsigned int y,
			     jt::association<unsigned int, 
			     unsigned int> &neighbours);
    bool mergeable(Segment_Properties &s1, Segment_Properties &s2) const;
    void merge_segments(pic2d<unsigned char> const &input,
			unsigned int label1, unsigned int label2,
			unsigned int startx, unsigned int starty);
    bool fill_segment(unsigned int label, unsigned int value,
		      pic2d<int> &image);
    void count_segments();

    string savedir;
    bool initialized;
    int thresh, ave_threshold, num_dilation, connectivity, label_num;
    float min_area, max_area, var_ratio_threshold, threshold, saliency_offset,
      exclusivity;

    vector<Segment_Properties*> segments;
    pic2d<int>* sobel;
    pic3d<unsigned char>* segments_orient;
    pic2d<int>* segment_image;
    static int max_segment, num_orientations;
    int orient_elements[13];
  };



  template<class T1, class T2> void 
    Segment_Properties::copy_segment(pic2d<T1> const &source, 
				     pic2d<T2> &dest) const {

    T2 write = T2(store>=0 ? store : label);

    for (int x=m.bounding(true,true); 	x<=m.bounding(false,true); x++)
      for (int y=m.bounding(true,false); y<=m.bounding(false,false); 
	   y++) 
	if (source.get(x,y) == T1(label)) 
	  dest.pset(x, y, write);
  };

  template<class T1, class T2> void 
    Segment_Properties::copy_segment(pic2d<T1> const &source, 
				     pic3d<T2> &dest,
				     unsigned int plane) const {

    T2 write = T2(store>=0 ? store : label);

    for (int x=m.bounding(true,true); x<=m.bounding(false,true); x++)
      for (int y=m.bounding(true,false); y<=m.bounding(false,false); 
	   y++) {
	if (source.get(x,y) == T1(label)) 
	  dest.pset(x, y, plane, write);	
      }
  }
  

}

#endif
