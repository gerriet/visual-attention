/* Featureberechnung: Exzentrizität
 * GB - 10.2002  
 */

#include "attention.h"
#include "feature/eccentricity.h"
#include "adt/image/pic2d.h"
#include "pic3d.h"
#include "adt/container/jqueue.h"
#include "adt/pair.h"
#include "adt/container/jassoc.h"
#include "math/moments2.h"
#include "vigra/stdimage.hxx"
#include "vigra/edgedetection.hxx"
#include "vigra/impex.hxx"


using jt::pic2d;
using jt::rasterimage;
using jt::queue;
using jt::pair;

namespace gtools {

  int Eccentricity_Feature::max_segment = 4000;
  int Eccentricity_Feature::num_orientations = 13;

  /** Setzt die Parameter und initialisiert die Datenstrukturen 
      - MUß vor compute ausgeführt werden
   */

  void Eccentricity_Feature::initialize(int const fsize, float thr, 
					unsigned int ave_thresh, int dilat,
					float minarea, float maxarea,
					float var_thresh, int connect,
					float sal_off) {
    size = fsize;
    sobel = new pic2d<int>(fsize,fsize);
    segments_orient = new pic3d<unsigned char>(fsize,fsize,num_orientations);
    segment_image = new pic2d<int>(fsize,fsize);
    segments.resize(256);
    segments[0] = new Segment_Properties(0);
    segments[1] = new Segment_Properties(1);
    threshold = thr;
    ave_threshold = ave_thresh;
    num_dilation = dilat;
    min_area = minarea;
    max_area = maxarea;
    connectivity = connect;
    var_ratio_threshold = var_thresh;
    saliency_offset = sal_off;
    initialized = true;
    if ((connect != 4) && (connect !=8))
      cerr << "**** Eccentricity_Feature: connectivity " << connect 
	   << " not supported (only 4 and 8)! ****" << endl;
    if (thr >= 1.0) 
      cerr << "**** Eccentricity_Feature: threshold should be smaller than 1 ****" 
	   << endl;
  };


  /** Zentrale Routine zur Merkmalsberechnung Exzentrizität,
      - erwartet, daß initialize und setinput ausgeführt wurden
      - Ergebnis findet sich anschließend in result
   */
  void Eccentricity_Feature::compute() {
    // 0. Preparation
    jt::stopwatch* timer = new jt::stopwatch();
    timer->restart();
    if (!initialized) initialize(input->w());
    if (result) {
      if (oldresult) delete oldresult;
      oldresult = result; 
    }
    result = new pic2d<unsigned char>(size, size);
    result->clear(0);

    if (verbose > 0)
      cout << "Ecc: " << size << " " << input->w() << endl;
    

    // 1. Sobelfiltering - Resultat in sobel 
    if (1)
    {
      std::string tempfilename("./sobelinput.png");
      input->save(tempfilename.c_str(), jt::rasterimage::jgp_png);
      vigra::ImageImportInfo info(tempfilename.c_str());
      vigra::BImage in(info.width(), info.height());
      importImage(info, destImage(in));
      vigra::BImage out(info.width(), info.height());
      out = 255;
      double threshold = 0.5;
      int scale = 2;
      cannyEdgeImage(srcImageRange(in), destImage(out),
                           scale, threshold, 0);
      exportImage(srcImageRange(out), "./canny.png");

    }

    pic2d<unsigned char> equal(*input);   
    equal.f_histeq();
    sobel_gradient(equal);
    sobel->save("./sobel.png");

    // 2. Determine segments - Resultat in segment_image
    initial_segment(equal);


    // 3. Expand segments (dilation)
    for (int dil=0; dil < num_dilation; dil++) {
      pic2d<int> dilated(size,size);
      dilated.clear(0);
      dilation(equal, dilated, dil);
    }

    // 4. Remove too small and too big segments
    unsigned int min_pixel = int(min_area/100*(size*size));
    unsigned int max_pixel = int(max_area/100*(size*size));
    for (int i=1; i < label_num; i++) 
      if ((segments[i]->size < min_pixel) || (segments[i]->size > max_pixel)) {
	fill_segment(i, 0, *segment_image);
	segments[i]->size = 0;
      }

    // 5. Count orientations + determine saliency
    count_segments();
    // Es fehlt noch die Umsetzung der Exklusivität

    cout << ">> Ecc time: " << timer->stop() << endl;  
  };



  /** Sobelfilterung von input, Ergebnisse in sobelx,sobely
   */
  void Eccentricity_Feature::sobel_gradient(pic2d<unsigned char> const &input,
					    bool do_x, bool do_y, 
					    unsigned int size) {
    int mask[size][size];
    int midrow = (size-1)/2;

    for (unsigned int x=0; x < size; x++)  // currently only 3*3 kernels
      for (unsigned int y=0; y < size; y++) 
	mask[x][y] = (midrow - y) * (size-1-abs(int(x-midrow)));  

    sobel->clear(0); 

    int size_x = input.w(); int size_y = input.h();
 
    for (int x = 1; x < size_x-1; x++) 
      for (int y = 1; y < size_y-1; y++) {
	int gx=0; int gy=0;
	for (unsigned int mx=0; mx < size; mx++) 
	  for (unsigned int my=0; my < size; my++) {
	    gx += mask[mx][my] * input.get(x-midrow+mx, y -midrow+my);
	    gy += mask[my][mx] * input.get(x-midrow+mx, y -midrow+my);
	  }	
	sobel->pset(x, y, (abs(gx) >? abs(gy)));
      }    
  };


  /** Initiale Bildsegmentierung per region growing */
  void Eccentricity_Feature::initial_segment(pic2d<unsigned char> const &input) {
    bool too_many_segments = true;
    while (too_many_segments) {
      too_many_segments = false;
      label_num=2; // damit beim späteren Umsortieren Platz ist
      segment_image->clear(0);
      pic2d<bool> processed(size,size); // did we touch this pixel?
      processed.clear(false);

      jt::array<unsigned> hist=sobel->histogram(0,255, 256);
      int target = int(size*size*threshold);
      thresh = 0;
      for (int sum = 0; ((sum < target) && (thresh < 250)); thresh++) 
	sum += hist[thresh];
      if (verbose > 1)
	cout << "Threshold at: " << thresh << " - " << target << " ( " << threshold << endl;
      for (unsigned int y=2; y < size-2; y++) {
	for (unsigned int x=2; x < size-2; x++) {
	  if (label_num < max_segment) {
	    if (!processed.get(x,y) && start_point(x,y)) {   
	      grow_region(input, x, y, label_num, processed);
	      label_num++;
	    }
	    processed.pset(x,y,true);
	  }
	  else 
	    too_many_segments = true;	  
	}
      }
      if (too_many_segments)
	threshold += threshold/2; // sollte die Anzahl der Segmente reduzieren
    }  
    if (verbose > 1)
      segment_image->save("/tmp/ecc_initial.png",rasterimage::jgp_png);


  };


  /** Führt von gewähltem Punkt(x,y) ein Bereichswachstum aus, 
      der Bereich wird mit label gefüllt und als processed gekennzeichnet
      Ergebnis landet in segment_image, Segmentdaten in segments[]
   */
  void Eccentricity_Feature::grow_region(pic2d<unsigned char> const &input,
					 unsigned int x, unsigned int y, 
					 unsigned int label, 
					 pic2d<bool> &processed) {
    int xpos[8] = {-1,1,0,0,-1,-1,1,1};
    int ypos[8] = {0,0,-1,1,-1,1,-1,1};

    typedef jt::pair<unsigned int, unsigned int> point;
    queue<point> pointqueue;
    pointqueue.add(point(x, y));
    processed.pset(x, y, true);
    if (label >= segments.size())
      segments.resize(segments.size()*2+1);
	
    segments[label] = new Segment_Properties(label);

    while (!pointqueue.empty()) {

      point p(pointqueue.get()); // ersten Punkt aus Queue nehmen
      int cx=p.head; 
      int cy=p.tail;  
      processed.pset(cx,cy,true);

      if (is_legal(cx,cy)) { // Punkt wird Segment hinzugefügt
	segment_image->pset(cx,cy,label);
	segments[label]->addpoint(cx,cy, input.get(cx,cy));
	
	// Nachbarpunkt in Queue einreihen und als verarbeitet markieren
	for (int i=0; i < connectivity; i++) {
	  int nx = cx + xpos[i];
	  int ny = cy + ypos[i];
	  if ((nx >= 0) && (nx < int(size-1)) && (ny >= 0) && (ny < int(size-1))) 
	    if (!processed.get(nx,ny)) {
	      pointqueue.add(point(nx,ny));
	      processed.pset(nx,ny,true);
	    }
	}
      } 
    }
  };
  

  /** Verschmelze Segmente von einem Punkt aus (soweit möglich) und füge nicht 
      eingeordnete Punkte zu Segmenten hinzu
      Ergebnis in dilated
   */
  void Eccentricity_Feature::dilation(pic2d<unsigned char> const &input,
				      pic2d<int> &dilated, unsigned int count) {
    for (unsigned int y=1; y < size-1; y++) 
      for (unsigned int x=1; x < size-1; x++) 
	if (segment_image->get(x,y) == 0) {

	  jt::association<unsigned int, unsigned int> neighbours;
	  // look_around -> bestimme Anzahl unterschiedlicher Label in der Umgebung
	  int num_neighbours = look_around(x, y, neighbours);

	  if (num_neighbours == 2) {
	    unsigned int label1 = neighbours.getindex(0);
	    unsigned int label2 = neighbours.getindex(1);
	    if (mergeable(*segments[label1], *segments[label2])) 
	      merge_segments(input, label1, label2, x, y);      
	  } else if (num_neighbours > 0) {
	    // Wenn überhaupt Nachbarn (!= 2): Füge zu häufigstem benachbartem Label 
	    unsigned int max=0, maxlabel=0;
	    for (int i=0; i < num_neighbours; i++) {
	      if (max < neighbours[neighbours[i]]) {
		maxlabel = neighbours.getindex(i);
		max = neighbours[maxlabel];
	      }  
	    }
	    dilated.pset(x,y, maxlabel);
	    segments[maxlabel]->addpoint(x,y, input.get(x,y));
	  }
	}

    // Werte für einen Schritt eintragen 
    //   (die Zwischenergebnisse dürfen den Schritt nicht verfälschen)
    int tmp;
    for (unsigned int y=0; y < size; y++) 
      for (unsigned int x=0; x < size; x++) 
	if ((tmp=dilated.get(x,y)) > 0)
	  segment_image->pset(x,y,tmp);
    
    if (verbose > 1) {
      dilated.save((string("/tmp/dilate")+num2string(count)+".png").c_str(),
		   rasterimage::jgp_png);
      segment_image->save((string("/tmp/eccstep")+num2string(count)+".png").c_str(),
			  rasterimage::jgp_png);
    }
  };

  /** Sammle Nachbarn eines Punktes
   */
  unsigned int Eccentricity_Feature::look_around(unsigned int x, unsigned int y,
						 jt::association<unsigned int, 
						 unsigned int> &neighbours) {
    int xpos[8] = {-1,1,0,0,-1,-1,1,1};
    int ypos[8] = {0,0,-1,1,-1,1,-1,1};
    unsigned int nbr_count = 0;
    for (int i=0; i < connectivity; i++) {
      unsigned int cx = x + xpos[i];
      unsigned int cy = y + ypos[i];
      if ((cx > 0) && (cx < size) && (cy > 0) && (cy < size)) {
	unsigned int label = segment_image->get(cx,cy);
	if (label > 0) 
	  if (neighbours.contains(label)) {
	    unsigned int num = neighbours[label]+1;
	    neighbours.del(label); //???
	    neighbours.add(label,num);
	  } else {
	    neighbours.add(label,1);
	    nbr_count++;
	  }
      }      
    }
    return(nbr_count);
  };

  // checks if neighbouring segments are similar enough to merge into one segment
  bool Eccentricity_Feature::mergeable(Segment_Properties &s1, 
				       Segment_Properties &s2) const {
    bool merge = (s1.size * s2.size > 0);
    if (merge)
      merge &= abs(s1.meangray() - s2.meangray()) <=  ave_threshold;
    if (merge) 
      merge &= ((s1.variance() >? s2.variance()) / (s1.variance() <? s2.variance()) 
		< var_ratio_threshold) ;
    return(merge);
  }

  /** Verbindet zwei ähnliche, benachbarte Segmente miteinander       
   */
  void Eccentricity_Feature::merge_segments(pic2d<unsigned char> const &input,
					    unsigned int label1, unsigned int label2, 
					    unsigned int startx, unsigned int starty) {
    unsigned int dest_label, source_label;   
    if (segments[label1]->size < segments[label2]->size) {
      dest_label = label2;
      source_label = label1;
    } else {
      dest_label = label1;
      source_label = label2;
    }
    segments[dest_label]->addpoint(startx,starty, input.get(startx,starty));

    if (!fill_segment(source_label, dest_label, *segment_image)) {
      segment_image->pset(startx,starty, source_label);
      segment_image->setcolor(dest_label);
      segment_image->fill(startx,starty);
    }

    segments[dest_label]->merge_with_segment(*segments[source_label]);
    delete segments[source_label];
    segments[source_label] = new Segment_Properties(source_label);
  }


  bool Eccentricity_Feature::fill_segment(unsigned int label, unsigned int value, 
					  pic2d<int> &image) {
    bool ok = ((segments[label]->in_x > 0) && (segments[label]->in_y > 0));
    if (ok) {
      image.setcolor(value);
      image.fill(segments[label]->in_x, segments[label]->in_y);
    }
    return(ok);
  }

  /* Entfernen aller leeren Segmente, Bestimmen der Momente und daraus der Orientierung
   */
  void Eccentricity_Feature::count_segments() {
    int real_counter = 1;
    int orient;
    for (unsigned int i=0; i < 14; i++)
      orient_elements[i]=0;

    for (int i=1; i < label_num; i++) 
      if (segments[i]->size > 0) {
	segments[real_counter] = segments[i];
	if (segments[real_counter]->exc() < 0.05)
	  orient = 12;
	else
	  orient = int(12*segments[real_counter]->angle()/ M_PI);

	segments[real_counter]->setvalue(i);
	segments[real_counter]->copy_segment(*segment_image, *segments_orient, orient);
	segments[real_counter]->setvalue(orient);
	orient_elements[orient]++;
	real_counter++;
      } 
    //    for (int i = real_counter; i < label_num; i++)
    // if (segments[i])
    //  delete segments[i];
    label_num = real_counter;
    for (int i=1; i < label_num; i++) 
      if (segments[i]->size > 0) {
	int orient = int(segments[i]->getvalue());
	segments[i]->setvalue((segments[i]->exc()-saliency_offset >? 0) * 
			      255/((1-saliency_offset)*(pow(orient_elements[orient],exclusivity))));
	segments[i]->copy_segment(*segment_image, *result);
	if (verbose > 1)
	  cout << *segments[i] << " V: " << segments[i]->getvalue() << endl;
      }
  }

  
  void Eccentricity_Feature::savestate() {
    sobel->save("/tmp/sobel.png",rasterimage::jgp_png);
    jt::array<unsigned> hist = sobel->histogram(0, 255, 10);
    for (unsigned int i=0; i < 10; i++)
      cout << i << " : " << hist[i] << endl;
    segment_image->save("/tmp/ecc_segment.png",rasterimage::jgp_png);
    result->save("/tmp/ecc_output.png",rasterimage::jgp_png);
    segments_orient->setbandimgproperties(5,3,255,1,0);
    segments_orient->save("/tmp/ecc_orient.png",rasterimage::jgp_png);
  }
  

  //////////////////////////////////////////////////////////////////////


  Segment_Properties::Segment_Properties(unsigned int thislabel) {
    label = thislabel;
    sum_gray = 0;
    gray_square = 0;
    size = 0;
    diameter = 0;
    in_x = -1;
    in_y = -1;
    store = -1;
  };

  void Segment_Properties::addpoint(int const x, int const y, int const value) {
    m.add(x,y);
    sum_gray += value;
    gray_square += value*value;
    
    size++;
    if (in_x < 0)
      in_x = x;
    if (in_y < 0)
      in_y = y;
  };

  void Segment_Properties::delpoint(int const x, int const y, int const value) {
    m.del(x,y);
    sum_gray -= value;
    gray_square -= value*value;
    if ((in_x == x) && (in_y == y)) {
      in_x = -1;
      in_y = -1;
    }
    size--;
  };

  void Segment_Properties::merge_with_segment(Segment_Properties const &other) {
    m.add_moment(other.m);
    sum_gray += other.sum_gray;
    gray_square += other.gray_square;
    size += other.size;
  }


  /** von 0 bei 9 Uhr über Pi/2 bei 12 Uhr bis Pi bei 3 Uhr
   */
  float Segment_Properties::angle() const {
    float p = (m.zm(2,0) + m.zm(0,2))/2;
    float q = m.zm(2,0)*m.zm(0,2) - m.zm(1,1)*m.zm(1,1);
    float lambda1 = p+sqrt(p*p-q);
    float lambda2 = p-sqrt(p*p-q);
    if ((abs(m.zm(2,0) - m.zm(0,2)) > 0.001) & (m.zm(1,1) != 0))
      if (lambda1 > lambda2)
	return(M_PI/2+atan2(2*m.zm(1,1), m.zm(2,0)-m.zm(0,2) )/2);
      else
	return(M_PI/2+atan2(m.zm(2,0)-m.zm(0,2), 2*m.zm(1,1) )/2);
    else
      return(0);
  }

  float Segment_Properties::exc() const {
    return(m.gbexc());
  }

  ostream& operator<< (ostream &o, Segment_Properties &s) {
    o << s.label << ": " << s.m.m(0,0) << " = " << s.m.m(1,0) << "*" 
      << s.m.m(0,1) << " " << s.meangray() << "," << s.variance() << " E:" 
      << s.m.exc() << " A: " << s.angle();
    return(o);
  };
}
