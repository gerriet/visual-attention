#include "feature/symmetry.h"
#include "adt/image/pic2d.h"
#include "pic3d.h"
#include "imagefilter.h"
#include "attention.h"

/*
 * Symmetry feature
 *
 * computing gabor-filter responses,
 * adding their values for different radii orthogonal around each point
 * and determining the maximum value
 *
 */

namespace gtools {

void SymmetryFeature::compute() {
  jt::stopwatch* timer = new jt::stopwatch();
  timer->restart();
  if (verbose > 1)
    cout << "Sym: " << size << " " << input->w() << " " << offset << " " 
	 << step << " " << a_num_bands << " " << orientations << " " << endl;
    if (result) {
      if (oldresult)
	{
	  pic2d<unsigned char>* tmp;
	  tmp = result;
	  result = oldresult;
	  oldresult = tmp;
	}
      else
	{
	  oldresult = result;	  
	}
      oldresult->clear();
    }
    else 
      {
	result = new pic2d<unsigned char>(size, size);
      }

  unsigned int const gabor_clip=255;

  pic3d<float> gabor_result(size, size, orientations);
  gabor_result.setbandimgproperties(5,1,255,1,0);
  pic2d<float> inputf = jt::cast<float, unsigned char>(*input);
  symmetrybands = new int[size*size*a_num_bands];

  //  int factor = size>128 ? 2*size : size;
  int factor = 4+size/12; // Warum?????? Sch***-Parameter

  // Step 1 - Gabor filtering
  if (int(size) != input->w()) {
    pic2d<float> tmp(size, size);
    resize(inputf, tmp);
    gabor12->apply_filter_mag(tmp, gabor_result, factor);
  }
  else
    gabor12->apply_filter_mag(inputf, gabor_result, factor);

  if (verbose > 1)
    cout << "Symgabortime: " << timer->lap() << endl;

  // Step 2 - Symmetry computation
  for (int z = 0; z < a_num_bands; z++)
    for (unsigned int y = 0; y < size; y++)
      for (unsigned int x = 0; x < size; x++)
	symmetrybands[(z*size+y)*size+x] = 0;

  // for each orientation
  for (int orientation = 0; orientation < orientations; orientation++) {
    unsigned char gabor[size*size];
    pic2d<float> gabor_single(gabor_result.getpic2d(orientation));
    gabor_single.clearborder(5); // Antworten der Gaborfilter am Rand unterdrücken
    for (unsigned int y = 0; y < size; y++)
      for (unsigned int x = 0; x < size; x++) 
	gabor[x+size*y] = ((unsigned char)(gabor_single.get(x,y)) 
			   <? gabor_clip);
 
    // Adding the contribution of one orientation to symmetrybands
    symmetry_intern(gabor, orientation*deltaalpha);
  } 

  if (verbose > 0)
    gabor_result.save(string("gabor_"+num2string(size)+".png").c_str(),
		      rasterimage::jgp_png);

  // Step 3 - addbands
  int ssize=size*size;
  int clip_threshold = 60;
  for (unsigned int y = 0; y < size; y++) {
    int yoff = y*size;
    for (unsigned int x = 0; x < size; x++) {
      int wert = 0;
      for (int i = 0; i < a_num_bands; i++)
	wert = wert >? (symmetrybands[x+yoff+i*ssize] + i*step
			- clip_threshold);

      result->pset(x, y, (255 <? wert));
    }
  }
  delete[] symmetrybands;
  if (verbose > 0)
    cout << ">> Sym time: " << timer->stop() << "   - ";
};


void SymmetryFeature::symmetry_intern(unsigned char const orientation_filter[], 
				      int const alpha) {
  /** for one given orientation (alpha) the symmetry evidence is accumulated 
   */

  float cos_a = cos((180-alpha)*M_PI/180);
  float sin_a = sin((180-alpha)*M_PI/180);

  for (int databand = 0; databand < a_num_bands; databand++) {
    // for each radius
    int radius = step * databand + offset;
    int box_length = int(M_PI * (radius+step) / orientations);
    int box_width = step;

    // ax und ay geben zentralen Punkt der summation area an
    int ax = -int(cos_a*(radius+step/2));
    int ay = int(sin_a*(radius+step/2));
    int area = int(sqrt(float((box_length+1)*(box_length+1)+
			(box_width+1)*(box_width+1))));
    int doff = databand*size*size;

    // one-time index computation (for speedup) of the summation areas
    int idz[area*area];
    int cnt=0;
    int maxx=0, maxy=0;

    // Überprüfe die Zugehörigkeit in einem Bereich (area), der garantiert
    // die summation area enthält; alle Resultate kommen in idz
    // die Koordinaten sind relativ zu ax/ay
    for (int x=-area; x < area; x++)
      for (int y=-area; y < area; y++) {
	float w = sin_a*y + cos_a*x;
	float l = cos_a*y - sin_a*x;
	if ((abs(2*w) <= box_width) && (abs(2*l) <= box_length)) {
	  idz[cnt] = x + size * y;
	  maxx = maxx >? abs(x);
	  maxy = maxy >? abs(y);
	  cnt++;
	}
      }        
    
    int fc = cnt * orientations/2;

    // x und y sind die Bildkoordinaten; Kernschleife - optimiert
    // bis auf die Division am Ende    
    for (unsigned int y = abs(ay)+maxy; y < size-(abs(ay)+maxy); y++) {
      int yoff = y*size + doff;
      int y1 = size * (y + ay);
      int y2 = size * (y - ay);
      unsigned int fromx = abs(ax)+maxx;
      unsigned int tox = size-(abs(ax)+maxx);
      for (unsigned int x = fromx; x < tox; x++) {
	int z1 = x + ax + y1;
	int z2 = x - ax + y2;
	int res = 0;
	for (int i=0; i < cnt; i++) 
	  res += orientation_filter[z1 + idz[i]] + 
	    orientation_filter[z2 - idz[i]];
	symmetrybands[x + yoff] += res/fc;
      }
    }
  }
}

void SymmetryFeature::initialize(int const fsize, int const off, 
				 int const stp, int const a_bands, 
				 int const orient, float const k0, 
				 float const r) {
    size = fsize;
    offset = (off<0 ? 8 : off);
    step = (stp<0 ? 3 : stp);
    orientations = orient;
    a_num_bands = a_bands;
    deltaalpha = 180 / orientations;
    if (gabor12)
      delete gabor12;
    gabor12 = new gaborfilter<float>(size, deltaalpha, k0, r, false, 1.0);
  };


 pic3d<int> SymmetryFeature::getbands(int rsize) {
   if (rsize < 0)
     rsize = size;
   pic3d<int> res(rsize, rsize, a_num_bands);
   for (int i=0; i < a_num_bands; i++) {
     res.setpic2d(i, pic2d<int>(&symmetrybands[i*size*size], size, size, 1));
   }
   return(res);
 }

  void SymmetryFeature::savestate() {
    getbands().save("/tmp/symbands.png",jt::rasterimage::jgp_png);
  }



 void SymmetryMultiFeature::compute() {
   jt::stopwatch* timer = new jt::stopwatch();
   timer->restart();
    if (result) {
      if (oldresult)
	{
	  pic2d<unsigned char>* tmp;
	  tmp = result;
	  result = oldresult;
	  oldresult = tmp;
	}
      else
	{
	  oldresult = result;	  
	}
      oldresult->clear();
    }
    else 
      {
	result = new pic2d<unsigned char>(size, size);
      }
   result->clear(0);

   for (int i = 0; i < scales; i++) {
     float factor = 0.5+i*0.5;
     feat[i]->setinput(input);
     feat[i]->compute();
     pic2d<unsigned char> tmp(result->w(), result->h());
     resize(feat[i]->current_result(), tmp);
     for (int y=0; y < result->h(); y++)
       for (int x=0; x < result->w(); x++) 
	 result->pset(x, y, (255 <? (result->get(x,y) >? 
				     int(factor*tmp.get(x,y)))));
   }
   if (verbose>0)
     cout << "SymM time: " << timer->stop();
 }

  
  void SymmetryMultiFeature::savestate() {
    unsigned int radii=0;
    int maxb=2;
    for (int i=0; i< scales; i++) {
      radii += feat[i]->a_num_bands;
      maxb = maxb >? feat[i]->a_num_bands;

      // Single pic saving
      for (int j=0; j < feat[i]->a_num_bands; j++) {
	pic2d<unsigned char> singleband((jt::cast<unsigned char,int>((feat[i]->getbands()).getpic2d(j))));
      	singleband.save(("/tmp/symband" + num2string(i) + "-" + 
      			 num2string(j)+".png").c_str(),
      			rasterimage::jgp_png);
      }
    }

    pic3d<int> bands(size, size, radii);
    bands.setbandimgproperties(maxb,1,255,1,0);

    int c=0;
    for (int i=0; i< scales; i++)
      for (int j=0; j < feat[i]->a_num_bands; j++) {
	pic2d<int> tmp(size,size);
	resize((feat[i]->getbands()).getpic2d(j),tmp);
	bands.setpic2d(c++, tmp);
      }
    bands.save("/tmp/symmulti.png", rasterimage::jgp_png);
    (current_result()).save("/tmp/symm_result.png", rasterimage::jgp_png);
  }
}


