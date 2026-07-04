/* Featureberechnung: Stereodisparität
 * GB - 05.2002 - 11.2002
 */

#include <string>
#include <strstream>
#include "adt/jadt.h"
#include "adt/image/pic2d.h"
#include "attention.h"
#include "feature/stereo.h"
#include "nf3d.h"
#include "pic3d.h"
#include "pic_complex.h"
#include "pixelop.h"
#include "unixspec/jtimer.h"

using jt::pic2d;
using jt::rasterimage;

namespace gtools {

/*
 * Stereo feature
 *
 * computing gabor-filter responses,
 * correlating them, WTA
 *
 */

  // to be changed !!!
int disp_to_num(int const num) {
  return(0 >? (9 <? (num-(num>5)-(num>7)-(num>9)-(num>10)-(num>11))));
};


  StereoFeature::StereoFeature(unsigned int n, unsigned int depth_range,
			       int x_size, int y_size, int numo, int anzahl,
			       int const min_disp, int const max_disp, 
			       int corr_method, 
			       int filtern, int xfield, int yfield) 
    : depth(depth_range), xsize(x_size), ysize(y_size),
      numorient(numo), anz(anzahl), mindisp(min_disp), maxdisp(max_disp),
      filternum(filtern), correlation_method(corr_method) {
    
    featurenum = n;
    ftype = stereo;
    lgabor = new pic3d<float>(xsize, ysize, numorient);
    rgabor = new pic3d<float>(xsize, ysize, numorient);
    saliencydistribution = new pic3d<unsigned char>(xsize, ysize, depth);
    confidence = new (pic3d<float>)(xsize, ysize, maxdisp-mindisp+1);
    dispwinner = new (pic2d<unsigned char>) (xsize, ysize);
    dispbest = new (pic2d<unsigned char>) (xsize, ysize);
    sigl = new(pic3d<unsigned char>) (xsize, ysize, numorient);
    sigr = new(pic3d<unsigned char>) (xsize, ysize, numorient);
    alldisp = new(pic3d<unsigned char>) (xsize, ysize, 
				       numorient * (maxdisp - mindisp + 1)); 
    savedir = ".";
    
    row_corr = new float[xsize*(maxdisp-mindisp+2)];
    
    win = 0; result = 0; oldresult = 0;
    gabor_stereo = new gaborfilter<float>(xsize, 15, 0.5, 0.5, false, 1.0);
    
    unsigned int orientations3[] = {0,3,9};
    unsigned int orientations5[] = {0,1,2,10,11};
    
    if (numorient == 3)
      gabor_stereo->select_orientations(numorient, orientations3);
    else if (numorient == 5)
      gabor_stereo->select_orientations(numorient, orientations5);
    else
      cerr << "Stereo: No suitable Gaborfilter found " << endl;
  
    exclusivity = 1.0;
    if (numorient == 3) {
      weight[0] = 0.5;
      weight[1] = 0.25;
      weight[2] = 0.25;
    } else {
      weight[0] = 0.4;
      weight[1] = 0.2;
      weight[2] = 0.1;
      weight[3] = 0.1;
      weight[4] = 0.2;
    };
  };
  
  /** Hauptroutine zur Berechnung des Stereofeatures */
  void StereoFeature::compute() {
    int fieldsize = 64;
    cout << "stereo compute" << endl;
    jt::stopwatch timer; timer.restart();
    
    if (result) {
      if (oldresult) 
	delete oldresult;
      oldresult = result;
    }    
    result = new pic2d<unsigned char>(xsize, ysize);
  
    input->save("linksinput.png",rasterimage::jgp_png);

    // 1. Gaborfilter
    gabor_stereo->apply_filter_mag(jt::cast<float, unsigned char>(*input), 
				   *lgabor, float(256));
    gabor_stereo->apply_filter_mag(jt::cast<float, unsigned char>(*stereoinput), 
				   *rgabor, float(256));
    
    cout << "Stereogabortime: " << timer.lap() << endl;
    
    // 2. Ähnlichkeit berechnen
    computedisp();
    
    // 3. Maxima selektieren
    compute_stereo_feature(xsize, ysize, fieldsize);
    
    cout << ">> Stereo time: " << timer.stop() << endl;
  }
  

  void StereoFeature::computedisp() {
    cout << "computedisp(min/max): " << mindisp << " " << maxdisp << " A: " << anz 
	 << " DL " << dl << " KSW: " << korrsw << " SIGW: " << sigw 
	 << " SIGW2: " << sigw2 << " Numb: " << numorient << endl;

    confidence->clear(0.0);
    alldisp->clear(0);
    sigl->clear(0);
    sigr->clear(0);
    
    for (int b=0; b < int(numorient); b++)
      for (int y=0; y < int(ysize); y++) {
	// Berechnungen für eine Bildzeile
	float leftrow[xsize], rightrow[xsize];

	for (unsigned int x=0; x < xsize; x++) {
	  leftrow[x]  = lgabor->get(x,y,b);
	  rightrow[x] = rgabor->get(x,y,b);
	};
	if (correlation_method==0)
	  lkfdisp(b, y, leftrow, rightrow);
	else if (correlation_method==1)
	  lkwindisp(b, y, leftrow, rightrow);
	else 
	  cerr << correlation_method << " - corr_method in computedisp not supported!" 
	       << endl;
	sort_in_row(b, y); // Eintragen der Werte aus row_corr 
      }
  };


  /** Store correlation values for one row and one band */
  void StereoFeature::sort_in_row(int b, int y) {
    int hdl=(dl-1)/2; 

    for (int x = 0; x <= int(xsize)-dl; x++) { 
      for (int disp = ( (x+mindisp >= 0) ? mindisp : (-x) );
	   disp <= ((int(xsize)-hdl-hdl-x) <? maxdisp);
	   disp++) {
	int disppos = disp - mindisp;
	float kd = row_corr[x+disppos*xsize];
	if (kd > 0) {
	  confidence->pset(x+hdl, y, disppos, confidence->get(x+hdl, y, disppos) +
			   weight[b] * kd);
	  alldisp->pset(x+hdl, y, disppos + b * (maxdisp-mindisp+1),
			(unsigned char)(kd*255));
	}
      }
    }
  };
  

void StereoFeature::lkfdisp(int b, int y, float* leftrow, float* rightrow) {
  /** lkfdisp
   *
   *  Adapted from a khoros glyph out of a Studienarbeit by Thomas Lieder.
   *  computes multiple similarity values between two gaborfilter responses
   *  using different disparity values, multiple hypothesis
   *
   *  Here, the computation is done for one image row and one band.
   *
   *  left- and rightrow: the right image is searched for correspondences
   *  (not necessarily left and right)
   *  dl = size of correlation window
   *  korrsw = lower cutoff for correlation value
   *  sigw = minimum variance of the signal for initial disparity
   *  sigw2 (> sigw) is used for the final result
   *  md = 
   *  mindisp < maxdisp : limits for disparity search 
   *  (typically mindisp is negative and maxdisp is 0 for leftinput,rightinput)
   *
   *  Results are stored in row_corr
   */

  int hdl=(dl-1)/2; // jetzt nicht mehr winkelabhängig    

  // Einmalige Berechnung der statistischen Werte der Signale 
  float mw_1[xsize-dl+1], sig_1[xsize-dl+1];
  float mw_2[xsize-dl+1], sig_2[xsize-dl+1];
  signal_stat(leftrow, mw_1, sig_1, xsize, dl, hdl);
  signal_stat(rightrow, mw_2, sig_2, xsize, dl, hdl);

  if (verbose > 0)
    for (int m = 0; m < int(xsize)-2*hdl; m++) {
      sigl->pset(m, y, b, short(255 <? sig_1[m]/10));
      sigr->pset(m, y, b, short(255 <? sig_2[m]/10));
    }
  for (unsigned int x = 0; x < xsize*(maxdisp-mindisp+1); x++) 
    row_corr[x] = 0.0;

  for (int x = 0; x <= int(xsize)-dl; x++) { 
    /* 1 - for one position (x,y) and orientation(b) */ 
    
    // m describes the start x-pos for the correlation in the second image
    // these are the minimum and maximum values
    
    if (sig_1[x] >= sigw) {    // Streuungsschwellwert ueberschritten ? 

      /* Korrelation */ // Problem mit Grenzen?
      for (int disp = ( (x+mindisp >= 0) ? mindisp : (-x) );
	   disp <= ((int(xsize)-hdl-hdl-x) <? maxdisp);
	   disp++) {
	// choose limits, so that source and destination area are valid

	float kd = 0.0;
	
	if (sig_2[x+disp] >= sigw) { //Streuungsschwellwert für 2tes Signal ?
	  for (int k=x; k < dl+x; k++) { // innerster Kern der Korrelation: 
                                   	 // nx*ny*orient*dl Durchläufe
	    kd += leftrow[k] * rightrow[k+disp];
	  }
	  float tmp = mw_1[x] - mw_2[x + disp];
	  kd /= dl;
	  kd -=  mw_1[x] *  mw_2[x + disp]  +  md * tmp * tmp;
	  kd /= sig_1[x] * sig_2[x + disp];
	}
	row_corr[x+(maxdisp-disp)*xsize] = kd;
      }
    }
  }
}

void StereoFeature::lkwindisp(int b, int y, float* leftrow, float* rightrow) {

  int hdl=(dl-1)/2; // jetzt nicht mehr winkelabhängig    

  // Einmalige Berechnung der statistischen Werte der Signale 
  float mw_1[xsize-dl+1], sig_1[xsize-dl+1];
  float mw_2[xsize-dl+1], sig_2[xsize-dl+1];
  float con_1[xsize-dl+1], con_2[xsize-dl+1];
  for (int m=0; m < int(xsize-hdl-hdl); m++) {
    con_1[m] = 0.0; con_2[m] = 0.0;
    for (int k=0; k < dl; k++) {
      con_1[m] += leftrow[m+k]*leftrow[m+k]*win[k];
      con_2[m] += rightrow[m+k]*rightrow[m+k]*win[k];
    }
  }
  signal_stat(leftrow, mw_1, sig_1, xsize, dl, hdl);
  signal_stat(rightrow, mw_2, sig_2, xsize, dl, hdl);
  
  if (verbose > 0)
    for (int m = 0; m < int(xsize)-2*hdl; m++) {
      sigr->pset(m, y, b, 255 <? short(sig_2[m]/10));
      sigl->pset(m, y, b, 255 <? short(sig_1[m]/10));
    }
  for (unsigned int x = 0; x < xsize*(maxdisp-mindisp+1); x++) 
    row_corr[x] = 0.0;

  for (int x = 0; x <= int(xsize)-dl; x++) { 
    /* 1 - for one position (x,y) and orientation(b) */ 
    
    // m describes the start x-pos for the correlation in the second image
    // these are the minimum and maximum values
    
    if (sig_1[x] >= sigw) {    // Streuungsschwellwert ueberschritten ? 

      /* Korrelation */ // Problem mit Grenzen?
      for (int disp = ( (x+mindisp >= 0) ? mindisp : (-x) );
	   disp <= ((int(xsize)-hdl-hdl-x) <? maxdisp);
	   disp++) {
	// choose limits, so that source and destination area are valid
	float kd = 0.0;
	
	if (sig_2[x+disp] >= sigw) {
	  for (int k=x; k < dl+x; k++) { // innerster Kern der Korrelation: 
                                   	 // nx*ny*orient*dl Durchläufe
	    kd += leftrow[k] * rightrow[k+disp] * win[k-x];
	  }
	  kd /= sqrt(con_1[x]) * sqrt(con_2[x+disp]);
	}	
	row_corr[x+(maxdisp-disp)*xsize] = kd;
      }
    }    
  }
}


void StereoFeature::complex_disp(int b, int y, float* leftreal, float* rightreal,
				 float* leftimag, float* rightimag) {

  int schalt, disp;
  int dispar[anz];
  float tmp;
  float korrko[anz];
  int hdl=(dl-1)/2; // jetzt nicht mehr winkelabhängig    

  // Einmalige Berechnung der statistischen Werte der Signale 
  float mw_y[xsize-dl+1], sig_y[xsize-dl+1];
  float mw_x[xsize-dl+1], sig_x[xsize-dl+1];
  signal_stat(rightreal, mw_y, sig_y, xsize, dl, hdl);
  signal_stat(leftreal,  mw_x, sig_x, xsize, dl, hdl);
  if (verbose > 0)
    for (int m = 0; m < int(xsize)-2*hdl; m++) {
      sigr->pset(m, y, b, 255 <? short(sig_y[m]/10));
      sigl->pset(m, y, b, 255 <? short(sig_x[m]/10));
    }

  for (int x = 0; x <= int(xsize)-dl; x++) { 
    /* 1 - for one position (x,y) and orientation(b) */ 
    
    // m describes the start x-pos for the correlation in the second image
    // these are the minimum and maximum values
    
    if (sig_x[x] >= sigw) {    // Streuungsschwellwert ueberschritten ? 
      int count = 0;
      for(int k = 0; k < anz; k++) {
	korrko[k] = 0;
	dispar[k] = 255;
      }
      /* Korrelation */ // Problem mit Grenzen?
      for (disp = ( (x+mindisp >= 0) ? mindisp : (-x) );
	   disp <= ((int(xsize)-hdl-hdl-x) <? maxdisp);
	   disp++) {
	// choose limits, so that source and destination area are valid
	int disppos = disp - mindisp;
	float kd = 0.0, kdi= 0.0;
	
	if (sig_y[x+disp] >= sigw) {
	  for (int k=x; k < dl+x; k++) { // innerster Kern der Korrelation: 
                                   	 // nx*ny*orient*dl Durchläufe
	    kd += leftreal[k] * rightreal[k+disp] + leftimag[k] * rightimag[k-disp];
	    kdi += leftimag[k] * rightreal[k+disp] + leftreal[k] * rightimag[k+disp]; 
	  }
	  tmp = mw_x[x] - mw_y[x + disp];
	  kd /= dl;
	  kd -=  mw_x[x] *  mw_y[x + disp]  +  md * tmp * tmp;
	  kd /= sig_x[x] * sig_y[x + disp];
	}
	
	if (kd > 0) {
	  alldisp->pset(x, y, disppos + b * (maxdisp-mindisp+1),
			(unsigned char)(kd*255));
	  if (kd > korrsw) {	  // Aehnlichkeitsschwelle ueberschritten ? 
	    schalt=1;

	    /* Sort values according to similarity */	       
	    for (int k=0; k < anz; k++) {
	      if ((kd > korrko[k]) && (schalt==1)) {
		for(int l = 1; l < (anz-k); l++) {
		  dispar[anz-l]=dispar[anz-l-1];
		  korrko[anz-l]=korrko[anz-l-1]; 
		}
		dispar[k]=abs(disp);
		korrko[k]=kd;
		schalt=0;
	      }
	    }
	    count++;
	  } 
	}
      }
      if (count != 0) 
	for(int k = 0; k < anz; k++) {
	  confidence->pset(x + hdl, y, b + numorient * k, korrko[k]);
	}   
    }    
  }
}


  void StereoFeature::signal_stat(float* input, float* out_mw, float* out_var,
				  int length, int dl, int hdl) {
    // Einmalige Berechnung der statistischen Werte eines Signals: Mittelwert und Varianz
    float tmp;
    out_mw[0] = 0;
    for (int m = 0; m < dl; m++)
      out_mw[0] += input[m];
    for (int m = 0; m < int(length)-2*hdl-1; m++) 
      out_mw[m+1] = out_mw[m] + input[m+dl] - input[m];
    for (int m = 0; m < int(length)-2*hdl; m++) 
      out_mw[m] /= dl;

    for (int m = 0; m < int(length)-(hdl+hdl); m++) {
      out_var[m]=0.0;
      for (int k = 0; k < dl; k++) { // zeitkritischer Anteil: nx*ny*dl*orient Durchläufe
	tmp = input[m+k] - out_mw[m];
	out_var[m] +=  tmp*tmp;
      }
      out_var[m]= sqrt(out_var[m]/float(dl));
    }
  }


// Compute stereo feature 
void StereoFeature::compute_stereo_feature(int const stereosize, 
					   int const internalsize, 
					   int const fieldsize) {

  // input gabor responses is in stereosize
  float  disp_rep[maxdisp-mindisp+5];

  unsigned int counter[maxdisp-mindisp+5];

  for (int i = mindisp; i <= maxdisp; i++) { 
    counter[i-mindisp] = 0;
    disp_rep[i-mindisp] = disp_to_num(i);
  }

  pic2d<unsigned char> dispresult(stereosize, stereosize, 1);

  for (int b=0; b < maxdisp-mindisp; b++) 
    confidence->setpic2d(b, (confidence->getpic2d(b)).f_blur(3));  

  confidence->setbandimgproperties(6,1,255,255);
  confidence->save("/tmp/confid.png",rasterimage::jgp_png);

  dispwinner->clear(0);
  // determine maximum value 
  // todo: exclusivity !!!
  for (int x = 0; x < coord_t(xsize); x++)
    for (int y = 0; y < coord_t(ysize); y++) {
      float na = confidence->get(x,y,0);
      int num=0;
      for (int d = 1; d < maxdisp-mindisp+1; d++)
	if (confidence->get(x, y, d) > na) {
	  num = d;
	  na = confidence->get(x,y,d);
	}
      if (mindisp < 0) {
	dispwinner->pset(x,y,((num>0)? -(mindisp+num):0));
	// GB18.12.02 : maxdisp eingefügt
	counter[num>0?maxdisp+1-(mindisp+num):0]++;
      } else {
	dispwinner->pset(x,y,num);
	counter[num]++;
      }
    }

  if (stereosize != internalsize)
    cerr << "PROBLEM! Resizing of initial disparity not implemented yet!" 
	 << endl;
  
  if (exclusivity > 1.0) {
    float sump=0;    
    for (int d = 0; d < maxdisp-mindisp+1; d++)
      sump += counter[d];
    int value[maxdisp-mindisp+2];
    for (int d = 0; d < maxdisp-mindisp+1; d++) {
      value[d] = int(float(d)/pow(exclusivity, counter[d]/sump));
    }
    for (int x=0; x < dispwinner->w(); x++)
      for (int y=0; y < dispwinner->h(); y++) {
	dispwinner->pset(x,y, value[dispwinner->get(x,y)]);
      }
  }
  // GROSSES PROBLEM: Bei Multiskalentechnik ist dies richtig: 
  multConstValue<unsigned char> (*dispwinner,
				 short(255 / (float(maxdisp + 1.5 - mindisp)))); 
  // Sonst eher die folgende Version ?????
  //  multConstValue<unsigned char> (*dispwinner,255/(float(1.5-mindisp))); 
  resize(*dispwinner, dispresult);
  clipminmax<unsigned char> cl(0,12); // ????
  cl.apply(dispresult);
  multConstValue<unsigned char> (dispresult,21);

  result->copy(*dispwinner);
};




pic3d<unsigned char> StereoFeature::getsaliencydistribution(int xs, int ys) {
  unsigned int saliency_table[maxdisp-mindisp+2];
  int factor=75;
  for (int i=0; i <= maxdisp-mindisp; i++) 
    saliency_table[i] = (unsigned int)(0.5+float(i*10)/float(maxdisp-mindisp));
  pic3d<unsigned char>* saldist;
  saldist = new pic3d<unsigned char>(xs, ys, 11);
  saldist->clear(0);

  // Wertebereich von confidence : [0,1]
  for (coord_t z = 0; z < coord_t(maxdisp-mindisp); z++) {
    int targetpos = saliency_table[z];
    for (coord_t x = 0; x < coord_t(xsize); x++)
      for (coord_t y = 0; y < coord_t(ysize); y++) {
	if (xs == confidence->w())
	  saldist->pset(x, y, targetpos, (unsigned char)
			(confidence->get(x,y,z)*factor >? 
			 saldist->get(x,y,targetpos)));
	else if (xs*2 == confidence->w())
	  saldist->pset(x/2, y/2, targetpos, (unsigned char)
			(confidence->get(x, y, z)*factor >? 
			 saldist->get(x/2,y/2,targetpos)));
	else if (xs*4 == confidence->w())
	  saldist->pset(x/4, y/4, targetpos, (unsigned char)
			(confidence->get(x, y, z)*factor >? 
			 saldist->get(x/4,y/4,targetpos)));
	else
	  cerr << "Problem with saldist and disparity sizes!!!!" << endl;
      }
  };

  // Normierung der Tiefenhypothese für jede 2D-Position
  for (coord_t x = 0; x < saldist->w(); x++)
    for (coord_t y = 0; y < saldist->h(); y++) {
      int n=0;
      for (coord_t z = 0; z < saldist->d(); z++) 
	n += saldist->get(x, y, z);
      n = ((255 - n) / 11 >? 0);
      for (coord_t z = 0; z < saldist->d(); z++) 
	saldist->pset(x, y, z, saldist->get(x, y, z)+n);
    };
  // Nur temporär!!!
  /*
  saldist->setbandimgproperties(8,1,255,1.0,0.0);
  saldist->save("/tmp/saldist.png",rasterimage::jgp_png);
  */
  return(*saldist);
};

 void StereoFeature::setparameters(int dlength, float korr_sw, float sig_w, 
				   int m_d, int sig_w_2, float excl) {
    dl = dlength;
    korrsw = korr_sw;
    sigw = sig_w;
    md = m_d;
    if (sig_w_2 == -1)
      sigw2 = sigw;
    else
      sigw2 = sig_w_2;
    exclusivity = excl;
    if (win) 
      delete[] win;
    win = new float[dl+1];
    int mid = dl/2;
    for (int i = 0; i <= dl; i++) 
      win[i] = exp(-(i-mid)*(i-mid)/(2*7.06));
  };

 // Problem with resizing is NOT solved
 void StereoFeature::setinput(pic2d<unsigned char>* leftinput,
			      pic2d<unsigned char>* rightinput) {
    pic2d<unsigned char>  *linput, *rinput;
    if ((leftinput->w() != int(xsize)) || (leftinput->h() != int(ysize))) {
      linput = new pic2d<unsigned char> (xsize, ysize);
      resize(*leftinput, *linput);
      input = linput;
    } else
      input = leftinput;
    if ((rightinput->w() != int(xsize)) || (rightinput->h() != int(ysize))) {
      rinput = new pic2d<unsigned char> (xsize, ysize);
      resize(*rightinput, *rinput);
      stereoinput = rinput;
    } else
      stereoinput = rightinput;
 };

 void StereoFeature::dump(ostream &os) {
   os << "StereoFeature properties : " << featurenum << " " 
      << savedir << endl
      << "Size:       " << xsize << endl
      << "Anz:        " << anz << endl
      << "DataLength: " << dl << endl
      << "MD:         " << md << endl
      << "Mindisp:    " << mindisp << endl
      << "Maxdisp:    " << maxdisp << endl
      << "Korrsw:     " << korrsw << endl
      << "Numbands:   " << numorient << endl
      << "Sigw:       " << sigw << endl
      << "Sigw2:      " << sigw2 << endl;
 };



void StereoFeature::savestate() {
  if (input)
    input->save((savedir+"/linput.png").c_str(), rasterimage::jgp_png);
  if (stereoinput)
    stereoinput->save((savedir+"/rinput.png").c_str(), rasterimage::jgp_png);
  if (lgabor){
    lgabor->setbandimgproperties(4,1,0,float(0.05));
    lgabor->save((savedir+"/lgabor.png").c_str(), rasterimage::jgp_png);
  }
  if (rgabor) {
    rgabor->setbandimgproperties(4,1,0,float(0.05));
    rgabor->save((savedir+"/rgabor.png").c_str(), rasterimage::jgp_png);
  }
  if (confidence) {
    confidence->setbandimgproperties(5,1,255,255);
    confidence->save((savedir+"/confidence.png").c_str(), rasterimage::jgp_png);
  }
  if (sigl) {
    sigl->setbandimgproperties(5,1,0);
    sigl->save((savedir+"/sigl.png").c_str(), rasterimage::jgp_png);
  }
  if (sigr) {    
    sigr->setbandimgproperties(5,1,0);
    sigr->save((savedir+"/sigr.png").c_str(), rasterimage::jgp_png);
  }
  if (dispwinner)
    dispwinner->save((savedir+"/dispwinner.png").c_str(), rasterimage::jgp_png);
  if (result)
    result->save((savedir+"/stereoresult.png").c_str(), rasterimage::jgp_png);
  if (alldisp) {
    alldisp->setbandimgproperties(6,1,255);
    alldisp->save((savedir+"/alldisp.png").c_str(), rasterimage::jgp_png);
  }
};

  /******************************************************************/
  /*                                                                */
  /*      Multiskalenversion des Stereofeatures                     */
  /*                                                                */
  /******************************************************************/

  StereoMultiFeature::StereoMultiFeature(unsigned int n, unsigned int depth_range,
					 int x_size, int y_size, int numo, int anzahl,
					 int const min_disp, int const max_disp, 
					 int corr_method, 
					 int filtern, int xfield, int yfield) {
    depth=depth_range;
    xsize=x_size; 
    ysize=y_size;
    numorient=numo;
    anz=anzahl; 
    mindisp=min_disp;
    maxdisp=max_disp;
    filternum=filtern;
    correlation_method=corr_method;
    initialized = false; 
    featurenum = n;
    ftype = stereo;
    lgabor = new pic3d<float>(xsize, ysize, numorient);
    rgabor = new pic3d<float>(xsize, ysize, numorient);
    saliencydistribution = new pic3d<unsigned char>(xsize, ysize, depth);
    confidence = new (pic3d<float>)(xsize, ysize, maxdisp-mindisp+1);
    dispwinner = new (pic2d<unsigned char>) (xsize, ysize);
    dispbest = new (pic2d<unsigned char>) (xsize, ysize);
    sigl = new(pic3d<unsigned char>) (xsize, ysize, numorient);
    sigr = new(pic3d<unsigned char>) (xsize, ysize, numorient);
    alldisp = new(pic3d<unsigned char>) (xsize, ysize, 
				       numorient * (maxdisp - mindisp + 1)); 
    savedir = ".";
    
    win = 0; result = 0; oldresult = 0;
  };


  /** Hauptroutine zur Berechnung des Stereofeatures */
  void StereoMultiFeature::compute() {
    int fieldsize = 64;
    if (initialized) {
      cout << "stereo compute" << endl;
      jt::stopwatch timer; timer.restart();
    
      if (result) {
	if (oldresult) 
	  delete oldresult;
	oldresult = result;
      }    
      result = new pic2d<unsigned char>(xsize, ysize);
      
      // 1. Gaborfilter
      for (int s=1; s <= scales; s++) {
	int sz = 1 << s-1;
	int xs = xsize / sz;
	int ys = ysize / sz;
	lgabor_m[s] = new pic3d<float>(xs, ys, numorient);
	rgabor_m[s] = new pic3d<float>(xs, ys, numorient);
	
	pic2d<float> tmpl(xs, ys);
	pic2d<float> tmpr(xs, ys);
	resize(jt::cast<float, unsigned char>(*input), tmpl);
	resize(jt::cast<float, unsigned char>(*stereoinput), tmpr);

	gaborf_list[s]->apply_filter_mag(tmpl, *(lgabor_m[s]), float(256));
	gaborf_list[s]->apply_filter_mag(tmpr, *(rgabor_m[s]), float(256));	
      }      
      cout << "Stereogabortime: " << timer.lap() << endl;
      
      // 2. Ähnlichkeit berechnen
      computedisp();
      
      // 3. Maxima selektieren
      compute_stereo_feature(xsize, ysize, fieldsize);
      
      cout << ">> Stereo time: " << timer.stop() << endl;
    } else
      cerr << "StereoMultiFeature not initalized !!!" << endl;
  }


  void StereoMultiFeature::computedisp() {
    cout << "compmultdisp(min/max): " << mindisp << " " << maxdisp << " A: " 
	 << anz << " DL " << dl << " SIGW: " << sigw 
	 << " SIGW2: " << sigw2 << " Numb: " << numorient << endl;

    confidence->clear(0.0);
    alldisp->clear(0);
    sigl->clear(0);
    sigr->clear(0);
    
    for (int b=0; b < int(numorient); b++)
      for (int y=0; y < int(ysize); y++) {
	// Berechnungen für eine Bildzeile
	//	for (unsigned int x=0; x < xsize; x++) 
	//  row_corr[x] = 0.0;

	bool first_scale = true;
	for (int s=scales; s >= 1; s--) { // von der kleinsten Auflösung bis zur höchsten
	  int scale_factor = 1 << s-1;
	  if (y % scale_factor == 0) {
	    // cout << b << " " << y << " " << s << endl;
	    int xs = int(xsize/scale_factor);
	    int ys = int(y/scale_factor);
	    float leftrow[xs], rightrow[xs];

	    for (int x=0; x < xs; x++) {
	      leftrow[x]  = lgabor_m[s]->get(x,ys,b);
	      rightrow[x] = rgabor_m[s]->get(x,ys,b);
	    };
	    if (correlation_method==0)
	      lkfdisp(b, ys, s, first_scale, dl-(s-1), leftrow, rightrow);
	    else if (correlation_method==1)
	      lkwindisp(b, ys, s, first_scale, dl-(s-1), leftrow, rightrow);
	    else 
	      cerr << correlation_method << " - corr_method in computedisp not supported!" 
		   << endl;
	  }
	  first_scale = false;
	}	
	sort_in_row(b, y); // Eintragen der Werte aus row_corr 
      }
  };


  void StereoMultiFeature::lkfdisp(int b, int y, int scale, bool first_scale, int dl,
				   float* leftrow, float* rightrow) {
  /** lkfdisp
   *  left- and rightrow: the right image is searched for correspondences
   *  (not necessarily left and right)
   *  mindisp < maxdisp : limits for disparity search 
   *  (typically mindisp is negative and maxdisp is 0 for leftinput,rightinput)
   *  Results are stored in row_corr
   */

  int hdl=(dl-1)/2;
  int scale_factor = 1 << scale - 1;
  int xs =  xsize / scale_factor;

  // Einmalige Berechnung der statistischen Werte der Signale 
  float mw_1[xs-dl+1], sig_1[xs-dl+1];
  float mw_2[xs-dl+1], sig_2[xs-dl+1];
  signal_stat(leftrow, mw_1, sig_1, xs, dl, hdl);
  signal_stat(rightrow, mw_2, sig_2, xs, dl, hdl);

  if ((verbose > 0) && (scale==1))
    for (int m = 0; m < int(xs)-2*hdl; m++) {
      sigl->pset(m, y, b, 255 <? short(sig_1[m]/10));
      sigr->pset(m, y, b, 255 <? short(sig_2[m]/10));
    }
  for (int x = 0; x < int(xs); x++) 
    for (int d = 0; d < (maxdisp-mindisp)/scale_factor+1; d++) 
      row_corr[scale][x][d] = 0.0;    

  for (int x = 0; x <= int(xs)-dl; x++) { 
    int mds =  int(mindisp/scale_factor);
    int mind = int(mindisp/scale_factor); // default für first_scale
    int maxd = int(maxdisp/scale_factor);
    /* 1 - for one position (x,y) and orientation(b) */ 
    
    // m describes the start x-pos for the correlation in the second image
    // these are the minimum and maximum values

    if (sig_1[x] >= sigw) {    // Streuungsschwellwert ueberschritten ? 
      // determine disparity borders
      // occasionally the whole range is searched
      if (!first_scale)// && ((x % 3 != 0) && (y % 3 != 0))) 
	disp_borders(anz_disp, xs, scale, x, mind, maxd);      

      /* Korrelation */
      for (int disp = ( (x+mind >= 0) ? mind : (-x) );
	   disp <= ((int(xs)-hdl-hdl-x) <? maxd);
	   disp++) {
	// choose limits, so that source and destination area are valid

	float kd = 0.0;       
	if (sig_2[x+disp] >= sigw) { //Streuungsschwellwert für 2tes Signal ?
	  for (int k=x; k < dl+x; k++) { // innerster Kern der Korrelation: 
                                   	 // nx*ny*orient*dl Durchläufe
	    kd += leftrow[k] * rightrow[k+disp];
	  }
	  float tmp = mw_1[x] - mw_2[x + disp];
	  kd /= dl;
	  kd -=  mw_1[x] *  mw_2[x + disp]  +  md * tmp * tmp;
	  kd /= sig_1[x] * sig_2[x + disp];
	}
	row_corr[scale][x+hdl][disp-mds] = kd;
      }
    }
  }
}

  void StereoMultiFeature::lkwindisp(int b, int y, int scale, bool first_scale, int dl,
				     float* leftrow, float* rightrow) {
    int hdl=(dl-1)/2;
    int scale_factor = 1 << scale - 1;
    int xs = xsize / scale_factor;

    cerr << "Warning: Multiscale version of lkwindisp is probably not working yet!" << endl;

    // Einmalige Berechnung der statistischen Werte der Signale 
    float mw_1[xsize-dl+1], sig_1[xsize-dl+1];
    float mw_2[xsize-dl+1], sig_2[xsize-dl+1];
    float con_1[xsize-dl+1], con_2[xsize-dl+1];
    for (int m=0; m < int(xsize-hdl-hdl); m++) {
      con_1[m] = 0.0; con_2[m] = 0.0;
      for (int k=0; k < dl; k++) {
	con_1[m] += leftrow[m+k]*leftrow[m+k]*win[k];
	con_2[m] += rightrow[m+k]*rightrow[m+k]*win[k];
      }
    }
    signal_stat(leftrow, mw_1, sig_1, xsize, dl, hdl);
    signal_stat(rightrow, mw_2, sig_2, xsize, dl, hdl);
    
    if (verbose > 0)
      for (int m = 0; m < int(xsize)-2*hdl; m++) {
	sigr->pset(m, y, b, 255 <? short(sig_2[m]/10));
	sigl->pset(m, y, b, 255 <? short(sig_1[m]/10));
      }

  for (int x = 0; x < xs; x++) 
    for (int d = 0; d < maxdisp-mindisp/scale_factor+1; d++) 
      row_corr[scale][x][d] = 0.0;    

  for (int x = 0; x <= int(xs-dl); x++) { 
    /* 1 - for one position (x,y) and orientation(b) */ 
    int mds =  int(mindisp/scale_factor);
    int mind = int(mindisp/scale_factor); // default für first_scale
    int maxd = int(maxdisp/scale_factor);
    
    // m describes the start x-pos for the correlation in the second image
    // these are the minimum and maximum values
    
    if (sig_1[x] >= sigw) {    // Streuungsschwellwert ueberschritten ? 
      if (!first_scale)// && ((x % 3 != 0) && (y % 3 != 0))) 
	disp_borders(anz_disp, xs, scale, x, mind, maxd);      

      /* Korrelation */ // Problem mit Grenzen?
      for (int disp = ( (x+mind >= 0) ? mind : (-x) );
	   disp <= ((int(xsize)-hdl-hdl-x) <? maxd);
	   disp++) {
	// choose limits, so that source and destination area are valid
	float kd = 0.0;
	
	if (sig_2[x+disp] >= sigw) {
	  for (int k=x; k < dl+x; k++) { // innerster Kern der Korrelation: 
                                   	 // nx*ny*orient*dl Durchläufe
	    kd += leftrow[k] * rightrow[k+disp] * win[k-x];
	  }
	  kd /= sqrt(con_1[x]) * sqrt(con_2[x+disp]);
	}	
	row_corr[x][scale][disp-mds] = kd;
      }
    }    
  }
}


  /** Bestimme Disparitätsgrenzen anhand der vorherigen Skala */
  void StereoMultiFeature::disp_borders(int anz, int size, int scale, int x, 
					int &from, int &to) const {

    /* Problem: hier müßte über die vorhergehenden Skalen summiert werden */

    float sum=0.0;
    int xpos = int(x/2);
    for (int d=0; d < anz/2; d++)
      sum += row_corr[scale+1][xpos][d];
    float maxsum = sum; 
    int maxto = anz/2;
    int sz = 1 << scale;
    for (int d = anz/2; d < (maxdisp-mindisp) / sz; d++) {
      sum += row_corr[scale+1][xpos][d] 
	- row_corr[scale+1][xpos][d-anz/2];
      if (sum > maxsum) {
	maxsum = sum; 
	maxto = d;
      }
    }
    if (maxsum == 0.0) {
      from=0;
      to=0;
    } else {
      to = maxto*2+mindisp;
      from = to-anz;
    }
  }


  /** Store correlation values for one row and one band in confidence + alldisp */
  void StereoMultiFeature::sort_in_row(int b, int y) {
    //    cout << "SIR:" << b << " " << y << endl;

    for (int s=1; s <= scales; s++) {
      int scale_factor = 1 << s-1 ;

      for (int x = 0; x < int(xsize); x++) { 
	int xpos = x / scale_factor;

	for (int d=mindisp; d <= maxdisp; d++) {
	  int disppos = d - mindisp;
	  int disp_scale = disppos/scale_factor; //?????

	  float kd = row_corr[s][xpos][disp_scale];
	  if (kd > 0) {
	    confidence->pset(x, y, disppos, confidence->get(x, y, disppos) +
			     weight[b] * kd);
	    alldisp->pset(x, y, disppos + b * (maxdisp-mindisp+1),
			  (unsigned char)(kd*255) + 
			  alldisp->get(x, y, disppos+b*(maxdisp-mindisp+1)));
	    if (verbose > 0)
	      scale_conf[s]->pset(xpos, y/scale_factor,disp_scale, weight[b]*kd);
	  }
	}
      }
    }
  };
  
  
  void StereoMultiFeature::initialize(int const anzdisp, int const numscales, 
				      int const sc_method, unsigned int orientations[]) {
    // orientations enthält die Nummern der Gaborfilter 
    scales = numscales;
    scale_method = sc_method;
    anz_disp = anzdisp;

    //    unsigned int orientations3[] = {0,3,9}; // 0°, +-45°
    //    unsigned int orientations5[] = {0,1,2,10,11}; // 0°, +-15°, +-30°
    
    if (numorient == 1) {
      weight[0] = 1;
    } else if(numorient == 3) {
      weight[0] = 0.5;
      weight[1] = 0.25;
      weight[2] = 0.25;
    } else if (numorient == 5) {
      weight[0] = 0.4;
      weight[1] = 0.2;
      weight[2] = 0.2;
      weight[3] = 0.1;
      weight[4] = 0.1;
    } else if (numorient == 7) {
      weight[0] = 0.35;
      weight[1] = 0.15;
      weight[2] = 0.15;
      weight[3] = 0.1;
      weight[4] = 0.1;
      weight[5] = 0.075;
      weight[6] = 0.075;
    } ;

    scale_step = xsize*(maxdisp-mindisp+1);
    row_corr.resize(numscales+1);
    for (int s=1; s <= scales; s++) {
      int scale_factor = 1 << s - 1;
      row_corr[s].resize(xsize/scale_factor+1);
      for (unsigned int d=0; d < xsize/scale_factor; d++) 
	row_corr[s][d].resize((maxdisp-mindisp)/scale_factor+2);      
    }

    gaborf_list.resize(scales);
    lgabor_m.resize(scales);
    rgabor_m.resize(scales);
    scale_conf.resize(scales);

    for (int s=1; s <= scales; s++) {
      int scale_factor = 1 << s-1;
      int xs = xsize/scale_factor;
      scale_conf[s] = new pic3d<float>(xs,xs,(maxdisp-mindisp)/scale_factor+1);
      scale_conf[s]->clear(0.0);
      gaborf_list[s] = new gaborfilter<float>(xs, 15, 0.5, 0.5, false, 1.0);
      gaborf_list[s]->select_orientations(numorient, orientations);
    }
    initialized = true;
  };


void StereoMultiFeature::savestate() {
  if (input)
    input->save((savedir+"/linput.png").c_str(), rasterimage::jgp_png);
  if (stereoinput)
    stereoinput->save((savedir+"/rinput.png").c_str(), rasterimage::jgp_png);
  if (lgabor){
    lgabor->setbandimgproperties(4,1,0,float(0.05));
    lgabor->save((savedir+"/lgabor.png").c_str(), rasterimage::jgp_png);
  }
  if (rgabor) {
    rgabor->setbandimgproperties(4,1,0,float(0.05));
    rgabor->save((savedir+"/rgabor.png").c_str(), rasterimage::jgp_png);
  }
  if (confidence) {
    confidence->setbandimgproperties(5,1,255,128);
    confidence->save((savedir+"/confidence.png").c_str(), rasterimage::jgp_png);
  }
  if (sigl) {
    sigl->setbandimgproperties(5,1,0);
    sigl->save((savedir+"/sigl.png").c_str(), rasterimage::jgp_png);
  }
  if (sigr) {    
    sigr->setbandimgproperties(5,1,0);
    sigr->save((savedir+"/sigr.png").c_str(), rasterimage::jgp_png);
  }
  if (dispwinner)
    dispwinner->save((savedir+"/dispwinner.png").c_str(), rasterimage::jgp_png);
  if (result)
    result->save((savedir+"/stereoresult.png").c_str(), rasterimage::jgp_png);
  if (alldisp) {
    alldisp->setbandimgproperties(6,1,255);
    alldisp->save((savedir+"/alldisp.png").c_str(), rasterimage::jgp_png);
  }
  for (int s=1; s <= scales; s++) {
    scale_conf[s]->setbandimgproperties(5,1,255,128);
    scale_conf[s]->save((savedir+"/scaleconf"+num2string(s)+".png").c_str(), 
		     rasterimage::jgp_png);
  }
};

}

