 /* ESAB2 class
 * GB 02.2001 - 12.2001
 * main class for attentional control - should be splitted
 */

#include <string>
#include <stdio.h>
#include "math.h"
//#include "adt/jadt.h"
#include "adt/image/pic2d.h"
#include "robot/orbclient.h"
#include "sensor/camera/camera.h"
#include "sensor/camera/filecam.h"
#include "sensor/camera/xilcam.h"
#include "filescam.h"
#include "orbcam.h"
#include "graycam.h"
#include "gdisplayer.h"
//#include "unixspec/jthreads.h"
#include "attention.h"
#include "feature/stereo.h"
#include "feature/symmetry.h"
#include "feature/eccentricity.h"
#include "feature/color.h"
#include "nf3d.h"
#include "nf2d.h"
#include "nf_2d_system.h"
#include "esab2.h"
#include "label.h"
#include "general-tools.h"
#include "pic3d.h"

using jt::rasterimage;
using jt::orbclient;
using jt::orbcam;
using jt::filecam;
using std::cout;

//#define debug

namespace gtools {

/* 
 * Implementation of the ESAB2 main class
 */

ESAB2::ESAB2(esab2_config config, unsigned int numfield) {
  configuration = config;
  string tmppath("/tmp/");
  numfields = numfield;
  current_x = 0; current_y = 0;
  next_x = 0; next_y = 0;
  unsigned int fcount = 0;
  stereofeat = -1; colorfeat = -1;
  do_scale = config.do_scale;
  
  // Initialize features

  // Symmetry
  if(do_scale) {
    SymmetryMultiFeature* symmfeature = new SymmetryMultiFeature(fcount++);
    symmfeature->setsize(config.featuresize, config.featuresize);
    if (true) {
      symmfeature->initialize(config.featuresize, 2);
      symmfeature->initialize_scale(0, config.featuresize/2, 6, 3, 4, 12, 0.75, 0.5);
      symmfeature->initialize_scale(1, config.featuresize/4, 6, 3, 4, 12, 0.75, 0.5);
    } else {
      symmfeature->initialize(config.featuresize, 3);
      symmfeature->initialize_scale(0, config.featuresize,   6, 3, 3, 12, 0.75, 0.5);
      symmfeature->initialize_scale(1, config.featuresize/2, 6, 3, 4, 12, 0.75, 0.5);
      //symmfeature->initialize_scale(2, config.featuresize/2, 6, 3, 4, 12, 0.75, 0.5);
      symmfeature->initialize_scale(2, config.featuresize/4, 6, 3, 4, 12, 0.75, 0.5);
    }
    featurelist.push_back(symmfeature); 
  } else {
    SymmetryFeature* symmfeature = new SymmetryFeature(fcount++);
    symmfeature->setsize(config.featuresize, config.featuresize);
    symmfeature->initialize(config.featuresize, 6, 3, 4);
    featurelist.push_back(symmfeature);
  }

  // Eccentricity
  Eccentricity_Feature* ecc = new Eccentricity_Feature(fcount++);
  ecc->initialize(configuration.featuresize, 0.78, 20, 4, 0.05, 30, 1.5, 4);
  //ecc->initialize(configuration.featuresize, 0.6, 20, 3, 0.1, 30, 1.5, 8);
  featurelist.push_back(ecc);

  // Color
  if (config.color_mode) {
    colorfeat = fcount;
    ColorFeature* colorfeat = new ColorFeature(fcount++);
    colorfeat->setsize(configuration.featuresize, configuration.featuresize);
    colorfeat->initialize(configuration.featuresize, 12, 6, 10, 0.0006, 0.12, 32, 0); // neu 07.10.04
    //    colorfeat->initialize(configuration.featuresize, 10, 5, 10, 0.0008, 0.12, 40, 0.02);
    // colorfeat->initialize(configuration.featuresize, 8, 5, 10, 0.0005, 0.15, 32, 0.0); // wie in a_sample/color_test.C

    featurelist.push_back(colorfeat);
    leftcolorimage=new(pic2d<unsigned char>) (configuration.inputsize, 
					      configuration.inputsize,3);
  }
  // Stereo
  if (config.stereo_mode) {
    stereofeat = fcount;
    if(do_scale) {
      StereoMultiFeature* stereofeature = 
	new StereoMultiFeature(fcount++, configuration.stereosize,
			  configuration.featuresize,
			  configuration.featuresize, 5, 3, 
			  configuration.mindisp, configuration.maxdisp, 0, 1,
			  configuration.fieldsize, configuration.fieldsize);
      stereofeature->setparameters(12, 0.0, 95, 1);
      unsigned int orientations[] = {0,1,11,2,10};
      stereofeature->initialize(10, 2, 0, orientations); // numdisp, numscales, scale_method
      stereofeature->setsavedir("/tmp");
      featurelist.push_back(stereofeature);
    } else {
      StereoFeature* stereofeature = 
	new StereoFeature(fcount++, configuration.stereosize,
			  configuration.featuresize,
			  configuration.featuresize, 5, 3, 
			  configuration.mindisp, configuration.maxdisp, 0, 1,
			  configuration.fieldsize, configuration.fieldsize);
    //    stereofeature->setparameters(15, 0.65, 40, 1);
      stereofeature->setparameters(12, 0.75, 75, 1);
      stereofeature->setsavedir("/tmp");
      featurelist.push_back(stereofeature);

    }
    rightgrayimage=new(pic2d<unsigned char>) (configuration.inputsize, 
					      configuration.inputsize);
  }
  numfeatures = fcount;
  unsigned int nfield=0;


  if (configuration.addmap.length() > 1) {
    offset = new pic2d<int>(configuration.addmap.c_str(),
			    rasterimage::jgp_ppm);
  } else {
    offset = new pic2d<int>(configuration.featuresize, 
			    configuration.featuresize);
    offset->clear(0);
  }

  if (configuration.subtractmap.length() > 1) {
    pic2d<int> tmp(configuration.subtractmap.c_str(),rasterimage::jgp_ppm);
    offset->copy(*offset - tmp);
  }

  // initialize Neural fields
  if (config.neuralmode == 3) { // 3D-Version of Neural fields
                                // local inhibition type
    nf = new NeuralField3D<float>(configuration.fieldsize, 
				  configuration.fieldsize,
				  configuration.stereosize);
    // alpha, global, resting, beta, inputmult
    nf->setparameters(0.4, 8.0, -0.25, 30, 0.006);
    //    nf->setparameters(0.5, 3.0, -0.27, 30, 0.018); // bis 19.11.
    //    nf->setparameters(0.5, 4.0, -0.27, 30, 0.015);
    nf->initialize();
    nf->setkernels(21,21,9);
    nfield = configuration.stereosize;
  } else if (config.neuralmode == 2) { // single 2D neural field 
    nf2d = new NeuralField2D<float>    // local inhibition type
      (configuration.fieldsize, configuration. fieldsize, false, 0, 0, 1, 1);
    nf2d->setparameters(0.33, 8, -0.33, 30, 0.003);
    nf2d->set_DoG_kernels(15, 15, 3.3, 0.12, 14, 0.03); 
    nf2d->set_thresh(0.002);
    //    nf2d->setparameters(0.33, 1, -0.25, 30, 0.003);
    //nf2d->set_DoG_kernels(15, 15, 3.3, 0.12, 14, 0.03); 
    //nf2d->set_thresh(0.002);
    //cout << "NF2d kernel:" << endl; nf2d->dumpkernel(cout); cout << endl;
    nfield = 1;
  } else if (config.neuralmode == 1) {     // system of 2D neural fields
    nf2ds = new NeuralField2DSystem<float> // global inhibition type 
      (configuration.fieldsize, configuration.fieldsize, 
       4, configuration.nfeatures, false, 0, 0, 1, 1);
    nf2ds->setparameters(0.3, 70, -0.25, 30, 0.007);
    nf2ds->set_G_kernels(11, 11, 3.5, 0.25);
    nf2ds->set_thresh(0.002);
    nf2ds->initialize();
    // quick hack for fixed initial feature weights
    for (unsigned int f = 0 ; f < configuration.nfeatures; f++)
      for (unsigned int o = 0; o < 4; o++) 
	nf2ds->weights.setweight(f, o, 0.4 + 1.5*(f==o) + 0.5 * (o>f));     
    nfield = 4;
  }

  neuralfield = new (pic3d<float>)
    (configuration.fieldsize, configuration.fieldsize, nfield);
  sigmoidfields = new (pic3d<unsigned char>)
    (configuration.fieldsize, configuration.fieldsize,nfield);
  neuralfield->setbandimgproperties(5, 1, 255, 20, 6.0);
  sigmoidfields->setbandimgproperties(5,1,255,1,0);  

  leftgrayimage=new(pic2d<unsigned char>) (configuration.inputsize, 
					   configuration.inputsize);

  featuremaps = new pic3d<unsigned char> (configuration.featuresize, 
					  configuration.featuresize, 
					  numfeatures);
  featuremaps->setbandimgproperties(5,1,255,1,0);

  segments = new (pic3d<unsigned int>) (configuration.fieldsize,
					 configuration.fieldsize,
					 configuration.stereosize);

  mastermap=new(pic2d<unsigned char>) (configuration.featuresize, 
				       configuration.featuresize);
  master3d = new pic3d<float> (configuration.featuresize, 
			       configuration.featuresize,
			       configuration.stereosize);
  master3d->setbandimgproperties(5,1,255,5,0);

  saliencydistribution = new pic3d<unsigned char> (configuration.featuresize,
						   configuration.featuresize,
						   configuration.stereosize);
  depthmap=new (pic2d<unsigned char>);

  // initialize displayers
  leftinputdisplayer = new (gdisplayer< pic2d<unsigned char> >)
    ("Left_input", configuration.do_display, configuration.do_save,
     true, true, "/tmp/left"),
  rightinputdisplayer = new (gdisplayer< pic2d<unsigned char> >)
    ("Right_input", configuration.do_display, configuration.do_save,
     true, true, "/tmp/right");
  grayinputdisplayer = new (gdisplayer< pic2d<unsigned char> >)
    ("Gray input", false, configuration.do_save, true, true, "/tmp/gray");
 
  featuredisplayer = new (gdisplayer< pic3d<unsigned char> >)
    ("Featuremaps", configuration.do_display, configuration.do_save,
     true, true, "/tmp/feature", rasterimage::jgp_png);
  masterdisplayer = new (gdisplayer<pic2d<unsigned char> >) 
    ("Mastermap_2D", configuration.do_display, configuration.do_save,
     true, true, "/tmp/master", rasterimage::jgp_png);

  saliency3ddisplayer = new(gdisplayer< pic3d<float> >)
    ("Saliency_3D", configuration.do_display, configuration.do_save,
     true, true, "/tmp/sal3d", rasterimage::jgp_png);
  neuralfielddisplayer = new (gdisplayer< pic3d<float> >) 
    ("NeuralFields", configuration.do_display, configuration.do_save, 
     true, true, "/tmp/neural");
  if (numfields > 1)
    nf->addnfdisplayer(neuralfielddisplayer, configuration.do_display);
  else {
    gdisplayer< pic2d<float> >* nfdisp = new (gdisplayer< pic2d<float> >)
      ("NeuralFields", configuration.do_display, configuration.do_save, 
     true, true, "/tmp/neural2d");
    nf2d->addnfdisplayer(nfdisp, configuration.do_display, configuration.do_save);
  }

  sigmoiddisplayer = new (gdisplayer< pic3d<unsigned char> >)
    ("SigmoidFields", configuration.do_display, configuration.do_save,
     true, true, "/tmp/sigmoid");

  // initialize camera(s) for input
  orbclient* Oclient;
  switch (configuration.inputtype) {
  case jaclfile:
    if (configuration.from_frame < configuration.to_frame) {
      if (configuration.color_mode)
	maincamera = new filescam("",configuration.colorfile.c_str(), 
					  configuration.filetype); 
      else
	maincamera = new filescam("",configuration.grayfile.c_str(), 
				  configuration.filetype);
      if (configuration.stereo_mode)
	stereocamera = new filescam("",configuration.stereofile.c_str(), 
				    configuration.filetype);      
    } else {
      if (configuration.color_mode)
	maincamera = new jt::filecam("",configuration.colorfile.c_str(), 
					 configuration.filetype); 
      else
	maincamera = new jt::filecam("",configuration.grayfile.c_str(), 
					 configuration.filetype);
      if (configuration.stereo_mode)
	stereocamera = new filecam("",configuration.stereofile.c_str(), 
				   configuration.filetype);      
    }
    break;
  case simulator: 
    filetype = rasterimage::jgp_jpeg;
    Oclient = new orbclient(configuration.simulserver.c_str(), 
			    configuration.simulport);
    
    Oclient->connect();
    maincamera= new orbcam(Oclient, "lcam",256,256);
    stereocamera = new orbcam(Oclient, "rcam", 256, 256);
    
    if (configuration.init_simulator) 
      execute_command(string("wget -q 'http://" + configuration.simulserver +
			     ":" + num2string(configuration.simulport) + 
			     "/orbital/fc/de/2003?simulator=nop'"));
    
    break;
  case xilinxcam:
    cout << "Using the XIL cameras: not working anymore" << endl;
    break; 
  case v4lcam:
    cout << "Using V4L camera" << endl;
    maincamera = new jt::v4lcam("/dev/video0");
    maincamera->set_color(true);
    maincamera->set_resolution(256, 256);
    break;
  case htcam:
    cerr << "Warning: htcam not implemented yet!" << endl;
  default:
    cerr << "Warning: unknown camera" << endl;
  }
}

/*
 * Destructor 
 */

ESAB2::~ESAB2() {
  if (configuration.color_mode)
    delete leftcolorimage;
  delete leftgrayimage;
  if (configuration.stereo_mode)
    delete rightgrayimage;
  delete mastermap;
  delete depthmap;
  delete featuremaps;
  if (configuration.nfields > 0) {
    delete neuralfield;
    delete sigmoidfields;
    delete nf;
  }
  if (configuration.neuralmode == 3)
    delete saliencydistribution;
  delete master3d;
  // Displayer löschen
  delete leftinputdisplayer;
  delete rightinputdisplayer;
  delete featuredisplayer;
  delete sigmoiddisplayer;
  delete masterdisplayer;
  delete neuralfielddisplayer;
  delete saliency3ddisplayer;
  delete maincamera;
  if (stereocamera)
    delete stereocamera;
}

/*
 * Grab the next frame from the camera(s)
 */

int ESAB2::getnextframe(int number) {
  actualframe = number;
  bool setcounters = (configuration.from_frame < configuration.to_frame) && 
    (configuration.inputtype == jaclfile);

  if (setcounters) 
    if (filescam* tmpcam = dynamic_cast<filescam*>(maincamera))
      tmpcam->setcounter(number);    

  if (configuration.color_mode) {
    maincamera->grab(*leftcolorimage);

    if (true) // GB031004: Nur fuer einen Fall
      {

      }

    leftgrayimage->copy(leftcolorimage->gray());
  } else
    maincamera->grab(*leftgrayimage);

  if (configuration.stereo_mode) {
    if (setcounters) 
      if (filescam* tmpcam = dynamic_cast<filescam*>(stereocamera))
	tmpcam->setcounter(number); 

    if (configuration.color_mode) {
      // funktioniert das?
      pic2d<unsigned char> tmp(rightgrayimage->w(), rightgrayimage->h(),3);
      stereocamera->grab(tmp);
      rightgrayimage->copy(tmp.gray());
    } else        
      stereocamera->grab(*rightgrayimage);
    leftgrayimage->save("graycam.png",rasterimage::jgp_png);
  }
  if (configuration.color_mode)
    leftinputdisplayer->update(*leftcolorimage, number); 
  else
    leftinputdisplayer->update(*leftgrayimage, number); 
  if (configuration.stereo_mode)
    rightinputdisplayer->update(*rightgrayimage, number); 
  for (unsigned int f = 0; f < numfeatures; f++) {
    if (featurelist[f]->ftype == AttentionFeature::color)
      featurelist[f]->setinput(leftcolorimage);
    else if ((featurelist[f]->ftype == AttentionFeature::stereo) || 
	     (featurelist[f]->ftype == AttentionFeature::stereomulti)) {
      StereoFeature* sf = dynamic_cast<StereoFeature*> (featurelist[f]);
      if (sf != NULL)
	sf->setinput(leftgrayimage, rightgrayimage);
      else
	cerr << "AttentionFeature error (cast for stereo input)" << endl;
    } else //usual feature (single graylevel input)
      featurelist[f]->setinput(leftgrayimage);     
  };  
  return (number + 1);
}

/*
 * Computes the feature maps and the master map for the current input images
 * providing the input into the neural fields
 */

int ESAB2::computefeatures(bool do_display) {

  show_time(string("Start Feature computation"));
  for (unsigned int f=0; f < numfeatures; f++) {
    std::cout << "Start Feature " << f << " von " << numfeatures << std::endl;
    std::cout.flush();
    featurelist[f]->compute();
    featuremaps->setpic2d(f,featurelist[f]->current_result());
    if (configuration.actionmode == feature_mode)
      (featurelist[f]->current_result()).save(string("/tmp/feat_"
						     +num2string(f)+".png").c_str(),
					      rasterimage::jgp_png);      
    if (configuration.do_display)
      featuredisplayer->display(*featuremaps, "Featuremaps");
  }

  featuredisplayer->update(*featuremaps, actualframe);
  
  // Put together into mastermap
  for (unsigned int x=0; x < configuration.featuresize; x++)
    for (unsigned int y=0; y < configuration.featuresize; y++) {
      float tmp=0.0;
      for (unsigned int z=0; z < configuration.nfeatures; z++)
	tmp += featuremaps->get(x,y,z) * configuration.featureweights[z];
      mastermap->pset(x, y, 0, (unsigned char)(0 >? 
					       (255 <? (tmp + offset->get(x,y))))); 
    }

  masterdisplayer->update(*mastermap, actualframe);

  std::cout << "Nach Mastermap update" << stereofeat << std::endl;
    std::cout.flush();

      StereoFeature* sf = stereofeat >= 0 ? dynamic_cast<StereoFeature*> (featurelist[stereofeat]) : NULL;


  // Generate Input for Neural fields
  if ((sf != NULL) && (configuration.neuralmode == 3)) { // Generate 3DNF Input
    std::cout << "Start 3dnf " << std::endl;
    std::cout.flush();
    pic3d<unsigned char> 
      tmp(sf->getsaliencydistribution(configuration.featuresize,
				      configuration.featuresize));
    for (unsigned int j=0; j < configuration.stereosize; j++)
      saliencydistribution->setpic2d(j, tmp.getpic2d(j));
  } else if (configuration.neuralmode == 1) { // Generate 2DS NF Input
    cerr << "2DS NF not implemented yet" << endl;
  }
  cout << endl << "Master: " << meanvalue(*mastermap) << endl;
  return(1);
}

int ESAB2::update_nf() {
  jt::stopwatch* timer = new jt::stopwatch(); timer->restart();
  pic3d<float> nf_input(configuration.fieldsize,configuration.fieldsize,
			configuration.nfields);
  nf_input.clear(0);
  int factor = configuration.featuresize/configuration.fieldsize;
  int scale = factor*factor*20; 	// Konstante als Parameter extrahieren

  for (int x=0; x < int(configuration.featuresize); x++)
    for (int y=0; y < int(configuration.featuresize); y++) {
      int xdest = x / factor; 
      int ydest = y / factor;
      float value = mastermap->get(x, y);
      for (unsigned int z=0; z < configuration.nfields; z++) {
	unsigned char result = (unsigned char)
	  ((value * saliencydistribution->get(x, y, z))/scale <? 255);
	nf_input.pset(xdest, ydest, z, result+nf_input.get(xdest, ydest, z));
	master3d->pset(x, y, z, result);
      }
    }
  saliency3ddisplayer->update(*master3d, actualframe);

  cout << "Mastermap-mean: ";
  for (int z=0; z < int(configuration.nfields); z++) 
    cout << meanvalue(nf_input.getpic2d(z)) << " ";
  cout << endl;

  nf->update(configuration.updateCycles, nf_input);
  nf->getSigmoidValues(*sigmoidfields);
  sigmoiddisplayer->update(*sigmoidfields, actualframe);
  nf->getActivityValues(*neuralfield);
  neuralfielddisplayer->update(*neuralfield, actualframe);
  unsigned int num_segments;
  label3d(*neuralfield, *segments, 8, float(0.01), 1, num_segments);
#ifdef debug
  segments->save("/tmp/segment.pnm");
#endif
  cout << "3DNF time: " << timer->stop() << endl;
  return(1);
}

  int ESAB2::update_fields() {
    if (configuration.neuralmode == 3)
      return(update_nf());
    else if (configuration.neuralmode == 2)
      return(update_nf_2d());
    else if (configuration.neuralmode == 1)
      return(update_nf_2ds());
    else 
      cerr << "Unknown neural mode: " << configuration.neuralmode 
	   << endl;
    return(0);
  }


int ESAB2::update_nf_2d() {
  jt::stopwatch* timer = new jt::stopwatch(); timer->restart();

  pic2d<float> nf_input(configuration.fieldsize, configuration.fieldsize, 4);
  resize(jt::cast<float, unsigned char>(*mastermap), nf_input, 1);

  nf2d->update(configuration.updateCycles, nf_input);
  nf2d->getSigmoidValues(*sigmoidfields);
  sigmoidfields->save("./sig2d.pnm");
  sigmoiddisplayer->update(*sigmoidfields, actualframe);
  nf2d->getActivityValues(*neuralfield);
  neuralfield->save("./nf2d.pnm");
  neuralfielddisplayer->update(*neuralfield, actualframe);
  //  label2d(*neuralfield, *segments, 8, float(0.01), 1, num_segments);
#ifdef debug
  segments->save("/tmp/segment.pnm");
#endif
  cout << "2DNF time: " << timer->stop() << endl;
  return(1);
}


int ESAB2::update_nf_2ds() {
  jt::stopwatch* timer = new jt::stopwatch(); timer->restart();

  nf2ds->update(configuration.updateCycles, *featuremaps, 0.3);
  nf2ds->getSigmoidValues(*sigmoidfields);
  sigmoidfields->save("./sig2d.pnm");
  sigmoiddisplayer->update(*sigmoidfields, actualframe);
  nf2ds->getActivityValues(*neuralfield);
  neuralfield->save("./nf2d.pnm");
  neuralfielddisplayer->update(*neuralfield, actualframe);
  //  label2d(*neuralfield, *segments, 8, float(0.01), 1, num_segments);
#ifdef debug
  segments->save("/tmp/segment.pnm");
#endif
  cout << "2DS NF time: " << timer->stop() << endl;
  return(1);
};


  void ESAB2::update_objectfiles(objectfiles& newfiles, objectfiles& oldfiles,
				 int framenum, int pan, int tilt, ofstream& oflog) {
    if (framenum < 1) {
      newfiles.addfromimages(*segments, neuralfield, 255, 0.1, 25);
    } else {
      newfiles.addfromimages(*segments, oldfiles, neuralfield,
			     255, 0.1, 25, pan, tilt);
    }
    cout << "OF created" << endl; newfiles.dump(cout);
    newfiles.check_consistency(oflog, "1", framenum);
  };


int ESAB2::adjustfields(int x_move , int y_move, unsigned int size) {
  /** The values inside the neural fields are shifted to match a movement
      of the sensor towards the new centre at x_move and y_move.
      The values are read from _ and written into _. */

  int tx, ty;
  int fsize = configuration.fieldsize;
  pic2d<float> singlemap (fsize, fsize,1);

  x_move *= fsize/size;
  y_move *= fsize/size;
  for (unsigned int z=0; z < configuration.nfields; z++) {
    singlemap = neuralfield->getpic2d(z);
    for (int x=0; x < fsize; x++)
      for (int y=0; y < fsize; y++) {
	tx = x + x_move;
	ty = y + y_move;
	if ((tx > 0) && (ty > 0) && (tx < fsize) && (ty < fsize))
	  neuralfield->pset(x, y, z, singlemap.get(x+ x_move,y + y_move));
	else
	  neuralfield->pset(x, y, z, -0.25); 
      }
  }
  nf->updateSigmoid();
  return(1);
}

int ESAB2::move_to_focus(int x, int y, unsigned int size) {
  float pan, tilt;
  pan = float((float(x) - float(size/2))) / size;
  pan_simul(pan, configuration.simulserver.c_str(), configuration.simulport);
  tilt = float((float(y) - float(size/2))) / size;
  tilt_simul(tilt, configuration.simulserver.c_str(), configuration.simulport);
  cout << endl << "movefocus(x,y - pan,tilt) [size]: " << x << "," << y 
       << " - " << pan << "," << tilt << "[" << size << "]" << endl;
  return(1);
}

int ESAB2::select_maximum(coord_t xlast, coord_t ylast, int diff) {
  return(1);
}

int ESAB2::dumpstate(bool do_config, bool do_save, 
		     rasterimage::filetype filetype,
		     bool do_display) {
  if (do_save) {
    // todo: extention benutzen für pnm
    char fileextension[100];
    char tmpname[100];
    if (filetype == rasterimage::jgp_jpeg)
      sprintf(fileextension,"%s","jpg");
    else
      sprintf(fileextension,"%s","pnm");
    if (configuration.color_mode)
      leftcolorimage->save("/tmp/leftcolorimage.jpg",filetype);
    else
      leftgrayimage->save("/tmp/leftgrayimage.jpg",filetype);
    if (configuration.stereo_mode)
      rightgrayimage->save("/tmp/rightgrayimage.jpg",filetype);
    mastermap->save("/tmp/mastermap.jpg",filetype);
    //    depthmap.save("/tmp/depthmap.jpg",jgp_jpeg);
    for (unsigned int i=0; i < numfeatures; i++) {
      sprintf(tmpname,"/tmp/feature_%d.%s",i,fileextension);
      featuremaps->getpic2d(i).save(tmpname,filetype);
    }
  }
  if (do_config) {
  cout << "Types (I/F)   : " ;
  switch (configuration.inputtype) {
  case jaclfile:
    cout << "Jaclfile   ";
    break;
  case khorosfile:
    cout << "Khorosfile ";
    break;
  case simulator:
    cout << "Simulator  ";
    break;
  default:
    cout << "Unknown    ";
  }
  cout << " - ";
  switch (configuration.filetype) {
  case rasterimage::jgp_ppm:
    cout << "ppm";
    break;
  case rasterimage::jgp_jpeg:
    cout << "jpg";
    break;
  case rasterimage::jgp_png:
    cout << "png";
    break;
  default:
    cout << "Unknown";
  }
  cout << endl << "Number (Features/Fields)      : " << configuration.nfeatures << 
    " - " << configuration.nfields << endl;
  cout << "Sizes (Input/Features/Fields) : " << configuration.inputsize << " - " 
       << configuration.featuresize << " - " << configuration.fieldsize << endl;
  }
  return(1);
}

// determines, if a gaze shift is necessary, computes the target coordinates
// and executes the gaze shift
// currently only the simulator is used
// returns, whether a gaze shift occured
bool ESAB2::overt_attention() {
  return((configuration.movecycle > 0) && (actualframe > 0) && 
	 ((configuration.movecycle == 1 ) || 
	  ((actualframe % configuration.movecycle) == 1))); 
};








/* Inputmodule class 
   - not ready for usage yet */

/*
void Inputmodule::addcamera(jt::camera &cam, int x, int y, 
		       bool color, bool stereo) {
  camerawithprops newcam;
  newcam.camera = &cam;
  newcam.prop.color = color;
  newcam.prop.stereo = stereo;
  newcam.prop.x = x;
  newcam.prop.y = y;
  cameras.push_back(newcam);
} 

Inputmodule::~Inputmodule() {
  //  for (campropiter i = cameras.begin(); i != cameras.end(); i++)
  //  delete *i;
}



bool Inputmodule::getold(pic2d<unsigned char> &output, bool color, 
			 bool stereo, int memory, int x, int y) {
  cerr << "Not implemented yet: Inputmodule::getold" << endl;
  return(false);
}

int Inputmodule::imagefit(imagewithprops* image, bool color, bool stereo, int x, int y) {
  int quality = 999;
  imageproperty iprop = image->prop;
  if (iprop.stereo == stereo) // left/right match
    if ((iprop.x > x) && (iprop.y > y)) { // at least image size
      if (iprop.color == color)
	quality = 2;
      else
	if (iprop.color && !color) 
	  quality = 4;
	else
	  return(quality);
      if ((x == iprop.x) && (y == iprop.y))
	quality--;
    }
  return(quality);
};

bool Inputmodule::getimage(pic2d<unsigned char> &output, imagewithprops &sourceimage,
			   bool color, bool stereo, int x, int y) {
  // todo
  if (color && !sourceimage.prop.color)
    // make gray from color image
    return(false);
  if ((x != sourceimage.prop.x) || (y != sourceimage.prop.y)) {
    pic2d<unsigned char> tmp(x,y);
    imagewithprops tmpp;
    resize(*(sourceimage.image), tmp, 0,x,y);
    tmpp.image = &tmp;
    tmpp.prop.color = color;
    tmpp.prop.stereo = stereo;
    tmpp.prop.x = x;
    tmpp.prop.y = y;
    storedimages.push_back(tmpp);
    output.copy(tmp);
  }
  else
    output.copy(*(sourceimage.image));
  return(true);
}

bool Inputmodule::get(pic2d<unsigned char> &output, bool color, bool stereo,
		      int memory, int x, int y) {

//  * Strategy is as follows: at first, the image is tried to be selected 
//  *  from the stored images; if not possible, it is acquired from the 
//  *  best matching camera.
//  *  Matches are: 
//  *  1. exact
//  *  2. larger image -> shrink
//  *  3. color image  -> gray
//  *  4. larger color -> shrink and gray
//  *  5. smaller image -> expand ?
//  *  ...
//  *  999 no match at all
//  *  Each image acquired or computed is stored for later retrieval.
//  *
  
  if (memory >0) 
    return(getold(output, color, stereo, memory, x, y));
  else {
    // check if available
    //    bool found = false;
    
    int bestquality = 999;
    imagewithprops bestimage;
    for (imagepropiter i = currentimages.begin(); 
	 (i != currentimages.end()) && (bestquality > 1); i++) {
      int quality = imagefit(i, color, stereo, x, y);
      bestquality = (quality<bestquality?quality:bestquality);    
      bestimage = *i;
    }
    if (bestquality < 999) {
      if (bestquality==1)
	return (bestimage.image);
      else
	cerr << "not implemented get modification" << endl;
    }
   

  }
  return(true);
}
*/
}
