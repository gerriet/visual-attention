#ifndef __READARGS__
#define __READARGS__

// Read arguments for k_sample
// GB .2001

#include "adt/image/pic2d.h"
#include "esab2.h"

std::string tmppath= "/tmp";

using jt::rasterimage;
typedef rasterimage::filetype jgp_filetype;

namespace gtools {

unsigned int do_test=0;

void k_sample_usage(const char *myname) {
  cerr << "Usage: " << myname << "" << endl;
  cerr << "-h           : help" << endl;
  cerr << "-d <file>    : definitions input file (not implemented yet)" << endl;
  cerr << "-m <mode>    : input mode - [s]imulator | [j]peg | [p]pm | pn[g] |"
    "[x]ilcam | [h]tcam | [v]4lcam" << endl;
  cerr << "-a <mode>    : action mode - [f]eature | [s] scanpath | [p] passive | "
       << endl << "                  [m]ove sensor (standard: m)" << endl;
  cerr << "-o <mode>    : output mode - [d]isplay | [s]ave | [b]oth | [n]one " 
       << endl<< "                standard: n" << endl;
  cerr << "-l <mode>    : logging mode - [s]creen | [f]ile | [b]oth | [n]one "
       << "                              standard: s" << endl;
  cerr << "-n <mode>    : neural field mode - [3]D field (local) | "
       << "[2]D field (local) | [S]ystem of 2d fields (global) " << endl;
  cerr << "-b <model>   : behaviour model - [e]xplore | [a]larm | "
       << "[v]isual search | [m]ulti-object-tracking | [s]earch-and-track " << endl;
  cerr << "-i <file>    : gray input file (standard: /tmp/tmpinput)" << endl;
  cerr << "-c <file>    : color input file" << endl;
  cerr << "-f <file>    : output feature files  (standard: /tmp/feature.x)" << endl;

  cerr << "-r <n1> <n2> : frames from n1 to n2" << endl;
  cerr << "-t <n>       : number of threads (standard: 1)" << endl;
  cerr << "--cycle <n>        : move sensor after every <n> cycle "
       << "(0 means no movement)" << endl;
  cerr << "-size <n>       : size (standard: 256)" << endl;
  cerr << "--ms            : multiscale mode on" << endl;
  cerr << "--mm <file>     : mastermap input" << endl;
  cerr << "--mm3 <file>     : mastermap 3D input" << endl;
  cerr << "--nf-size <n>      : size of neural fields (standard: " 
       << nf_size << ")" << endl;
  cerr << "--feature-size <n> : size of features (standard: " << 256
       << ")" << endl;
  cerr << "--stereo <file>    : other stereo-channel " << endl; 
  cerr << "--size-stereo <n>  : use size for stereo computation (standard: "
       << ")" << endl;
  cerr << "--disp <min> <max> : Disparity search range (standard: -25 0 )" 
       << endl;
  cerr << "--multiply <f> <c> : change multiply factor for feature <f> to <c>" 
       << endl;
  cerr << "--addmap <pnmfile> : values to add to the feature-computation" << endl;
  cerr << "--submap <pnmfile> : values to subtract from the feature-computation" 
       << endl;
  cerr << "--temp-path <path> : use as temporary path (default: " 
       << tmppath << ") " << endl;
  cerr << "--log-file <file>  : File for logging (in conjunction with " << endl 
       << "                     -l b or -l f) (default: /tmp/k_sample.log)" 
       << endl;  
  cerr << "Simulator options (only useful in conjunction with -m s" << endl;
  cerr << "--sim-server <name>: name (or IP) of the Simulation server " << endl;
  cerr << "--sim-port <port>  : portnumber for the Simulation server " << endl;
  cerr << "--no-sim-init      : don't initialize the simulator " << endl;
  cerr << "Misc Options" << endl;
  cerr << "--test             : strange test-mode" << endl;
  exit(1);
}

esab2_config interpret_arguments(int argc, char *argv[], 
				 string outputfile, string tmpfile) {
  float restvalue = 0.25;

  esab2_config config;

  // Insert Standard values
  config.outputmode = none;
  config.actionmode = move_sensor_mode;
  config.neuralmode = 3;
  //   config.featureweights[0] = 1.25; // Das war die Standard-Diss-Version
  //  config.featureweights[1] = 0.25;
  //  config.featureweights[2] = 1.5;
  //  config.featureweights[3] = 0.75;
  config.featureweights[0] = 1.2;
  config.featureweights[1] = 0.35;
  config.featureweights[2] = 1.6;
  config.featureweights[3] = 0.7;
  config.nfields = 10;
  config.inputsize = 256;
  config.featuresize = 256;
  config.fieldsize = 64;
  config.mindisp = -25;
  config.maxdisp = 0;
  config.init_simulator = 1;
  config.num_threads = 1;
  config.stereosize = 10;
  config.updateCycles = 20;
  config.movecycle = 0;
  config.simulserver=string("localhost");
  config.simulport=8090;
  config.from_frame = 0;
  config.to_frame = 0;
  config.do_scale = false;
  config.subtractmap = "";
  config.addmap = "";

  do_test = 0;
  config.loggingmode = screen;
  config.logfile = string("/tmp/k_sample.log");
  config.behavior = nomodel;

  // Übergabeparameter einlesen
  for (int narg = 1; narg < argc; narg++) {
    if (strcmp(argv[narg],"-h")==0) 
      k_sample_usage(argv[0]); 
    else if (strcmp(argv[narg],"-i")==0) {
      narg++;
      config.grayfile = string(argv[narg]);
      config.color_mode = false;
    } 
    else if (strcmp(argv[narg],"-o")==0) {
      narg++;
      char omode = argv[narg][0];
      if (omode == "s"[0]) 
	config.outputmode = save;
      else if (omode == "d"[0]) 
	config.outputmode = display;
      else if (omode == "b"[0])
	config.outputmode = both;
      else if (omode == "n"[0])
	config.outputmode = none;
      else
	cerr << "Unknown output mode: " << argv[narg];
    }
    else if (strcmp(argv[narg],"-l")==0) {
      narg++;
      char lmode = argv[narg][0];
      if (lmode == "s"[0])
	config.loggingmode = screen;
      else if (lmode == "f"[0])
	config.loggingmode = file;
      else if (lmode == "n"[0])
	config.loggingmode = nonelog;
      else if (lmode == "b"[0])
	config.loggingmode = bothlog;
      else
	cerr << "Unknown logging mode: " << argv[narg];
    }
    else if (strcmp(argv[narg],"-n")==0) {
      narg++;
      char nmode = argv[narg][0];
      if (nmode == "2"[0])
	config.neuralmode = 2; // single 2d nf (local inhibition)
      else if (nmode == "3"[0])
	config.neuralmode = 3; // single 3d nf (local inhibition)
      else 
	config.neuralmode = 1; // system of 2d nf (global inhibition)
    }
    else if (strcmp(argv[narg],"-c")==0) {
      narg++;
      config.colorfile = string(argv[narg]);
    } 
    else if (strcmp(argv[narg],"-s")==0) {
      narg++;
      config.inputsize = atoi(argv[narg]);
    }
    else if (strcmp(argv[narg],"-r")==0) {
      narg++;
      config.from_frame=atoi(argv[narg]);
      narg++;
      config.to_frame=atoi(argv[narg]);
    } 
    else if (strcmp(argv[narg],"-a")==0) {
      narg++;
      char amode = argv[narg][0];
      if (amode == "f"[0]) 
	config.actionmode = feature_mode;
      else if (amode == "s"[0]) 
	config.actionmode = scanpath_mode;
      else if (amode == "p"[0])
	config.actionmode = passive_mode;
      else if (amode == "m"[0])
	config.actionmode = move_sensor_mode;
      else
	cerr << "Unknown action mode: " << argv[narg];
    }
    else if (strcmp(argv[narg],"-m")==0) {
      narg++;
      char imode = argv[narg][0];
      if (imode == "s"[0]) {
	config.inputtype=simulator;
	config.filetype=rasterimage::jgp_jpeg;
      }
      else if (imode == "j"[0]) {
	config.inputtype=jaclfile;
	config.filetype=rasterimage::jgp_jpeg;
      }
      else if (imode =="p"[0]) {
	config.inputtype=jaclfile;
	config.filetype=rasterimage::jgp_ppm;
      }      
      else if (imode =="g"[0]) {
	config.inputtype=jaclfile;
	config.filetype=rasterimage::jgp_png;
      }
      else if (imode == "x"[0]) {
	config.inputtype=xilinxcam;
	config.filetype=rasterimage::jgp_jpeg;
      }      
      else if (imode == "v"[0]) {
	config.inputtype=v4lcam;
	config.filetype=rasterimage::jgp_jpeg;
      }
      else
	cerr << "Unknown input mode: " << argv[narg];
    }
    else if (strcmp(argv[narg],"-b")==0) {
      narg++;
      char bmode = argv[narg][0];
      if (bmode == "e"[0])
	config.behavior = explore;
      else if (bmode == "v"[0])
	config.behavior = visualsearch;
      else if (bmode == "m"[0])
	config.behavior = mot;
      else if (bmode == "s"[0])
	config.behavior = searchandtrack;
      else if (bmode == "a"[0])
	config.behavior = alarm;
      else
	cerr << "Unkown behavior model: " << argv[narg];
    }
    else if (strcmp(argv[narg],"-t")==0) {
      narg++;
      config.num_threads=atoi(argv[narg]);
    }
    else if (strcmp(argv[narg],"--ms")==0) { // Multiscale on
      config.do_scale = true;
    }
    else if (strcmp(argv[narg],"--cycle")==0) { // Movement each cycle ...
      narg++;
      config.movecycle = atoi(argv[narg]);
    }
    else if (strcmp(argv[narg],"--temp-path")==0) {
      narg++;
      //      tmppath = argv[narg];
    }
    else if (strcmp(argv[narg],"--stereo")==0) {
      narg++;
      config.stereofile = argv[narg];
      config.stereo_mode = 1;
    }
    else if (strcmp(argv[narg],"--multiply")==0) {
      narg++;
      config.featureweights[atoi(argv[narg])] = atoi(argv[++narg]);
    }
    else if (strcmp(argv[narg],"--submap")==0) {
      narg++;
      config.subtractmap = string(argv[narg]);
    }
    else if (strcmp(argv[narg],"--addmap")==0) {
      narg++;
      config.addmap = string(argv[narg]);
    }
    else if (strcmp(argv[narg],"--size-stereo")==0) {
      narg++;
      config.stereosize = atoi(argv[narg]);
    } 
    else if (strcmp(argv[narg],"--disp")==0) {
      narg++;
      config.mindisp = atoi(argv[narg]);
      narg++;
      config.maxdisp = atoi(argv[narg]);
    }
    else if (strcmp(argv[narg],"--nf-size")==0) {
      narg++;
      config.fieldsize = atoi(argv[narg]); 
    }
    else if (strcmp(argv[narg],"--feature-size")==0) {
      narg++;
      config.featuresize = atoi(argv[narg]); 
    }
    else if (strcmp(argv[narg],"--no-sim-init")==0) {
      narg++;
      config.init_simulator = 0;
    } 
    else if (strcmp(argv[narg],"--sim-server")==0) {
      narg++;
      config.simulserver = argv[narg];
    }
    else if (strcmp(argv[narg],"--sim-port")==0) {
      narg++;
      config.simulport = atoi(argv[narg]);
    }
    else if (strcmp(argv[narg],"--log-file")==0) {
      narg++;
      config.logfile = string(argv[narg]);
    }
    else if (strcmp(argv[narg],"--test")==0) {
      do_test = 1;
    } else {      
      cout << "Unknown option: " << argv[narg] << endl;
      exit(1);
    }    
   }  
   config.stereo_mode = (config.stereofile.length() > 0) 
     || (config.inputtype == simulator);
   config.color_mode = (config.colorfile.length() >0) 
     || (config.inputtype == simulator) || (config.inputtype == v4lcam);
   config.do_save = true; config.do_display = true;
   if (config.outputmode == none) {
    config.do_save = false; config.do_display = false;
  } else if (config.outputmode == display) 
    config.do_save = false;
  else if (config.outputmode == save)
    config.do_display=false;
   if (!config.stereo_mode) {
     config.nfields = 1;
     if (config.neuralmode == 3)
       config.neuralmode = 2;
   }

   config.nfeatures = 2 + config.stereo_mode + config.color_mode;
   for (unsigned int i= 0; i < config.nfeatures; i++)
     config.featureweights[i] /= config.nfeatures;
   config.sigmoidfile = string("/tmp/sigmoid");
   config.neuralfile = string("/tmp/nf");
   config.restvalue = restvalue;
   return(config);
}

}

#endif
