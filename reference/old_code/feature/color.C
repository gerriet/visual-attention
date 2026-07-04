/* Featureberechnung Farbkontrast
 * GB 02.2001 - 07.2002
 * anhand von Khoros-Sourcen von Maik Bollmann, Christoph Justkowski und ????
 */

#include <string>
#include "adt/jadt.h"
#include "adt/image/pic2d.h"
#include "attention.h"
#include "pic_complex.h"
#include "imagefilter.h"
#include "nf3d.h"
#include "pic3d.h"
#include "unixspec/jtimer.h"
#include "feature/color.h"

using jt::pic2d;
using jt::rasterimage;

namespace gtools {

/*
 * Color feature
 *
 * computing a color segmentation and evaluating the mean contrast
 * to the neighboring segments
 */


  void ColorFeature::initialize(int fsize, float thres, float t_par, 
				int attr_t,float min_seg, 
				float max_seg, float msal,
				float prior) {
    size = fsize;
    threshold  = thres;
    thres_par  = t_par;
    attr_thres = attr_t;
    min_seg_size = int(min_seg*size*size);
    max_seg_size = int(max_seg*size*size);
    max_sal = msal;
    priorize_exclusivity = prior;
    setsize(size,size);
    result = NULL;
    oldresult = NULL;
    delete munsell;
    munsell = new pic3d<float>(size,size,3);
  }


  void ColorFeature::compute() {
    jt::stopwatch* timer = new jt::stopwatch();
    timer->restart();
    cout << "col compute" << endl;
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
     
    munsell->clear();
    rgb_to_munsell(*input, *munsell, false);

    if (munsell->w() == result->w()) 
      {
      evalColor(*munsell, *result);
      std::cout << "Nach EvalColor " << std::endl;
      std::cout.flush();
      }
    else {
      pic2d<unsigned char> tmp(munsell->w(), munsell->h());
      evalColor(*munsell, tmp);
      resize(tmp, *result);
    }
    cout << ">> Col time: " << timer->stop() << "  - ";
    cout.flush();
  };
  


  void ColorFeature::evalColor(pic3d<float> const &munsell,
			       pic2d<unsigned char> &result) {
    /** Main saliency computation from MTM (munsell) input image 
	- leider eine Altlast */

    double threshold_1;
    list<struct region*> regionen;
    struct region *region_start, *last_region, *current_region;
    struct neighbour_list *n_ptr;
    int i, j, row, col, pos;
    double inp_vect[3]; 
    
    int label_vect[4], label_vect_orig[4];
    struct region  *reg_ptr_vect[4], *reg_ptr_vect_orig[4];
    double grad, dist_vect[4];
    int merge_vect[4];
    int merge_case;
    signed int direction; /*direction of the line scan for segmentation */
    int count, info;
    int L; /* absolute number of labels */
    double ML, Mx, My; /* Munsell (or CIE) coordinates in cartesian space */
    double MH, MC;     /* Munsell (or CIE) hue and saturation in polar space */
    int div = 1;

    info = 0;
    int w = munsell.w(); int h = munsell.h();
    int npoints=w*h;
    if (output_data) delete[] output_data;
    output_data=new int[npoints+1];

    for (int i=0; i < w*h; i++) 
      output_data[i] = 0; 
    
    /*alloc & init region_start*/
    region_start = (struct region *)malloc(sizeof(struct region));
    region_start->label = 1;
    
    region_start->n_of_points = 1;
    region_start->mean[0] = (double)munsell.get(0,0,0);
    region_start->mean[1] = (double)munsell.get(0,0,1);
    region_start->mean[2] = (double)munsell.get(0,0,2);
    region_start->sigma_2[0] = 0.0;
    region_start->sigma_2[1] = 0.0;
    region_start->sigma_2[2] = 0.0;
    region_start->pg_x=region_start->pg_y = 0.0;
    region_start->perimeter = 0;
    region_start->mf_gradient = 0.0;
    
    region_start->next = NULL;
    region_start->prev = NULL;
    region_start->neighbours = NULL;
    last_region = region_start;
    current_region = NULL;
    
    /*
     * calculate the labels and their neighborhood
     */
    output_data[0] = 1;
    L = 2; /* for next region */
    pos = 0;
    direction = 1; /* forward */
    
    /*** main loop ***/
    for (count=1; count<w*h; count++) { 
      /* pos = row*w+col; */
      pos = pos + direction;
      row = pos / w;
      col = pos % w;
      
      for (j=0; j < 3; j++)
	inp_vect[j] = (double)munsell.get(col,row,j);
      
      merge_case = 0;
      
      /* fill label_vect_orig with neighb. labels */
      if (direction == 1) 
	label_vect_orig[0] = (col>0          ? output_data[pos-1]   : 0);
      else
	label_vect_orig[0] = (col<w-1        ? output_data[pos+1]   : 0);
      
      label_vect_orig[1] = (row>0            ? output_data[pos-w]   : 0);
      label_vect_orig[2] = (row>0 && col>0   ? output_data[pos-w-1] : 0);
      label_vect_orig[3] = (row>0 && col<w-1 ? output_data[pos-w+1] : 0);
      
      /* copy label_vect_orig to label_vect */
      for (j=0; j<4; j++)
	label_vect[j] = label_vect_orig[j];
      
      /* repeated labels in label_vect set to be zero */
      for (j=0; j<3; j++) 
	for (i=j+1; i<4; i++) 
	  if (label_vect[j] == label_vect[i])
	    label_vect[i] = 0;
      
      /* fill reg_ptr_vect_orig with proper pointers */
      for (j=0; j<4; j++)
	reg_ptr_vect_orig[j] = search_reg_label (region_start, label_vect_orig[j]);
      
      
      /* fill reg_ptr_vect with proper pointers */
      for (j=0; j<4; j++) {
	if (label_vect[j] != 0) {
	  reg_ptr_vect[j] = search_reg_label (region_start, label_vect[j]);
	  /* fill dist_vect */
	  dist_vect[j] = euclid_metric(inp_vect, reg_ptr_vect[j]->mean);
	} else {
	  reg_ptr_vect[j] = NULL;
	  /* dist_vect[j] set to -1, invalid */
	  dist_vect[j] = -1.0;
	}
      }
      
      /* for var. threshold after Yokoya */
      threshold_1 = win_sigma(munsell, pos, w, h) * thres_par;
      
      /* fill merge_vect */
      for (j=0; j<4; j++) 
	if ((dist_vect[j] < threshold+threshold_1) && (dist_vect[j] != -1.0)) {
	    merge_vect[j] = 1; /* candidate to merge */
	    merge_case++;
	  } else
	    merge_vect[j] = 0;
      
      if (merge_case == 0) { /* no proper region to merge */
	current_region = (struct region *)malloc(sizeof(struct region)); /* alloc memory for new region */
	last_region->next = current_region;
	current_region->label = L;
	current_region->n_of_points = 1;
	
	for (j=0; j<3; j++) {
	  current_region->mean[j] = inp_vect[j];
	  current_region->sigma_2[j] = 0;
	}
	
	current_region->pg_x = (double)col;
	current_region->pg_y = (double)row;
	current_region->perimeter = 0;
	current_region->mf_gradient = 0.0;
	current_region->neighbours = NULL;
	current_region->prev = last_region;
	current_region->next = NULL; 
	
	last_region = current_region;
	
	for (j=0; j<2; j++)
	  make_neighbours (last_region, reg_ptr_vect_orig[j]);
	
	output_data[pos]=L;
	L++;
      } else if (merge_case==1) { /* one proper region to merge */
	j=0;
	
	while (merge_vect[j] != 1)
	  j++;
	
	add_pix_to_reg (inp_vect, reg_ptr_vect[j], col, row, j);
	output_data[pos] = reg_ptr_vect[j]->label;
	
	for (i=0; i<2; i++)
	  make_neighbours (reg_ptr_vect_orig[j], reg_ptr_vect_orig[i]);
      } else {
	j = min_ind(dist_vect); /* index of region */
	add_pix_to_reg (inp_vect, reg_ptr_vect[j], col, row, j); /* merge with the region with minimal dist */
	output_data[pos] = reg_ptr_vect[j]->label;
	
	for (i=0; i<2; i++) { 
	  if (reg_ptr_vect_orig[j] != reg_ptr_vect_orig[i])
	    make_neighbours(reg_ptr_vect_orig[j], reg_ptr_vect_orig[i]);
	}	      
      } /* end of else of if(merge_case==0) */
      
      /* alter direction & shift inp. pointer */
      if ((col==w-1 && direction==1) || (col==0 && direction==-1)) {
	direction *= -1;
	pos += w;
	pos -= direction;
      }
    } /*** end main loop for(count) ***/


    /*
     * compute the perimeter and mf_gradient for each region
     */
    current_region = region_start;
    
    for (j=0; j<L-1; j++) {
      grad = 0.0;
      n_ptr = current_region->neighbours;
      
      if (n_ptr == NULL)
	cerr << "n_ptr is NULL!";
      else
	while (n_ptr != NULL) {
	  current_region->perimeter += n_ptr->border_length;
	  grad += (euclid_metric(current_region->mean, n_ptr->reg->mean))*(double)n_ptr->border_length;
	  n_ptr = n_ptr->next;
	}
      current_region->mf_gradient = grad/(double)current_region->perimeter;
      current_region = current_region->next;
    } /* end of for(j) */
       
  /*
   * count the labels falling in each of the 12 color intervalls
   * (currently only a version with 12 feature maps)
   */
    current_region = region_start;
    
    for (i=0; i<12; i++)
      quantity[i] = 0;
    if (orients) delete[] orients;
    orients = new int[L+1];
    
    for (j=0; j<L-1; j++) {
      ML = current_region->mean[0];
      Mx = current_region->mean[1];
      My = current_region->mean[2];
      
      /* hue in degrees */
      MH = int(57.29578 * atan2(My,Mx) + 0.5);
      if (MH < 0)
	MH += 360;
    
      /* saturation */
      MC = sqrt(Mx*Mx + My*My);
      orients[j] = 0;
      
      if (current_region->mf_gradient >= attr_thres) {
	if (MH>=15 && MH<345)
	  orients[j] = int((MH+15)/30);
	quantity[orients[j]]++;
      }
      current_region = current_region->next;
    } /* end of for(j) */
    
    
    /*
     * value the color contrast of the segments by the frequency of their appearance
     */

    current_region = region_start;
    if (attr_values) delete[] attr_values;
    attr_values=new float[L+1];
    
    for (j=0; j<L-1; j++) {
      ML = current_region->mean[0];
      Mx = current_region->mean[1];
      My = current_region->mean[2];
      
      /* hue in degrees */
      MH = (int)(57.29578 * atan2(My,Mx) + 0.5);
      if (MH < 0)
	MH += 360;
      
      /* saturation */
      MC = sqrt(Mx*Mx + My*My);
      if (MC >= attr_thres) 
	div = quantity[orients[j]];
      else
	div = 0;
      if ((priorize_exclusivity > 0.0) && (div > 1))
	current_region->mf_gradient /= pow(div, priorize_exclusivity);
      if (current_region->n_of_points >= min_seg_size && 
	  current_region->n_of_points <= max_seg_size) {
	attr_values[j] = saliency(float(current_region->mf_gradient/ max_sal));
      } else
	attr_values[j] = 0.0;

      if (verbose > 1)
	cout << j << ": " << current_region->mf_gradient << " - " 
	     << attr_values[j] << " * " << current_region->mf_gradient/max_sal 
	     << endl;
      
      // delete regions
      struct region* reg = current_region->next;
      delete current_region;
      current_region = reg;
    } /* end of for(j) */   

    /* compute the output_map */
    for (int y=0; y < h; y++)
      for (int x=0; x < w; x++) 
	result.pset(x, y, (unsigned char)
		    (attr_values[output_data[x+y*w]-1]*255));
  };
  
  void ColorFeature::savestate() const {
    pic2d<int> segment(output_data, size, size, 1);
    segment.save("/tmp/colorsegment.png",rasterimage::jgp_png);
    pic3d<int> orient12(size, size, 12);
    orient12.clear(0);
    for (unsigned int x = 0; x < size; x++)
      for (unsigned int y = 0; y < size; y++) {
	int seg = output_data[x+y*size]-1;
	if ((orients[seg] < 0) || (orients[seg] > 11)) 
	  cout << seg << ": " << x << " x " << y << " - " << orients[seg] 
	       << endl;
	orient12.pset(x, y, orients[seg], seg);
      }
    orient12.setbandimgproperties(6,2,255);
    orient12.save("/tmp/color_orient.png",rasterimage::jgp_png);
    munsell->setbandimgproperties(5,1,255);
    munsell->save("/tmp/munsell.png",rasterimage::jgp_png);
  }  

  /*win_sigma*/
  double win_sigma(pic3d<float> const &input, int pos, int w, int h) {
    signed int row, col;
    double data_table[9][3];
    int mask_table[9];
    double temp_vect[3]={0, 0, 0};
    double window_mean[3], dat_vect[3];
    double dist, square_sum=0;
    double sigma=0;
    int n=0;
    int m=0;
    signed int i, j, k, y, x;
    
    row=pos/w;
    col=pos%w;
    
    for(k=-1; k<=1; k++) { //rows in window
      for(j=-1; j<=1; j++) { // columns in window
	y=row+k;
	x=col+j;
	if(y>=0 && y<h && x>=0 && x<w) {/*in range*/	    
	  n++;
	  for(i=0; i<3; i++) { // all bands
	    data_table[m][i]=input.get(x,y,i);
	    temp_vect[i]+=data_table[m][i];/*sum for mean*/
	  }
	} else { // outer range
	  for(i=0; i<3; i++){ /*all bands*/
	    data_table[m][i]=0;/*pad to zero*/
	    mask_table[m]=1;
	  }
	}
	m++;
      }
    }
    
    for(j=0; j<3; j++)
      window_mean[j]=temp_vect[j]/(double)n;
    
    for(j=0; j<9; j++)   {
      if(mask_table[j]!=1) {
	for(i=0; i<3; i++)    
	  dat_vect[i]=data_table[j][i];
	dist=euclid_metric(dat_vect, window_mean);
	dist*=dist;
	square_sum+=dist;
       
      }
      sigma=sqrt(square_sum/(double)n);
    }
    return sigma;
  }


/*updated_sigma_2*/
double updated_sigma_2(double inp_vect[3], struct region *reg_ptr, int k) {
  if(reg_ptr!=NULL) {
    double dn=(double)reg_ptr->n_of_points;
    double mean=(double)reg_ptr->mean[k];/*mean value before merging*/
    return (1/(dn+1)) * (dn*reg_ptr->sigma_2[k] + ((dn/(dn+1))*((inp_vect[k]-mean)*(inp_vect[k]-mean))));/*updated sigma_2*/
    }
   else 
    return -1.0 ; 
}


/*max_el*/
double max_el(double inp_vect[3]) {
  double max=inp_vect[0];
  for(int j=1; j<4; j++)
    max = max >? inp_vect[j];

  return max;
}


/*min_el*/
double min_el(double inp_vect[3]) {
  double min=inp_vect[0];
  for(int j=1; j<4; j++)
    min = min <? inp_vect[j]; 

  return min;
}
  

/*add_pixel_to_region*/
int add_pix_to_reg (double data_vect[3], struct region *reg_ptr, 
		    int x, int y, int i) {
  double pgx, pgy, n;
  double mean, dn;
 
  for(int j=0; j<3; j++)  {
      n=(double)reg_ptr->n_of_points;
      dn=(double)reg_ptr->n_of_points;
      mean=reg_ptr->mean[j];/*mean value before merging*/
      reg_ptr->mean[j]=(n*reg_ptr->mean[j]+data_vect[j]) / (n+1);/*update mean*/
      reg_ptr->sigma_2[j]=(1/(dn+1)) * (dn*reg_ptr->sigma_2[j] + ((dn/(dn+1))*((data_vect[j]-mean)*(data_vect[j]-mean))));/*update sigma_2*/
    }
  pgx=(reg_ptr->pg_x*n+(double)x)/(n+1);
  pgy=(reg_ptr->pg_y*n+(double)y)/(n+1);
  reg_ptr->pg_x=pgx;
  reg_ptr->pg_y=pgy;
  reg_ptr->n_of_points++;
  return 0;
}


/*search_reg_label*/
struct region *search_reg_label(struct region *start_region, int search_label)
{
  struct region *reg_ptr;
  
  reg_ptr=start_region;
  while(reg_ptr->label!=search_label && reg_ptr->next!=NULL)
      reg_ptr=reg_ptr->next;    
  if(reg_ptr->label==search_label) 
      return reg_ptr;
  else
      return NULL;
}


/*euclid_metric*/
double euclid_metric(double vect_1[3], double vect_2[3]) {
  double delta[3];
  double dist_2=0.0;
  for(int j=0; j<3; j++)    {
      delta[j]=vect_1[j]-vect_2[j];
      delta[j]*=delta[j];
      dist_2+=delta[j];
    }
  return(fabs(sqrt (dist_2)));
}


/*euclid_metric_ptr*/
double euclid_metric_ptr(double *vect_1[3], double *vect_2[3]) {
  double delta[3];
  double dist_2=0.0;
  for(int j=0; j<3; j++)    {
      delta[j]=*vect_1[j]-*vect_2[j];
      delta[j]*=delta[j];
      dist_2+=delta[j];
    }
  return(fabs(sqrt (dist_2)));
}


/*min_ind*/
int min_ind(double vect[4]) {
  int min_ind=0;
  int ind=1;
  while(ind<4) {
    if(vect[min_ind]==-1) {
      min_ind++;
      ind++;
    }
    else if(vect[min_ind]<=vect[ind] || vect[ind]==-1)
      ind++;
    else {
      min_ind=ind;
      ind++;
    }
  }
  return min_ind;
}



/*search_label_in_n_list*/
struct neighbour_list *search_label_in_n_list(int label, struct region *reg_ptr) {
  struct neighbour_list *searcher;
  searcher=reg_ptr->neighbours;
  if(searcher==NULL)
    return NULL;
  while(searcher->label!=label && searcher->next!=NULL)
    searcher=searcher->next;
  if(searcher->label==label)
    return searcher;
  else
    return NULL;
}


/*new_neighbour*/
int new_neighbour(struct region *reg_ptr_1, struct region *reg_ptr_2) {
  struct neighbour_list *temp;
  if(reg_ptr_2==NULL)
    return 1;
  temp=(struct neighbour_list *)malloc(sizeof(struct neighbour_list));
  temp->label=reg_ptr_1->label;
  temp->border_length=1;
  temp->reg=reg_ptr_1;
  if(reg_ptr_2->neighbours == NULL)
    temp->next=NULL;
  else
    temp->next=reg_ptr_2->neighbours;
  reg_ptr_2->neighbours=temp;
    
  return 0;
}



/*make_neighbours*/
int make_neighbours(struct region *reg_ptr_1, struct region *reg_ptr_2) {
  if(reg_ptr_1==NULL || reg_ptr_2==NULL || reg_ptr_1==reg_ptr_2)
    return 1;
  else {
      struct neighbour_list *list_ptr;
      int label;

      label=reg_ptr_1->label;/*reg_1 in neighbours reg_2*/
      list_ptr=search_label_in_n_list(label, reg_ptr_2);
      if(list_ptr==NULL)
	new_neighbour(reg_ptr_1,reg_ptr_2);
      else
	list_ptr->border_length++;
	       
      label=reg_ptr_2->label;/*reg_2 in neighbours reg_1*/
      list_ptr=search_label_in_n_list(label, reg_ptr_1);
      if(list_ptr==NULL)
	new_neighbour(reg_ptr_2,reg_ptr_1);
      else
	list_ptr->border_length++;

      return 0;
    }
}

/*updated_mean*/
double updated_mean(double inp_vect[3], struct region *reg_ptr, int k) {
  double mean, dn;
  if(reg_ptr!=NULL) {
      dn=(double)reg_ptr->n_of_points;
      mean=(double)reg_ptr->mean[k];/*mean value before merging*/
      return (double)(dn*mean+inp_vect[k]) / (dn+1);/*updated mean*/
  } else 
    return -1.0 ; 
}

}
