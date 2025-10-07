# Modern Visual Attention Systems - Research & Implementations

## Overview

Your PhD work (2003-2005) on neural field-based visual attention was **ahead of its time**. The field has evolved significantly, splitting into several directions:

1. **Bottom-up saliency** (biologically-inspired, similar to your work)
2. **Deep learning saliency** (data-driven, dominates since 2015)
3. **Attention mechanisms in neural networks** (Transformers, 2017+)
4. **Active vision** (embodied AI, robotics)
5. **Hybrid systems** (combining classical + deep learning)

---

## 1. Classical Bottom-Up Saliency (Closest to Your Work)

### **1.1 Itti-Koch Model Evolution**

Your original code references "Koch et al." - this lineage continues:

#### **Original: Itti, Koch & Niebur (1998)**
- The foundational work
- Multi-scale feature extraction
- Winner-take-all + inhibition of return
- **Your work extended this with neural fields**

#### **iLab Neuromorphic Vision Toolkit (iNVT)**
- **Website**: http://ilab.usc.edu/toolkit/
- **Status**: Last updated ~2013, partially obsolete
- **Language**: C++
- **Features**: Implements Itti-Koch model
- **Comparison to your work**: Similar approach, but you added:
  - Neural field dynamics
  - 3D attention with stereo
  - More sophisticated integration

**Verdict**: Historical reference, not actively maintained

---

#### **SaliencyToolbox (Matlab)**
- **Website**: http://www.saliencytoolbox.net/
- **Status**: Last updated 2013
- **Comparison**: Educational, but outdated

---

### **1.2 Modern Classical Implementations**

#### **OpenSaliency (2020s revival)**
- **Not a single toolkit**, but OpenCV includes saliency
- **OpenCV Saliency Module**: `cv::saliency`
  - Implements several classical algorithms
  - Spectral Residual (SR)
  - Fine Grained (FG)
  - Itti-Koch variant (StaticSaliency)

**Code Example:**
```cpp
#include <opencv2/saliency.hpp>

cv::Ptr<cv::saliency::StaticSaliencySpectralResidual> saliency =
    cv::saliency::StaticSaliencySpectralResidual::create();

cv::Mat saliency_map;
saliency->computeSaliency(image, saliency_map);
```

**Comparison to your work**:
- ❌ No neural field dynamics
- ❌ No stereo/3D
- ✅ Fast, simple, modern API
- ✅ Good for baseline comparisons

---

#### **VOCUS2 (Visual Object Detection with a Computational Attention System)**
- **Paper**: Frintrop et al., "Traditional Saliency Reloaded" (2015)
- **Code**: https://github.com/GeeeG/VOCUS2
- **Language**: C++
- **Status**: Maintained until ~2018

**Features**:
- Center-surround with DoG filters
- Multiple feature channels (color, intensity, orientation)
- Configurable weights
- Fast processing

**Comparison to your work**:
- Similar feature extraction
- ✅ Well-engineered, modular
- ❌ Simpler integration (no neural fields)
- ❌ No stereo
- ✅ Active research group validated it

**Verdict**: Good modern classical baseline

---

### **1.3 Spectral/Frequency-Based Methods**

#### **Spectral Residual (Hou & Zhang, 2007)**
- Extremely simple: saliency from Fourier spectrum
- Very fast
- Included in OpenCV

#### **GBVS (Graph-Based Visual Saliency, Harel et al., 2007)**
- **Code**: http://www.vision.caltech.edu/~harel/share/gbvs.php
- **Language**: Matlab
- **Status**: Classic, still cited

**Key idea**: Treat saliency as graph activation
- Create graph from feature maps
- Use Markov chains for activation spread
- Similar philosophy to your neural fields!

**Comparison**:
- ✅ Graph dynamics similar to your neural field dynamics
- ✅ Biologically plausible
- ❌ No code maintenance
- 🤔 **Interesting**: Could compare graph propagation vs neural field

---

## 2. Deep Learning Saliency (Dominant Since 2015)

### **2.1 CNN-Based Saliency**

#### **DeepGaze (2014-2017)**
- **Papers**: DeepGaze I, II, III
- **Code**: https://github.com/matthias-k/DeepGaze
- **Language**: Python (PyTorch)
- **Approach**: CNN trained on eye-tracking data

**Key Innovation**: Learn saliency from where humans look

**Comparison**:
- ✅ State-of-the-art accuracy on benchmarks
- ❌ Data-driven, not interpretable
- ❌ No stereo/3D
- ❌ Requires GPU, training data
- 🤔 Your work is more **mechanistic**, theirs is **predictive**

---

#### **SalGAN (2017)**
- **Paper**: "SalGAN: Visual Saliency Prediction with GANs"
- **Code**: https://github.com/imatge-upc/saliency-salgan-2017
- **Approach**: GAN trained on eye-tracking datasets

**Performance**: Excellent on standard benchmarks

**Comparison**:
- Pure deep learning, no interpretability
- **Different goal**: Predict human gaze, not "interesting" regions
- Your work could be **ground truth** for their training!

---

### **2.2 Modern Saliency Frameworks**

#### **PyTorch Saliency**
- **Code**: https://github.com/xuebinqin/BASNet
- **BASNet (2019)**: Boundary-Aware Salient Object Detection
- **U²-Net (2020)**: Better architecture

**State-of-the-art** for salient object detection

**Comparison**:
- ✅ Very accurate object segmentation
- ❌ Not attention mechanism (detects objects)
- ❌ Single image only
- 🤔 Could **combine**: Use your attention to guide where they look

---

#### **EML-NET (2021)**
- **Paper**: "Extremely Multilabel Deep Learning"
- **Approach**: Multi-task learning for saliency

**Trend**: Modern work is about **predicting human eye movements**, not computational attention

---

### **2.3 Practical Tools**

#### **OpenCV DNN Saliency**
```cpp
// Modern: Use pre-trained DNN
cv::dnn::Net net = cv::dnn::readNet("salgan.pb");
cv::Mat blob = cv::dnn::blobFromImage(image);
net.setInput(blob);
cv::Mat saliency = net.forward();
```

**Trade-off**: Accurate but black-box

---

## 3. Attention in Neural Networks (Transformers)

### **3.1 Self-Attention Mechanisms**

#### **Vision Transformers (ViT, 2020)**
- **Paper**: "An Image is Worth 16x16 Words"
- **Code**: https://github.com/google-research/vision_transformer

**Key Idea**: Treat image patches as tokens, use Transformer attention

**Comparison**:
- **Completely different paradigm**
- Attention weights are **learned**, not computed
- ✅ Can visualize attention maps
- ❌ Not interpretable mechanistically
- 🤔 Interesting: Their attention ≈ your saliency?

---

#### **DINO (2021) - Self-Supervised ViT**
- **Paper**: "Emerging Properties in Self-Supervised Vision Transformers"
- **Code**: https://github.com/facebookresearch/dino

**Amazing discovery**: Self-attention maps correspond to **semantic segmentation**

**Example**:
```python
import torch
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
attention = model.get_last_selfattention(image)
# Visualize: attention shows object boundaries!
```

**Comparison**:
- 🤯 **Emergent behavior**: Unsupervised attention discovers objects
- Your work is **hand-crafted**, theirs is **emergent**
- 🤔 **Research question**: How do these relate?

---

## 4. Stereo & 3D Attention (Closest to Your Unique Contribution)

### **4.1 Depth-Based Attention**

Your work on **stereo-based attention with 3D neural fields** was novel. Modern approaches:

#### **RGB-D Saliency**
- **Paper**: Peng et al., "RGBD Salient Object Detection" (2014+)
- **Code**: Multiple implementations on GitHub
- **Approach**: Use depth as additional feature channel

**Example Papers**:
- "Depth-Induced Multi-Scale Recurrent Attention Network" (2020)
- "JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework" (2020)

**Comparison**:
- ✅ Use depth, like you
- ❌ Deep learning, not neural fields
- ❌ Single image, no dynamics
- 🤔 **Your innovation**: Neural field **dynamics** in 3D

---

#### **Point Cloud Attention**
- **Paper**: "Point Cloud Saliency" (2019+)
- **Approach**: Saliency on 3D point clouds (LiDAR)

**Modern context**: Autonomous driving, robotics

**Comparison**:
- Different data type (point cloud vs stereo)
- Similar goal: 3D spatial attention
- **Your work could extend**: Neural fields on point clouds?

---

### **4.2 Active Vision & Embodied AI**

#### **Active Vision Toolbox (AVT)**
- **Status**: Research prototypes, no unified toolkit
- **Examples**:
  - DOVES (Dynamic Vision Sensor models)
  - Neuromorphic vision systems

**Comparison**:
- Your `move_sensor_mode` was **active vision**
- Modern: Focus on robotic platforms (ROS integration)

---

#### **Habitat-Sim (Facebook AI, 2019+)**
- **Code**: https://github.com/facebookresearch/habitat-sim
- **Purpose**: 3D environment for embodied AI
- **Attention**: Agents learn where to look

**Comparison**:
- **Reinforcement learning** to learn attention
- Your work: **Computational model** of attention
- 🤔 Could **combine**: Use your model as RL reward

---

## 5. Modern Implementations You Can Use

### **5.1 Python Libraries**

#### **pySaliencyMap**
- **Code**: https://github.com/akisato-/pySaliencyMap
- **Features**: Itti-Koch model in Python
- **Status**: ~2018, simple

**Use case**: Quick prototyping, teaching

---

#### **Saliency Benchmarking**
- **MIT Saliency Benchmark**: http://saliency.mit.edu/
- **Dataset**: Eye-tracking on 1000+ images
- **Use**: Compare algorithms

**Datasets**:
- MIT1003
- CAT2000
- SALICON (15K images)

**Comparison**: Test your model vs modern DL

---

### **5.2 C++ Libraries**

#### **OpenCV (Best for new projects)**
```cpp
#include <opencv2/saliency.hpp>

// Multiple algorithms available:
auto sr = cv::saliency::StaticSaliencySpectralResidual::create();
auto fg = cv::saliency::StaticSaliencyFineGrained::create();

// For motion/video:
auto motion = cv::saliency::MotionSaliencyBinWangApr2014::create();
```

**Verdict**: **Start here** for your Phase 1

---

#### **PCL (Point Cloud Library)**
For 3D/depth work:
```cpp
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>

// Compute normals, curvature → saliency in 3D
```

---

### **5.3 ROS2 Packages (For Robotics)**

#### **attention_system_ros**
- Various ROS packages for visual attention
- Example: `ros_visual_saliency`
- Integrate with robot cameras

**Your work**: Could be wrapped in ROS node

---

## 6. Research Directions (2020s)

### **6.1 Current Hot Topics**

#### **Explainable AI (XAI) with Attention**
- **Trend**: Use attention to explain CNN decisions
- **Papers**: Grad-CAM, Attention maps in CNNs
- **Connection**: Your work is inherently explainable!

#### **Attention for Video Understanding**
- **Trend**: Temporal attention in videos
- **Papers**: Video Transformers, Temporal Attention
- **Your work**: Had temporal component (`move_sensor_mode`)

#### **Multi-Modal Attention**
- **Trend**: Combine vision, audio, text
- **Example**: CLIP (OpenAI), Flamingo (DeepMind)
- **Your work**: Multi-feature integration was multi-modal!

#### **Neuromorphic Vision**
- **Trend**: Event cameras, spiking neural networks
- **Connection**: Your neural fields are somewhat similar to SNNs
- **Hardware**: Intel Loihi, IBM TrueNorth

---

### **6.2 Gaps Your Work Could Fill**

#### **1. Interpretable 3D Attention**
- **Gap**: Modern DL methods are black-box
- **Your contribution**: Mechanistic model with neural dynamics
- **Modern angle**: Combine with ViT attention visualization

#### **2. Dynamic Attention (Temporal Coherence)**
- **Gap**: Most modern work is single-image
- **Your contribution**: Neural field maintains state across time
- **Modern angle**: Compare to LSTM/Transformer temporal attention

#### **3. Biologically-Plausible Deep Learning**
- **Trend**: Make DNNs more brain-like
- **Your contribution**: Neural field dynamics from neuroscience
- **Modern angle**: Use your model as regularizer for DL training

#### **4. Active Vision for Robotics**
- **Gap**: Where should robot look next?
- **Your contribution**: Sensor movement based on attention
- **Modern angle**: Integrate with modern SLAM, object detection

---

## 7. Specific Comparisons to Your Work

### **What Made Your Work Unique (2003-2005)**

✅ **Neural field dynamics** (not just feature integration)
✅ **3D attention** with stereo depth
✅ **Temporal coherence** across frames
✅ **Active vision** (sensor movement)
✅ **Multi-feature integration** with learned weights
✅ **Object tracking** through attention

### **What's Changed Since Then**

**Then (2003-2005)**:
- Computational power: limited
- Cameras: low resolution, slow
- Main application: Robotics
- Approach: Hand-crafted features
- Validation: Qualitative, small datasets

**Now (2025)**:
- Computational power: abundant (GPUs)
- Cameras: 4K, high frame rate, cheap depth sensors
- Main application: Autonomous vehicles, AR/VR, smartphones
- Approach: Deep learning dominates
- Validation: Large datasets (SALICON, MIT1003), eye-tracking ground truth

### **Where Your Work Still Stands Out**

✅ **Interpretability**: You can explain WHY attention is where it is
✅ **Efficiency**: Neural fields are fast (no GPU needed)
✅ **Dynamics**: Temporal evolution, not single-frame
✅ **3D**: Native stereo support
✅ **Modularity**: Easy to add features
✅ **Real-time**: Ran on 2003 hardware!

---

## 8. Recommended Modern Baselines for Comparison

If you rebuild your system, compare against:

### **Classical Baselines**
1. **OpenCV Spectral Residual** (dead simple)
2. **VOCUS2** (fair comparison, similar approach)
3. **GBVS** (graph-based, similar philosophy)

### **Deep Learning Baselines**
1. **DeepGaze II** (state-of-the-art eye-tracking prediction)
2. **BASNet** (salient object detection)
3. **DINO self-attention maps** (emergent attention)

### **Stereo/3D Baselines**
1. **RGB-D saliency** (Peng et al. or recent papers)
2. **Point cloud saliency** (if you have LiDAR)

### **Evaluation Datasets**
1. **MIT1003** (eye-tracking, 1003 images)
2. **CAT2000** (categorized scenes)
3. **SALICON** (large-scale, 15K images)
4. **KITTI** (if doing stereo/autonomous driving)

---

## 9. Modern Research Questions You Could Address

### **Hybrid Models**
**Question**: Can neural field dynamics improve deep learning saliency?

**Approach**:
1. Use CNN for feature extraction
2. Use your neural field for integration
3. Compare to pure DL

**Impact**: Best of both worlds (learned features + interpretable dynamics)

---

### **Temporal Attention**
**Question**: How do neural fields compare to Transformers for video attention?

**Approach**:
1. Implement both on same video dataset
2. Compare: accuracy, speed, interpretability
3. Analyze: Where do they agree/disagree?

**Impact**: Understand emergent vs designed attention

---

### **Active Vision**
**Question**: Can neural field attention guide robotic vision?

**Approach**:
1. Integrate with ROS2
2. Use attention to decide camera movements
3. Compare to learned policies (RL)

**Impact**: Efficient, interpretable active vision

---

### **3D Scene Understanding**
**Question**: Does 3D neural field attention help object recognition?

**Approach**:
1. Use your stereo attention
2. Feed focused regions to object detector (YOLO)
3. Compare to full-image detection

**Impact**: Faster, more robust detection

---

## 10. Recommended Reading (2020s Papers)

### **Survey Papers**
1. **"Salient Object Detection: A Survey"** (Borji et al., 2019)
   - Comprehensive overview
   - Compares 300+ methods

2. **"Saliency in Deep Learning"** (Simonyan et al., 2020)
   - How attention works in DNNs

### **Key Recent Papers**

**Classical Evolution**:
- "Traditional Saliency Reloaded" (Frintrop, 2015) - VOCUS2
- "Revisiting Video Saliency" (Wang, 2018)

**Deep Learning**:
- "DeepGaze II" (Kümmerer, 2017) - SOTA eye-tracking
- "DINO" (Caron, 2021) - Emergent attention

**3D/Stereo**:
- "RGB-D Saliency with Cross-Modal Transfer" (Piao, 2019)
- "Depth-Induced Multi-Scale Attention" (Chen, 2020)

**Active Vision**:
- "Learning to Look Around" (Jayaraman, 2018) - RL for attention
- "Active Vision for Manipulation" (Zeng, 2020) - Robotics

---

## 11. Code Repositories to Study

### **GitHub Collections**

**Saliency Zoo**:
- https://github.com/topics/saliency-detection
- 100+ implementations

**Awesome Saliency**:
- https://github.com/jiwei0921/SOD-CNNs-based-code-summary-
- Curated list of papers + code

### **Specific Implementations**

**Classical**:
```
xuebinqin/BASNet              # State-of-the-art salient object detection
GeeeG/VOCUS2                  # Modern classical (closest to yours)
matthias-k/DeepGaze           # DL baseline
```

**Video**:
```
wenguanwang/VideoSaliency     # Video attention
```

**3D/Depth**:
```
jiwei0921/RGBD-SOD            # RGB-D saliency collection
```

---

## 12. Bottom Line: How Does Your Work Compare?

### **Then (2005)**
Your work was **cutting-edge research**:
- Neural field integration was novel
- 3D stereo attention was unique
- Active vision was forward-thinking

### **Now (2025)**
Your work is **historically important** but:
- ❌ Deep learning dominates benchmarks
- ❌ No longer state-of-the-art accuracy
- ✅ Still relevant for interpretability
- ✅ Still relevant for real-time/embedded
- ✅ Still relevant for robotics
- ✅ Unique 3D neural field approach

### **Future (2025+)**
Your work could be **relevant again** because:
- ✅ Trend toward interpretable AI
- ✅ Neuromorphic computing (Loihi chips)
- ✅ Embodied AI needs efficient attention
- ✅ Hybrid models (classical + DL)
- ✅ Edge computing (no GPU)

---

## 13. Recommendation for Your Project

### **Phase 1: Modern Reimplementation**
- Build system with modern C++ + OpenCV
- Compare to OpenCV saliency baselines
- Validate on MIT1003 or CAT2000

### **Phase 2: Hybrid Approach**
- Replace hand-crafted features with CNN features
- Keep neural field dynamics
- Test if interpretability + accuracy possible

### **Phase 3: Modern Application**
- Active vision for robot (ROS2)
- Or: Efficient attention for mobile (Android/iOS)
- Or: Explainable AI for medical imaging

### **Research Contribution**
**Title idea**: *"Neural Field Dynamics for Interpretable Visual Attention: A Modern Perspective"*

**Contributions**:
1. Modern reimplementation of neural field attention
2. Comparison to DL methods (accuracy vs interpretability)
3. Hybrid model combining both
4. Application to robotics/XAI

**Impact**: Bridge classical neuroscience-inspired models and modern deep learning

---

## 14. Key Contacts / Groups Still Active

**Classical Saliency**:
- **Simone Frintrop** (Universität Hamburg) - VOCUS2, still active
- **Laurent Itti** (USC) - Original Itti-Koch, moved to neuromorphic
- **Christof Koch** (Allen Institute) - Now consciousness research

**Deep Learning Saliency**:
- **Matthias Kümmerer** (Tübingen) - DeepGaze
- **Ali Borji** (Survey papers)

**Active Vision**:
- **Deva Ramanan** (CMU) - Active perception
- **Jitendra Malik** (Berkeley) - Embodied AI

**Neuromorphic**:
- **Tobi Delbruck** (ETH Zurich) - Event cameras
- **Garrick Orchard** (Intel) - Neuromorphic vision

---

## Conclusion

Your 20-year-old work remains **conceptually relevant**:
- Neural field dynamics are elegant and interpretable
- 3D attention with stereo is still understudied
- Active vision is more important than ever (robotics)

Modern implementations mostly went the **deep learning route** (accuracy over interpretability).

**Your opportunity**: Build a **hybrid system** that combines:
- Modern CNN features (accurate)
- Your neural field dynamics (interpretable)
- 3D attention (unique)
- Active vision (practical)

This could be **publishable** in 2025, framed as:
*"Interpretable AI meets classical neuroscience-inspired models"*

The field needs **explainable, efficient, dynamic** attention models. Your work checked all those boxes 20 years ago. Time to update it! 🚀
