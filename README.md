## P2DARL: Disaster-Aware Path Planning Based on Reinforcement Learning for Post-Earthquake Emergency Response

This project implements a post-disaster path planning system based on deep reinforcement learning, integrating disaster information extraction, environmental modeling, and intelligent path planning capabilities.



## Directory Structure

```
P2DARL/
├── Disaster Information Extraction/
│   ├── ChangeOS/
│   └── D-LinkNet/
├── Post-Earthquake Environmental Modeling/
└── Disaster-Aware Path Planning Agent/
```



## Resources of Weighted and Test Data

https://pan.baidu.com/s/1B-cbCcT0moAFGG7c79DOSw?pwd=2fgm



## Installation

### 1. Pre-trained Weights Setup
1. Download model weights from "Resources of Weighted":
   - ChangeOS model weights
   - D-LinkNet model weights
2. Merge the downloaded weights into the `P2DARL/Disaster Information Extraction` directory

### 2. Environment Requirements
- Python 3.7+
- PyTorch
- CUDA (recommended)
- ArcGIS Pro (required for kernel density analysis)



## Implementation Pipeline

### 1. Image Format Conversion
Use `TifClip2PNG.py` in `Post-Earthquake Environmental Modeling` directory to convert remote sensing images from .tif to .png format for disaster information extraction.

### 2. Disaster Information Extraction
1. Building Damage Assessment
   - Use `eval.py` in `ChangeOS` directory
   - Input: Pre-disaster and post-disaster high-resolution remote sensing images
   - Output: Building damage assessment results

2. Road Network Extraction
   - Use `test.py` in `D-LinkNet` directory
   - Input: Post-disaster high-resolution remote sensing images
   - Output: Road network extraction results

### 3. Environmental Modeling
1. Image Processing
   - Use `PNG_stitching.py` to merge sliding window inference results
   
2. Damage Assessment Analysis
   - Use `kernel_density.py` for kernel density analysis of building damage
   - Use `heatmap_pretreatment.py` and `heatmap_points.py` for peak point extraction
   
3. Road Network Processing
   - Use `zhang_method.py` for road extraction refinement
   - Use `shp2npy.py` to convert refined roads to environment arrays for path planning

### 4. Path Planning
Use `main_P2DARL_train.py` in the Disaster-Aware Path Planning Agent directory for multi-objective reinforcement learning path planning.



## Important Notes

1. Input images must meet resolution and format requirements
2. ArcGIS Pro license is required for kernel density analysis
3. GPU is recommended for model training and inference