# Semi-automatic quantification of incomplete ablation lesions after pulmonary vein isolation (PVI)
Author: Marta Nuñez-Garcia (marnugar@gmail.com)

## About
Implementation of the method described in: [*Mind the gap: quantification of incomplete ablation patterns after pulmonary vein isolation using minimum path search*. Marta Nuñez-Garcia, Oscar Camara, Mark D O’Neill, Reza Razavi, Henry Chubb, and Constantine Butakoff. Medical Image Analysis (2018) 51, 1-12](https://www.sciencedirect.com/science/article/abs/pii/S1361841518307965). Please cite this reference when using this code. Preprint available at: [arXiv:1806.06387.](https://arxiv.org/abs/1806.06387) 
The code runs in Linux and Windows.

We present a methodology to detect and quantify incomplete ablation lesions (or gaps) after Radiofrequency Pulmonary Vein Isolation (RF-PVI) in a reliable, reproducible and observer-independent way. We provide an unambiguous definition of the gap as the minimal portion of healthy tissue around a pulmonary vein (PV). Based on this objective definition of gap we propose a quantitative and highly reproducible index, the *Relative Gap Measure (RGM)*, which represents the proportion of the vein not encircled by scar (i.e. portion of incompleteness). More metrics such as the total gap length and the number of gaps are also provided.

Example of gap quantification result:
![Example](https://github.com/martanunez/PV_gap_quantification/blob/master/gaps.png)


The method is modified with respect to the pipeline described in the paper as follows:
- The automatic parcellation of the left atrium (LA) and the definition of the gap search areas (step 2) are done by using the flattening standardization method described in [*https://github.com/martanunez/LA_flattening*](https://github.com/martanunez/LA_flattening). 3D mesh registration is no longer used highly reducing execution time. 
- Only one given scar segmentation is considered (multi-threshold result integration is not included).


Schematic representation of the proposed graph based algorithm (from the paper):

![Algorithm](https://github.com/martanunez/PV_gap_quantification/blob/master/graph-based-algorithm.png)

## Code
[Python](https://www.python.org/) scripts depending (basically) on [VTK](https://vtk.org/) and [VMTK](http://www.vmtk.org/). 

## Pipeline
The pipeline is split in 5 parts. Please, refer to [*https://github.com/martanunez/LA_flattening*](https://github.com/martanunez/LA_flattening) to run the first 4 steps (i.e. LA flattening).

The parcellation defined in a 2D reference template (see figure below) is then transferred to the 2D version of the LA obtained with steps 1 to 4 and to the 3D LA mesh afterwards.
The definition of the areas where gaps will be searched is related to the chosen PVI strategy. In this work we considered the two most common PVI scenarios in clinical routine:
- Independent-encirclement : the four PVs are independently isolated by creating PV-specific continuous lesions that completely surround each of them.
- Joint-encirclement : the two ipsilateral veins (i.e., on the same side, right or left PVs) are jointly isolated by a lesion that simultaneously encircles the two of them.

Searching areas depicted in a two-dimensional representation of the LA:
<img src=https://github.com/martanunez/PV_gap_quantification/blob/master/searching_areas.png width="70%">


## Instructions
Clone the repository:
```
git clone https://github.com/martanunez/PV_gap_quantification

cd PV_gap_quantification
```

## Usage

**LA flattening:**
```
Run steps 1, 2, 3, and 4 as explained in https://github.com/martanunez/LA_flattening
```

**Gap quantification:**
```
5_compute_RGM_4veins.py [-h] [--meshfile PATH]
5_compute_RGM_lateral_veins.py [-h] [--meshfile PATH]

Arguments:
  -h, --help            show this help message and exit
  --meshfile PATH       path to input mesh
```

## Usage example
**LA flattening:**
```
python 1_mesh_standardisation.py --meshfile data/mesh.vtk --pv_dist 5 --laa_dist 5 --vis 1
python 2_close_holes_project_info.py --meshfile_open data/mesh_crinkle_clipped.vtk --meshfile_open_no_mitral  data/mesh_clipped_mitral.vtk --meshfile_closed data/mesh_clipped_c.vtk
python 3_divide_LA.py --meshfile data/mesh_clipped_c.vtk
python 4_flat_atria.py --meshfile data/mesh_clipped_c.vtk
```
**Gap quantification:**
```
python 5_compute_RGM_4veins.py --meshfile data/mesh.vtk
python 5_compute_RGM_lateral_veins.py --meshfile data/mesh.vtk
```

## Dependencies
The scripts in this repository were successfully run with:
1. **Ubuntu 16.04**
    - [Python](https://www.python.org/) 3.5.2
    - [VMTK](http://www.vmtk.org/) 1.4
    - [VTK](https://vtk.org/) 9.1.0
2. **Ubuntu 16.04**
    - [Python](https://www.python.org/) 2.7.12
    - [VMTK](http://www.vmtk.org/) 1.4
    - [VTK](https://vtk.org/) 8.1.0
3. **Windows 8.1**
    - [Python](https://www.python.org/) 3.6.4
    - [VMTK](http://www.vmtk.org/) 1.4
    - [VTK](https://vtk.org/) 8.1.0
  
Other required packages are: NumPy, SciPy, xlsxwriter, Matplotlib, joblib, and python-tk. 
*FillSurfaceHoles* and *DistanceTransformMesh* binaries can be compiled from source using [FillSurfaceHoles](https://github.com/cbutakoff/tools/tree/master/FillSurfaceHoles) and [DistTransform](https://github.com/cbutakoff/tools/tree/master/DistTransform).

### Python packages installation
To install VMTK follow the instructions [here](http://www.vmtk.org/download/). The easiest way is installing the VMTK [conda](https://docs.conda.io/en/latest/) package (it additionally includes VTK, NumPy, etc.). It is recommended to create an environment where VMTK is going to be installed and activate it:

```
conda create --name vmtk_env
conda activate vmtk_env
```
Then, install vmtk:
```
conda install -c vmtk vtk itk vmtk
```
<!--Activate the environment when needed using:
```
source activate vmtk_env
```-->
**Nevertheless,** for Ubuntu, the easiest option for me was to build VMTK from source (slowly but surely). Instructions can be found [here.](http://www.vmtk.org/download/)
In brief:
```
git clone https://github.com/vmtk/vmtk.git
mkdir vmtk-build
cd vmtk-build
ccmake ../vmtk
make 
```
And edit the ~/.bashrc file,
```
gedit ~/.bashrc
```
adding the following line:  source /home/{your_path_to_vmtk}/vmtk/vmtk-build/Install/vmtk_env.sh


## Important note
You may need to slightly modify vmtkcenterlines.py from the VMTK package if you encounter the following error when running 1_mesh_standardisation.py:

```
     for i in range(len(self.SourcePoints)/3):
TypeError: 'float' object cannot be interpreted as an integer
```

Find vmtkcenterlines.py file and edit as follows:

Line 128: change ***for i in range(len(self.SourcePoints)/3):*** by ***for i in range(len(self.SourcePoints)//3):***

Line 133: change ***for i in range(len(self.TargetPoints)/3):*** by ***for i in range(len(self.TargetPoints)//3):*** 
## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
