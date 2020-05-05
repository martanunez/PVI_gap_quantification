# Semi-automatic quantification of incomplete Pulmonary Vein Isolation (PVI)
Author: Marta Nuñez-Garcia (marnugar@gmail.com)

## About
Implementation of the method described in: [*Mind the gap: quantification of incomplete ablation patterns after pulmonary vein isolation using minimum path search*. Marta Nuñez-Garcia, Oscar Camara, Mark D O’Neill, Reza Razavi, Henry Chubb, and Constantine Butakoff. Medical Image Analysis (2018) 51, 1-12](https://www.sciencedirect.com/science/article/abs/pii/S1361841518307965). Please cite this reference when using this code. Preprint available at: [arXiv:1806.06387.](https://arxiv.org/abs/1806.06387) 

The method is modified with respect to the pipeline described in the paper as follows:
- The automatic parcellation of the left atrium (LA) and definition of the gap searching areas (step 2) is done by using the flattening standardization method described in [*https://github.com/martanunez/LA_flattening*](https://github.com/martanunez/LA_flattening). 3D mesh registration is no longer used reducing execution time. 
- Only one given scar segmentation is considered (multi-threshold result integration is not included).


Original pipeline (from the paper):

![Original pipeline](https://github.com/martanunez/PV_gap_quantification/blob/master/pipeline_gaps.png)

## Code
[Python](https://www.python.org/) scripts depending (basically) on [VTK](https://vtk.org/) and [VMTK](http://www.vmtk.org/). 

## Pipeline
The pipeline is split in 5 parts. Please, refer to [*https://github.com/martanunez/LA_flattening*](https://github.com/martanunez/LA_flattening) to run the first 4 steps (i.e. LA flattening).

The parcellation defined in a 2D reference template is then transferred to the 2D version of the LA obtained with steps 1 to 4 and to the 3D LA mesh afterwards.
The definition of the areas where gaps will be searched is related to the chosen PVI strategy. In this work we considered the two most common PVI scenarios in clinical routine:
- Independent-encirclement : the four PVs are independently isolated by creating PV-specific continuous lesions that completely surround each of them.
- Joint-encirclement : the two ipsilateral veins (i.e., on the same side, right or left PVs) are jointly isolated by a lesion that simultaneously encircles the two of them.

Searching areas depicted in a two-dimensional representation of the LA:
![Searching areas depicted in a two-dimensional representation of the LA](https://github.com/martanunez/PV_gap_quantification/blob/master/searching_areas.png | width=100)

- **5_compute_RGM_4veins:**  
- **5_compute_RGM_lateral_veins:** 


## Instructions
Clone the repository:
```
git clone https://github.com/martanunez/PV_gap_quantification

cd PV_gap_quantification
```

## Usage
```
5_compute_RGM_4veins.py [-h] [--meshfile PATH]

Arguments:
  -h, --help            show this help message and exit
  --meshfile PATH       path to input mesh


5_compute_RGM_lateral_veins.py [-h] [--meshfile PATH]

Arguments:
  -h, --help            show this help message and exit
  --meshfile PATH       path to input mesh


```

## Usage example
```
python 5_compute_RGM_4veins.py --meshfile data/mesh.vtk

and/or

python 5_compute_RGM_lateral_veins.py --meshfile data/mesh.vtk
```

## Dependencies
The scripts in this repository were successfully run with:
1. Ubuntu 16.04
    - [Python](https://www.python.org/) 2.7.12
    - [VMTK](http://www.vmtk.org/) 1.4
    - [VTK](https://vtk.org/) 8.1.0
2. Windows 8.1
    - [Python](https://www.python.org/) 3.6.4
    - [VMTK](http://www.vmtk.org/) 1.4
    - [VTK](https://vtk.org/) 8.1.0
  
Other required packages are: NumPy, SciPy, xlsxwriter, Matplotlib, joblib, and python-tk.  

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
You can also build VMTK from source if you wish, for example, to use a specific VTK version. Instructions can be found [here.](http://www.vmtk.org/download/)


## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
