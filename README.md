# PLuM: Real Time 6 DOF Pose Estimation of Known Geometries in Point Cloud Data
<!-- About PLuM -->
The Pose Lookup Method, PLuM, is a real-time method for 6-DOF pose estimation of known geometries in point cloud data.
The original paper can be found at: https://doi.org/10.3390/s23063085, and the C++ code available here is designed to be easy to understand and ready to hack or integrate with your projects.
The purpose of this file is to provide a simple to understand implementation of the algorithm - the aim is to keep it readable!
The GPU results reported in the paper are performed with a slightly different algorithm using different variable types etc. to suit GPU operations. Please contact v.bhandari@uq.edu.au for more details about the GPU version.\
**This repository is being actively updated.**

<!-- The three applications. -->
Three applications are provided:
* ***raycastModel***: generate a point cloud of a known geometry model for a given sensor to model pose.
* ***generateLookupTable***: generate a lookup table of the known geometry for use in the PLuM algorithm.
* ***plum***: the PLuM 6-DOF pose estimation algroithm. Requires a lookup table of the known geometry, and the point cloud to interpret.

<!-- The Maximum Sum of Evidence (MSoE) methdo. -->
PLuM stems from the Maximum Sum of Evidence (MSoE) method detailed here: https://doi.org/10.3390/s21196473.

## Dependencies
* The *Eigen* library for matrix operations: https://eigen.tuxfamily.org/dox/GettingStarted.html or https://linux.how2shout.com/how-to-install-eigen-c-library-on-ubuntu-22-04-or-20-04/ 
* Open3D for lookup table generation and raycasting: http://www.open3d.org/docs/release/compilation.html#compilation
* Boost library for Gaussian noise generation: https://linux.how2shout.com/how-to-install-boost-c-on-ubuntu-20-04-or-22-04/ 

## Installation
Clone the repository.
```bash
git clone https://github.com/vb44/Pose-Lookup-Method.git
```

Create a build folder in the repository.
```bash
mkdir build && cd build
```

Run CMake.
```bash
cmake ../src/
```

Make the executables.
```bash
make
```

## Example usage
We use the Stanford bunny model from https://www.thingiverse.com/thing:11622 as an example. The file is available in the *sample_data* folder.

### Generate a lookup table for the geometry
A lookup table is used instead of raycasting operations to limit computational resources.
The lookup table is generated such that the model is located in the positive XYZ axes.
This allows for direct accessing of elements from the lookup table.
A detailed explanation of the parameters is provided in the paper.
Example usage:
```bash
./generateLookup --modelFileName ../sample_data/StanfordBunny.stl --outputFileName ../sample_data/bunny_lookupTable --lookupToModel 0,0,0,50,40,10 --maxXYZ 100,100,100 --stepSize 1 --sigma 1
```

### Generate a point cloud of the geometery
Raycasting provides a method to generate a point cloud of the geometry.
This can be avoided if sensor data is available.
Example usage:
```bash
./raycastModel --modelFileName ../sample_data/StanfordBunny.stl --outputFileName ../sample_data/bunny_pointcloud --sensorToModel 0,0,30,30,20,-5 --headingRange -90,90,9 --elevationRange 0,90,4.5
```
### Run PLuM
PLuM requires a config file with information about the path to the point cloud file, the lookup table, and the search heuristic settings.
An example config file is dispalyed below. 

*testConfig*
```text
pointCloudFile                  : ../sample_data/bunny_pointcloud
LOOKUP_TABLE_lookupTableFile    : ../sample_data/bunny_lookupTable
LOOKUP_TABLE_lookupToModel      : 0,0,0,50,40,10
LOOKUP_TABLE_maxXYZ             : 100,100,100
LOOKUP_TABLE_stepSize           : 1
SEARCH_HEURISTIC_seed           : 0,0,20,25,15,-10
SEARCH_HEURISTIC_minDeviation   : -180,-180,-180,-10,-10,-10 
SEARCH_HEURISTIC_maxDeviation   : 180,180,180,10,10,10
SEARCH_HEURISTIC_stepSizes      : 60,60,60,2,2,2
SEARCH_HEURISTIC_rotSigma       : 0.05
SEARCH_HEURISTIC_transSigma     : 3
SEARCH_HEURISTIC_noIterations   : 40
SEARCH_HEURISTIC_resampleSize   : 1000
``` 
To run PLuM:
```bash
./plum --configFile ../sample_data/testConfig
```