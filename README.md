# BundleAdjustment_Benchmarks
Bundle adjustment benchmarking code.

Project is configured via CMake, which creates separate project for each currently available solver.

### Benchmarks:
 * Bundle_Adjustment_QRKit - QRKIT
 * Bundle_Adjustment_QRChol - QRCHOL
 * Bundle_Adjustment_MoreQR - MOREQR
 * Bundle_Adjustment_SPQR - QRSPQR
 * Bundle_Adjustment_Cholesky - CHOLESKY
 
All benchmarks share the same code and only where needed, type of the solver is distinguished by a symbol definition:
* QRKIT - QRKit block diagonal solver on the left block and dense QR on the remainig lower right block
* QRCHOL - QRKit block diagonal solver on the left block and Cholesky LDLT on the remainig lower right block
* MOREQR - same as QRKIT, but performing 2 QR decompositions in each step
* QRSPQR - SuiteSparseQR on the whole matrix
* CHOLESKY - Cholesky LDLT on the whole matrix

### Switching machine precision 
is done using typedef in BATypeUtils.h:
 * Single precision (Float32) - typedef float Scalar;
 * Double precision (Double64) - typedef double Scalar;

### How To Run: 

All projects expect input data file in th same format. Find some samples in the data folder of this repository.

#### Usage:

project_name.exe problem_data_file.txt

#### For example:

Bundle_Adjustment_QRKit.exe data/problem-16-22106-pre.txt

Bundle_Adjustment_Cholesky.exe data/problem-21-11315-pre.txt

### Acknowledgements

The data samples are taken from
http://grail.cs.washington.edu/projects/bal/

Some parts of the code are ported Eigen C++ implementation of 
https://github.com/chzach/SSBA
