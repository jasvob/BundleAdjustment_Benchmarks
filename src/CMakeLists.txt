cmake_minimum_required(VERSION 3.0)

PROJECT(BundleAdjustment_Bench)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---------------------------------------------------------------		
# Find dependencies
#---------------------------------------------------------------

# OpenSubDiv
IF(WIN32)
    #Warning!! If MSVC, include <iso646.h> within the code to define words like "and", "not"... as operators.
    SET(SSP_PATH "Z:/suitesparse-metis-for-windows-master/suitesparse-metis-for-windows-master/build/install" CACHE PATH "Path to SuiteSparse")
    SET(SSP_INCLUDE_DIRS "${SSP_PATH}/include")
    SET(SSP_LIBRARY_DIRS "${SSP_PATH}/lib")
    SET(SSP_BLAS_LIBRARY_DIRS "${SSP_PATH}/lib64/lapack_blas_windows")
    SET(SSP_LIB "libamd.lib;libbtf.lib;libcamd.lib;libccolamd.lib;libcholmod.lib;libcolamd.lib;libcxsparse.lib;libklu.lib;libldl.lib;libspqr.lib;libumfpack.lib;metis.lib;suitesparseconfig.lib;libblas.lib;liblapack.lib")
ELSE()
    MESSAGE(FATAL_ERROR "Unrecognized OS")
ENDIF()
INCLUDE_DIRECTORIES(${SSP_INCLUDE_DIRS})
LINK_DIRECTORIES(${SSP_LIBRARY_DIRS} ${SSP_BLAS_LIBRARY_DIRS})

# Eigen
#SET(CMAKE_PREFIX_PATH "Z:/Eigen_PR/build;Z:/Eigen_PR;Z:/OpenSubdiv/install/include;C:/Program Files/Eigen3/include/eigen3")
SET(CMAKE_PREFIX_PATH "Z:/eigen_sparse_qr;Z:/OpenSubdiv/install/include;C:/Program Files/Eigen3/include/eigen3")
FIND_PACKAGE(Eigen3)
# Point to Eigen directly, if developing Eigen at the same time SET(EIGEN3_INCLUDE_DIR "C:/dev/eigen_pr")
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# For debug builds
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

IF(UNIX)
    ADD_DEFINITIONS("-std=c++11")
ENDIF()
if(WIN32)
	ADD_DEFINITIONS("-D_USE_MATH_DEFINES")
ENDIF()

#---------------------------------------------------------------
#Set the projects
#---------------------------------------------------------------		
# Main
ADD_EXECUTABLE(Bundle_Adjustment_Bench
	bundle_adjustment_bench.cpp
	Logger.cpp
	Optimization/BAFunctor.cpp
	CameraMatrix.cpp
	DistortionFunction.cpp
	Logger.h
	Optimization/BAFunctor.h
	CameraMatrix.h
	DistortionFunction.h
	Utils.h
	MathUtils.h
	)
	
TARGET_LINK_LIBRARIES(Bundle_Adjustment_Bench ${SSP_LIB})

# Tests
ADD_EXECUTABLE(Bundle_Adjustment_Solver_Test
	bundle_adjustment_solver_test.cpp
	Logger.cpp
	Optimization/BAFunctor.cpp
	CameraMatrix.cpp
	DistortionFunction.cpp
	Logger.h
	Optimization/BAFunctor.h
	CameraMatrix.h
	DistortionFunction.h
	Utils.h
	MathUtils.h
	BlockAngularSparseQR_Ext.h
	BlockDiagonalSparseQR_Ext.h
	BandedBlockedSparseQR_Ext.h
	SparseBlockCOO_Ext.h
	SparseQR_Ext.h
	)
	
TARGET_LINK_LIBRARIES(Bundle_Adjustment_Solver_Test ${SSP_LIB})

ADD_EXECUTABLE(Bundle_Adjustment_Solver_New
	bundle_adjustment_solver_new.cpp
	Logger.cpp
	Optimization/BAFunctor.cpp
	CameraMatrix.cpp
	DistortionFunction.cpp
	Logger.h
	Optimization/BAFunctor.h
	CameraMatrix.h
	DistortionFunction.h
	Utils.h
	MathUtils.h
	BlockAngularSparseQR_Ext.h
	BlockDiagonalSparseQR_Ext.h
	BandedBlockedSparseQR_Ext.h
	SparseBlockCOO_Ext.h
	SparseQR_Ext.h
	)
	
TARGET_LINK_LIBRARIES(Bundle_Adjustment_Solver_New ${SSP_LIB})

ADD_EXECUTABLE(LevMar_Solver_Test
	levmar_solver_test.cpp
	Logger.cpp
	Optimization/BAFunctor.cpp
	CameraMatrix.cpp
	DistortionFunction.cpp
	Logger.h
	Optimization/BAFunctor.h
	CameraMatrix.h
	DistortionFunction.h
	Utils.h
	MathUtils.h
	BlockAngularSparseQR_Ext.h
	BlockDiagonalSparseQR_Ext.h
	BandedBlockedSparseQR_Ext.h
	SparseBlockCOO_Ext.h
	SparseQR_Ext.h
	)
	
TARGET_LINK_LIBRARIES(LevMar_Solver_Test ${SSP_LIB})

ADD_EXECUTABLE(SimpleLM_Test
	simplelm_test.cpp
	Logger.cpp
	CameraMatrix.cpp
	DistortionFunction.cpp
	Logger.h
	Optimization/SimpleLM.h
	CameraMatrix.h
	DistortionFunction.h
	Utils.h
	MathUtils.h
	BlockAngularSparseQR_Ext.h
	BlockDiagonalSparseQR_Ext.h
	BandedBlockedSparseQR_Ext.h
	SparseBlockCOO_Ext.h
	SparseQR_Ext.h
	)
	
TARGET_LINK_LIBRARIES(SimpleLM_Test ${SSP_LIB})

