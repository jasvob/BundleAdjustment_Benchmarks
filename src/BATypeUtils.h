#ifndef BA_TYPE_UTILS_H
#define BA_TYPE_UTILS_H

#include <Eigen/Eigen>

//typedef float Scalar;
typedef double Scalar;

typedef Eigen::Matrix<Scalar, 2, 1> Vector2X;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3X;
typedef Eigen::Matrix<Scalar, 4, 1> Vector4X;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXX;
typedef Eigen::Matrix<Scalar, 2, 2> Matrix2X;
typedef Eigen::Matrix<Scalar, 2, Eigen::Dynamic> Matrix2XX;
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Matrix3XX;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix3X;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4X;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXX;
typedef Eigen::Matrix<Scalar, 2, 3> Matrix2x3X;
typedef Eigen::Matrix<Scalar, 2, 6> Matrix2x6X;
typedef Eigen::Matrix<Scalar, 3, 4> Matrix3x4X;
typedef Eigen::Matrix<Scalar, 3, 6> Matrix3x6X;

#endif