#ifndef CAMERA_MATRIX_H
#define CAMERA_MATRIX_H

#include <vector>
#include <Eigen/Eigen>

#include "DistortionFunction.h"

typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;

template<typename T> T sqr(T t) { return t * t; }

class CameraMatrix {
public:
	typedef std::vector<CameraMatrix> Vector;

	CameraMatrix();
	CameraMatrix(double f, double cx, double cy);
	CameraMatrix(const Eigen::Matrix3d &K, const Matrix3x4d &RT);
	CameraMatrix(const Matrix3x4d &P);
	~CameraMatrix();

	CameraMatrix& operator=(const CameraMatrix &cam);

	void setIntrinsic(const Eigen::Matrix3d &K);
	void setRotation(const Eigen::Matrix3d &R);
	void setTranslation(const Eigen::Vector3d &T);
	void setCameraCenter(const Eigen::Vector3d &c);
	void setOrientation(const Matrix3x4d &RT);
	void setSize(double w, double h);

	const Eigen::Matrix3d& getIntrinsic() const;
	const Eigen::Matrix3d& getRotation() const;
	const Eigen::Vector3d& getTranslation() const;
	const Eigen::Vector2d& getSize() const;
	double getWidth() const;
	double getHeight() const;
	const Eigen::Vector3d& getCameraCenter() const;

	const Eigen::Vector3d opticalAxis() const;
	const Eigen::Vector3d upVector() const;
	const Eigen::Vector3d rightVector() const;

	const Matrix3x4d getExtrinsic() const;
	const Matrix3x4d getOrientation() const;
	const Matrix3x4d getProjection() const;

	double getFocalLength() const;
	double getAspectRatio() const;

	Eigen::Vector2d getPrincipalPoint() const;
	Eigen::Vector2d projectPoint(const Eigen::Vector3d& X) const;
	Eigen::Vector2d projectPoint(const DistortionFunction &distortion, const Eigen::Vector3d &X) const;
	Eigen::Vector3d unprojectPixel(const Eigen::Vector2d &p, double depth = 1) const;

	Eigen::Vector3d intersectRayWithPlane(const Eigen::Vector4d &plane, int x, int y) const;
	
	Eigen::Vector3d transformPointIntoCameraSpace(const Eigen::Vector3d &p) const;
	Eigen::Vector3d transformPointFromCameraSpace(const Eigen::Vector3d &p) const;
	Eigen::Vector3d transformDirectionFromCameraSpace(const Eigen::Vector3d &dir) const;
	Eigen::Vector3d transformDirectionIntoCameraSpace(const Eigen::Vector3d &dir) const;

	Eigen::Vector2d transformPointIntoNormalizedCoordinate(const Eigen::Vector2d &p) const;
	Eigen::Vector2d transformPointFromNormalizedCoordinate(const Eigen::Vector2d &p) const;

	Eigen::Vector3d getRay(const Eigen::Vector2d& p) const;
	bool isOnGoodSide(const Eigen::Vector3d& p) const;

private:
	void updateCachedValues(bool intrinsic, bool orientation);

	Eigen::Matrix3d m_K, m_R;
	Eigen::Vector3d m_T;
	Eigen::Matrix3d m_invK, m_Rt;
	Eigen::Vector3d m_center;
	Eigen::Vector2d m_size;

};

#endif

