#ifndef CAMERA_MATRIX_H
#define CAMERA_MATRIX_H

#include <vector>
#include <Eigen/Eigen>

#include "BATypeUtils.h"

#include "DistortionFunction.h"

template<typename T> T sqr(T t) { return t * t; }

class CameraMatrix {
public:
	typedef std::vector<CameraMatrix> Vector;

	CameraMatrix();
	CameraMatrix(Scalar f, Scalar cx, Scalar cy);
	CameraMatrix(const Matrix3X &K, const Matrix3x4X &RT);
	CameraMatrix(const Matrix3x4X &P);
	~CameraMatrix();

	CameraMatrix& operator=(const CameraMatrix &cam);

	void setIntrinsic(const Matrix3X &K);
	void setRotation(const Matrix3X &R);
	void setTranslation(const Vector3X &T);
	void setCameraCenter(const Vector3X &c);
	void setOrientation(const Matrix3x4X &RT);
	void setSize(Scalar w, Scalar h);

	const Matrix3X& getIntrinsic() const;
	const Matrix3X& getRotation() const;
	const Vector3X& getTranslation() const;
	const Vector2X& getSize() const;
	Scalar getWidth() const;
	Scalar getHeight() const;
	const Vector3X& getCameraCenter() const;

	const Vector3X opticalAxis() const;
	const Vector3X upVector() const;
	const Vector3X rightVector() const;

	const Matrix3x4X getExtrinsic() const;
	const Matrix3x4X getOrientation() const;
	const Matrix3x4X getProjection() const;

	Scalar getFocalLength() const;
	Scalar getAspectRatio() const;

	Vector2X getPrincipalPoint() const;
	Vector2X projectPoint(const Vector3X& X) const;
	Vector2X projectPoint(const DistortionFunction &distortion, const Vector3X &X) const;
	Vector3X unprojectPixel(const Vector2X &p, Scalar depth = 1) const;

	Vector3X intersectRayWithPlane(const Vector4X &plane, int x, int y) const;
	
	Vector3X transformPointIntoCameraSpace(const Vector3X &p) const;
	Vector3X transformPointFromCameraSpace(const Vector3X &p) const;
	Vector3X transformDirectionFromCameraSpace(const Vector3X &dir) const; 
	Vector3X transformDirectionIntoCameraSpace(const Vector3X &dir) const;

	Vector2X transformPointIntoNormalizedCoordinate(const Vector2X &p) const;
	Vector2X transformPointFromNormalizedCoordinate(const Vector2X &p) const;

	Vector3X getRay(const Vector2X& p) const;
	bool isOnGoodSide(const Vector3X& p) const;

private:
	void updateCachedValues(bool intrinsic, bool orientation);

	Matrix3X m_K, m_R;
	Vector3X m_T;
	Matrix3X m_invK, m_Rt;
	Vector3X m_center;
	Vector2X m_size;

};

#endif

