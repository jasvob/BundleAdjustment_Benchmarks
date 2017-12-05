#include "CameraMatrix.h"

CameraMatrix::CameraMatrix() 
	: m_K(Matrix3X::Identity()),
	m_R(Matrix3X::Identity()),
	m_T(Vector3X::Zero()) {

	this->updateCachedValues(true, true);
}

CameraMatrix::CameraMatrix(Scalar f, Scalar cx, Scalar cy)
	: m_K(Matrix3X::Identity()),
	m_R(Matrix3X::Identity()),
	m_T(Vector3X::Zero()) {

	m_K(0, 0) = f;
	m_K(1, 1) = f;
	m_K(0, 2) = cx;
	m_K(1, 2) = cy;

	this->updateCachedValues(true, true);
}

CameraMatrix::CameraMatrix(const Matrix3X &K, const Matrix3x4X &RT) 
	: m_K(K) {

	m_R = RT.leftCols(3);
	m_T = RT.col(3);
	
	this->updateCachedValues(true, true);
}

CameraMatrix::CameraMatrix(const Matrix3x4X &P) {
	Matrix3X K = P.leftCols(3);

	// Get affine matrix (rq-decomposition of M)
	// See Hartley & Zissermann, p552 (1st ed.)
	Scalar h = sqrt(sqr(K(2, 1)) + sqr(K(2, 2)));
	Scalar s = K(2, 1) / h;
	Scalar c = -K(2, 2) / h;

	Matrix3X Rx = Matrix3X::Zero();
	Rx(0, 0) = 1.0;
	Rx(1, 1) = c; Rx(2, 2) = c;
	Rx(1, 2) = -s; Rx(2, 1) = s;

	K = K * Rx;

	h = sqrt(sqr(K(2, 0)) + sqr(K(2, 2)));
	s = K(2, 0) / h;
	c = -K(2, 2) / h;

	Matrix3X Ry = Matrix3X::Zero();
	Ry(1, 1) = 1.0;
	Ry(0, 0) = c; Ry(2, 2) = c;
	Ry(0, 2) = -s; Ry(2, 0) = s;

	K = K * Ry;

	h = sqrt(sqr(K(1, 0)) + sqr(K(1, 1)));
	s = K(1, 0) / h;
	c = -K(1, 1) / h;

	Matrix3X Rz = Matrix3X::Zero();
	Rz(2, 2) = 1.0;
	Rz(0, 0) = c; Rz(1, 1) = c;
	Rz(0, 1) = -s; Rz(1, 0) = s;

	K = K * Rz;

	Matrix3X sign = Matrix3X::Identity();
	sign(0, 0) = (K(0, 0) < 0) ? -1 : 1;
	sign(1, 1) = (K(1, 1) < 0) ? -1 : 1;
	sign(2, 2) = (K(2, 2) < 0) ? -1 : 1;

	K = K * sign;

	Matrix3X R = Rx * Ry * Rz * sign;
	R.transposeInPlace();

	Vector3X P4 = P.col(3);
	Vector3X T = K.inverse() * P4;

	K = K * (1.0 / K(2, 2));

	m_K = K;
	m_R = R;
	m_T = T;

	this->updateCachedValues(true, true);
}

CameraMatrix::~CameraMatrix() {
}

CameraMatrix& CameraMatrix::operator=(const CameraMatrix &cam) {
	m_K = cam.getIntrinsic();
	m_R = cam.getRotation();
	m_T = cam.getTranslation();
	m_size = cam.getSize();

	this->updateCachedValues(true, true);

	return *this;
}


void CameraMatrix::setIntrinsic(const Matrix3X &K) {
	this->m_K = K;
	this->updateCachedValues(true, false);
}

void CameraMatrix::setRotation(const Matrix3X &R) {
	this->m_R = R;
	this->updateCachedValues(false, true);
}

void CameraMatrix::setTranslation(const Vector3X &T) {
	this->m_T = T;
	this->updateCachedValues(false, true);
}

void CameraMatrix::setCameraCenter(const Vector3X &c) {
	this->setTranslation(-1.0 * (m_R * c));
}

void CameraMatrix::setOrientation(const Matrix3x4X &RT) {
	this->m_R = RT.leftCols(3);
	this->m_T = RT.col(3);
	this->updateCachedValues(false, true);
}

void CameraMatrix::setSize(Scalar w, Scalar h) {
	this->m_size = Vector2X(w, h);
}

const Matrix3X& CameraMatrix::getIntrinsic() const {
	return this->m_K;
}

const Matrix3X& CameraMatrix::getRotation() const {
	return this->m_R;
}

const Vector3X& CameraMatrix::getTranslation() const {
	return this->m_T;
}

const Vector2X& CameraMatrix::getSize() const {
	return this->m_size;
}

Scalar CameraMatrix::getWidth() const {
	return this->m_size(0);
}

Scalar CameraMatrix::getHeight() const {
	return this->m_size(1);
}

const Vector3X& CameraMatrix::getCameraCenter() const {
	return this->m_center;
}

const Vector3X CameraMatrix::opticalAxis() const {
	return this->transformDirectionFromCameraSpace(Vector3X(0.0, 0.0, 1.0));
}

const Vector3X CameraMatrix::upVector() const {
	return this->transformDirectionFromCameraSpace(Vector3X(0.0, 1.0, 0.0));
}

const Vector3X CameraMatrix::rightVector() const {
	return this->transformDirectionFromCameraSpace(Vector3X(1.0, 0.0, 0.0));
}

Vector3X CameraMatrix::getRay(const Vector2X& p) const {
	return m_invK * Vector3X(p(0), p(1), 1.0);
}

bool CameraMatrix::isOnGoodSide(const Vector3X& p) const {
	return this->transformPointIntoCameraSpace(p)(2) > 0;
}

const Matrix3x4X CameraMatrix::getExtrinsic() const {
	Matrix3x4X RT;
	RT.leftCols(3) = this->m_R;
	RT.col(3) = this->m_T;

	return RT;
}

const Matrix3x4X CameraMatrix::getOrientation() const {
	Matrix3x4X RT;
	RT.leftCols(3) = this->m_R;
	RT.col(3) = this->m_T;

	return RT;
}

const Matrix3x4X CameraMatrix::getProjection() const {
	Matrix3x4X RT = this->getOrientation();

	return this->m_K * RT;
}

Scalar CameraMatrix::getFocalLength() const {
	return this->m_K(0, 0);
}

Scalar CameraMatrix::getAspectRatio() const {
	return this->m_K(1, 1) / this->m_K(0, 0);
}

Vector2X CameraMatrix::getPrincipalPoint() const {
	return Vector2X(this->m_K(0, 2), this->m_K(1, 2));
}

Vector2X CameraMatrix::projectPoint(const Vector3X& X) const {
	Vector3X q = m_K * (m_R * X + m_T);
	
	return Vector2X(q(0) / q(2), q(1) / q(2));
}

Vector2X CameraMatrix::projectPoint(const DistortionFunction &distortion, const Vector3X &X) const {
	Vector3X XX = m_R * X + m_T;
	Vector2X p;
	p(0) = XX(0) / XX(2);
	p(1) = XX(1) / XX(2);
	p = distortion(p);

	return Vector2X(
		m_K(0, 0) * p(0) + m_K(0, 1) * p(1) + m_K(0, 2),
						   m_K(1, 1) * p(1) + m_K(1, 2)
	);	
}

Vector3X CameraMatrix::unprojectPixel(const Vector2X &p, Scalar depth) const {
	Vector3X pp;
	pp.segment<2>(0) = p;
	pp(2) = 1.0;

	Vector3X ray = m_invK * pp;
	ray(0) *= (depth / ray(2));
	ray(1) *= (depth / ray(2));
	ray(2) *= depth;
	ray = m_Rt * ray;

	return m_center + ray;
}

Vector3X CameraMatrix::intersectRayWithPlane(const Vector4X &plane, int x, int y) const {
	Vector3X ray = this->getRay(Vector2X(x, y));
	Scalar rho = (-(plane.segment<3>(0).dot(m_center)) - plane(3)) / plane.segment<3>(0).dot(ray);
	
	return m_center + rho * ray;
}

Vector3X CameraMatrix::transformPointIntoCameraSpace(const Vector3X &p) const {
	return m_R * p + m_T;
}

Vector3X CameraMatrix::transformPointFromCameraSpace(const Vector3X &p) const {
	return m_Rt * (p - m_T);
}

Vector3X CameraMatrix::transformDirectionFromCameraSpace(const Vector3X &dir) const {
	return m_Rt * dir;
}

Vector3X CameraMatrix::transformDirectionIntoCameraSpace(const Vector3X &dir) const {
	return m_R * dir;
}

Vector2X CameraMatrix::transformPointIntoNormalizedCoordinate(const Vector2X &p) const {
	return Vector2X(
		m_K(0, 0) * p(0) + m_K(0, 1) * p(1) + m_K(0, 2),
						   m_K(1, 1) * p(1) + m_K(1, 2)
	);
}

Vector2X CameraMatrix::transformPointFromNormalizedCoordinate(const Vector2X &p) const {
	return Vector2X(
		m_invK(0, 0) * p(0) + m_invK(0, 1) * p(1) + m_invK(0, 2),
						   m_invK(1, 1) * p(1) + m_invK(1, 2)
	);
}

void CameraMatrix::updateCachedValues(bool intrinsic, bool orientation) {
	if (intrinsic) {
		m_invK = m_K.inverse();
	}

	if (orientation) {
		m_Rt = m_R.transpose();
		m_center = m_Rt * (-1.0 * m_T);
	}
}