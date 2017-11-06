#include "CameraMatrix.h"

CameraMatrix::CameraMatrix() 
	: m_K(Eigen::Matrix3d::Identity()),
	m_R(Eigen::Matrix3d::Identity()),
	m_T(Eigen::Vector3d::Zero()) {

	this->updateCachedValues(true, true);
}

CameraMatrix::CameraMatrix(double f, double cx, double cy)
	: m_K(Eigen::Matrix3d::Identity()),
	m_R(Eigen::Matrix3d::Identity()),
	m_T(Eigen::Vector3d::Zero()) {

	m_K(0, 0) = f;
	m_K(1, 1) = f;
	m_K(0, 2) = cx;
	m_K(1, 2) = cy;

	this->updateCachedValues(true, true);
}

CameraMatrix::CameraMatrix(const Eigen::Matrix3d &K, const Matrix3x4d &RT) 
	: m_K(K) {

	m_R = RT.leftCols(3);
	m_T = RT.col(3);
	
	this->updateCachedValues(true, true);
}

CameraMatrix::CameraMatrix(const Matrix3x4d &P) {
	Eigen::Matrix3d K = P.leftCols(3);

	// Get affine matrix (rq-decomposition of M)
	// See Hartley & Zissermann, p552 (1st ed.)
	double h = sqrt(sqr(K(2, 1)) + sqr(K(2, 2)));
	double s = K(2, 1) / h;
	double c = -K(2, 2) / h;

	Eigen::Matrix3d Rx = Eigen::Matrix3d::Zero();
	Rx(0, 0) = 1.0;
	Rx(1, 1) = c; Rx(2, 2) = c;
	Rx(1, 2) = -s; Rx(2, 1) = s;

	K = K * Rx;

	h = sqrt(sqr(K(2, 0)) + sqr(K(2, 2)));
	s = K(2, 0) / h;
	c = -K(2, 2) / h;

	Eigen::Matrix3d Ry = Eigen::Matrix3d::Zero();
	Ry(1, 1) = 1.0;
	Ry(0, 0) = c; Ry(2, 2) = c;
	Ry(0, 2) = -s; Ry(2, 0) = s;

	K = K * Ry;

	h = sqrt(sqr(K(1, 0)) + sqr(K(1, 1)));
	s = K(1, 0) / h;
	c = -K(1, 1) / h;

	Eigen::Matrix3d Rz = Eigen::Matrix3d::Zero();
	Rz(2, 2) = 1.0;
	Rz(0, 0) = c; Rz(1, 1) = c;
	Rz(0, 1) = -s; Rz(1, 0) = s;

	K = K * Rz;

	Eigen::Matrix3d sign = Eigen::Matrix3d::Identity();
	sign(0, 0) = (K(0, 0) < 0) ? -1 : 1;
	sign(1, 1) = (K(1, 1) < 0) ? -1 : 1;
	sign(2, 2) = (K(2, 2) < 0) ? -1 : 1;

	K = K * sign;

	Eigen::Matrix3d R = Rx * Ry * Rz * sign;
	R.transposeInPlace();

	Eigen::Vector3d P4 = P.col(3);
	Eigen::Vector3d T = K.inverse() * P4;

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


void CameraMatrix::setIntrinsic(const Eigen::Matrix3d &K) {
	this->m_K = K;
	this->updateCachedValues(true, false);
}

void CameraMatrix::setRotation(const Eigen::Matrix3d &R) {
	this->m_R = R;
	this->updateCachedValues(false, true);
}

void CameraMatrix::setTranslation(const Eigen::Vector3d &T) {
	this->m_T = T;
	this->updateCachedValues(false, true);
}

void CameraMatrix::setCameraCenter(const Eigen::Vector3d &c) {
	this->setTranslation(-1.0 * (m_R * c));
}

void CameraMatrix::setOrientation(const Matrix3x4d &RT) {
	this->m_R = RT.leftCols(3);
	this->m_T = RT.col(3);
	this->updateCachedValues(false, true);
}

void CameraMatrix::setSize(double w, double h) {
	this->m_size = Eigen::Vector2d(w, h);
}

const Eigen::Matrix3d& CameraMatrix::getIntrinsic() const {
	return this->m_K;
}

const Eigen::Matrix3d& CameraMatrix::getRotation() const {
	return this->m_R;
}

const Eigen::Vector3d& CameraMatrix::getTranslation() const {
	return this->m_T;
}

const Eigen::Vector2d& CameraMatrix::getSize() const {
	return this->m_size;
}

double CameraMatrix::getWidth() const {
	return this->m_size(0);
}

double CameraMatrix::getHeight() const {
	return this->m_size(1);
}

const Eigen::Vector3d& CameraMatrix::getCameraCenter() const {
	return this->m_center;
}

const Eigen::Vector3d CameraMatrix::opticalAxis() const {
	return this->transformDirectionFromCameraSpace(Eigen::Vector3d(0.0, 0.0, 1.0));
}

const Eigen::Vector3d CameraMatrix::upVector() const {
	return this->transformDirectionFromCameraSpace(Eigen::Vector3d(0.0, 1.0, 0.0));
}

const Eigen::Vector3d CameraMatrix::rightVector() const {
	return this->transformDirectionFromCameraSpace(Eigen::Vector3d(1.0, 0.0, 0.0));
}

Eigen::Vector3d CameraMatrix::getRay(const Eigen::Vector2d& p) const {
	return m_invK * Eigen::Vector3d(p(0), p(1), 1.0);
}

bool CameraMatrix::isOnGoodSide(const Eigen::Vector3d& p) const {
	return this->transformPointIntoCameraSpace(p)(2) > 0;
}

const Matrix3x4d CameraMatrix::getExtrinsic() const {
	Matrix3x4d RT;
	RT.leftCols(3) = this->m_R;
	RT.col(3) = this->m_T;

	return RT;
}

const Matrix3x4d CameraMatrix::getOrientation() const {
	Matrix3x4d RT;
	RT.leftCols(3) = this->m_R;
	RT.col(3) = this->m_T;

	return RT;
}

const Matrix3x4d CameraMatrix::getProjection() const {
	Matrix3x4d RT = this->getOrientation();

	return this->m_K * RT;
}

double CameraMatrix::getFocalLength() const {
	return this->m_K(0, 0);
}

double CameraMatrix::getAspectRatio() const {
	return this->m_K(1, 1) / this->m_K(0, 0);
}

Eigen::Vector2d CameraMatrix::getPrincipalPoint() const {
	return Eigen::Vector2d(this->m_K(0, 2), this->m_K(1, 2));
}

Eigen::Vector2d CameraMatrix::projectPoint(const Eigen::Vector3d& X) const {
	Eigen::Vector3d q = m_K * (m_R * X + m_T);
	
	return Eigen::Vector2d(q(0) / q(2), q(1) / q(2));
}

Eigen::Vector2d CameraMatrix::projectPoint(const DistortionFunction &distortion, const Eigen::Vector3d &X) const {
	Eigen::Vector3d XX = m_R * X + m_T;
	Eigen::Vector2d p;
	p(0) = XX(0) / XX(2);
	p(1) = XX(1) / XX(2);
	p = distortion(p);

	return Eigen::Vector2d(
		m_K(0, 0) * p(0) + m_K(0, 1) * p(1) + m_K(0, 2),
						   m_K(1, 1) * p(1) + m_K(1, 2)
	);	
}

Eigen::Vector3d CameraMatrix::unprojectPixel(const Eigen::Vector2d &p, double depth) const {
	Eigen::Vector3d pp;
	pp.segment<2>(0) = p;
	pp(2) = 1.0;

	Eigen::Vector3d ray = m_invK * pp;
	ray(0) *= (depth / ray(2));
	ray(1) *= (depth / ray(2));
	ray(2) *= depth;
	ray = m_Rt * ray;

	return m_center + ray;
}

Eigen::Vector3d CameraMatrix::intersectRayWithPlane(const Eigen::Vector4d &plane, int x, int y) const {
	Eigen::Vector3d ray = this->getRay(Eigen::Vector2d(x, y));
	double rho = (-(plane.segment<3>(0).dot(m_center)) - plane(3)) / plane.segment<3>(0).dot(ray);
	
	return m_center + rho * ray;
}

Eigen::Vector3d CameraMatrix::transformPointIntoCameraSpace(const Eigen::Vector3d &p) const {
	return m_R * p + m_T;
}

Eigen::Vector3d CameraMatrix::transformPointFromCameraSpace(const Eigen::Vector3d &p) const {
	return m_Rt * (p - m_T);
}

Eigen::Vector3d CameraMatrix::transformDirectionFromCameraSpace(const Eigen::Vector3d &dir) const {
	return m_Rt * dir;
}

Eigen::Vector3d CameraMatrix::transformDirectionIntoCameraSpace(const Eigen::Vector3d &dir) const {
	return m_R * dir;
}

Eigen::Vector2d CameraMatrix::transformPointIntoNormalizedCoordinate(const Eigen::Vector2d &p) const {
	return Eigen::Vector2d(
		m_K(0, 0) * p(0) + m_K(0, 1) * p(1) + m_K(0, 2),
						   m_K(1, 1) * p(1) + m_K(1, 2)
	);
}

Eigen::Vector2d CameraMatrix::transformPointFromNormalizedCoordinate(const Eigen::Vector2d &p) const {
	return Eigen::Vector2d(
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