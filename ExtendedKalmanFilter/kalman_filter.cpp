#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &Hj_in, 
						MatrixXd &R_laser_in, MatrixXd &R_radar_in, 
						MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  Hj = Hj_in;
  R_laser = R_laser_in;
  R_radar = R_radar_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */

	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_laser;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

	Tools tools;
  	MatrixXd Hj = tools.CalculateJacobian(x_);
	
	float px_p = x_(0);
	float py_p = x_(1);
	float vx_p = x_(2);
	float vy_p = x_(3);
	
	float rho = sqrt(px_p*px_p+py_p*py_p);
	
	float phi = 0;
	if (fabs(px_p) > 0.0001){
		phi = atan2(py_p,px_p);
	}
	
	float rho_dot = 0;
	if (fabs(rho) > 0.0001){
		rho_dot = (px_p*vx_p+py_p*vy_p)/rho;
	}
	
	MatrixXd h(3,1);
	h << rho, phi, rho_dot;
	
	VectorXd y = z - h;	
	
	while (y(1)> M_PI) {
	y(1) -= 2 * M_PI;
	}
	while (y(1)< -M_PI) {
	y(1) += 2 * M_PI;
	}

	MatrixXd Ht = Hj.transpose();
	MatrixXd S = Hj * P_ * Ht + R_radar;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;
	
	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * Hj) * P_;

}
