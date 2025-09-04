#pragma once
#include <ceres/ceres.h>
#include <Eigen/Core>

// Residual for point-to-camera constraint with scaled camera positions.
// Observation ray v_obs_world (in world coordinates) should match the predicted
// ray d_ik * (X - (C0 + s * dir)).
// - C0: base (fixed) camera center (3D)   ← marked constant in the problem
// - X : 3D point (3D)
// - s : scalar scale parameter (1D)       ← optimization variable
// - dir: world direction vector (3D)      ← marked constant in the problem
// - d_ik: per-observation scale (1D)      ← same role as in original glomap

namespace glomap {

// v_obs_world ≈ d_ik * ( X - ( C_root + s * dir ) )
struct PtCamDirErrorScaledRoot {
  PtCamDirErrorScaledRoot(const Eigen::Vector3d& v_obs_world,
                          const Eigen::Vector3d& C_root_fixed,
                          const Eigen::Vector3d& dir_fixed)
      : v_obs_world_(v_obs_world),
        C_root_fixed_(C_root_fixed),
        dir_fixed_(dir_fixed) {}

  template <typename T>
  bool operator()(const T* const X,     // 3D point
                  const T* const s,     // 1D scalar along dir
                  const T* const d_ik,  // 1D observation scale
                  T* residuals) const {
    const Eigen::Matrix<T,3,1> Xv   (X[0], X[1], X[2]);
    const Eigen::Matrix<T,3,1> Croot = C_root_fixed_.cast<T>();
    const Eigen::Matrix<T,3,1> dir   = dir_fixed_.cast<T>();
    const Eigen::Matrix<T,3,1> vobs  = v_obs_world_.cast<T>();

    const Eigen::Matrix<T,3,1> C = Croot + (*s) * dir;
    const Eigen::Matrix<T,3,1> vpred = (*d_ik) * (Xv - C);

    const Eigen::Matrix<T,3,1> r = vobs - vpred;
    residuals[0] = r[0];
    residuals[1] = r[1];
    residuals[2] = r[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& v_obs_world,
                                     const Eigen::Vector3d& C_root_fixed,
                                     const Eigen::Vector3d& dir_fixed) {
    // residual 3; parameter blocks: X(3), s(1), d_ik(1)
    return new ceres::AutoDiffCostFunction<PtCamDirErrorScaledRoot, 3, 3, 1, 1>(
        new PtCamDirErrorScaledRoot(v_obs_world, C_root_fixed, dir_fixed));
  }

  Eigen::Vector3d v_obs_world_;
  Eigen::Vector3d C_root_fixed_;
  Eigen::Vector3d dir_fixed_;
};

} // namespace glomap

