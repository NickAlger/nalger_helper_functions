#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <numeric>
#include <algorithm>

#include <math.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "brent_minimize.h"
#include "aabbtree.h"

namespace ELLIPSOID {

inline bool point_is_in_ellipsoid( const Eigen::VectorXd mu,
                                   const Eigen::MatrixXd Sigma,
                                   const Eigen::VectorXd p,
                                   const double tau )
{
    Eigen::VectorXd z = p - mu;
    return Sigma.ldlt().solve(z).dot(z) <= (tau * tau);
}

inline std::tuple<Eigen::VectorXd, Eigen::VectorXd> ellipsoid_bounding_box( const Eigen::VectorXd mu,
                                                                            const Eigen::MatrixXd Sigma,
                                                                            const double tau )
{
    Eigen::VectorXd diag_Sigma = Sigma.diagonal();
    Eigen::VectorXd widths = (diag_Sigma.array().sqrt() * tau).matrix();
    return std::make_tuple(mu - widths, mu + widths);
}

inline bool boxes_intersect( const Eigen::VectorXd A_min,
                             const Eigen::VectorXd A_max,
                             const Eigen::VectorXd B_min,
                             const Eigen::VectorXd B_max )
{
    return (( A_min.array() <= B_max.array() ).all() &&
            ( B_min.array() <= A_max.array() ).all());
}

double K_fct(const double s,
             const Eigen::VectorXd lambdas,
             const Eigen::VectorXd v,
             const double tau)
{
    double K = 0.0;
    for ( int ii=0; ii<lambdas.size(); ++ii )
    {
        K += (v(ii) * v(ii)) / ( 1.0 + s * (lambdas(ii) - 1.0) );
    }
    K = 1.0 - ( s * (1.0 - s) / (tau * tau) ) * K;
    return K;
}

bool ellipsoids_intersect( const Eigen::VectorXd mu_A,
                           const Eigen::MatrixXd Sigma_A,
                           const Eigen::VectorXd mu_B,
                           const Eigen::MatrixXd Sigma_B,
                           double tau )
{
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> A_box = ellipsoid_bounding_box(mu_A, Sigma_A, tau);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> B_box = ellipsoid_bounding_box(mu_B, Sigma_B, tau);
//    if (true)
    if (boxes_intersect(std::get<0>(A_box), std::get<1>(A_box),
                        std::get<0>(B_box), std::get<1>(B_box)))
    {
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma_A, Sigma_B);
        Eigen::VectorXd lambdas = es.eigenvalues();
        Eigen::VectorXd v = es.eigenvectors().transpose() * (mu_A - mu_B);

        std::function<double(double)> K = [lambdas, v, tau](const double s) {return K_fct(s, lambdas, v, tau);};

        std::tuple<double, double, int, int> sol = BRENT::brent_minimize( K, 0.0, 1.0, 1.0e-8, 200 );
        return (std::get<1>(sol) >= 0);
    }
    else
    {
        return false;
    }
}


class EllipsoidBatchPicker
{
private:
    AABB::AABBTree               aabb;
    std::vector<Eigen::VectorXd> all_points;
    std::vector<Eigen::VectorXd> all_mu;
    std::vector<Eigen::MatrixXd> all_Sigma;
    int                          num_pts;
    int                          spatial_dim;
    double                       tau;
    std::vector<bool>            is_in_batch;
    std::vector<bool>            is_pickable;

public:
    std::vector<std::vector<int>> batches;
    std::vector<double>           squared_distances;

    EllipsoidBatchPicker( const std::vector<Eigen::VectorXd> all_points_input,
                          const std::vector<Eigen::VectorXd> all_mu_input,
                          const std::vector<Eigen::MatrixXd> all_Sigma_input,
                          const double                       tau_input)
    {
        tau = tau_input;
        num_pts = all_points_input.size();
        if (all_mu_input.size() != num_pts)
        {
            throw std::invalid_argument( "Different number of points and mu" );
        }
        if (all_Sigma_input.size() != num_pts)
        {
            throw std::invalid_argument( "Different number of points and Sigma" );
        }
        if ( num_pts == 0 )
        {
            throw std::invalid_argument( "No ellipsoids provided" );
        }
        spatial_dim = all_points_input[0].size();

        squared_distances.reserve(num_pts);
        is_pickable.reserve(num_pts);
        is_in_batch.reserve(num_pts);
        all_points.reserve(num_pts);
        all_mu.reserve(num_pts);
        all_Sigma.reserve(num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            all_points.push_back(all_points_input[ii]);
            all_mu.push_back(all_mu_input[ii]);
            all_Sigma.push_back(all_Sigma_input[ii]);
            is_pickable.push_back(true);
            is_in_batch.push_back(false);
            squared_distances.push_back(-1.0);
        }

        Eigen::MatrixXd box_mins(spatial_dim, num_pts);
        Eigen::MatrixXd box_maxes(spatial_dim, num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(all_mu[ii],
                                                                                    all_Sigma[ii],
                                                                                    tau);
            box_mins.col(ii) = std::get<0>(B);
            box_maxes.col(ii) = std::get<1>(B);
        }
        aabb.build_tree(box_mins, box_maxes);
    }

    std::vector<int> pick_batch()
    {
        for ( int ii=0; ii<num_pts; ++ii )
        {
            is_pickable[ii] = (!is_in_batch[ii]);
        }

        std::vector<int> candidate_inds(num_pts);
        std::iota(candidate_inds.begin(), candidate_inds.end(), 0);
        stable_sort(candidate_inds.begin(), candidate_inds.end(),
            [this](int i1, int i2) {return squared_distances[i1] > squared_distances[i2];});

        std::vector<int> next_batch;
        for ( int idx : candidate_inds )
        {
            if ( is_pickable[idx] )
            {
                next_batch.push_back(idx);
                is_pickable[idx] = false;
                is_in_batch[idx] = true;
                std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(all_mu[idx], all_Sigma[idx], tau);
                Eigen::VectorXi possible_collisions = aabb.box_collisions(std::get<0>(B), std::get<1>(B));
                for ( int jj=0; jj<possible_collisions.size(); ++jj )
                {
                    int idx2 = possible_collisions[jj];
                    if ( is_pickable[idx2] )
                    {
                        if ( ellipsoids_intersect(all_mu[idx2], all_Sigma[idx2],
                                                  all_mu[idx],  all_Sigma[idx],
                                                  tau) )
                        {
                            is_pickable[idx2] = false;
                        }
                    }

                }
            }
        }
        batches.push_back(next_batch);

        for ( int ind : next_batch )
        {
            for ( int ii=0; ii<num_pts; ++ii )
            {
                double old_dsq = squared_distances[ii];
                double new_dsq = (all_points[ind] - all_points[ii]).squaredNorm();
                if ( new_dsq < old_dsq || old_dsq < 0.0 )
                {
                    squared_distances[ii] = new_dsq;
                }
            }
        }
        return next_batch;
    }
};

} // end namespace ELLIPSOID