#pragma once

#include <iostream>
#include <list>
#include <queue>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


// Nearest neighbor in subtree to query point
struct SubtreeResult
{
    int index; // index of nearest neighbor
    double distance_squared; // distance squared to query point

    const bool operator < ( const SubtreeResult& other ) const
    {
        return ( distance_squared < other.distance_squared );
    }
};


// Node in KD tree
template <int K>
struct KDNode { Matrix<double,K,1> point;
                int                left;       // index of left child
                int                right; };   // index of right child


template <int K>
struct PointWithIndex { Matrix<double,K,1> point;
                        int                index; };


template <int K>
class KDTree
{
private:
    vector<KDNode<K>> nodes; // All nodes in the tree
    vector<int>       perm_i2e; // permutation from internal ordering to external ordering

    // creates subtree and returns the index for root of subtree
    int make_subtree( int                         start,
                      int                         stop,
                      int                         depth,
                      vector<PointWithIndex<K>> & points,
                      int &                       counter )
    {
        int num_pts_local = stop - start;
        int current_node_ind = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1)
        {
            current_node_ind = counter;
            counter = counter + 1;

            int axis = depth % K;
            sort( points.begin() + start, points.begin() + stop,
                  [axis](PointWithIndex<K> u, PointWithIndex<K> v) {return u.point(axis) > v.point(axis);} );

            int mid = start + (num_pts_local / 2);

            int left_start = start;
            int left_stop = mid;

            int right_start = mid + 1;
            int right_stop = stop;

            int left = make_subtree(left_start, left_stop, depth + 1, points, counter);
            int right = make_subtree(right_start, right_stop, depth + 1, points, counter);

            nodes[current_node_ind] = KDNode<K> { points[mid].point, left, right };
            perm_i2e[current_node_ind] = points[mid].index;
        }
        return current_node_ind;
    }

    // finds num_neighbors nearest neighbors of query in subtree
    void query_subtree( const Matrix<double,K,1> &                             query_point,
                        priority_queue<SubtreeResult, vector<SubtreeResult>> & nn,
                        int                                                    cur_index,
                        int                                                    depth,
                        int                                                    num_neighbors ) const
    {
        KDNode<K> cur = nodes[cur_index];

        const Matrix<double,K,1> delta = query_point - cur.point;
        double dsq_cur = delta.squaredNorm();
        SubtreeResult cur_result = {cur_index, dsq_cur};

        if ( nn.size() < num_neighbors )
        {
            nn.push( cur_result );
        }
        else if ( dsq_cur < nn.top().distance_squared )
        {
            nn.pop();
            nn.push( cur_result );
        }

        int axis = depth % K;
        double displacement_to_splitting_plane = delta(axis);

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0)
        {
            A = cur.left;
            B = cur.right;
        }
        else
        {
            A = cur.right;
            B = cur.left;
        }

        if (A >= 0)
        {
            query_subtree( query_point, nn, A, depth+1, num_neighbors );
        }

        if (B >= 0)
        {
            if ( displacement_to_splitting_plane*displacement_to_splitting_plane
                 < nn.top().distance_squared )
            {
                query_subtree( query_point, nn, B, depth+1, num_neighbors );
            }
        }
    }

public:
    KDTree( ) {}

    KDTree( const Ref<const Matrix<double,K,Dynamic>> input_points )
    {
        int num_pts = input_points.cols();

        // Copy points into std::vector of tuples which will be re-ordered
        vector< PointWithIndex<K> > points(num_pts); // (coords, original_index)
        for ( int ii=0; ii<num_pts; ++ii)
        {
            points[ii].point = input_points.col(ii);
            points[ii].index = ii;
        }

        nodes.reserve(num_pts);
        perm_i2e.resize(num_pts, 1);
        int counter = 0;
        int zero = make_subtree(0, num_pts, 0, points, counter);
    }

    // Many queries, many neighbors each
    pair<MatrixXi, MatrixXd> query( const Ref<const Matrix<double,K,Dynamic>> query_points, int num_neighbors ) const
    {
        int num_queries = query_points.cols();

        MatrixXi closest_point_inds(num_neighbors, num_queries);
        MatrixXd squared_distances(num_neighbors, num_queries);

        for ( int ii=0; ii<num_queries; ++ii )
        {
            vector<SubtreeResult> nn_container;
            nn_container.reserve(2*num_neighbors);
            priority_queue<SubtreeResult, vector<SubtreeResult>> nn(less<SubtreeResult>(), move(nn_container));

            query_subtree( query_points.col(ii), nn, 0, 0, num_neighbors );

            for ( int kk=0; kk<num_neighbors; ++kk )
            {
                int jj = num_neighbors - kk - 1;
                const SubtreeResult n_kk = nn.top();
                nn.pop();
                closest_point_inds(jj,ii) = perm_i2e[n_kk.index];
                squared_distances(jj,ii) = n_kk.distance_squared;
            }
        }
        return make_pair(closest_point_inds, squared_distances);
    }

    // one query, one neighbor
    pair<int, double> query( const Matrix<double,K,1> & query_point ) const
    {
        vector<SubtreeResult> nn_container;
        nn_container.reserve(2);
        priority_queue<SubtreeResult, vector<SubtreeResult>> nn(less<SubtreeResult>(), move(nn_container));

        query_subtree( query_point, nn, 0, 0, 1 );

        SubtreeResult n_ii = nn.top();
        return make_pair(perm_i2e[n_ii.index], n_ii.distance_squared);
    }

};

