#include <iostream>
#include <numeric>
#include <math.h>
#include <Eigen/Dense>
#include <chrono>

#include "kdtree.h"

using namespace Eigen;
using namespace std;
using namespace KDT;

/*
Example compile and run commands for my computer (put your own include directories):
g++ -o "kdtree_example" "kdtree_example.cpp" -std=c++17 -pthread -lpthread -O3 -I../include -I/home/nick/anaconda3/envs/fenics3/include/eigen3  -I/home/nick/repos/thread-pool
./kdtree_example
*/

int main()
{
    int dim = 2; // spatial dimension
    int num_points = 10;
    int num_queries = 5;
    int num_neighbors = 3;

    MatrixXd points = MatrixXd::Random(dim, num_points);
    KDTree kdtree = KDTree(points); // <----------------------------------------------------- Build the kd-tree

    MatrixXd query_points = MatrixXd::Random(dim, num_queries);
    pair<MatrixXi, MatrixXd> result = kdtree.query(query_points, num_neighbors); // <---- find nearest neighbors

    MatrixXi inds              = result.first;  // shape=(num_neighbors, num_queries)
    MatrixXd squared_distances = result.second; // shape=(num_neighbors, num_queries)


    // PRINT RESULTS

    cout << "indices of nearest " << num_neighbors << " points to " << num_queries << " query points:" << endl;
    cout << inds << endl;
    cout << endl;

    cout << "squared distances to nearest points:" << endl;
    cout << squared_distances << endl;
    cout << endl;

    MatrixXd first_nearest_points(dim, num_queries);
    MatrixXd second_nearest_points(dim, num_queries);
    MatrixXd third_nearest_points(dim, num_queries);
    for ( int ii=0; ii<num_queries; ++ii )
    {
        first_nearest_points.col(ii) = points.col(inds(0,ii));
        second_nearest_points.col(ii) = points.col(inds(1,ii));
        third_nearest_points.col(ii) = points.col(inds(2,ii));
    }

    cout << "query points:" << endl;
    cout << query_points << endl << endl;

    cout << "nearest point to each query:" << endl;
    cout << first_nearest_points << endl << endl;

    cout << "second nearest point to each query:" << endl;
    cout << first_nearest_points << endl << endl;

    cout << "third nearest point to each query:" << endl;
    cout << first_nearest_points << endl << endl;


    // CHECK CORRECTNESS BY BRUTE FORCE

    vector<vector<double>> pairwise_distances(num_queries, vector<double>(num_points));
    for ( int qq=0; qq<num_queries; ++qq )
    {
        for (int pp=0; pp<num_points; ++pp)
        {
            pairwise_distances[qq][pp] = (points.col(pp) - query_points.col(qq)).norm();
        }
    }

    MatrixXi inds_true(num_neighbors, num_queries);
    MatrixXd squared_distances_true(num_neighbors, num_queries);
    for ( int qq=0; qq<num_queries; ++qq )
    {
        vector<int> sort_inds(num_points);
        std::iota(sort_inds.begin(), sort_inds.end(), 0);
        vector<double> distances = pairwise_distances[qq];

        sort(sort_inds.begin(),
             sort_inds.end(),
             [&distances](int ii, int jj){return distances[ii] < distances[jj];});

        for ( int pp=0; pp<num_neighbors; ++pp )
        {
            inds_true(pp, qq) = sort_inds[pp];
            double d = distances[sort_inds[pp]];
            squared_distances_true(pp, qq) = d*d;
        }
    }

    double err_ind = (inds - inds_true).norm();
    double err_dsq = (squared_distances - squared_distances_true).norm();

    cout << "error in indices:" << err_ind <<  endl;
    cout << "error in squared distances:" << err_dsq <<  endl;
    cout << endl;


    // TIMING

    constexpr int dim2 = 3; // 3D
    int num_points2 = 100000; // 1e5 points
    int num_queries2 = 100000; // 1e5 query points
    int num_neighbors2 = 20;

    cout << "Timing test:" << endl;
    cout << "dim=" << dim2 << endl;
    cout << "num_points=" << num_points2 << endl;
    cout << "num_queries=" << num_queries2 << endl;
    cout << "num_neighbors=" << num_neighbors2 << endl << endl;

    MatrixXd points2 = MatrixXd::Random(dim2, num_points2);

    auto build_t1 = std::chrono::high_resolution_clock::now();
    KDTree kdtree2 = KDTree(points2); // <----------------------------------------------------- Build the kd-tree
    auto build_t2 = std::chrono::high_resolution_clock::now();

    cout << "build time=" << std::chrono::duration_cast<std::chrono::milliseconds>(build_t2-build_t1).count() << "ms" << endl;

    MatrixXd query_points2 = MatrixXd::Random(dim2, num_queries2);

    // Single threaded version
    auto query_t1 = std::chrono::high_resolution_clock::now();
    pair<MatrixXi, MatrixXd> result2 = kdtree2.query(query_points2, num_neighbors2); // <---- find nearest neighbors
    auto query_t2 = std::chrono::high_resolution_clock::now();

    cout << "query time=" << std::chrono::duration_cast<std::chrono::milliseconds>(query_t2-query_t1).count() << "ms" << endl;

    // Multithreaded version
    query_t1 = std::chrono::high_resolution_clock::now();
    result2 = kdtree2.query_multithreaded(query_points2, num_neighbors2); // <---- multithreaded version
    query_t2 = std::chrono::high_resolution_clock::now();

    cout << "multithreaded query time=" << std::chrono::duration_cast<std::chrono::milliseconds>(query_t2-query_t1).count() << "ms" << endl;
}