#include <iostream>
#include <numeric>
#include <math.h>
#include <Eigen/Dense>
#include <chrono>

#include "aabbtree.h"

using namespace Eigen;
using namespace std;
using namespace AABB;

/*
Example compile and run commands for my computer (put your own include directories)
g++ -o "aabbtree_example" "aabbtree_example.cpp" -std=c++17 -pthread -lpthread -O3 -I../include -I/home/nick/anaconda3/envs/fenics3/include/eigen3  -I/home/nick/repos/thread-pool
./aabbtree_example
*/

// Helper functions to check correctness of our AABB code
bool box_contains_point( const VectorXd & box_min,
                         const VectorXd & box_max,
                         const VectorXd & point )
{
    return ((box_min.array() <= point.array()) && (point.array() <= box_max.array())).all();
}

bool box_intersects_ball( const VectorXd & box_min,
                          const VectorXd & box_max,
                          const VectorXd & ball_center,
                          const double &   ball_radius )
{
    VectorXd closest_point_on_box = ball_center.cwiseMin(box_max).cwiseMax(box_min);
    return ((closest_point_on_box - ball_center).norm() <= ball_radius);
}

int main()
{
    int dim = 2; // spatial dimension
    int num_boxes = 20;

    MatrixXd random_points1 = 2.0*MatrixXd::Random(dim, num_boxes);
    MatrixXd random_points2 = 2.0*MatrixXd::Random(dim, num_boxes);

    MatrixXd box_mins = random_points1.cwiseMin(random_points2);
    MatrixXd box_maxes = random_points1.cwiseMax(random_points2);

    cout << "Boxes:" << endl;
    for ( int ii=0; ii<num_boxes; ++ii )
    {
        cout << ii << ": ";
        cout << "[" << box_mins(0,ii) << "," << box_maxes(0,ii) << "]";
        for ( int kk=1; kk<dim; ++kk )
        {
            cout << " x [" << box_mins(kk,ii) << "," << box_maxes(kk,ii) << "]";
        }
        cout << endl;
    }
    cout << endl;

    AABBTree aabbtree = AABBTree( box_mins, box_maxes ); // <--------------------------------------- Build the aabb-tree


    // POINT COLLISION
    VectorXd point = VectorXd::Random(dim);

    cout << "Point:" << endl;
    cout << point << endl << endl;

    VectorXi boxes_containing_point = aabbtree.point_collisions(point); // <---------- find boxes that contain the point

    cout << "Point is in the following boxes:" << endl;
    cout << boxes_containing_point << endl << endl;


    // BALL COLLISION
    VectorXd ball_center = VectorXd::Random(dim);
    double ball_radius = 0.25;

    cout << "Ball center:" << endl;
    cout << ball_center << endl << endl;
    cout << "Ball radius:" << endl;
    cout << ball_radius << endl << endl;

    VectorXi boxes_intersecting_ball = aabbtree.ball_collisions(ball_center, ball_radius); // <---- find boxes that intersect ball

    cout << "Ball intersects the following boxes:" << endl;
    cout << boxes_intersecting_ball << endl << endl;


    // VECTORIZED POINT COLLISION
    int num_points = 4;
    MatrixXd points = MatrixXd::Random(dim, num_points);

    cout << "Points array (each column is one point):" << endl;
    cout << points << endl << endl;

    vector<VectorXi> boxes_containing_points = aabbtree.point_collisions_vectorized(points); // <---- find boxes that contain the points

    for ( int ii=0; ii<num_points; ++ii )
    {
        cout << "Point " << ii << " is in the following boxes:" << endl;
        cout << boxes_containing_points[ii] << endl << endl;
    }


    // VECTORIZED BALL COLLISION
    int num_balls = 3;
    MatrixXd ball_centers = MatrixXd::Random(dim, num_balls);
    VectorXd ball_radii = VectorXd::Random(num_balls).cwiseAbs(); // abs to make radii all nonnegative

    cout << "Ball centers array (each column is the center of one ball):" << endl;
    cout << ball_centers << endl << endl;

    cout << "Ball radii array (each entry is the radius of one ball)" << endl;
    cout << ball_radii << endl << endl;

    vector<VectorXi> boxes_intersecting_balls = aabbtree.ball_collisions_vectorized(ball_centers, ball_radii); // <---- find boxes that intersect the balls

    for ( int ii=0; ii<num_balls; ++ii )
    {
        cout << "Ball " << ii << " intersects the following boxes:" << endl;
        cout << boxes_intersecting_balls[ii] << endl << endl;
    }


    // TIMING

    int dim2 = 3;
    int num_boxes2 = 100000; // 1e5 boxes
    int num_points2 = 100000; // 1e5 points
    int num_balls2 = 100000; // 1e5 balls

    double scaling_factor = pow((double) num_boxes2, 1.0/((double) dim2));

    MatrixXd box_centers = MatrixXd::Random(dim2, num_boxes2);
    MatrixXd box_widths =  5.0 * MatrixXd::Random(dim2, num_boxes2) / scaling_factor;

    MatrixXd box_mins2 =  box_centers - box_widths;
    MatrixXd box_maxes2 = box_centers + box_widths;

    MatrixXd points2 =       MatrixXd::Random(dim2, num_points2);
    MatrixXd ball_centers2 = MatrixXd::Random(dim2, num_balls2);
    VectorXd ball_radii2 =   VectorXd::Random(num_balls2).cwiseAbs() / scaling_factor;

    cout << "Timing:" << endl;
    cout << "space dimension=" << dim2 << ", num_boxes=" << num_boxes2 << ", num_points=" << num_points2 << ", num_balls=" << num_balls2 << endl;

    // build aabb tree
    auto build_t1 = std::chrono::high_resolution_clock::now();
    AABBTree aabbtree2 = AABBTree( box_mins2, box_maxes2 );
    auto build_t2 = std::chrono::high_resolution_clock::now();

    cout << "build time=" << std::chrono::duration_cast<std::chrono::milliseconds>(build_t2-build_t1).count() << "ms" << endl << endl;;

    // point collisions
    auto point_t1 = std::chrono::high_resolution_clock::now();
    vector<VectorXi> boxes_containing_points2 = aabbtree2.point_collisions_vectorized(points2);
    auto point_t2 = std::chrono::high_resolution_clock::now();

    cout << "point collision time=" << std::chrono::duration_cast<std::chrono::milliseconds>(point_t2-point_t1).count() << "ms" << endl;

    VectorXi point_collision_counts(num_points2);
    for ( int ii=0; ii<num_points2; ++ii )
    {
        point_collision_counts[ii] = boxes_containing_points2[ii].size();
    }

    double average_number_of_point_collisions = ((double)point_collision_counts.sum()) / ((double) num_points2);
    cout << "Average number of point collisions: " << average_number_of_point_collisions << endl << endl;

    // ball collisions
    auto ball_t1 = std::chrono::high_resolution_clock::now();
    vector<VectorXi> boxes_intersecting_balls2 = aabbtree2.ball_collisions_vectorized(ball_centers2, ball_radii2);
    auto ball_t2 = std::chrono::high_resolution_clock::now();

    cout << "ball collision time=" << std::chrono::duration_cast<std::chrono::milliseconds>(ball_t2-ball_t1).count() << "ms" << endl;

    VectorXi ball_collision_counts(num_balls2);
    for ( int ii=0; ii<num_balls2; ++ii )
    {
        ball_collision_counts[ii] = boxes_intersecting_balls2[ii].size();
    }

    double average_number_of_ball_collisions = ((double)ball_collision_counts.sum()) / ((double) num_balls2);
    cout << "Average number of ball collisions: " << average_number_of_ball_collisions << endl << endl;


    // CHECK CORRECTNESS BY BRUTE FORCE
    int dim3 = 3;
    int num_boxes3 = 1072;
    int num_points3 = 6335;
    int num_balls3 = 5916;

    cout << "Correctness checks:" << endl;
    cout << "space dimension=" << dim3 << ", num_boxes=" << num_boxes3 << ", num_points=" << num_points3 << ", num_balls=" << num_balls3 << endl << endl;

    double scaling_factor3 = pow((double) num_boxes3, 1.0/((double) dim3));

    MatrixXd box_centers3 = MatrixXd::Random(dim3, num_boxes3);
    MatrixXd box_widths3 =  5.0 * MatrixXd::Random(dim3, num_boxes3).cwiseAbs() / scaling_factor3;

    MatrixXd box_mins3 =  box_centers3 - box_widths3;
    MatrixXd box_maxes3 = box_centers3 + box_widths3;

    MatrixXd points3 =       MatrixXd::Random(dim3, num_points3);
    MatrixXd ball_centers3 = MatrixXd::Random(dim3, num_balls3);
    VectorXd ball_radii3 =   VectorXd::Random(num_balls3).cwiseAbs() / scaling_factor3;

    AABBTree aabbtree3 = AABBTree( box_mins3, box_maxes3 );
    vector<VectorXi> boxes_containing_points3 = aabbtree3.point_collisions_vectorized(points3);
    vector<VectorXi> boxes_intersecting_balls3 = aabbtree3.ball_collisions_vectorized(ball_centers3, ball_radii3);

    VectorXi point_collision_counts3(num_points3);
    for ( int ii=0; ii<num_points3; ++ii )
    {
        point_collision_counts3[ii] = boxes_containing_points3[ii].size();
    }

    double average_number_of_point_collisions3 = ((double)point_collision_counts3.sum()) / ((double) num_points3);
    cout << "Average number of point collisions: " << average_number_of_point_collisions3 << endl;

    vector<vector<int>> sorted_boxes_point(num_points3);
    for ( int ii=0; ii<num_points3; ++ii )
    {
        for ( int jj=0; jj<boxes_containing_points3[ii].size(); ++jj )
        {
            sorted_boxes_point[ii].push_back(boxes_containing_points3[ii](jj));
        }
        sort(sorted_boxes_point[ii].begin(), sorted_boxes_point[ii].end());
    }

    vector<vector<int>> sorted_boxes_point_brute(num_points3);
    for ( int ii=0; ii<num_points3; ++ii )
    {
        for ( int bb=0; bb<num_boxes3; ++bb )
        {
            if ( box_contains_point(box_mins3.col(bb), box_maxes3.col(bb), points3.col(ii)) )
            {
                sorted_boxes_point_brute[ii].push_back(bb);
            }
        }
        sort(sorted_boxes_point_brute[ii].begin(), sorted_boxes_point_brute[ii].end());
    }

    bool point_collision_is_correct = (sorted_boxes_point == sorted_boxes_point_brute);
    if (point_collision_is_correct)
    {
        cout << "Point collision check: PASS" << endl << endl;
    }
    else
    {
        cout << "Point collision check: FAIL" << endl << endl;
    }

    //

    VectorXi ball_collision_counts3(num_balls3);
    for ( int ii=0; ii<num_balls3; ++ii )
    {
        ball_collision_counts3[ii] = boxes_intersecting_balls3[ii].size();
    }

    double average_number_of_ball_collisions3 = ((double)ball_collision_counts3.sum()) / ((double) num_points3);
    cout << "Average number of ball collisions: " << average_number_of_ball_collisions3 << endl;

    vector<vector<int>> sorted_boxes_ball(num_balls3);
    for ( int ii=0; ii<num_balls3; ++ii )
    {
        for ( int jj=0; jj<boxes_intersecting_balls3[ii].size(); ++jj )
        {
            sorted_boxes_ball[ii].push_back(boxes_intersecting_balls3[ii](jj));
        }
        sort(sorted_boxes_ball[ii].begin(), sorted_boxes_ball[ii].end());
    }

    vector<vector<int>> sorted_boxes_ball_brute(num_balls3);
    for ( int ii=0; ii<num_balls3; ++ii )
    {
        for ( int bb=0; bb<num_boxes3; ++bb )
        {
            if ( box_intersects_ball(box_mins3.col(bb), box_maxes3.col(bb), ball_centers3.col(ii), ball_radii3(ii)) )
            {
                sorted_boxes_ball_brute[ii].push_back(bb);
            }
        }
        sort(sorted_boxes_ball_brute[ii].begin(), sorted_boxes_ball_brute[ii].end());
    }

    bool ball_collision_is_correct = (sorted_boxes_ball == sorted_boxes_ball_brute);
    if (ball_collision_is_correct)
    {
        cout << "Ball collision check: PASS" << endl << endl;
    }
    else
    {
        cout << "Ball collision check: FAIL" << endl << endl;
    }

}
