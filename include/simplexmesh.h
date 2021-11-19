#pragma once

#include <iostream>
#include <list>
#include <stdexcept>

#include "thread_pool.hpp"

#include <math.h>
#include <Eigen/Dense>

#include "kdtree.h"
#include "aabbtree.h"
#include "geometric_sort.h"


namespace SMESH {

std::vector<std::vector<int>> powerset(int N)
{
    std::vector<std::vector<int>> pset;
    if (N < 1)
    {
        std::vector<int> empty_set;
        pset.push_back( empty_set );
    }
    else
    {
        pset = powerset(N-1);
        int sz0 = pset.size();
        for ( int ii=0; ii<sz0; ++ii )
        {
            std::vector<int> x = pset[ii];
            x.push_back(N-1);
            pset.push_back(x);
        }
    }
    return pset;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> make_simplex_transform_operator( const Eigen::MatrixXd & simplex_vertices )
{
    int K = simplex_vertices.rows();
    int M = simplex_vertices.cols();

    Eigen::MatrixXd S(M, K); // first return
    Eigen::VectorXd b(M);    // second return

    if ( M == 1 )
    {
        S.setZero();
        b.setOnes();
    }
    else
    {
        Eigen::VectorXd v0(K);
        v0 = simplex_vertices.col(0);
        Eigen::MatrixXd dV(K,M-1);
        for ( int jj=1; jj<M; ++jj )
        {
            dV.col(jj-1) = simplex_vertices.col(jj) - v0;
        }
        Eigen::MatrixXd S0(M-1, K);
        S0 = dV.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(K,K));
        Eigen::Matrix<double, 1, Eigen::Dynamic> ones_rowvec(M-1);
        ones_rowvec.setOnes();
        S.bottomRightCorner(M-1, K) = S0;
        S.row(0) = -ones_rowvec * S0;

        Eigen::VectorXd e0(M);
        e0.setZero();
        e0(0) = 1.0;
        b = e0 - S * v0;
    }
    return std::make_pair(S, b);
}

struct Simplex { Eigen::MatrixXd V;   // simplex vertices
                 Eigen::MatrixXd A;   // coordinate transform matrix
                 Eigen::VectorXd b;}; // coordinate transform vector

std::pair<Eigen::VectorXd, Eigen::VectorXd> compute_pointcloud_bounding_box( const Eigen::MatrixXd & points )
{
    int dim = points.rows();
    int num_points = points.cols();

    Eigen::VectorXd box_min(dim);
    Eigen::VectorXd box_max(dim);

    for ( int kk=0; kk<dim; ++kk )
    {
        double min_k = points(kk, 0);
        double max_k = points(kk, 0);
        for ( int vv=1; vv<num_points; ++vv)
        {
            double candidate_k = points(kk, vv);
            if (candidate_k < min_k)
            {
                min_k = candidate_k;
            }
            if (candidate_k > max_k)
            {
                max_k = candidate_k;
            }
        }
        box_min(kk) = min_k;
        box_max(kk) = max_k;
    }
    return std::make_pair(box_min, box_max);
}


class SimplexMesh
{
private:
    Eigen::MatrixXi              faces;    // boundary face simplices of dimension dim-1. shape=(dim,num_faces)
    std::vector<Eigen::VectorXi> subfaces; // sub-simplices of boundary faces of dimension 0 through dim-1 (includes boundary faces)

    std::vector<Eigen::VectorXi> face2subface;

    AABB::AABBTree cell_aabbtree;
    AABB::AABBTree face_aabbtree;
    KDT::KDTree   face_kdtree;

    std::vector< Simplex > cell_simplices;
    std::vector< Simplex > subface_simplices;

    int dim;
    int num_vertices;
    int num_cells;
    int num_faces;
    int num_subfaces;

    int default_sleep_duration;
    int default_number_of_threads;

    void eval_CG1_helper( Eigen::MatrixXd &                         functions_at_points,
                          const std::vector<int> &                  function_inds,
                          const std::vector<int> &                  point_inds,
                          const Eigen::VectorXi &                   all_simplex_inds,
                          const Eigen::MatrixXd &                   all_affine_coords,
                          const Eigen::Ref<const Eigen::MatrixXd> & functions_at_vertices ) const
    {
        for ( int point_ind : point_inds )
        {
            int      simplex_ind   = all_simplex_inds(point_ind);
            Eigen::VectorXd affine_coords = all_affine_coords.col(point_ind);

            for ( int kk=0; kk<dim+1; ++kk ) // for simplex vertex
            {
                int vv = cells(kk, simplex_ind);
                for ( int ll : function_inds ) // for each function
                {
                    functions_at_points(ll, point_ind) += affine_coords(kk) * functions_at_vertices(ll, vv);
                }
            }
        }
    }

    Eigen::VectorXd closest_point_one_query( const Eigen::VectorXd & query ) const
    {
        Eigen::VectorXd point = vertices.col(0);
        if ( point_is_in_mesh( query )(0) )
        {
            point = query;
        }
        else
        {
            // 1. Find a set of candidate boundary faces, one of which contains the closest point
            std::pair<Eigen::VectorXi, Eigen::VectorXd> kd_result = face_kdtree.query( query, 1 );
            double dist_estimate = (1.0 + 1e-14) * sqrt(kd_result.second(0));
            Eigen::VectorXi face_inds = face_aabbtree.ball_collisions( query, dist_estimate );

            // 2. Determine unique set of boundary entities to visit
            std::vector<int> entities;
            entities.reserve(AABB::power_of_two(dim));
            for ( int ii=0; ii<face_inds.size(); ++ii )
            {
                const Eigen::VectorXi & subface_inds = face2subface[face_inds(ii)];
                for ( int jj=0; jj<subface_inds.size(); ++jj )
                {
                    entities.push_back(subface_inds(jj));
                }
            }
            std::sort( entities.begin(), entities.end() );
            entities.erase( unique( entities.begin(), entities.end() ), entities.end() );

            // 3. Project query onto the affine subspaces associated with each subface.
            // 4. Discard "bad" projections that to not land in their subface.
            // 5. Return closest "good" projection.
            double dsq_best = (point - query).squaredNorm();
            for ( int ee=0; ee<entities.size(); ++ee )
            {
                const Simplex & E = subface_simplices[entities[ee]];
                Eigen::VectorXd projected_affine_coords = E.A * query + E.b;
                if ( (projected_affine_coords.array() >= 0.0).all() ) // projection is in subface simplex
                {
                    Eigen::VectorXd projected_query = E.V * projected_affine_coords;
                    double dsq = (projected_query - query).squaredNorm();
                    if ( dsq < dsq_best )
                    {
                        point = projected_query;
                        dsq_best = dsq;
                    }
                }
            }
        }
        return point;
    }

    std::pair<int,Eigen::VectorXd> first_point_collision_one_query( const Eigen::VectorXd & point ) const
    {
        Eigen::VectorXi candidate_inds =  cell_aabbtree.point_collisions( point );
        int num_candidates = candidate_inds.size();

        int simplex_ind = -1;
        Eigen::VectorXd affine_coords(dim+1);
        for ( int jj=0; jj<num_candidates; ++jj ) // for each candidate simplex that the point might be in
        {
            int candidate_simplex_ind = candidate_inds(jj);
            const Simplex & S = cell_simplices[candidate_simplex_ind];
            affine_coords = S.A * point + S.b;
            if ( (affine_coords.array() >= 0.0).all() ) // point is in simplex
            {
                simplex_ind = candidate_simplex_ind;
                break;
            }
        }
        return std::make_pair(simplex_ind, affine_coords);
//        return first_point_collision_from_candidates(candidate_inds, point);
    }

public:
    Eigen::MatrixXd    vertices; // shape=(dim,num_vertices)
    Eigen::MatrixXi    cells;    // interior simplices of volumetric dimension. shape=(dim+1,num_cells)
    thread_pool pool;

    SimplexMesh( const Eigen::Ref<const Eigen::MatrixXd> input_vertices, // shape=(dim,num_vertices)
                 const Eigen::Ref<const Eigen::MatrixXi> input_cells )   // shape=(dim+1,num_cells)
    {
        // ------------------------    Input checking and copying    ------------------------
        dim = input_vertices.rows();
        num_vertices = input_vertices.cols();
        num_cells = input_cells.cols();

        if ( input_cells.rows() != dim+1 )
        {
            throw std::invalid_argument( "simplices have wrong dimension." );
        }

        if ( num_vertices < 1 )
        {
            throw std::invalid_argument( "no vertices provided" );
        }

        if ( num_cells < 1 )
        {
            throw std::invalid_argument( "no cells provided" );
        }

        if ( input_cells.minCoeff() < 0 )
        {
            throw std::invalid_argument( "at least one vertex index in input_cells is negative" );
        }

        if ( input_cells.maxCoeff() >= num_vertices )
        {
            throw std::invalid_argument( "at least one vertex index in input_cells >= num_vertices" );
        }

        vertices = input_vertices; // copy
        cells    = input_cells;    // copy


        // ------------------------    Multithreading stuff    ------------------------
        default_sleep_duration = pool.sleep_duration;
        default_number_of_threads = pool.get_thread_count();


        // ------------------------    CELLS    ------------------------
        // Generate cell simplices and transform operators
        cell_simplices.resize(num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            Eigen::MatrixXd simplex_vertices(dim, dim+1);
            for (int jj=0; jj<dim+1; ++jj )
            {
                simplex_vertices.col(jj) = vertices.col(cells(jj, ii));
            }
            std::pair<Eigen::MatrixXd, Eigen::VectorXd> STS = make_simplex_transform_operator( simplex_vertices );
            cell_simplices[ii] = Simplex { simplex_vertices, // V
                                           STS.first,        // A
                                           STS.second };     // b
        }

        // Generate cell AABB tree
        Eigen::MatrixXd cell_box_mins (dim, num_cells);
        Eigen::MatrixXd cell_box_maxes(dim, num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            std::pair<Eigen::VectorXd, Eigen::VectorXd> BB = compute_pointcloud_bounding_box( cell_simplices[ii].V );
            cell_box_mins.col(ii) = BB.first;
            cell_box_maxes.col(ii) = BB.second;
        }
        cell_aabbtree.build_tree( cell_box_mins, cell_box_maxes );


        // ------------------------    FACES    ------------------------
        // For all faces (dim-1 dimensional simplex which has dim vertices), compute how many cells they are part of.
        std::map<std::vector<int>, int> face_counts; // face -> cell count
        for ( int cc=0; cc<num_cells; ++cc )
        {
            for ( int opposite_vertex_ind=0; opposite_vertex_ind<dim+1; ++opposite_vertex_ind )
            {
                std::vector<int> face;
                for ( int kk=0; kk<dim+1; ++kk )
                {
                    if ( kk != opposite_vertex_ind )
                    {
                        face.push_back(cells(kk, cc));
                    }
                }
                std::sort( face.begin(), face.end() ); // sort for comparison purposes

                if ( face_counts.find(face) == face_counts.end() ) // if this face isnt in the map yet
                {
                    face_counts[face] = 1;
                }
                else
                {
                    face_counts[face] += 1;
                }
            }
        }

        // Faces (faces on the boundary) are the faces that are part of only one cell
        std::vector<Eigen::VectorXi> faces_vector;
        for ( auto it = face_counts.begin(); it != face_counts.end(); ++it )
        {
            std::vector<int> face = it->first;
            Eigen::VectorXi F(dim);
            for ( int kk=0; kk<dim; ++kk)
            {
                F(kk) = face[kk];
            }
            int count = it->second;
            if ( count == 1 )
            {
                faces_vector.push_back(F);
            }
        }

        num_faces = faces_vector.size();
        faces.resize(dim, num_faces);
        for ( int ii=0; ii<num_faces; ++ii )
        {
            faces.col(ii) = faces_vector[ii];
        }


        // Create kdtree of vertices that are on faces (i.e., on the boundary)
        std::set<int> face_vertex_inds;
        for ( int bb=0; bb<num_faces; ++bb )
        {
            for ( int kk=0; kk<dim; ++kk )
            {
                face_vertex_inds.insert(faces(kk, bb));
            }
        }

        int num_face_vertices = face_vertex_inds.size();
        Eigen::MatrixXd face_vertices(dim,num_face_vertices);
        int vv=0;
        for ( auto it  = face_vertex_inds.begin();
                   it != face_vertex_inds.end();
                 ++it )
        {
            face_vertices.col(vv) = vertices.col( *it );
            vv += 1;
        }
        face_kdtree.build_tree( face_vertices );


        // Create face AABB tree
        Eigen::MatrixXd face_box_mins(dim, num_faces);
        Eigen::MatrixXd face_box_maxes(dim, num_faces);
        for ( int bb=0; bb<num_faces; ++bb )
        {
            Eigen::MatrixXd face_vertices(dim,dim);
            for (int jj=0; jj<dim; ++jj )
            {
                face_vertices.col(jj) = vertices.col(faces(jj, bb));
            }
            std::pair<Eigen::VectorXd, Eigen::VectorXd> BB = compute_pointcloud_bounding_box( face_vertices );
            face_box_mins.col(bb) = BB.first;
            face_box_maxes.col(bb) = BB.second;
        }
        face_aabbtree.build_tree( face_box_mins, face_box_maxes );


        // ------------------------    SUBFACES    ------------------------
        // Construct all boundary entities (faces-of-faces, etc)
        std::vector<std::vector<int>> pset = powerset(dim); // powerset(3) = [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
        std::map<std::vector<int>, std::vector<int>> subface2face_map;
        for ( int bb=0; bb<num_faces; ++bb )
        {
            for ( int ii=0; ii<pset.size(); ++ii )
            {
                std::vector<int> vertex_subset = pset[ii];
                if ( !vertex_subset.empty() )
                {
                    int num_subface_vertices = vertex_subset.size();
                    std::vector<int> subface_vertex_inds;
                    for ( int jj=0; jj<num_subface_vertices; ++jj )
                    {
                        subface_vertex_inds.push_back(faces(vertex_subset[jj], bb));
                    }
                    std::sort( subface_vertex_inds.begin(), subface_vertex_inds.end() );


                    if ( subface2face_map.find(subface_vertex_inds) == subface2face_map.end() ) // if this subface isnt in the map yet
                    {
                        std::vector<int> faces_containing_this_subface;
                        faces_containing_this_subface.push_back(bb);
                        subface2face_map[subface_vertex_inds] = faces_containing_this_subface;
                    }
                    else
                    {
                        subface2face_map[subface_vertex_inds].push_back(bb);
                    }
                }
            }
        }

        std::vector<std::vector<int>> face2subface_vector( num_faces );
        int subface_number = 0;
        for ( auto it  = subface2face_map.begin();
                   it != subface2face_map.end();
                 ++it )
        {
            std::vector<int> subface = it->first;
            int num_subface_vertices = subface.size();
            Eigen::VectorXi SF(num_subface_vertices);
            for ( int kk=0; kk<num_subface_vertices; ++kk)
            {
                SF(kk) = subface[kk];
            }
            subfaces.push_back(SF);

            std::vector<int> faces_containing_this_subface = it->second;
            for ( int bb=0; bb<faces_containing_this_subface.size(); ++bb )
            {
                int face_ind = faces_containing_this_subface[bb];
                face2subface_vector[face_ind].push_back(subface_number);
            }
            subface_number += 1;
        }

        face2subface.resize(num_faces);
        for ( int bb=0; bb<num_faces; ++bb )
        {
            std::vector<int> entities_for_this_face_vector = face2subface_vector[bb];
            Eigen::VectorXi entities_for_this_face(entities_for_this_face_vector.size());
            for ( int jj=0; jj<entities_for_this_face_vector.size(); ++jj )
            {
                entities_for_this_face(jj) = entities_for_this_face_vector[jj];
            }
            face2subface[bb] = entities_for_this_face;
        }

        num_subfaces = subfaces.size();

        subface_simplices.resize(num_subfaces);
        for ( int ee=0; ee<num_subfaces; ++ee )
        {
            int num_vertices_in_subface = subfaces[ee].size();
            Eigen::MatrixXd subface_vertices(dim, num_vertices_in_subface);
            for (int vv=0; vv<num_vertices_in_subface; ++vv )
            {
                subface_vertices.col(vv) = vertices.col(subfaces[ee](vv));
            }
            std::pair<Eigen::MatrixXd, Eigen::VectorXd> STS = make_simplex_transform_operator( subface_vertices );
            subface_simplices[ee] = Simplex { subface_vertices, // V
                                              STS.first,       // A
                                              STS.second };    // b
        }
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> point_is_in_mesh( const Eigen::Ref<const Eigen::MatrixXd> query_points ) const
    {
        return (first_point_collision(query_points).first.array() >= 0);
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> point_is_in_mesh_multithreaded( const Eigen::Ref<const Eigen::MatrixXd> query_points )
    {
        return (first_point_collision_multithreaded(query_points).first.array() >= 0);
    }

    Eigen::MatrixXd closest_point( const Eigen::Ref<const Eigen::MatrixXd> query_points ) const
    {
        int num_queries = query_points.cols();
        Eigen::MatrixXd closest_points;
        closest_points.resize(dim, num_queries);

        for ( int ii=0; ii<num_queries; ++ii )
        {
            closest_points.col(ii) = closest_point_one_query( query_points.col(ii) );
        }

        return closest_points;
    }

    Eigen::MatrixXd closest_point_multithreaded( const Eigen::Ref<const Eigen::MatrixXd> query_points )
    {
        int num_queries = query_points.cols();
        Eigen::MatrixXd closest_points;
        closest_points.resize(dim, num_queries);

        std::vector<int> shuffle_inds(num_queries); // randomize ordering to make work even among threads
        std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
        std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

        auto loop = [&](const int &a, const int &b)
        {
            for ( int ii=a; ii<b; ++ii )
            {
                closest_points.col(shuffle_inds[ii]) = closest_point_one_query( query_points.col(shuffle_inds[ii]) );
            }
        };

        pool.parallelize_loop(0, num_queries, loop);
        return closest_points;
    }

    std::pair<Eigen::VectorXi,Eigen::MatrixXd> first_point_collision( const Eigen::Ref<const Eigen::MatrixXd> points ) const
    {
        int num_pts = points.cols();
        Eigen::VectorXi all_simplex_inds(num_pts);
        Eigen::MatrixXd all_affine_coords(dim+1, num_pts);

        for ( int ii=0; ii<num_pts; ++ii )
        {
            std::pair<int,Eigen::VectorXd> IC = first_point_collision_one_query( points.col(ii) );
            all_simplex_inds(ii) = IC.first;
            all_affine_coords.col(ii) = IC.second;
        }

        return std::make_pair(all_simplex_inds, all_affine_coords);
    }

    std::pair<Eigen::VectorXi,Eigen::MatrixXd> first_point_collision_multithreaded( const Eigen::Ref<const Eigen::MatrixXd> points )
    {
        int num_pts = points.cols();
        Eigen::VectorXi all_simplex_inds(num_pts);
        Eigen::MatrixXd all_affine_coords(dim+1, num_pts);

        std::vector<int> shuffle_inds(num_pts); // randomize ordering to make work even among threads
        std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
        std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

        for ( int ii=0; ii<num_pts; ++ii )
        {
            std::pair<int,Eigen::VectorXd> IC = first_point_collision_one_query( points.col(ii) );
            all_simplex_inds(ii) = IC.first;
            all_affine_coords.col(ii) = IC.second;
        }
        auto loop = [&](const int &a, const int &b)
        {
            for ( int ii=a; ii<b; ++ii )
            {
                std::pair<int,Eigen::VectorXd> IC = first_point_collision_one_query( points.col(shuffle_inds[ii]) );
                all_simplex_inds(shuffle_inds[ii]) = IC.first;
                all_affine_coords.col(shuffle_inds[ii]) = IC.second;
            }
        };

        pool.parallelize_loop(0, num_pts, loop);
        return std::make_pair(all_simplex_inds, all_affine_coords);
    }

    // ------------    SimplexMesh::eval_CG1()    --------------
    // INPUT:
    //   Finite element function nodal values:
    //      functions_at_vertices = [[f_1, f_2, f_3, f_4, ..., f_N],
    //                               [g_1, g_2, g_3, g_4, ..., g_N],
    //                               [h_1, h_2, h_3, h_4, ..., h_N]]
    //      - shape = (num_functions, num_vertices)
    //      - f(x) = sum_{i=1}^N f_i phi_i(x)
    //      - g(x) = sum_{i=1}^N g_i phi_i(x)
    //      - h(x) = sum_{i=1}^N h_i phi_i(x)
    //      - phi_i is CG1 FEM basis function (hat function)
    //
    //   Points to evaluate finite element functions at:
    //      points = [[p1_x, p2_x, p3_x, ..., pM_x],
    //                [p1_y, p2_y, p3_y, ..., pM_y]]
    //      - shape = (dim, num_pts)
    //      - dim = spatial dimension
    //      - pi = [pi_x, pi_y] is ith point
    //
    // OUTPUT:
    //   Finite element functions evaluated at points:
    //      function_at_points = [[f(p1), f(p2), ..., f(pM)],
    //                            [g(p1), g(p2), ..., g(pM)],
    //                            [h(p1), h(p2), ..., h(pM)]]
    //       - shape = (num_functions, num_pts)
    Eigen::MatrixXd eval_CG1( const Eigen::Ref<const Eigen::MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
                              const Eigen::Ref<const Eigen::MatrixXd> points,
                              bool use_reflection ) const // shape=(dim, num_pts)
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        Eigen::MatrixXd functions_at_points(num_functions, num_pts);
        functions_at_points.setZero();

        Eigen::VectorXi all_simplex_inds(num_pts);
        Eigen::MatrixXd all_affine_coords(dim+1, num_pts);

        std::vector<int> function_inds(num_functions);
        std::iota(function_inds.begin(), function_inds.end(), 0);

        std::vector<int> point_inds;
        point_inds.reserve(num_pts);

        for ( int ii=0; ii<num_pts; ++ii )
        {
            Eigen::VectorXd point = points.col(ii);
            std::pair<Eigen::VectorXi,Eigen::MatrixXd> IC = first_point_collision( point );
            if ( use_reflection )
            {
               if ( IC.first(0) < 0 ) // if point is outside mesh
                {
                    point = 2.0 * closest_point( point ) - point; // reflect point across boundary
                    IC = first_point_collision( point );
                }
            }

            all_simplex_inds[ii] = IC.first(0);
            all_affine_coords.col(ii) = IC.second.col(0);
            if ( IC.first(0) >= 0 ) // if point is inside mesh
            {
                point_inds.push_back(ii);
            }
        }

        eval_CG1_helper( functions_at_points,
                         function_inds,
                         point_inds,
                         all_simplex_inds,
                         all_affine_coords,
                         functions_at_vertices );

        return functions_at_points;
    }

    Eigen::MatrixXd eval_CG1_multithreaded( const Eigen::Ref<const Eigen::MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
                                            const Eigen::Ref<const Eigen::MatrixXd> points,
                                            bool use_reflection ) // shape=(dim, num_pts)
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        Eigen::MatrixXd functions_at_points(num_functions, num_pts);
        functions_at_points.setZero();

        Eigen::VectorXi all_simplex_inds(num_pts);
        Eigen::MatrixXd all_affine_coords(dim+1, num_pts);

        std::vector<int> function_inds(num_functions);
        std::iota(function_inds.begin(), function_inds.end(), 0);

        auto loop = [&](const int & start, const int & stop)
        {
            std::vector<int> point_inds;
            point_inds.reserve(stop-start);

            for ( int ii=start; ii<stop; ++ii )
            {
                Eigen::VectorXd point = points.col(ii);
                std::pair<Eigen::VectorXi,Eigen::MatrixXd> IC = first_point_collision( point );
                if ( use_reflection )
                {
                   if ( IC.first(0) < 0 ) // if point is outside mesh
                    {
                        point = 2.0 * closest_point( point ) - point; // reflect point across boundary
                        IC = first_point_collision( point );
                    }
                }

                all_simplex_inds[ii] = IC.first(0);
                all_affine_coords.col(ii) = IC.second.col(0);
                if ( IC.first(0) >= 0 ) // if point is inside mesh
                {
                    point_inds.push_back(ii);
                }
            }

            eval_CG1_helper( functions_at_points,
                             function_inds,
                             point_inds,
                             all_simplex_inds,
                             all_affine_coords,
                             functions_at_vertices );
        };

        pool.parallelize_loop(0, num_pts, loop);
        return functions_at_points;
    }


    void set_sleep_duration(int duration_in_microseconds)
    {
        pool.sleep_duration = duration_in_microseconds;
    }

    void reset_sleep_duration_to_default()
    {
        pool.sleep_duration = default_sleep_duration;
    }

    void set_thread_count(int num_threads)
    {
        pool.reset(num_threads);
    }

    void reset_thread_count_to_default()
    {
        pool.reset(default_number_of_threads);
    }

};

} // end namespace SMESH


