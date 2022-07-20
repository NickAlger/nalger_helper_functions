#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "kdtree.h"
#include "aabbtree.h"
#include "simplexmesh.h"
#include "geometric_sort.h"
#include "brent_minimize.h"
#include "ellipsoid.h"

namespace py = pybind11;

using namespace Eigen;
using namespace std;

using namespace KDT;
using namespace AABB;
using namespace SMESH;
using namespace GSORT;
using namespace BRENT;
using namespace ELLIPSOID;


PYBIND11_MODULE(nalger_helper_functions_cpp, m) {
    m.doc() = "python bindings for helpers written in c++";

    py::class_<KDTree>(m, "KDTree")
        .def(py::init< const Ref<const MatrixXd> >())
        .def_readwrite("block_size", &KDTree::block_size)
        .def("query", &KDTree::query, "many querys, many neighbor")
        .def("query_multithreaded", &KDTree::query_multithreaded, "many querys, many neighbor");

    py::class_<AABBTree>(m, "AABBTree")
        .def(py::init< const Ref<const MatrixXd>,
                       const Ref<const MatrixXd> >())
        .def("point_collisions", &AABBTree::point_collisions)
        .def("point_collisions_vectorized", &AABBTree::point_collisions_vectorized)
        .def("box_collisions", &AABBTree::box_collisions)
        .def("box_collisions_vectorized", &AABBTree::box_collisions_vectorized)
        .def("ball_collisions", &AABBTree::ball_collisions)
        .def("ball_collisions_vectorized", &AABBTree::ball_collisions_vectorized);

    py::class_<SimplexMesh>(m, "SimplexMesh")
        .def(py::init< const Ref<const MatrixXd>,
                       const Ref<const MatrixXi> >())
        .def("closest_point", &SimplexMesh::closest_point)
        .def("closest_point_multithreaded", &SimplexMesh::closest_point_multithreaded)
        .def("point_is_in_mesh", &SimplexMesh::point_is_in_mesh)
        .def("point_is_in_mesh_multithreaded", &SimplexMesh::point_is_in_mesh_multithreaded)
        .def("first_point_collision", &SimplexMesh::first_point_collision)
        .def("first_point_collision_multithreaded", &SimplexMesh::first_point_collision_multithreaded)
        .def("eval_CG1", &SimplexMesh::eval_CG1)
        .def("eval_CG1_multithreaded", &SimplexMesh::eval_CG1_multithreaded)
        .def("set_sleep_duration", &SimplexMesh::set_sleep_duration)
        .def("reset_sleep_duration_to_default", &SimplexMesh::reset_sleep_duration_to_default)
        .def("set_thread_count", &SimplexMesh::set_thread_count)
        .def("reset_thread_count_to_default", &SimplexMesh::reset_thread_count_to_default);

//    m.def("closest_point_in_simplex_vectorized", &closest_point_in_simplex_vectorized);
    m.def("geometric_sort", &geometric_sort);
    m.def("brent_minimize", &brent_minimize);
    m.def("ellipsoids_intersect", &ellipsoids_intersect);
}


//import numpy as np
//from nalger_helper_functions import brent_minimize
//brent_minimize(lambda x: np.cos(np.exp(x)), -1.0, 1.5, 1e-7, 200)
//Out[3]: (1.1447298817285088, -0.9999999999999999, 12, 13)