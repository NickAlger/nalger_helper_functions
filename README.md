# nalger_helper_functions
A hodgepodge of Python and C++ helper functions I've written during my research that may be broadly useful. Primarily related to numerical methods/computational science/finite elements.

The files are mostly stand-alone, so you can typically just copy-paste the ones you need into your project, and edit them as needed to your specifications.

Or, you can install the python package as follows:
1) **Edit the makefile** to point to the Eigen include directory and threadpool include directory on your computer. Eigen can be downloaded here: https://eigen.tuxfamily.org/index.php?title=Main_Page and threadpool can be downloaded here: https://github.com/bshoshany/thread-pool
2) Open a terminal in the nalger_helper_functions base directory (directory containing the makefile, not the subdirectory of the same name with the .py files), and **run the makefile**:
 ```none
 make
 ```
3) **Install the python package**:
 ```none
 pip install .
 ```
4) **Import the package in python**, and off you go. E.g., in a python script or jupyter notebook, 
 ```python
 import nalger_helper_functions as nhf
 ```

## Notes:

- If you don't care about the c++ functions (fast static KD-tree, AABB-tree, and simplex mesh tools), then you can skip steps 1 and 2 and just do steps 3 and 4.
- If you only care about the c++ functions, they are header-only, so you can point your compiler to the include directory, and #include the desired .h files in your c++ code.
