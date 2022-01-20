Rust 2D Triangular Mesh utilities
=================================

These mesh generation and manipulations utilities come in several
crates.  More precisely:

- `mesh`: A set of traits for meshes providing several functions to
  manipulate them.  In particular, it allows to export meshes and
  graphs of functions defined on their nodes to LaTeX, SciLab,
  Matlab, and Mathematica.
- `mesh-triangle`: This crate is a binding to the [Triangle][] library
  which was awarded the James Hardy Wilkinson Prize in Numerical
  Software in 2003.  If `libtriangle-dev` is not installed on your
  system, it will install a local copy for this library.


[Triangle]: http://www.cs.cmu.edu/~quake/triangle.html



![Rust-mesh](http://math.umons.ac.be/an/en/software/rust-mesh.png)
