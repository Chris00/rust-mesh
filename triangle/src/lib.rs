//! Interface for the Triangle 2D mesh generator.
//!
//! [Triangle][] is a two-dimensional quality mesh generator and
//! delaunay triangulator which was awarded the 2003 Wilkinson Prize.
//!
//! Jonathan Richard Shewchuk, the author of Triangle, would like
//! that, if you use Triangle in a scientific publication, you
//! acknowledge it with the following citation:
//!   Jonathan Richard Shewchuk, ``Triangle: Engineering a 2D Quality
//!   Mesh Generator and Delaunay Triangulator,'' in Applied
//!   Computational Geometry: Towards Geometric Engineering (Ming
//!   C. Lin and Dinesh Manocha, editors), volume 1148 of Lecture
//!   Notes in Computer Science, pages 203-222, Springer-Verlag,
//!   Berlin, May 1996.
//!
//! [Triangle]: http://www.cs.cmu.edu/~quake/triangle.html

use std::ptr;
use triangle_sys as sys;

/// Planar Straight Line Graph datastructure.
#[derive(Debug, Clone)]
pub struct Pslg {
    /// Vector of points coordinates (x,y).
    pub point: Vec<(f64, f64)>,
    /// Vector of points markers.  Points inside the domain receive the
    /// marker 0, so assign markers ≥ 1] to distinguish different
    /// parts of the boundary.  It must either be empty (in which case
    /// it is equivalent to all the markers being 1), or it must be of
    /// size `point.len()`.
    pub point_marker: Vec<u32>,
    // /// Vector of point attributes.  Attributes are typically
    // /// floating-point values of physical quantities (such as mass or
    // /// conductivity) associated with the nodes.
    // pub point_attribute: Vec<>,
    /// Vector of of segments endpoints.  Segments are edges whose
    /// presence in the triangulation is enforced (although each
    /// segment may be subdivided into smaller edges).
    pub segment: Vec<(usize, usize)>,
    /// Vector of segment markers.  It must either be empty (in which
    /// case it is equivalent to all the markers being 1), or it must
    /// be of size `segment.len()`.
    pub segment_marker: Vec<u32>,
    /// Vector of holes.  For each hole, the array specifies a point (x,y)
    /// inside the hole.
    pub hole: Vec<(f64, f64)>,
    /// of regional attributes and area constraints.  For a region
    /// `i`, `region[i].0` is the couple (x, y) of coordinates of a
    /// point inside the region (the region is bounded by segments),
    /// `region[i].1` is the regional attribute, and `region[i].2` is
    /// the maximum area.  If you wish to specify a regional attribute
    /// but not a maximum area for a given region, set `region[i].2`
    /// to a negative value.
    pub region: Vec<((f64, f64), f64, f64)>, // FIXME
}

impl Default for Pslg {
    /// Return a new [`Pslg`] with all fields being empty vectors.
    fn default() -> Self {
        Pslg {point: vec![], point_marker: vec![], //point_attribute: vec![],
              segment: vec![], segment_marker: vec![],
              hole: vec![], region: vec![] }
    }
}

impl Pslg {
    fn triangulateio(&self) -> &sys::triangulateio {
        &sys::triangulateio {
            //pointlist: self.point.as_mut_ptr(),
            pointlist: ptr::null_mut(),
            pointattributelist: ptr::null_mut(),
            pointmarkerlist: ptr::null_mut(),
            numberofpoints: 0,
            numberofpointattributes: 0,
            trianglelist: ptr::null_mut(),
            triangleattributelist: ptr::null_mut(),
            trianglearealist: ptr::null_mut(),
            neighborlist: ptr::null_mut(),
            numberoftriangles: 0,
            numberofcorners: 3,
            numberoftriangleattributes: 0,
            segmentlist: ptr::null_mut(),
            segmentmarkerlist: ptr::null_mut(),
            numberofsegments: 0,
            holelist: ptr::null_mut(),
            numberofholes: 0,
            regionlist: ptr::null_mut(),
            numberofregions: 0,
            edgelist: ptr::null_mut(),
            edgemarkerlist: ptr::null_mut(),
            normlist: ptr::null_mut(),
            numberofedges: 0,
        }
    }

    pub fn triangulate(&self) -> Triangle {
        todo!()
    }
}

/// Mesh datastructure for Triangle.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vector of points coordinates (x,y).
    pub point: Vec<(f64, f64)>,
    /// Vector of points markers.  Points inside the domain receive the
    /// marker 0, so assign markers ≥ 1] to distinguish different
    /// parts of the boundary.  It must either be empty (in which case
    /// it is equivalent to all the markers being 1), or it must be of
    /// size `point.len()`.
    pub point_marker: Vec<u32>,
    // /// Vector of point attributes.  Attributes are typically
    // /// floating-point values of physical quantities (such as mass or
    // /// conductivity) associated with the nodes.
    // pub point_attribute: Vec<>,
    /// Vector of of segments endpoints.  Segments are edges whose
    /// presence in the triangulation is enforced (although each
    /// segment may be subdivided into smaller edges).
    pub segment: Vec<(usize, usize)>,
    /// Vector of segment markers.  It must either be empty (in which
    /// case it is equivalent to all the markers being 1), or it must
    /// be of size `segment.len()`.
    pub segment_marker: Vec<u32>,
    /// Vector of holes.  For each hole, the array specifies a point (x,y)
    /// inside the hole.
    pub hole: Vec<(f64, f64)>,
    /// of regional attributes and area constraints.  For a region
    /// `i`, `region[i].0` is the couple (x, y) of coordinates of a
    /// point inside the region (the region is bounded by segments),
    /// `region[i].1` is the regional attribute, and `region[i].2` is
    /// the maximum area.  If you wish to specify a regional attribute
    /// but not a maximum area for a given region, set `region[i].2`
    /// to a negative value.
    pub region: Vec<((f64, f64), f64, f64)>, // FIXME

    /// Vector of triangle corners: for each triangle, give the 3
    /// indices p₁, p₂ and p₃ of its corners such that p₁ ≤ p₂ ≤ p₃.
    triangle: Vec<(usize, usize, usize)>,
    // /// Vector of triangle attributes.
    // triangle_attribute: vec<>,
    // /// Triangle neighbors.
    // neighbor: Vec<>, // FIXME
    /// Vector of edge endpoints.
    edge: Vec<(usize, usize)>,
    /// Vector of edge markers.  Edges inside the domain receive the
    /// marker 0.  It must either be empty (meaning that the
    /// information is not provided) or it must be of size `edge.len()`.
    edge_marker: Vec<usize>,
}

impl Default for Mesh {
    /// Return a new [`Pslg`] with all fields being empty vectors.
    fn default() -> Self {
        Mesh {point: vec![], point_marker: vec![], //point_attribute: vec![],
              segment: vec![], segment_marker: vec![],
              hole: vec![], region: vec![],
              triangle: vec![],
              edge: vec![],
              edge_marker: vec![],
        }
    }
}

impl Mesh {
    pub fn refine(&self) -> Triangle {
        todo!()
    }

    // FIXME: what should slices look like?
    pub fn sub(&self ) -> Self {
        todo!()
    }

    pub fn permute_triangles(&mut self, p: &mesh2d::Permutation) {
        todo!()
    }
}

impl mesh2d::Mesh for Mesh {
    fn n_points(&self) -> usize { self.point.len() }
    fn point(&self, i: usize) -> (f64, f64) { self.point[i] }
    fn n_triangles(&self) -> usize { self.triangle.len() }
    fn triangle(&self, i: usize) -> (usize, usize, usize) {
        self.triangle[i]
    }
}

impl mesh2d::Permutable for Mesh {
    fn permute_points(&mut self, p: &mesh2d::Permutation) {
        todo!()
    }
}

/// Voronoi diagram.
#[derive(Debug, Clone)]
pub struct Voronoi {
    /// Vector of points coordinates (x,y).
    pub point: Vec<(f64, f64)>,
    // pub point_attribute: Vec<>,
    /// Vector of edge endpoints.
    pub edge: Vec<(usize, usize)>,
    /// Normal vectors.
    pub normal: Vec<(f64, f64)>,
}

/// Trait for functions used to determine whether or not a selected
/// triangle is too big (and needs to be refined).
///
/// `triunsuitable(p1, p2, p3, area)` must return `true` if the
/// triangle is too big.  The arguments are as follow:
/// - `p1` is the triangle's origin vertex.
/// - `p2` is the triangle's destination vertex.
/// - `p3` is the triangle's apex vertex.
/// - `area` is the area of the triangle.
pub trait Triunsuitable: Fn((f64,f64), (f64,f64), (f64,f64), f64) -> bool {}

/// Triangulate or refine the mesh (see [`Pslg::triangulate`] and
/// [`Mesh::refine`]).
pub struct Triangle {
    mesh_to_date: bool,
    mesh: Mesh, // only valid if `mesh_to_date`
    voronoi_to_date: bool,
    voronoi: Voronoi, // only valid if `voronoi_to_date`
    delaunay: bool,
    min_angle: f64,
    max_area: f64,
    region_area: f64, // only for `triangulate`
    max_steiner: i32,
    edge: bool,
    neighbor: bool,
    subparam: bool,
    triangle_area: Vec<f64>, // only for `refine`
    triunsuitable: Box<dyn Triunsuitable>,
    check_finite: bool,
    debug: bool,
    verbose: i32, // FIXME
}

impl Triangle {
    /// Return (and compute if it wasn't) the
    fn mesh(&mut self) -> &Mesh {
        if ! self.mesh_to_date {
            // FIXME
            let m = Mesh::default();
            self.mesh = m;
        }
        &self.mesh
    }
}

