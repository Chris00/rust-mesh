//! Low-level bindings to [Triangle][].
//!
//! [Triangle]: http://www.cs.cmu.edu/~quake/triangle.html
//!
//! ## License of the Triangle C library
//!
//! These programs may be freely redistributed under the condition
//! that the copyright notices (including the copy of this notice in
//! the code comments and the copyright notice printed when the `-h'
//! switch is selected) are not removed, and no compensation is
//! received.  Private, research, and institutional use is free.  You
//! may distribute modified versions of this code UNDER THE CONDITION
//! THAT THIS CODE AND ANY MODIFICATIONS MADE TO IT IN THE SAME FILE
//! REMAIN UNDER COPYRIGHT OF THE ORIGINAL AUTHOR, BOTH SOURCE AND
//! OBJECT CODE ARE MADE FREELY AVAILABLE WITHOUT CHARGE, AND CLEAR
//! NOTICE IS GIVEN OF THE MODIFICATIONS.  Distribution of this code
//! as part of a commercial system is permissible ONLY BY DIRECT
//! ARRANGEMENT WITH THE AUTHOR.  (If you are not directly supplying
//! this code to a customer, and you are instead telling them how they
//! can obtain it for free, then you are not required to make any
//! arrangement with me.)

use ::std::os::raw::{c_char, c_int};

/// Structure to pass data into and out of the [`triangulate`] function.
///
/// Arrays are used to store points, triangles, markers, and so forth.
/// In all cases, the first item in any array is stored starting at
/// index `0`.  However, that item is item number `1` unless the `z`
/// switch is used, in which case it is item number `0`.  Hence, you
/// may find it easier to index points (and triangles in the neighbor
/// list) if you use the `z` switch.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct triangulateio {
    /// Array of point coordinates.  The first point's x coordinate is
    /// at index `0` and its y coordinate at index `1`, followed by
    /// the coordinates of the remaining points.  Each point occupies
    /// two `f64`
    pub pointlist: *mut f64,
    /// An array of point attributes.  Each point's attributes occupy
    /// `numberofpointattributes` floats `f64`.
    pub pointattributelist: *mut f64,
    /// An array of point markers; one int per point.
    pub pointmarkerlist: *mut c_int,
    pub numberofpoints: c_int,
    pub numberofpointattributes: c_int,
    /// An array of triangle corners.  The first triangle's first
    /// corner is at index `0`, followed by its other two corners in
    /// counterclockwise order, followed by any other nodes if the
    /// triangle represents a nonlinear element.  Each triangle
    /// occupies `numberofcorners` ints.
    pub trianglelist: *mut c_int,
    /// An array of triangle attributes.  Each triangle's attributes
    /// occupy `numberoftriangleattributes` floats.
    pub triangleattributelist: *mut f64,
    /// An array of triangle area constraints; one REAL per triangle.
    /// **Input only.**
    pub trianglearealist: *mut f64,
    /// An array of triangle neighbors; three ints per triangle.
    /// **Output only.**
    pub neighborlist: *mut c_int,
    pub numberoftriangles: c_int,
    pub numberofcorners: c_int,
    pub numberoftriangleattributes: c_int,
    /// An array of segment endpoints.  The first segment's endpoints
    /// are at indices `0` and `1`, followed by the remaining
    /// segments.  Two ints per segment.
    pub segmentlist: *mut c_int,
    /// An array of segment markers; one int per segment.
    pub segmentmarkerlist: *mut c_int,
    pub numberofsegments: c_int,
    /// An array of holes.  The first hole's x and y coordinates are
    /// at indices `0` and `1`, followed by the remaining holes.  Two
    /// floats per hole.  Input only, although the pointer is copied
    /// to the output structure for your convenience.
    pub holelist: *mut f64,
    pub numberofholes: c_int,
    /// An array of regional attributes and area constraints.  The
    /// first constraint's x and y coordinates are at indices `0` and
    /// `1`, followed by the regional attribute at index `2`, followed
    /// by the maximum area at index `3`, followed by the remaining
    /// area constraints.  Four `f64` floats per area constraint.
    /// Note that each regional attribute is used only if you select
    /// the `A` switch, and each area constraint is used only if you
    /// select the `a` switch (with no number following), but omitting
    /// one of these switches does not change the memory layout.
    /// **Input only**, although the pointer is copied to the output
    /// structure for your convenience.
    pub regionlist: *mut f64,
    pub numberofregions: c_int,
    /// An array of edge endpoints.  The first edge's endpoints are at
    /// indices `0` and `1`, followed by the remaining edges.  Two
    /// ints per edge.  **Output only.**
    pub edgelist: *mut c_int,
    /// An array of edge markers; one int per edge.  **Output only.**
    pub edgemarkerlist: *mut c_int,
    /// An array of normal vectors, used for infinite rays in Voronoi
    /// diagrams.  The first normal vector's x and y magnitudes are at
    /// indices `0` and `1`, followed by the remaining vectors.  For
    /// each finite edge in a Voronoi diagram, the normal vector
    /// written is the zero vector.  Two floats per edge.  **Output only.**
    pub normlist: *mut f64,
    pub numberofedges: c_int,
}

extern "C" {
    /// Triangulate the PSLG or refine the mesh.
    ///
    /// `triswitches` string containing the command line switches you
    /// wish to invoke.  No initial dash is required.
    ///
    /// `in_`, `out`, and `vorout` are descriptions of the input, the
    /// output, and the Voronoi output.  If the `v` (Voronoi output)
    /// switch is not used, `vorout` may be NULL.  `in` and `out` may
    /// never be NULL.
    pub fn triangulate(
        triswitches: *mut c_char,
        in_: *mut triangulateio,
        out: *mut triangulateio,
        vorout: *mut triangulateio,
    );
}
