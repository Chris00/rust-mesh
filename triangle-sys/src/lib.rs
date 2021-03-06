//! Low-level bindings to [Triangle][].
//!
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
//!
//! [Triangle]: http://www.cs.cmu.edu/~quake/triangle.html

use ::std::{os::raw::{c_char, c_int, c_void},
            ptr};

/// Structure to pass data into and out of the [`triangulate`]
/// function.
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
    /// two `f64` and they must both be *finite*.
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

/// Convenience constant to help only fill the required fields using
/// the struct update syntax.
pub const EMPTY_TRIANGULATEIO: triangulateio = triangulateio {
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
};

/// Type of the `triunsuitable` functions.
/// See [`triangulate`] for more information.
pub type Triunsuitable
    = unsafe extern "C" fn(triorg: *mut f64, tridest: *mut f64,
                           triapex: *mut f64, area: f64,
                           user_data: *mut c_void) -> bool;

extern "C" {
    /// Triangulate the PSLG or refine the mesh (depending on the
    /// switches).
    ///
    /// `triswitches` string containing the command line switches you
    /// wish to invoke.  No initial dash is required.
    ///
    /// `in_`, `out`, and `vorout` are descriptions of the input, the
    /// output, and the Voronoi output.  If the `v` (Voronoi output)
    /// switch is not used, `vorout` may be NULL.  `in` and `out` may
    /// never be NULL.
    ///
    /// `triunsuitable` is a function used to determine whether or not
    /// a triangle is too big (and needs to be refined).  More
    /// precisely, `triunsuitable(p1, p2, p3, area, u)` must
    /// return `true` if the triangle is too big.  The arguments are
    /// as follow:
    /// - `p1` is the triangle's origin vertex;
    /// - `p2` is the triangle's destination vertex;
    /// - `p3` is the triangle's apex vertex;
    /// - `area` is the area of the triangle;
    /// - `u` is the `user_data` value given to `triangulate`.
    pub fn triangulate(
        triswitches: *const c_char,
        in_: *const triangulateio,
        out: *mut triangulateio,
        vorout: *mut triangulateio,
        triunsuitable: Triunsuitable,
        user_data: *mut c_void,
    );

    /// The pointers of [`triangulateio`] **must** be deallocated
    /// using this function.
    pub fn trifree(memptr: *mut c_void);
}

/// Default `triunsuitable` function.
///
/// This function is a Rust transcription of the default one used in
/// Triangle code.  It can be used as a default [`Triunsuitable`]
/// function.
pub extern "C" fn default_triunsuitable(
    triorg: *mut f64, tridest: *mut f64, triapex: *mut f64,
    _area: f64, _user_data: *mut c_void) -> bool {
    let triorg = unsafe { &*(triorg as *mut [f64; 2]) };
    let tridest = unsafe { &*(tridest as *mut [f64; 2]) };
    let triapex = unsafe { &*(triapex as *mut [f64; 2]) };
    let dxoa = triorg[0] - triapex[0];
    let dyoa = triorg[1] - triapex[1];
    let dxda = tridest[0] - triapex[0];
    let dyda = tridest[1] - triapex[1];
    let dxod = triorg[0] - tridest[0];
    let dyod = triorg[1] - tridest[1];
    // Find the squares of the lengths of the triangle's three edges.
    let oalen = dxoa * dxoa + dyoa * dyoa;
    let dalen = dxda * dxda + dyda * dyda;
    let odlen = dxod * dxod + dyod * dyod;
    // Find the square of the length of the longest edge.
    let maxlen = if dalen > oalen { dalen } else { oalen };
    let maxlen = if odlen > maxlen { odlen } else { maxlen };
    maxlen > 0.05 * (triorg[0] * triorg[0] + triorg[1] * triorg[1]) + 0.02
}



#[cfg(test)]
mod test {
    use ::std::ptr;
    use super::triangulateio;

    #[test]
    fn test_triunsuitable() {
        let triswitches = b"zpDu"; // u ??? triunsuitable
        let square = [[0.,0.], [1.,0.], [1.,1.], [0.,1.]];
        let segments = [[0,1], [1,2], [2,3], [3,0]];
        let input = triangulateio {
            pointlist: square.as_ptr() as *mut _,
            numberofpoints: square.len() as i32,
            segmentlist: segments.as_ptr() as *mut _,
            numberofsegments: segments.len() as i32,
            .. super::EMPTY_TRIANGULATEIO
        };
        let mut output = triangulateio {
            .. super::EMPTY_TRIANGULATEIO
        };
        let mut vorout = triangulateio {
            .. super::EMPTY_TRIANGULATEIO
        };
        unsafe {
            super::triangulate(
                triswitches.as_ptr() as *const i8,
                &input, &mut output, &mut vorout,
                super::default_triunsuitable,
                ptr::null_mut());
        }
    }
}
