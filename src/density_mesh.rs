//! Compatibility with the crate "density-mesh-core".

use density_mesh_core::{mesh::DensityMesh, coord::Coord};
use crate::MeshBase;

impl MeshBase for DensityMesh {
    fn n_points(&self) -> usize { self.points.len() }

    fn point(&self, i: usize) -> (f64, f64) {
        let Coord{x,y} = self.points[i];
        (x as f64, y as f64)
    }

    fn n_triangles(&self) -> usize { self.triangles.len() }

    fn triangle(&self, i: usize) -> (usize, usize, usize) {
        let t = self.triangles[i];
        sort(t.a, t.b, t.c)
    }
}

/// Return the triple made of `p1`, `p2`, and `p3` sorted in
/// increasing order.
fn sort(p1: usize, p2: usize, p3: usize) -> (usize, usize, usize) {
    if p1 <= p2 {
        if p2 <= p3 { (p1, p2, p3) }
        else if p1 <= p3 { (p1, p3, p2) } else { (p3, p1, p2) }
    } else { // p2 < p1
        if p1 <= p3 { (p2, p1, p3) }
        else if p2 <= p3 { (p2, p3, p1) } else { (p3, p2, p1) }
    }
}
