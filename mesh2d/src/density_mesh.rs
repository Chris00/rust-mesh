//! Compatibility with the crate "density-mesh-core".

use density_mesh_core::{mesh::DensityMesh, coord::Coord};
use crate::Mesh;

macro_rules! impl_mesh {
    ($m: ty) => {
        impl Mesh for $m {
            fn n_points(&self) -> usize { self.points.len() }

            fn point(&self, i: usize) -> (f64, f64) {
                let Coord{x,y} = self.points[i];
                (x as f64, y as f64)
            }

            fn n_triangles(&self) -> usize { self.triangles.len() }

            fn triangle(&self, i: usize) -> (usize, usize, usize) {
                let t = self.triangles[i];
                super::sort3(t.a, t.b, t.c)
            }
        }
    }
}

impl_mesh!(DensityMesh);
impl_mesh!(&DensityMesh);
