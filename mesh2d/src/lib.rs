//! Generic mesh structure to be used with various meshers.  Also
//! provide some functions to help to display and design geometries.

use std::{collections::{VecDeque, HashSet, HashMap},
          fmt::{self, Display, Formatter},
          fs::File,
          io::{self, Write},
          ops::Index,
          path::{Path, PathBuf}};
use rgb::{RGB, RGB8};

#[cfg(feature = "density-mesh-core")]
mod density_mesh;

////////////////////////////////////////////////////////////////////////
//
// Permutations

/// An immutable permutation.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Permutation(Vec<usize>);

impl Display for Permutation {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        if self.0.len() == 0 {
            write!(f, "Permutation()")
        } else {
            write!(f, "Permutation({}", self.0[0])?;
            for i in self.0.iter().skip(1) {
                write!(f, ", {}", i)?
            }
            write!(f, ")")
        }
    }
}

/// Check that `p` is a valid permutation.
fn is_perm(p: &Vec<usize>) -> bool {
    let len = p.len();
    let mut taken = vec![false; len];
    for &j in p.iter() {
        if j >= len || taken[j] { return false }
        taken[j] = true;
    }
    true
}

impl Permutation {
    /// Create a new permutation from the iterable value `c`.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Permutation;
    /// let p = Permutation::new([1, 0, 2]);
    /// ```
    pub fn new(c: impl IntoIterator<Item=usize>) -> Option<Self> {
        let p: Vec<usize> = c.into_iter().collect();
        if is_perm(&p) { Some(Permutation(p)) } else { None }
    }

    /// Return the length of the permutation (i.e., the permutation
    /// acts on {0,..., `len()`-1}).
    pub fn len(&self) -> usize { self.0.len() }

    /// Return the inverse of the permutation `self`.
    pub fn inv(&self) -> Self {
        let mut inv = self.0.clone();
        for (i, &j) in self.0.iter().enumerate() {
            inv[j] = i;
        }
        Permutation(inv)
    }

    /// Permute `x` with the permutation `self`.
    /// More precisely, if we denote `x_after` the slice after this
    /// transformation, one has `x_after[self[i]] == x[i]` for all `i`.
    /// In other words, `self[i]` gives the new location of the data
    /// at index `i`.
    ///
    /// Panic if the slice `x` and the permutation `self` have
    /// different lengths.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Permutation;
    /// let p = Permutation::new([2, 0, 3, 1]).unwrap();
    /// let mut x = ["a", "b", "c", "d"];
    /// p.apply(&mut x);
    /// assert_eq!(x, ["b", "d", "a", "c"]);
    /// ```
    pub fn apply<T>(&self, x: &mut [T]) {
        if self.len() != x.len() {
            panic!("mesh2d::Permutation::apply: the permutation and slice \
                    must have the same length.")
        }
        let mut replaced = vec![false; self.len()];
        let mut start = 0_usize;
        while start < self.len() {
            if replaced[start] { start += 1;
                                 continue }
            // Deal with cycle starting at `start`.
            let mut i = self[start];
            if i == start { start += 1; // no need to update `replaced[start]`
                            continue }
            while i != start {
                x.swap(start, i);
                replaced[i] = true;
                i = self[i];
            }
            // x[start] holds the last value
            start += 1; // ⇒ no need to update `replaced[start]`
        }
    }
}

impl Index<usize> for Permutation {
    type Output = usize;
    fn index(&self, i: usize) -> &usize { &self.0[i] }
}

/// Conversion error when a vector is not a permutation.
#[derive(Debug, PartialEq, Eq)]
pub struct NotPermutation;

impl TryFrom<Vec<usize>> for Permutation {
    type Error = NotPermutation;
    fn try_from(x: Vec<usize>) -> Result<Self, Self::Error> {
        if is_perm(&x) { Ok(Permutation(x)) }
        else { Err(NotPermutation) }
    }
}

// This will make easy for, say, the crate "permutation" to convert
// permutations generated from this library.
impl AsRef<Vec<usize>> for Permutation {
    /// Return a vector representing the permutation.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Permutation;
    /// let p = Permutation::new([1, 0, 2]).unwrap();
    /// assert_eq!(&p.as_ref()[..], [1, 0, 2]);
    /// ```
    fn as_ref(&self) -> &Vec<usize> { &self.0 }
}


////////////////////////////////////////////////////////////////////////
//
// Basic colors

const BLACK: RGB8 = RGB {r: 0, g: 0, b: 0};
const GREY: RGB8 = RGB {r: 150, g: 150, b: 150};

////////////////////////////////////////////////////////////////////////
//
// Meshes

/// A bounding box in ℝ².  It is required that `xmin` ≤ `xmax` and
/// `ymin` ≤ `ymax` unless the box is empty in which case `xmin` =
/// `ymin` = +∞ and `xmax` = `ymax` = -∞.
pub struct BoundingBox {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

/// Trait describing minimal characteristics of a mesh.
pub trait Mesh {
    /// Return the number of points in the mesh.
    fn n_points(&self) -> usize;
    /// Return the coordinates (x,y) of the point of index `i` (where
    /// it is assumed that 0 ≤ `i` < `n_points()`).  The coordinates
    /// **must** be finite numbers.
    fn point(&self, i: usize) -> (f64, f64);
    /// Number of triangles in the mesh.
    fn n_triangles(&self) -> usize;
    /// The 3 corners (p₁, p₂, p₃) of the triangle `i`:
    /// 0 ≤ pₖ < `n_points()` are the indices of the corresponding
    /// points.
    fn triangle(&self, i: usize) -> (usize, usize, usize);

    /// Return the bounding box enclosing the mesh.  If the mesh is
    /// empty, `xmin` = `ymin` = +∞ and `xmax` = `ymax` = -∞.
    fn bounding_box(&self) -> BoundingBox {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for i in 0 .. self.n_points() {
            let (x, y) = self.point(i);
            if x < xmin { xmin = x }
            if x < xmax { xmax = x }
            if y < ymin { ymin = y }
            if y > ymax { ymax = y }
        }
        BoundingBox{ xmin, xmax, ymin, ymax }
    }

    /// Returns the number of nonzero super-diagonals + 1 (for the
    /// diagonal) of symmetric band matrices for P1 finite elements
    /// inner products.  It is the maximum on all triangles T of
    /// max(|i1 - i2|, |i2 - i3|, |i3 - i1|) where i1, i2, and i3 are
    /// the indices of the nodes of the three corners of the triangle T.
    fn band_height_p1(&self) -> usize {
        let mut kd = 0_usize;
        for i in 0 .. self.n_triangles() {
            let (i1, i2, i3) = sort3(self.triangle(i));
            kd = kd.max(i3 - i2).max(i3 - i1).max(i2 - i1);
        }
        kd + 1
    }

    /// Same as [`band_height_p1`][Mesh::band_height_p1] except that
    /// it only consider the nodes `i` such that `predicate(i)` is `true`.
    fn band_height_p1_filter<P>(&self, mut predicate: P) -> usize
    where P: FnMut(usize) -> bool {
        let mut kd = 0_usize;
        for i in 0 .. self.n_triangles() {
            let (i1, i2, i3) = sort3(self.triangle(i));
            if predicate(i1) {
                if predicate(i2) {
                    if predicate(i3) {
                        kd = kd.max(i3 - i2).max(i3 - i1).max(i2 - i1);
                    } else { // i3 excluded
                        kd = kd.max(i2 - i1);
                    }
                } else /* i2 excluded */ if predicate(i3) {
                    kd = kd.max(i3 - i1);
                }
            } else /* i1 excluded */ if predicate(i1) && predicate(i3) {
                kd = kd.max(i3 - i2);
            }
        }
        kd + 1
    }

    /// Graph the vector `z` defined on the mesh `self` using Scilab.
    /// The value of the function at the point [`point(i)`][Mesh::point]
    /// is given by `z[i]`.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Mesh;
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.scilab(&[1., 2., 3., 4.]).save("/tmp/foo");
    /// # Ok(()) }
    /// ```
    fn scilab<'a, Z>(&'a self, z: &'a Z) -> Scilab<'a, Self>
    where Z: P1 + 'a {
        if z.len() != self.n_points() {
            panic!("mesh2d::Mesh::scilab: z.len() = {} but expected {}",
                   z.len(), self.n_points());
        }
        Scilab { mesh: self, z,
                 longitude: 70.,  azimuth: 60.,
                 mode: Mode::Triangles,
                 draw_box: DrawBox::Full,
                 edge_color: None }
    }

    /// Graph the vector `z` defined on the mesh `self` using Matlab.
    /// The value of the function at the point [`point(i)`][Mesh::point]
    /// is given by `z[i]`.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Mesh;
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.matlab(&[1., 2., 3., 4.]).save("/tmp/foo");
    /// # Ok(()) }
    /// ```
    fn matlab<'a, Z>(&'a self, z: &'a Z) -> Matlab<'a, Self>
    where Z: P1 + 'a {
        if z.len() != self.n_points() {
            panic!("mesh2d::Mesh::matlab: z.len() = {} but expected {}",
                   z.len(), self.n_points());
        }
        Matlab { mesh: self, z,
                 edge_color: EdgeColor::Color(BLACK),
                 line_style: LineStyle::Solid,
                 face_alpha: 1. }
    }

    /// Graph the vector `z` defined on the mesh `self` using Matplotlib.
    /// The value of the function at the point [`point(i)`][Mesh::point]
    /// is given by `z[i]`.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Mesh;
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.matplotlib(&[1., 2., 3., 4.]).save("/tmp/foo");
    /// # Ok(()) }
    /// ```
    fn matplotlib<'a, Z>(&'a self, z: &'a Z) -> Matplotlib<'a, Self>
    where Z: P1 + 'a {
        if z.len() != self.n_points() {
            panic!("mesh2d::Mesh::matplotlib: z.len() = {} but expected {}",
                   z.len(), self.n_points());
        }
        Matplotlib { mesh: self, z }
    }

    /// LaTeX output of the mesh `self` and of vectors defined on this
    /// mesh.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Mesh;
    /// use rgb::{RGB, RGB8};
    /// const RED: RGB8 = RGB {r: 255, g: 0, b: 0};
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// let levels = [(1.5, RED)];
    /// mesh.latex().sub_levels(&[1., 2., 3., 4.], levels).save("/tmp/foo")?;
    /// # Ok(()) }
    /// ```
    fn latex<'a>(&'a self) -> LaTeX<'a, Self> {
        LaTeX { mesh: self,  edges: Edges::new(self),
                color: None,
                boundary_color: None,
                action: Action::Mesh, levels: vec![] }
    }


    fn mathematica<'a, Z>(&'a self, z: &'a Z) -> Mathematica<'a, Self>
    where Z: P1 + 'a {
        if z.len() != self.n_points() {
            panic!("mesh2d::Mesh::mathematica: z.len() = {} but \
                    expected {}", z.len(), self.n_points());
        }
        Mathematica { mesh: self, z }
    }
}


/// Return the triple made of `p1`, `p2`, and `p3` sorted in
/// increasing order.  Helper function.
fn sort3((p1, p2, p3): (usize, usize, usize)) -> (usize, usize, usize) {
    if p1 <= p2 {
        if p2 <= p3 { (p1, p2, p3) }
        else if p1 <= p3 { (p1, p3, p2) } else { (p3, p1, p2) }
    } else { // p2 < p1
        if p1 <= p3 { (p2, p1, p3) }
        else if p2 <= p3 { (p2, p3, p1) } else { (p3, p2, p1) }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Band reduction

fn arg_min_deg(deg: &Vec<i32>) -> usize {
    let mut i = 0;
    let mut degi = i32::MAX;
    for (j, &degj) in deg.iter().enumerate() {
        if degj >= 0 && degj < degi {
            i = j;
            degi = degj
        }
    }
    i
}

macro_rules! add_node {
    ($node: ident, $q: ident, $free: ident,
     $p: ident, $deg: ident, $nbh: ident) => {
        $p[$node] = $free;
        $free += 1;
        $deg[$node] = -1; // `$node` has been put in the permutation `$p`
        let mut nbhs: Vec<_> =
            $nbh[$node].iter().filter(|&&i| $deg[i] >= 0).collect();
        nbhs.sort_by_key(|&&i| $deg[i]);
        for &i in nbhs {
            $q.push_back(i)
        }
    }
}

// Cannot make it a polymorphic fn and then pass `self` (unsized).
macro_rules! cuthill_mckee_base {
    ($m: ident) => {{
        let n = $m.n_points();
        // Permutation.
        let mut p: Vec<usize> = vec![0; n];
        // Degree of adjacency of each node.  -1 = done with that node.
        let mut deg: Vec<i32> = vec![0; n];
        // List of adjacent nodes:
        let mut nbh = vec![HashSet::new(); n];
        for t in 0 .. $m.n_triangles() {
            let (p1, p2, p3) = $m.triangle(t);
            if nbh[p1].insert(p2) { deg[p1] += 1 }
            if nbh[p1].insert(p3) { deg[p1] += 1 }
            if nbh[p2].insert(p1) { deg[p2] += 1 }
            if nbh[p2].insert(p3) { deg[p2] += 1 }
            if nbh[p3].insert(p1) { deg[p3] += 1 }
            if nbh[p3].insert(p2) { deg[p3] += 1 }
        }
        let mut free = 0_usize; // first free position in `perm`
        let mut q = VecDeque::new();
        while free < n {
            let i = arg_min_deg(&deg);
            add_node!(i, q, free,  p, deg, nbh);
            while let Some(c) = q.pop_front() {
                if deg[c] >= 0 {
                    add_node!(c, q, free,  p, deg, nbh);
                }
            }
        }
        p
    }}
}

/// Meshes accepting permutations of points indices.
pub trait Permutable: Mesh {
    /// Transform the mesh `self` so that the points indices are
    /// transformed through the permutation `p`: if `point_after` is
    /// the result after permutation, one must have:
    /// `point_after(p[i]) == point(i)`
    /// See [`Permutation::apply`].
    fn permute_points(&mut self, p: &Permutation);

    // Transform the mesh `self` so that the triangle indices are
    // transformed through the permutation `p`.  See
    // [`permute_points`][Permutable::permute_points] and
    // [`Permutation::apply`].
    //fn permute_triangles(&mut self, p: &Permutation);

    /// Transform the mesh `self` so that the labelling of the points
    /// has been changed to lower its band (as computed by
    /// [`band_height_p1`][Mesh::band_height_p1]).  Return the
    /// permutation `p` of the points as it is needed to transfer
    /// vectors defined on the initial labeling to the new one.  The
    /// relation `p[i] == j` means that `j` is the new label for the
    /// point initially labeled `i` (see [`Permutation::apply`]).
    // Inspired from http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
    fn cuthill_mckee(&mut self) -> Permutation {
        let p = Permutation(cuthill_mckee_base!(self));
        self.permute_points(&p);
        p
    }

    /// Same as [`cuthill_mckee`][Permutable::cuthill_mckee] but use
    /// the Reverse CutHill-McKee algorithm.
    fn cuthill_mckee_rev(&mut self) -> Permutation {
        let mut p = cuthill_mckee_base!(self);
        let n = p.len() - 1;
        for pi in &mut p { *pi = n - *pi } // reverse positions
        let p = Permutation(p);
        self.permute_points(&p);
        p
    }


    // A Generalized GPS Algorithm For Reducing The Bandwidth And Profile
    // Of A Sparse Matrix, Q. Wang, Y. C. Guo, and X. W. Shi
    // http://www.jpier.org/PIER/pier90/09.09010512.pdf
    // fn ggps<M>(&mut self) -> Permutation {
    //     let n = self.n_points();
    //     let mut p: Vec<usize> = vec![0; n];
    //     // Degree of adjacency of each node.  -1 = done with that node.
    //     let mut deg = vec![0; n];
    //     for i in 0 .. self.n_edges() {
    //         let (p1, p2) = self.edge(i);
    //         deg[p1] += 1;
    //         deg[p2] += 1;
    //     }

    //     todo!();
    //     Permutation(p)
    // }
}


////////////////////////////////////////////////////////////////////////
//
// Vectors

/// Represent a P1 function defined on a mesh: the value at all points
/// of the mesh needs to be known.
pub trait P1 {
    /// The length of the vector.
    fn len(&self) -> usize;
    /// The value of the function at the point of index `i`.
    fn index(&self, i: usize) -> f64;
}

impl<const N: usize> P1 for [f64; N] {
    fn len(&self) -> usize { <[f64]>::len(self) }
    fn index(&self, i: usize) -> f64 { self[i] }
}

impl<const N: usize> P1 for &[f64; N] {
    fn len(&self) -> usize { <[f64]>::len(*self) }
    fn index(&self, i: usize) -> f64 { self[i] }
}

impl P1 for Vec<f64> {
    fn len(&self) -> usize { <[f64]>::len(self) }
    fn index(&self, i: usize) -> f64 { self[i] }
}

impl P1 for &Vec<f64> {
    fn len(&self) -> usize { <[f64]>::len(*self) }
    fn index(&self, i: usize) -> f64 { self[i] }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::P1;
    use ndarray::Array1;

    impl P1 for Array1<f64> {
        fn len(&self) -> usize { Array1::len(self) }
        fn index(&self, i: usize) -> f64 { self[i] }
    }

    impl P1 for &Array1<f64> {
        fn len(&self) -> usize { Array1::len(self) }
        fn index(&self, i: usize) -> f64 { self[i] }
    }

    impl P1 for Array1<f32> {
        fn len(&self) -> usize { ndarray::Array1::len(self) }
        fn index(&self, i: usize) -> f64 { self[i] as f64 }
    }

    impl P1 for &ndarray::Array1<f32> {
        fn len(&self) -> usize { ndarray::Array1::len(self) }
        fn index(&self, i: usize) -> f64 { self[i] as f64 }
    }
}

////////////////////////////////////////////////////////////////////////
//
// LaTeX Output

enum Action<'a> {
    Mesh,
    Levels(&'a dyn P1),
    SuperLevels(&'a dyn P1), // levels in increasing order
    SubLevels(&'a dyn P1) // levels in decreasing order
}

/// Map edges to 0 if inside and 1 if on the boundary.
/// If (p₁, p₂) is a key, we require p₁ ≤ p₂.
struct Edges(HashMap<(usize, usize), i8>);

impl Edges {
    fn new<M: Mesh + ?Sized>(m: &M) -> Self {
        let mut e = HashMap::new();
        for t in 0 .. m.n_triangles() {
            let (p1, p2, p3) = sort3(m.triangle(t));
            let cnt = e.entry((p1, p2)).or_insert(2);
            *cnt -= 1;
            let cnt = e.entry((p1, p3)).or_insert(2);
            *cnt -= 1;
            let cnt = e.entry((p2, p3)).or_insert(2);
            *cnt -= 1;
        }
        Edges(e)
    }

    fn on_boundary(&self, i1: usize, i2: usize) -> bool {
        match self.0.get(&if i1 <= i2 { (i1, i2) } else { (i2, i1) }) {
            Some(&cnt) => cnt == 1,
            None => false
        }
    }
}

/// LaTeX output.  Created by [`Mesh::latex`].
pub struct LaTeX<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    edges: Edges,
    color: Option<Option<RGB8>>, // if specified, and then None ⇔ not drawn
    boundary_color: Option<Option<RGB8>>,
    action: Action<'a>,
    levels: Vec<(f64, RGB8)>,
}

/// Only keep levels that are finite.
fn sanitize_levels<L>(l: L) -> Vec<(f64, RGB8)>
where L: IntoIterator<Item=(f64, RGB8)>{
    l.into_iter().filter(|(x, _)| x.is_finite()).collect()
}

macro_rules! do_not_sort_levels { ($l: expr) => { $l }}
macro_rules! sort_levels_incr { ($l: expr) => {{
    let mut levels = $l;
    levels.sort_by(|(l1, _), (l2, _)| l1.partial_cmp(l2).unwrap());
    levels }}}
macro_rules! sort_levels_decr { ($l: expr) => {{
    let mut levels = $l;
    levels.sort_by(|(l1, _), (l2, _)| l2.partial_cmp(l1).unwrap());
    levels }}}

macro_rules! meth_levels {
    ($(#[$meta:meta])* $meth: ident, $act: ident, $sort: ident) => {
        $(#[$meta])*
        pub fn $meth<Z, L>(self, z: &'a Z, levels: L) -> LaTeX<'a, M>
        where Z: P1 + 'a,
              L: IntoIterator<Item=(f64, RGB8)> {
            if z.len() != self.mesh.n_points() {
                panic!("mesh2d::LaTeX::{}: z.len() == {}, expected {}",
                       stringify!($meth), z.len(), self.mesh.n_points()) }
            let levels = $sort!(sanitize_levels(levels));
            LaTeX { action: Action::$act(z), levels, .. self }
        }
    }
}

fn mid((x1, y1): (f64, f64), (x2, y2): (f64, f64)) -> (f64, f64) {
    (0.5 * (x1 + x2), 0.5 * (y1 + y2))
}

fn intercept((x1, y1): (f64, f64), z1: f64, (x2, y2): (f64, f64), z2: f64,
             level: f64) -> (f64, f64) {
    let d = z1 - z2;
    let a = level - z2;
    let b = z1 - level;
    ((a * x1 + b * x2) / d,  (a * y1 + b * y2) / d)
}

fn line<W: Write>(w: &mut W, c: RGB8,
                  (x0,y0): (f64,f64), (x1,y1): (f64,f64)) -> io::Result<()> {
    write!(w, "  \\meshline{{{},{},{}}}{{{:.12}}}{{{:.12}}}\
               {{{:.12}}}{{{:.12}}}\n", c.r, c.g, c.b, x0, y0, x1, y1)
}

fn point<W: Write>(w: &mut W, c: RGB8,
                   i: usize, (x,y): (f64,f64)) -> io::Result<()> {
    write!(w, "  \\meshpoint{{{},{},{}}}{{{}}}{{{:.12}}}{{{:.12}}}\n",
           c.r, c.g, c.b, i, x, y)
}

fn triangle<W: Write>(w: &mut W, c: RGB8,
                      (x1,y1): (f64,f64), (x2,y2): (f64,f64),
                      (x3,y3): (f64,f64)) -> io::Result<()> {
    write!(w, "  \\meshtriangle{{{},{},{}}}{{{:.12}}}{{{:.12}}}\
               {{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}\n",
           c.r, c.g, c.b, x1, y1, x2, y2, x3, y3)
}

fn quadrilateral<W: Write>(
    w: &mut W, c: RGB8, (x1,y1): (f64,f64), (x2,y2): (f64,f64),
    (x3,y3): (f64,f64), (x4,y4): (f64,f64)) -> io::Result<()> {
    write!(w, "  \\meshquadrilateral{{{},{},{}}}{{{:.12}}}{{{:.12}}}\
               {{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}\n",
           c.r, c.g, c.b, x1, y1, x2, y2, x3, y3, x4, y4)
}

/// Levels considered equal (to draw level curves).
// FIXME: need to make it customizable?
fn level_eq(l1: f64, l2: f64) -> bool {
    (l1 - l2).abs() <= 1e-8 * (l1.abs() + l2.abs())
}

macro_rules! keep_order { ($o: expr) => { $o } }
macro_rules! reverse_order { ($o: expr) => { $o.reverse() } }

/// Designed for super-levels, reverse the ordering for sub-levels.
macro_rules! write_xxx_levels {
    ($w: ident, $m: ident, $z: ident, $levels: expr, $rev: ident) => {
        for &(l, color) in &$levels {
            for t in 0 .. $m.n_triangles() {
                let (i1, i2, i3) = $m.triangle(t);
                let p1 = $m.point(i1);
                let z1 = $z.index(i1);
                let p2 = $m.point(i2);
                let z2 = $z.index(i2);
                let p3 = $m.point(i3);
                let z3 = $z.index(i3);
                use std::cmp::Ordering::*;
                // Only finite values have been kept.
                match ($rev!(z1.partial_cmp(&l).unwrap()),
                       $rev!(z2.partial_cmp(&l).unwrap()),
                       $rev!(z3.partial_cmp(&l).unwrap())) {
                    (Greater, Greater | Equal, Greater | Equal)
                        | (Equal, Greater, Greater | Equal)
                        | (Equal, Equal, Greater) => {
                            // (Equal, Equal, Equal) not accepted
                            // because strict superlevel.
                            triangle($w, color, p1, p2, p3)?
                        }
                    (Greater, Greater, Less) => { // Cut edges 1-3, 3-2
                        quadrilateral($w, color, p2, p1,
                                      intercept(p1,z1, p3,z3, l),
                                      intercept(p3,z3, p2,z2, l))?
                    }
                    (Greater, Less, Greater) => { // Cut edges 3-2, 2-1
                        quadrilateral($w, color, p1, p3,
                                      intercept(p3,z3, p2,z2, l),
                                      intercept(p2,z2, p1,z1, l))?
                    }
                    (Less, Greater, Greater) => { // Cut edges 2-1, 1-3
                        quadrilateral($w, color, p3, p2,
                                      intercept(p2,z2, p1,z1, l),
                                      intercept(p1,z1, p3,z3, l))?
                    }
                    // (Greater, Equal, Equal) rightly matched before.
                    (Greater, Equal | Less, Equal | Less) => {
                        triangle($w, color, p1,
                                 intercept(p1,z1, p2,z2, l),
                                 intercept(p1,z1, p3,z3, l))?
                    }
                    (Equal | Less, Greater, Equal | Less) => {
                        triangle($w, color, p2,
                                 intercept(p2,z2, p1,z1, l),
                                 intercept(p2,z2, p3,z3, l))?
                    }
                    (Equal | Less, Equal | Less, Greater) => {
                        triangle($w, color, p3,
                                 intercept(p3,z3, p1,z1, l),
                                 intercept(p3,z3, p2,z2, l))?
                    }
                    // Nothing to color
                    (Equal | Less, Equal | Less, Equal | Less) => {}
                }
            }
        }
    }
}


/// # LaTeX output
///
/// LaTex output is given in terms of three macros
/// `\meshline{R,G,B}{x1}{y1}{x2}{y2}`,
/// `\meshpoint{R,G,B}{point number}{x}{y}`,
/// `\meshtriangle{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}`, and
/// `\meshquadrilateral{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}{x4}{y4}` to
/// respectively plot edges, points, (filled) triangles, and (filled)
/// quadrilaterals.  You can easily customize them in your LaTeX file.
/// If you do not provide your own implementations, default ones will
/// be used.  The LaTeX package `tikz` is required.
///
/// # Example
/// ```
/// use mesh2d::Mesh;
/// use rgb::{RGB, RGB8};
/// const RED: RGB8 = RGB {r: 255, g: 0, b: 0};
/// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
/// // Assume `mesh` is a value of a type implementing `Mesh`
/// // and that, for this example, `mesh.n_points()` is 4.
/// let levels = [(1.5, RED)];
/// mesh.latex().sub_levels(&[1., 2., 3., 4.], levels).save("/tmp/foo")?;
/// # Ok(()) }
/// ```
///
/// [tikz]: https://sourceforge.net/projects/pgf/
impl<'a, M> LaTeX<'a, M>
where M: Mesh {
    /// Specify the color of edges inside the domain.
    /// A value of `None` says that inside edges will not be drawn.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Mesh;
    /// use rgb::{RGB, RGB8};
    /// const GREY: RGB8 = RGB {r: 150, g: 150, b: 150};
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.latex().color(Some(GREY));
    /// # Ok(()) }
    /// ```
    pub fn color(self, color: Option<RGB8>) -> Self {
        LaTeX { color: Some(color), .. self }
    }

    /// Specify the color of edges on the boundary of the domain.
    /// If `None`, the boundary will not be drawn.
    pub fn boundary_color(self, color: Option<RGB8>) -> Self {
        LaTeX { boundary_color: Some(color), .. self }
    }

    meth_levels!(/// Specify that one wants to draw the level curves of `z`.
        level_curves, Levels, do_not_sort_levels);
    meth_levels!(
        /// Specify that one wants to draw the strict super-levels
        /// `levels` of `z`.
        super_levels, SuperLevels, sort_levels_incr);
    meth_levels!(
        /// Specify that one wants to draw the strict sub-levels
        /// `levels` of `z`.
        sub_levels, SubLevels, sort_levels_decr);

    /// Write the mesh `self` to the writer `w`.
    fn write_mesh<W>(&self, w: &mut W,
                     default_color: Option<RGB8>) -> io::Result<()>
    where W: Write {
        let m = self.mesh;
        // Write points
        write!(w, "  % {} points\n", m.n_points())?;
        for i in 0 .. m.n_points() { point(w, BLACK, i, m.point(i))? }
        // Write lines
        let color = self.color.unwrap_or(default_color);
        let brdy_color = self.boundary_color.unwrap_or(Some(BLACK));
        write!(w, "  % {} triangles\n", m.n_triangles())?;
        for (&(p1, p2), &cnt) in self.edges.0.iter() {
            if cnt == 0 /* inside */ {
                if let Some(c) = color {
                    line(w, c, m.point(p1), m.point(p2))?;
                }
            } else {
                if let Some(c) = brdy_color {
                    line(w, c, m.point(p1), m.point(p2))?;
                }
            }
        }
        Ok(())
    }

    fn write_levels<W>(&self, w: &mut W, z: &dyn P1) -> io::Result<()>
    where W: Write {
        write!(w, "{}", LATEX_BEGIN)?;
        let m = self.mesh;
        self.write_mesh(w, None)?;
        let e = Edges::new(m);
        // Write level lines for each triangle.
        for t in 0 .. m.n_triangles() {
            let (i1, i2, i3) = m.triangle(t);
            let p1 = m.point(i1);
            let z1 = z.index(i1);
            let p2 = m.point(i2);
            let z2 = z.index(i2);
            let p3 = m.point(i3);
            let z3 = z.index(i3);
            for &(l, color) in &self.levels {
                // Draw the level curve [l] on the triangle [t] except
                // if that curve is on the boundary.
                if level_eq(l, z1) {
                    if level_eq(l, z2) {
                        if level_eq(l, z3) {
                            // The entire triangle is at the same
                            // level.  Try to remove boundary edges.
                            if e.on_boundary(i1, i2) {
                                if e.on_boundary(i1, i3)
                                    || e.on_boundary(i2, i3) {
                                        triangle(w, color, p1, p2, p3)?
                                    }
                                else {
                                    line(w, color, p3, mid(p1, p2))?
                                }
                            } else { // i1-i2 not on boundary
                                if e.on_boundary(i1, i3) {
                                    if e.on_boundary(i2, i3) {
                                        triangle(w, color, p1, p2, p3)?
                                    } else {
                                        line(w, color, p2, mid(p1, p3))?
                                    }
                                } else { // i1-i3 not on boundary
                                    if e.on_boundary(i2, i3) {
                                        line(w, color, p1, mid(p2, p3))?
                                    } else {
                                        triangle(w, color, p1, p2, p3)?
                                    }
                                }
                            }
                        } else { // l = z1 = z2 ≠ z3
                            if ! e.on_boundary(i1, i2) {
                                line(w, color, p1, p2)?
                            }
                        }
                    } else { // l = z1 ≠ z2
                        if level_eq(l, z3) { // l = z1 = z3 ≠ z2
                            if ! e.on_boundary(i1, i3) {
                                line(w, color, p1, p3)?
                            } else {
                                if (z2 < l && l < z3) || (z3 < l && l < z2) {
                                line(w, color, p1,
                                     intercept(p2, z2, p3, z3, l))?
                                }
                            }
                        }
                    }
                } else if l < z1 {
                    if level_eq(l, z2) {
                        if level_eq(l, z3) { // l = z2 = z3 < z1
                            if ! e.on_boundary(i2, i3) {
                                line(w, color, p2, p3)?
                            }
                        } else if l > z3 { // z3 < l = z2 < z1
                            line(w, color, p2, intercept(p1,z1, p3,z3, l))?
                        } else { // l = z2 < min{z1, z3}
                            // Corner point, inside the domain.
                            // Ususally this happens because the level
                            // line passes through a triangle corner.
                            point(w, color, i2, p2)?
                        }
                    } else if l < z2 {
                        if level_eq(l, z3) { // l = z3 < min{z1, z2}
                            point(w, color, i3, p3)?
                        } else if l > z3 { // z3 < l < min{z1, z2}
                            line(w, color, intercept(p1,z1, p3,z3, l),
                                 intercept(p2,z2, p3,z3, l))?
                        }
                    } else { // z2 < l < z1
                        line(w, color, intercept(p1,z1, p2,z2, l),
                             if level_eq(l, z3) { p3 }
                             else if l < z3 { intercept(p2,z2, p3,z3, l) }
                             else { intercept(p1,z1, p3,z3, l) })?
                    }
                } else { // l > z1
                    // Symmetric of `l < z1` with all inequalities reversed
                    if level_eq(l, z2) {
                        if level_eq(l, z3) {
                            if ! e.on_boundary(i2, i3) {
                                line(w, color, p2, p3)?
                            }
                        } else if l < z3 { // z1 < l = z2 < z3
                            line(w, color, p2, intercept(p1,z1, p3,z3, l))?
                        } else { // Corner point, inside the domain
                            point(w, color, i2, p2)?
                        }
                    } else if l > z2 {
                        if level_eq(l, z3) {
                            point(w, color, i3, p3)?
                        } else if l < z3 {
                            line(w, color, intercept(p1,z1, p3,z3, l),
                                 intercept(p2,z2, p3,z3, l))?
                        }
                    } else { // z1 < l < z2
                        line(w, color, intercept(p1,z1, p2,z2, l),
                             if level_eq(l, z3) { p3 }
                             else if l > z3 { intercept(p2,z2, p3,z3, l) }
                             else { intercept(p1,z1, p3,z3, l) })?
                    }
                }
            }
        }
        write!(w, "{}", LATEX_END)
    }

    fn write_superlevels<W>(&self, w: &mut W, z: &dyn P1) -> io::Result<()>
    where W: Write {
        write!(w, "{}", LATEX_BEGIN)?;
        let m = self.mesh;
        write_xxx_levels!(w, m, z, self.levels, keep_order);
        // Write the mesh (by default, only the boundary) on top of
        // the filled region.
        self.write_mesh(w, None)?;
        write!(w, "{}", LATEX_END)
    }

    fn write_sublevels<W>(&self, w: &mut W, z: &dyn P1) -> io::Result<()>
    where W: Write {
        write!(w, "{}", LATEX_BEGIN)?;
        let m = self.mesh;
        write_xxx_levels!(w, m, z, self.levels, reverse_order);
        self.write_mesh(w, None)?;
        write!(w, "{}", LATEX_END)
    }

    /// Write the mesh or the levels/superlevels/sublevels to `w`.
    pub fn write<W>(&self, w: &mut W) -> Result<(), io::Error>
    where W: Write {
        match &self.action {
            Action::Mesh => {
                write!(w, "{}", LATEX_BEGIN)?;
                self.write_mesh(w, Some(GREY))?;
                write!(w, "{}", LATEX_END)
            }
            Action::Levels(u) => {
                self.write_levels(w, &**u)
            }
            Action::SuperLevels(u) => {
                self.write_superlevels(w, &**u)
            }
            Action::SubLevels(u) => {
                self.write_sublevels(w, &**u)
            }
        }
    }

    /// Same as [`write`] except that it writes to the given file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut f = File::create(path)?;
        self.write(&mut f)?;
        f.flush()
    }
}

// We need to put paths in a scope otherwise one gets "TeX
// capacity exceeded".
const LATEX_BEGIN: &str =
    r#"\begin{pgfscope}
  % Written by the Rust "mesh2d" crate.
  % \meshline{R,G,B}{x1}{y1}{x2}{y2}
  \providecommand{\meshline}[5]{
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfusepathqstroke
    \end{pgfscope}}
  % \meshpoint{R,G,B}{point number}{x}{y}
  \providecommand{\meshpoint}[4]{}
  % \meshtriangle{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}
  \providecommand{\meshtriangle}[7]{
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfpathlineto{\pgfpointxy{#6}{#7}}
      \pgfusepathqfillstroke
    \end{pgfscope}}
  % \meshquadrilateral{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}{x4}{y4}
  \providecommand{\meshquadrilateral}[9]{
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfpathlineto{\pgfpointxy{#6}{#7}}
      \pgfpathlineto{\pgfpointxy{#8}{#9}}
      \pgfusepathqfillstroke
    \end{pgfscope}}
"#;

const LATEX_END: &str = "\\end{pgfscope}\n";

////////////////////////////////////////////////////////////////////////
//
// Scilab Output

/// The mode of drawing for [`Scilab`].
#[derive(Debug, Clone, Copy)]
pub enum Mode {
    /// Draw the triangles on the surface.
    Triangles,
    /// Only draw the triangles (no surface color).
    TrianglesOnly,
    /// Only draw the colored surface.
    NoTriangles
}

/// How to draw the box around the plot for [`Scilab`].
#[derive(Debug, Clone, Copy)]
pub enum DrawBox {
    None,
    Behind,
    BoxOnly,
    Full
}

/// Scilab Output.  Created by [`Mesh::scilab`].
pub struct Scilab<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    z: &'a dyn P1,
    longitude: f64,
    azimuth: f64,
    mode: Mode,
    draw_box: DrawBox,
    edge_color: Option<RGB8>,
}

impl<'a, M> Scilab<'a, M>
where M: Mesh {
    /// Set sets the longitude to `l` degrees of the observation point.
    pub fn longitude(self, l: f64) -> Self {
        Scilab { longitude: l, .. self }
    }

    /// Sets the azimuth to `a` degrees of the observation point.
    pub fn azimuth(self, a: f64) -> Self {
        Scilab { azimuth: a, .. self }
    }

    /// Set the mode for the drawing.
    pub fn mode(self, mode: Mode) -> Self { Scilab { mode, .. self } }

    /// Specify how what the box arounf the plot should look like.
    pub fn draw_box(self, draw_box: DrawBox) -> Self {
        Scilab { draw_box, .. self }
    }

    /// Color the edges using `color`.
    pub fn edge(self, color: RGB8) -> Self {
        Scilab { edge_color: Some(color), .. self }
    }

    /// Saves the mesh data and the function values `z` on the mesh
    /// (see [`Mesh::scilab`]) so that when Scilab runs the created
    /// `path`.sci script, the graph of the function is drawn.  Note
    /// that this method also creates `path`.x.dat, `path`.y.dat, and
    /// `path`.z.dat to hold Scilab matrices.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let path = path.as_ref();
        let sci = path.with_extension("sci");
        let xf = path.with_extension ("x.dat");
        let yf = path.with_extension("y.dat");
        let zf = path.with_extension("z.dat");
        let mode = match self.mode {
            Mode::Triangles => 1,
            Mode::TrianglesOnly => 0,
            Mode::NoTriangles => -1 };
        let draw_box = match self.draw_box {
            DrawBox::None => 0,
            DrawBox::Behind => 2,
            DrawBox::BoxOnly => 3,
            DrawBox::Full => 4 };
        let mut f = File::create(&sci)?;
        let filename = |f: &'a PathBuf| -> &'a str {
            f.file_name().map(std::ffi::OsStr::to_str).flatten()
                .expect("mesh2d::Scilab::save: non UTF-8 file name") };
        write!(f, "mode(0);\n\
                   // Run in Scilab with: exec('{}')\n\
                   // Written by the Rust mesh2d crate.\n\
                   rust = struct('f', scf(), 'e', null, \
                   'x', fscanfMat('{}'), 'y', fscanfMat('{}'), \
                   'z', fscanfMat('{}'));\n\
                   clf();\n\
                   rust.e = gce();\n\
                   rust.e.hiddencolor = -1;\n\
                   rust.f.color_map = jetcolormap(100);\n",
               sci.display(), filename(&xf), filename(&yf), filename(&zf))?;
        match self.edge_color {
            Some(c) if mode >= 0 => {
                write!(f, "rust.f.color_map(1,:) = [{}, {}, {}];\n\
                           xset('color', 1);\n",
                       c.r as f64 / 255.,  c.g as f64 / 255.,
                       c.b as f64 / 255.)?;
            }
            _ => ()
        }
        write!(f, "plot3d1(rust.x, rust.y, rust.z, theta={}, alpha={}, \
                   flag=[{},2,{}]);\n\
                   disp('Save: xs2pdf(rust.f, ''{}.pdf'')');\n",
               self.longitude, self.azimuth, mode, draw_box,
               filename(&sci))?;
        f.flush()?;
        macro_rules! save_mat {
            ($fn: ident, $m: expr, $coord: expr) => {
                let mut f = File::create($fn)?;
                for t in 0 .. $m.n_triangles() {
                    write!(f, "{:16e} ", $coord($m.triangle(t).0))?;
                }
                write!(f, "\n")?;
                for t in 0 .. $m.n_triangles() {
                    write!(f, "{:16e} ", $coord($m.triangle(t).1))?;
                }
                write!(f, "\n")?;
                for t in 0 .. $m.n_triangles() {
                    write!(f, "{:16e} ", $coord($m.triangle(t).2))?;
                }
                write!(f, "\n")?;
                f.flush()?;
            }
        }
        save_mat!(xf, self.mesh, |i| self.mesh.point(i).0);
        save_mat!(yf, self.mesh, |i| self.mesh.point(i).1);
        save_mat!(zf, self.mesh, |i| self.z.index(i));
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////
//
// Matlab Output

/// Specification of the color of the edges for [`Matlab`] output.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum EdgeColor {
    None,
    Flat,
    Interp,
    Color(RGB8),
}

impl Display for EdgeColor {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::None => write!(f, "'none'"),
            Self::Flat => write!(f, "'flat'"),
            Self::Interp => write!(f, "'interp'"),
            Self::Color(c) =>
                write!(f, "[{} {} {}]", c.r as f64 / 255.,
                       c.g as f64 / 255., c.b as f64 / 255.),
        }
    }
}

/// Specifies the Matlab line style.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDotted,
    None,
}

impl Display for LineStyle {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            LineStyle::Solid => write!(f, "'-'"),
            LineStyle::Dashed => write!(f, "'--'"),
            LineStyle::Dotted => write!(f, "':'"),
            LineStyle::DashDotted => write!(f, "'-.'"),
            LineStyle::None => write!(f, "'none'"),
        }
    }
}

/// Matlab Output.  Created by [`Mesh::matlab`].
pub struct Matlab<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    z: &'a dyn P1,
    edge_color: EdgeColor,
    line_style: LineStyle,
    face_alpha: f64,
}

impl<'a, M> Matlab<'a, M>
where M: Mesh {
    /// Saves the mesh data and the function values `z` on the mesh
    /// (see [`Mesh::matlab`]) so that when Matlab runs the
    /// created `path`.m script, the graph of the function is drawn.
    /// It is mandated by Matlab that the filename should contain only
    /// alphanumeric characters. Other characters in the filename will
    /// be replaced by underscore.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let path = path.as_ref();
        let sanitize_char = |c| {
            match c {'0' ..= '9' | 'a' ..= 'z' | 'A' ..= 'Z' => c,
                     _ => '_' }};
        let fname: String = path.with_extension("").file_name()
            .expect("mesh2d::Matlab::save: a filename is required.")
            .to_str().unwrap_or("rust_matlab")
            .chars().map(sanitize_char).collect();
        let mat = path.with_file_name(fname).with_extension("m");
        let pdf = mat.clone().with_extension("pdf");
        let mut f = File::create(&mat)?;
        let m = self.mesh;
        write!(f, "%% Run in Matlab with: run {}\n\
                   %% Created by the Rust mesh2d crate.\n\
                   %% print({:?}, \"-dpdf\")\n",
               mat.display(), pdf.file_name().unwrap())?;
        write!(f, "mesh_x = [")?;
        for i in 0 .. m.n_points() {
            write!(f, "{:.16e} ", m.point(i).0)?; }
        write!(f, "];\nmesh_y = [")?;
        for i in 0 .. m.n_points() {
            write!(f, "{:.16e} ", m.point(i).1)?; }
        write!(f, "];\nmesh_z = [")?;
        for i in 0 .. m.n_points() {
            write!(f, "{:.16e} ", self.z.index(i))?; }
        write!(f, "];\nmesh_triangles = [")?;
        for t in 0 .. m.n_triangles() {
            let (p1, p2, p3) = m.triangle(t);
            // Matlab uses indexing from 1.
            write!(f, "{} {} {}; ", p1 + 1, p2 + 1, p3 + 1)?; }
        write!(f, "];\ntrisurf(mesh_triangles, mesh_x, mesh_y, mesh_z, \
                   'FaceAlpha', {}, 'EdgeColor', {}, 'LineStyle', {});\n",
               self.face_alpha.clamp(0., 1.), self.edge_color,
               self.line_style)
    }
}

////////////////////////////////////////////////////////////////////////
//
// Matplotlib Output

/// Matplotlib Output.  Created by [`Mesh::matplotlib`].
pub struct Matplotlib<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    z: &'a dyn P1,
}

impl<'a, M> Matplotlib<'a, M>
where M: Mesh {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let py = path.as_ref().with_extension("py");
        let mut f = File::create(&py)?;
        let m = self.mesh;
        write!(f, "#!/usr/bin/env python3\n\
                   # -*- coding:utf-8 -*-\n\
                   import matplotlib.pyplot\n\
                   class Mesh:\n    \
                     \"\"\"Created by the Rust mesh2d crate\"\"\"\n    \
                     def __init__(self):\n        \
                       self.x = [")?;
        for i in 0 .. m.n_points() {
            write!(f, "{:.16e}, ", m.point(i).0)?; }
        write!(f, "]\n        self.y = [")?;
        for i in 0 .. m.n_points() {
            write!(f, "{:.16e}, ", m.point(i).1)?; }
        write!(f, "]\n        self.z = [")?;
        for i in 0 .. m.n_points() {
            write!(f, "{:.16e}, ", self.z.index(i))?; }
        // https://matplotlib.org/stable/api/tri_api.html
        write!(f, "]\n        self.triangles = [")?;
        for t in 0 .. m.n_triangles() {
            let (p0, p1, p2) = m.triangle(t);
            let (x0, y0) = m.point(p0);
            let (x1, y1) = m.point(p1);
            let (x2, y2) = m.point(p2);
            let dx1 = x1 - x0;  let dy1 = y1 - y0;
            let dx2 = x2 - x0;  let dy2 = y2 - y0;
            let e1 = - dx2 * dx1 - dy2 * dy1;
            let e2 = dx2 * dy1 - dy2 * dx1;
            if e2.atan2(e1) >= 0. {
                write!(f, "[{}, {}, {}], ", p0, p1, p2)?;
            } else {
                write!(f, "[{}, {}, {}], ", p0, p2, p1)?;
            }
        }
        write!(f, "]\n\n    \
                   def plot(self, linewidth=0.2, cmap='viridis'):\n        \
                   ax = matplotlib.pyplot.figure()\
                   .add_subplot(projection='3d')\n        \
                   ax.plot_trisurf(self.x, self.y, self.z, \
                   triangles=self.triangles, antialiased=True, cmap=cmap, \
                   linewidth=linewidth)\n\
                   \n    \
                   def mesh(self):\n        \
                   ax = matplotlib.pyplot.figure().add_subplot()\n        \
                   ax.triplot(self.x, self. y, self.triangles, lw=1.0)\
                   \n\n\
                   if __name__ == \"__main__\":\n    \
                   import sys\n    \
                   if len(sys.argv) >= 2 \
                   and sys.argv[1] == '--mesh':\n        \
                   Mesh().mesh()\n    \
                   else:\n        \
                   Mesh().plot()\n    \
                   matplotlib.pyplot.show()")
    }
}


////////////////////////////////////////////////////////////////////////
//
// Mathematica Output

/// Mathematica Output.  Created by [`Mesh::mathematica`].
pub struct Mathematica<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    z: &'a dyn P1,
}

fn write_f64(f: &mut File, x: f64) -> Result<(), io::Error> {
    let x = format!("{:.16e}", x);
    match x.find('e') {
        None => write!(f, "{}", x.trim_end_matches('0')),
        Some(e) => {
            write!(f, "{}", &x[0..e].trim_end_matches('0'))?;
            let exp = &x[e+1 ..];
            if exp != "0" { write!(f, "*^{}", exp)?; }
            Ok(())
        }
    }
}

fn write_3d_point(f: &mut File,
                  (x, y): (f64, f64), z: f64) -> Result<(), io::Error> {
    write!(f, "{{")?; write_f64(f, x)?;
    write!(f, ",")?;  write_f64(f, y)?;
    write!(f, ",")?;  write_f64(f, z)?;  write!(f, "}}")
}

impl<'a, M> Mathematica<'a, M>
where M: Mesh {
    /// Saves the mesh data and the function values `z` on the mesh
    /// (see [`Mesh::mathematica`]) so that when Mathematica runs
    /// the created `path`.m script, the graph of the function is drawn.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let path = path.as_ref();
        let allowed_char = |c: &char| {
            match c {'0' ..= '9' | 'a' ..= 'z' | 'A' ..= 'Z' => true,
                     _ => false }};
        let fname: String = path.with_extension("").file_name()
            .expect("mesh2d::Mathematica::save: a filename is required.")
            .to_str().unwrap_or("RustMathematica")
            .chars().filter(allowed_char).collect();
        let pkg =
            if fname.is_empty() { "RustMathematica".to_string() }
            else { let mut p = fname.chars();
                   p.next().unwrap().to_uppercase().chain(p).collect() };
        let mat = path.with_file_name(fname).with_extension("m");
        let mut f = File::create(&mat)?;
        let m = self.mesh;
        write!(f, "(* Created by the Rust mesh2d crate. *)\n")?;
        if m.n_points() == 0 {
            write!(f, "\"No points in the mesh!\"\n")?;
            return Ok(())
        }
        write!(f, "{}`xyz = {{", pkg)?;
        write_3d_point(&mut f, m.point(0), self.z.index(0))?;
        for i in 1 .. m.n_points() {
            write!(f, ", ")?;
            write_3d_point(&mut f, m.point(i), self.z.index(i))?;
        }
        write!(f, "}};\n\
                   {}`triangles = {{", pkg)?;
        // Mathematica indices start at 1.
        let (i1, i2, i3) = m.triangle(0);
        write!(f, "{{{},{},{}}}", i1+1, i2+1, i3+1)?;
        for t in 1 .. m.n_triangles() {
            let (i1, i2, i3) = m.triangle(t);
            write!(f, ", {{{},{},{}}}", i1+1, i2+1, i3+1)?;
        }
        write!(f, "}};\n\
                   Graphics3D[GraphicsComplex[{0}`xyz, \
                   Polygon[{0}`triangles]]]\n", pkg)
    }
}



#[cfg(test)]
mod tests {

    #[test]
    fn cuthill_mckee() {
        use super::*;

        #[derive(Debug, PartialEq, Eq, Clone)]
        struct M {
            n_points: usize,
            triangles: Vec<(usize, usize, usize)>,
        }
        impl Mesh for M {
            fn n_points(&self) -> usize { self.n_points }
            fn point(&self, _: usize) -> (f64, f64) { todo!() }
            fn n_triangles(&self) -> usize { self.triangles.len() }
            fn triangle(&self, i: usize) -> (usize, usize, usize) {
                self.triangles[i]
            }
        }
        impl Permutable for M {
            fn permute_points(&mut self, p: &Permutation) {
                for t in &mut self.triangles {
                    *t = sort3((p[t.0], p[t.1], p[t.2]))
                }
            }
        }
        let mut before = M {
            n_points: 8,
            triangles: vec![(1,5,7), (1,2,7), (0,2,4), (3,6,7), (5,6,7)]
        };
        let mut after = M {
            n_points: 8,
            triangles: vec![(2,3,4), (3,4,5), (5,6,7), (0,1,3), (0,2,3)]
        };
        let p = before.clone().cuthill_mckee();
        assert_eq!(p, Permutation::new([0, 3, 2, 6, 1, 5, 7, 4]).unwrap());
        let p = before.cuthill_mckee_rev();
        assert_eq!(before.band_height_p1(), after.band_height_p1());
        assert_eq!(p, Permutation::new([7, 4, 5, 1, 6, 2, 0, 3]).unwrap());
        before.triangles.sort();
        after.triangles.sort();
        assert_eq!(before, after);
    }
}
