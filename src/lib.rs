//! Generic mesh structure to be used with various meshers.  Also
//! provide some functions to help to display and design geometries.

use std::{collections::{VecDeque, HashSet, HashMap},
          fmt::{self, Display, Formatter},
          fs::File,
          io::{self, Write},
          ops::Index,
          path::{Path, PathBuf}};
use rgb::{RGB, RGB8};

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

macro_rules! default_mesh_color {
    ($m: expr) => { |i| { if $m.edge_marker(i) == 0 { Some(GREY) }
                          else { Some(BLACK) } }}}

macro_rules! default_level_color {
    ($s: ident) => { |i| { if $s.mesh.edge_marker(i) != 0 { Some(BLACK) }
                           else { None } }}}

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

/// Trait describing the base methods a mesh must have.
pub trait MeshBase {
    /// Return the number of points in the PSLG.
    fn n_points(&self) -> usize;
    /// Return the coordinates (x,y) of the point of index `i` (where
    /// it is assumed that 0 ≤ `i` < `n_points()`).  The coordinates
    /// must be finite numbers.
    fn point(&self, i: usize) -> (f64, f64);
    /// Number of triangles in the mesh.
    fn n_triangles(&self) -> usize;
    /// The 3 corners (p₁, p₂, p₃) of the triangle `i`:
    /// 0 ≤ pₖ < `n_points()` are the indices of the corresponding
    /// points.  We **require** that p₁ ≤ p₂ ≤ p₃.
    fn triangle(&self, i: usize) -> (usize, usize, usize);

    /// Return the bounding box enclosing the PSLG.  If the PSLG is
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
            let (i1, i2, i3) = self.triangle(i);
            // No absolute values as it is required that i3 ≥ i2 ≥ i1
            kd = kd.max(i3 - i2).max(i3 - i1).max(i2 - i1);
        }
        kd + 1
    }

    /// Same as [`band_height_p1`][MeshBase::band_height_p1] except that
    /// it only consider the nodes `i` such that `predicate(i)` is `true`.
    fn band_height_p1_filter<P>(&self, mut predicate: P) -> usize
    where P: FnMut(usize) -> bool {
        let mut kd = 0_usize;
        for i in 0 .. self.n_triangles() {
            let (i1, i2, i3) = self.triangle(i);
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
    /// The value of the function at the point [`point(i)`][MeshBase::point]
    /// is given by `z[i]`.
    ///
    /// # Example
    /// ```
    /// use mesh2d::Mesh;
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.scilab([1., 2., 3., 4.]).save("/tmp/foo");
    /// # Ok(()) }
    /// ```
    fn scilab<'a, Z>(&'a self, z: Z) -> Scilab<'a, Self>
    where Z: P1 + 'a {
        if z.len() != self.n_points() {
            panic!("mesh2d::MeshBase::scilab: z.len() = {} but expected {}",
                   z.len(), self.n_points());
        }
        Scilab { mesh: self, z: Box::new(z),
                 longitude: 70.,  azimuth: 60.,
                 mode: Mode::Triangles,
                 draw_box: DrawBox::Full,
                 edge_color: None }
    }

}

/// Trait describing various characteristics of a mesh.
pub trait Mesh: MeshBase {
    /// Return the number of edges in the mesh.
    fn n_edges(&self) -> usize;
    /// Return (p₁, p₂) the point indices of the enpoints of edge `i`.
    /// We **require** that p₁ ≤ p₂.
    fn edge(&self, i: usize) -> (usize, usize);
    /// Return the marker of the edge `i` where 0 ≤ `i` < `n_edges()`.
    /// By convention, edges inside the domain receive the marker `0`.
    fn edge_marker(&self, i: usize) -> i32;

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
    /// mesh.latex().sub_levels([1., 2., 3., 4.], levels).save("/tmp/foo")?;
    /// # Ok(()) }
    /// ```
    fn latex<'a>(&'a self) -> LaTeX<'a, Self> {
        LaTeX { mesh: &self,  edge_color: None,
                action: Action::Mesh, levels: vec![] }
    }


}

/// Structure to promote a [`MeshBase`] to a [`Mesh`].
pub struct Mesh2D<B> where B: MeshBase {
    mesh: B,
    edge: Vec<(usize, usize)>,
    edge_marker: Vec<i8>,
}

impl<B: MeshBase> MeshBase for Mesh2D<B> {
    fn n_points(&self) -> usize { self.mesh.n_points() }
    fn point(&self, i: usize) -> (f64, f64) { self.mesh.point(i) }
    fn n_triangles(&self) -> usize { self.mesh.n_triangles() }
    fn triangle(&self, i: usize) -> (usize, usize, usize) {
        self.mesh.triangle(i)
    }
}

impl<B: MeshBase> Mesh for Mesh2D<B> {
    fn n_edges(&self) -> usize { self.edge.len() }
    fn edge(&self, i: usize) -> (usize, usize) { self.edge[i] }
    fn edge_marker(&self, i: usize) -> i32 { self.edge_marker[i] as i32 }
}


impl<B: MeshBase> Mesh2D<B> {
    pub fn new(b: B) -> Self {
        let mut e = HashMap::new();
        for t in 0 .. b.n_triangles() {
            // p1 ≤ p2 ≤ p3 required by the specification.
            let (p1, p2, p3) = b.triangle(t);
            let cnt = e.entry((p1, p2)).or_insert(-1);
            *cnt += 1;
            let cnt = e.entry((p1, p3)).or_insert(-1);
            *cnt += 1;
            let cnt = e.entry((p2, p3)).or_insert(-1);
            *cnt += 1;
        }
        let n = e.len();
        let mut edge = Vec::with_capacity(n);
        let mut edge_marker = Vec::with_capacity(n);
        for ((p1, p2), cnt) in e.drain() {
            edge.push((p1, p2));
            if cnt > 1 { panic!("mesh2d::Mesh2D::new: an edge cannot be part \
                                 od more than 2 triangles.") }
            edge_marker.push(cnt);
        }
        Mesh2D { mesh: b,  edge,  edge_marker }
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
        let mut nbh: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0 .. $m.n_edges() {
            let (p1, p2) = $m.edge(i);
            nbh[p1].push(p2);
            deg[p1] += 1;
            nbh[p2].push(p1);
            deg[p2] += 1;
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

/// Meshes that accepting permutations of points and triangles indices.
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
    /// [`band_height_p1`][MeshBase::band_height_p1]).  Return the
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

#[cfg(ndarray)]
impl P1 for ndarray::Array1<f64> {
    fn len(&self) -> usize { todo!() }
    fn index(&self, i: usize) -> f64 { todo!() }
}

////////////////////////////////////////////////////////////////////////
//
// LaTeX Output

enum Action<'a> {
    Mesh,
    Levels(Box<dyn P1 + 'a>),
    SuperLevels(Box<dyn P1 + 'a>), // levels in increasing order
    SubLevels(Box<dyn P1 + 'a>) // levels in decreasing order
}

/// LaTeX output.  Created by [`Mesh::latex`].
pub struct LaTeX<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    // The "dyn" option was chosen because it avoids having a type
    // parameter for the closure passed on.  This type may confuse the user.
    edge_color: Option<Box<dyn Fn(usize) -> Option<RGB8> + 'a>>,
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
        pub fn $meth<Z, L>(self, z: Z, levels: L) -> LaTeX<'a, M>
        where Z: P1 + 'a,
              L: IntoIterator<Item=(f64, RGB8)> {
            if z.len() != self.mesh.n_points() {
                panic!("mesh2d::LaTeX::{}: z.len() == {}, expected {}",
                       stringify!($meth), z.len(), self.mesh.n_points()) }
            let levels = $sort!(sanitize_levels(levels));
            LaTeX { action: Action::$act(Box::new(z)), levels, .. self }
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

fn fill_triangle<W: Write>(w: &mut W, c: RGB8,
                           (x1,y1): (f64,f64), (x2,y2): (f64,f64),
                           (x3,y3): (f64,f64)) -> io::Result<()> {
    write!(w, "  \\meshfilltriangle{{{},{},{}}}{{{:.12}}}{{{:.12}}}\
               {{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}\n",
           c.r, c.g, c.b, x1, y1, x2, y2, x3, y3)
}

fn fill_quadrilateral<W: Write>(
    w: &mut W, c: RGB8, (x1,y1): (f64,f64), (x2,y2): (f64,f64),
    (x3,y3): (f64,f64), (x4,y4): (f64,f64)) -> io::Result<()> {
    write!(w, "  \\meshfillquadrilateral{{{},{},{}}}{{{:.12}}}{{{:.12}}}\
               {{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}\n",
           c.r, c.g, c.b, x1, y1, x2, y2, x3, y3, x4, y4)
}

/// Levels considered equal (to draw level curves).
// FIXME: need to make it customizable?
fn level_eq(l1: f64, l2: f64) -> bool {
    (l1 - l2).abs() <= 1e-8 * (l1.abs() + l2.abs())
}

/// Hold the boundary edges (to draw level curves).
type BdryEdges = HashSet<(usize, usize)>;

fn boundary_edges(m: &impl Mesh) -> BdryEdges {
    let mut bdry = HashSet::new();
    for i in 0 .. m.n_edges() {
        // (i1, i2) = m.edge(i) ⇒ i1 ≤ i2
        if m.edge_marker(i) != 0 { bdry.insert(m.edge(i)); }
    }
    bdry
}

fn on_boundary(bdry: &BdryEdges, i1: usize, i2: usize) -> bool {
    bdry.contains(&if i1 <= i2 { (i1, i2) } else { (i2, i1) })
}

macro_rules! keep_order { ($o: expr) => { $o } }
macro_rules! reverse_order { ($o: expr) => { $o.reverse() } }

/// Designed for super-levels, reverse the ordering for sub-levels.
macro_rules! write_xxx_levels {
    ($w: ident, $m: ident, $z: ident, $levels: expr, $rev: ident) => {
        for t in 0 .. $m.n_triangles() {
            let (i1, i2, i3) = $m.triangle(t);
            let p1 = $m.point(i1);
            let z1 = $z.index(i1);
            let p2 = $m.point(i2);
            let z2 = $z.index(i2);
            let p3 = $m.point(i3);
            let z3 = $z.index(i3);
            for &(l, color) in &$levels {
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
                            fill_triangle($w, color, p1, p2, p3)?
                        }
                    (Greater, Greater, Less) => { // Cut edges 1-3, 3-2
                        fill_quadrilateral($w, color, p2, p1,
                                           intercept(p1,z1, p3,z3, l),
                                           intercept(p3,z3, p2,z2, l))?
                    }
                    (Greater, Less, Greater) => { // Cut edges 3-2, 2-1
                        fill_quadrilateral($w, color, p1, p3,
                                           intercept(p3,z3, p2,z2, l),
                                           intercept(p2,z2, p1,z1, l))?
                    }
                    (Less, Greater, Greater) => { // Cut edges 2-1, 1-3
                        fill_quadrilateral($w, color, p3, p2,
                                           intercept(p2,z2, p1,z1, l),
                                           intercept(p1,z1, p3,z3, l))?
                    }
                    // (Greater, Equal, Equal) rightly matched before.
                    (Greater, Equal | Less, Equal | Less) => {
                        fill_triangle($w, color, p1,
                                      intercept(p1,z1, p2,z2, l),
                                      intercept(p1,z1, p3,z3, l))?
                    }
                    (Equal | Less, Greater, Equal | Less) => {
                        fill_triangle($w, color, p2,
                                      intercept(p2,z2, p1,z1, l),
                                      intercept(p2,z2, p3,z3, l))?
                    }
                    (Equal | Less, Equal | Less, Greater) => {
                        fill_triangle($w, color, p3,
                                      intercept(p3,z3, p1,z1, l),
                                      intercept(p3,z3, p2,z2, l))?
                    }
                    // Nothing to color
                    (Equal | Less, Equal | Less, Equal | Less) => {}
                }
            }
        }
        write!($w, "{}", LATEX_END)?;
    }
}


/// # LaTeX output
///
/// LaTex output is given in terms of three macros
/// `\meshline{R,G,B}{x1}{y1}{x2}{y2}`,
/// `\meshpoint{R,G,B}{point number}{x}{y}`, and
/// `\meshtriangle{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}` to respectively
/// plot edges, points and (filled) triangles.  You can easily
/// customize them in your LaTeX file.  If you do not provide your own
/// implementations, default ones will be used.  The LaTeX package
/// `tikz` is required.
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
/// mesh.latex().sub_levels([1., 2., 3., 4.], levels).save("/tmp/foo")?;
/// # Ok(()) }
/// ```
///
/// [tikz]: https://sourceforge.net/projects/pgf/
impl<'a, M> LaTeX<'a, M>
where M: Mesh {
    /// Use the function `edge_color` to color the edges.
    /// The return value of `edge_color(i)` specifies the color of the
    /// edge numbered `i`.  If `edge_color(i)` returns `None`, the
    /// edge is not drawn.
    ///
    /// # Example
    /// To draw only the boundary of the mesh (the default when drawing
    /// level cuves or sets), use the following.
    /// ```
    /// use mesh2d::Mesh;
    /// use rgb::{RGB, RGB8};
    /// const BLACK: RGB8 = RGB {r: 0, g: 0, b: 0};
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.latex().edge(|i| if mesh.edge_marker(i) != 0 { Some(BLACK) }
    ///                       else { None });
    /// # Ok(()) }
    /// ```
    pub fn edge<E>(self, edge_color: E) -> Self
    where E: Fn(usize) -> Option<RGB8> + 'a {
        LaTeX { edge_color: Some(Box::new(edge_color)), .. self }
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
    // Pass any `edge_color` so one can use the same value to both
    // `write` and `save` (because `self.edge` consumes `self`).
    fn write_mesh_begin<W, E>(&self, w: &mut W, edge_color: E) -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        write!(w, "{}", LATEX_BEGIN)?;
        // Write points
        write!(w, "  % {} points", m.n_points())?;
        for i in 0 .. m.n_points() { point(w, BLACK, i, m.point(i))? }
        // Write lines (on top of points)
        write!(w, "% {} triangles\n", m.n_triangles())?;
        for i in 0 .. m.n_edges() {
            match edge_color(i) {
                Some(c) => {
                    let (p1, p2) = m.edge(i);
                    line(w, c, m.point(p1), m.point(p2))?;
                }
                None => ()
            }
        }
        Ok(())
    }

    fn write_levels<W,E>(&self, w: &mut W, edge_color: E, z: &dyn P1)
                         -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        self.write_mesh_begin(w, edge_color)?;
        let bdry = boundary_edges(m);
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
                            if on_boundary(&bdry, i1, i2) {
                                if on_boundary(&bdry, i1, i3)
                                    || on_boundary(&bdry, i2, i3) {
                                        triangle(w, color, p1, p2, p3)?
                                    }
                                else {
                                    line(w, color, p3, mid(p1, p2))?
                                }
                            } else { // i1-i2 not on boundary
                                if on_boundary(&bdry, i1, i3) {
                                    if on_boundary(&bdry, i2, i3) {
                                        triangle(w, color, p1, p2, p3)?
                                    } else {
                                        line(w, color, p2, mid(p1, p3))?
                                    }
                                } else { // i1-i3 not on boundary
                                    if on_boundary(&bdry, i2, i3) {
                                        line(w, color, p1, mid(p2, p3))?
                                    } else {
                                        triangle(w, color, p1, p2, p3)?
                                    }
                                }
                            }
                        } else { // l = z1 = z2 ≠ z3
                            if ! on_boundary(&bdry, i1, i2) {
                                line(w, color, p1, p2)?
                            }
                        }
                    } else { // l = z1 ≠ z2
                        if level_eq(l, z3) { // l = z1 = z3 ≠ z2
                            if ! on_boundary(&bdry, i1, i3) {
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
                            if ! on_boundary(&bdry, i2, i3) {
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
                            if ! on_boundary(&bdry, i2, i3) {
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

    fn write_superlevels<W,E>(&self, w: &mut W, edge_color: E, z: &dyn P1)
                              -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        self.write_mesh_begin(w, edge_color)?;
        write_xxx_levels!(w, m, z, self.levels, keep_order);
        Ok(())
    }

    fn write_sublevels<W,E>(&self, w: &mut W, edge_color: E, z: &dyn P1)
                            -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        self.write_mesh_begin(w, edge_color)?;
        write_xxx_levels!(w, m, z, self.levels, reverse_order);
        Ok(())
    }

    /// Write the mesh or the levels/superlevels/sublevels to `w`.
    pub fn write<W>(&self, w: &mut W) -> Result<(), io::Error>
    where W: Write {
        match &self.action {
            Action::Mesh => match &self.edge_color {
                None => {
                    self.write_mesh_begin(w, default_mesh_color!(self.mesh))?;
                    write!(w, "{}", LATEX_END)
                }
                Some(e) => {
                    self.write_mesh_begin(w, e)?;
                    write!(w, "{}", LATEX_END)
                }
            }
            Action::Levels(u) => match &self.edge_color {
                None =>
                    self.write_levels(w, default_level_color!(self), &**u),
                Some(e) =>
                    self.write_levels(w, e, &**u)
            }
            Action::SuperLevels(u) => match &self.edge_color {
                None =>
                    self.write_superlevels(w, default_level_color!(self), &**u),
                Some(e) =>
                    self.write_superlevels(w, e, &**u)
            }
            Action::SubLevels(u) => match &self.edge_color {
                None =>
                    self.write_sublevels(w, default_level_color!(self), &**u),
                Some(e) =>
                    self.write_sublevels(w, e, &**u)
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
  \providecommand{\meshline}[5]{%
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfusepath{stroke}
    \end{pgfscope}}
  % \meshpoint{R,G,B}{point number}{x}{y}
  \providecommand{\meshpoint}[4]{}\n";
  % \meshtriangle{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}\n";
  \providecommand{\meshtriangle}[7]{%
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfpathlineto{\pgfpointxy{#6}{#7}}
      \pgfusepath{fill}
    \end{pgfscope}}\n";
  % \meshfilltriangle{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}
  \providecommand{\meshfilltriangle}[7]{%
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfpathlineto{\pgfpointxy{#6}{#7}}
      \pgfusepath{fill}
    \end{pgfscope}}\n";
  % \meshfillquadrilateral{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}{x4}{y4}
  \providecommand{\meshfillquadrilateral}[9]{%
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfpathlineto{\pgfpointxy{#6}{#7}}
      \pgfpathlineto{\pgfpointxy{#8}{#9}}
      \pgfusepath{fill}
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

/// Scilab Output.  Created by [`MeshBase::scilab`].
pub struct Scilab<'a, M>
where M: MeshBase + ?Sized {
    mesh: &'a M,
    z: Box<dyn P1 + 'a>,
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

    /// Use the function `edge_color` to color the edges.
    /// See [`LaTeX::edge`] for details.
    pub fn edge(self, edge_color: RGB8) -> Self {
        Scilab { edge_color: Some(edge_color), .. self }
    }

    /// Saves the mesh data and the function values `z` on the mesh
    /// (see [`MeshBase::scilab`]) so that when Scilab runs the created
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
                   // Written by the Rust Mesh module.\n\
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


////////////////////////////////////////////////////////////////////////
//
// Mathematica Output




#[cfg(test)]
mod tests {

    #[test]
    fn cuthill_mckee() {
        use super::*;

        #[derive(Debug, PartialEq, Eq, Clone)]
        struct M {
            n_points: usize,
            edge: Vec<(usize, usize)>,
        }
        impl MeshBase for M {
            fn n_points(&self) -> usize { self.n_points }
            fn point(&self, _: usize) -> (f64, f64) { todo!() }
            fn n_triangles(&self) -> usize { todo!() }
            fn triangle(&self, _: usize) -> (usize, usize, usize) { todo!() }
        }
        impl Mesh for M {
            fn n_edges(&self) -> usize { self.edge.len() }
            fn edge(&self, i: usize) -> (usize, usize) { self.edge[i] }
            fn edge_marker(&self, _: usize) -> i32 { todo!() }
        }
        impl Permutable for M {
            fn permute_points(&mut self, p: &Permutation) {
                for e in &mut self.edge {
                    if p[e.0] <= p[e.1] { *e = (p[e.0], p[e.1]) }
                    else { *e = (p[e.1], p[e.0]) }
                }
            }
        }
        fn band(x: &M) -> usize {
            let mut b = 0;
            for (i,j) in &x.edge {
                b = b.max(if i <= j { j - i } else { i - j })
            }
            b
        }
        let mut before = M {
            n_points: 8,
            edge: vec![(0,4), (1,2), (1,5), (1,7), (2,4), (3,6), (5,7)]
        };
        let mut after = M {
            n_points: 8,
            edge: vec![(0,1), (2,3), (2,4), (3,4), (4,5), (5,6), (6,7)]
        };
        let p = before.clone().cuthill_mckee();
        assert_eq!(p, Permutation::new([0, 3, 2, 6, 1, 4, 7, 5]).unwrap());
        let p = before.cuthill_mckee_rev();
        assert_eq!(band(&before), band(&after));
        assert_eq!(p, Permutation::new([7, 4, 5, 1, 6, 3, 0, 2]).unwrap());
        before.edge.sort();
        after.edge.sort();
        assert_eq!(before, after);
    }
}
