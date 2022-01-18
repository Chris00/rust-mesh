//! Generic mesh structure to be used with various meshers.  Also
//! provide some functions to help to display and design geometries.

use std::{collections::VecDeque,
          io::{self, Write},
          fmt::{self, Display, Formatter},
          fs,
          ops::Index,
          path::Path};
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
    /// use mesh::Permutation;
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
    /// use mesh::Permutation;
    /// let p = Permutation::new([2, 0, 3, 1]).unwrap();
    /// let mut x = ["a", "b", "c", "d"];
    /// p.apply(&mut x);
    /// assert_eq!(x, ["b", "d", "a", "c"]);
    /// ```
    pub fn apply<T>(&self, x: &mut [T]) {
        if self.len() != x.len() {
            panic!("mesh::Permutation::apply: the permutation and slice must have the same length.")
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
    /// use mesh::Permutation;
    /// let p = Permutation::new([1, 0, 2]).unwrap();
    /// assert_eq!(&p.as_ref()[..], [1, 0, 2]);
    /// ```
    fn as_ref(&self) -> &Vec<usize> { &self.0 }
}


////////////////////////////////////////////////////////////////////////
//
// PSLG & meshes

/// A bounding box in ℝ².  It is required that `xmin` ≤ `xmax` and
/// `ymin` ≤ `ymax` unless the box is empty in which case `xmin` =
/// `ymin` = +∞ and `xmax` = `ymax` = -∞.
pub struct BoundingBox {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

/// Trait implemented by Planar Straight Line Graphs.
pub trait Pslg {
    /// Return the number of points in the PSLG.
    fn n_points(&self) -> usize;
    /// Return the coordinates (x,y) of the point of index `i` (where
    /// it is assumed that 0 ≤ `i` < `n_points()`).  The coordinates
    /// must be finite numbers.
    fn point(&self, i: usize) -> (f64, f64);

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
}

/// Trait describing various characteristics of a mesh.
pub trait Mesh: Pslg {
    /// Number of triangles in the mesh.
    fn n_triangles(&self) -> usize;
    /// The 3 corners (p₁, p₂, p₃) of the triangle `i`:
    /// 0 ≤ pₖ < `n_points()` are the indices of the corresponding
    /// points.  We **require** that p₁ ≤ p₂ ≤ p₃.
    fn triangle(&self, i: usize) -> (usize, usize, usize);

    /// Return the number of edges in the mesh.
    fn n_edges(&self) -> usize;
    /// Return (p₁, p₂) the point indices of the enpoints of edge `i`.
    /// We **require** that p₁ ≤ p₂.
    fn edge(&self, i: usize) -> (usize, usize);
    /// Return the marker of the edge `i` where 0 ≤ `i` < `n_edges()`.
    /// By convention, edges inside the domain receive the marker `0`.
    fn edge_marker(&self, i: usize) -> i32;

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

    /// Same as [`band_height_p1`][Mesh::band_height_p1] except that
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

    /// LaTeX output of the mesh `self` and of vectors defined on this
    /// mesh.
    ///
    /// # Example
    /// ```
    /// use mesh::Mesh;
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
        LaTeX { mesh: &self,  edge_color: None,
                action: Action::Mesh, z: None, levels: vec![] }
    }

    /// Graph the vector `z` defined on the mesh `self` using Scilab.
    /// The value of the function at the point [`point(i)`][Pslg::point]
    /// is given by `z[i]`.
    ///
    /// # Example
    /// ```
    /// use mesh::Mesh;
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.scilab(&[1., 2., 3., 4.]).save("/tmp/foo");
    /// # Ok(()) }
    /// ```
    fn scilab<'a, Z>(&'a self, z: &'a Z) -> Scilab<'a, Self, Z>
    where Z: IntoIterator<Item=f64> {
        // It would be nice to panic if z.len() ≠ self.n_points() but,
        // with this generality, it will be done when we use the iterrator.
        Scilab { mesh: self, z,
                 longitude: 70.,  azimuth: 60.,
                 mode: Mode::Triangles,
                 draw_box: DrawBox::Full }
    }

}


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

const BLACK: RGB8 = RGB {r: 0, g: 0, b: 0};
const GREY: RGB8 = RGB {r: 150, g: 150, b: 150};

macro_rules! default_mesh_color {
    ($s: ident) => { |i| { if $s.mesh.edge_marker(i) == 0 { Some(GREY) }
                           else { Some(BLACK) } }}}

macro_rules! default_level_color {
    ($s: ident) => { |i| { if $s.mesh.edge_marker(i) != 0 { Some(BLACK) }
                           else { None } }}}

#[derive(Debug, Clone, Copy)]
enum Action { Mesh, Levels, SuperLevels, SubLevels }

/// LaTeX output.  Created by [`Mesh::latex`].
pub struct LaTeX<'a, M>
where M: Mesh + ?Sized {
    mesh: &'a M,
    // The "dyn" option was chosen because it avoids having a type
    // parameter for the closure passed on.  This type may confuse the user.
    edge_color: Option<Box<dyn Fn(usize) -> Option<RGB8> + 'a>>,
    action: Action,
    z: Option<Box<dyn P1 + 'a>>,
    levels: Vec<(f64, RGB8)>,
}

fn valid_levels<L>(l: L) -> Vec<(f64, RGB8)>
where L: IntoIterator<Item=(f64, RGB8)>{
    l.into_iter().filter(|(x, _)| x.is_finite()).collect()
}

/// # LaTeX output
///
/// LaTex output is given in terms of three macros
/// `\meshline{R,G,B}{x1}{y1}{x2}{y2}`,
/// `\meshpoint{point number}{x}{y}`, and
/// `\meshtriangle{R,G,B}{x1}{y1}{x2}{y2}{x3}{y3}` to respectively
/// plot edges, points and (filled) triangles.  You can easily
/// customize them in your LaTeX file.  If you do not provide your own
/// implementations, default ones will be used.  The LaTeX package
/// `tikz` is required.
///
/// # Example
/// ```
/// use mesh::Mesh;
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
    /// Use the function `edge_color` to color the edges. 
    /// The return value of `edge_color(i)` specifies the color of the
    /// edge numbered `i`.  If `edge_color(i)` returns `None`, the
    /// edge is not drawn.
    ///
    /// # Example
    /// To draw only the boundary of the mesh (the default when drawing
    /// level cuves or sets), use the following.
    /// ```
    /// use mesh::Mesh;
    /// use rgb::{RGB, RGB8};
    /// const BLACK: RGB8 = RGB {r: 0, g: 0, b: 0};
    /// # fn test<M: Mesh>(mesh: M) -> std::io::Result<()> {
    /// // Assume `mesh` is a value of a type implementing `Mesh`
    /// // and that, for this example, `mesh.n_points()` is 4.
    /// mesh.latex().edge(|i| if mesh.edge_marker(i) != 0 { Some(BLACK) }
    ///                       else { None });
    /// # Ok(()) }
    /// ```
    pub fn edge<E>(self, edge_color: E) -> LaTeX<'a, M>
    where E: Fn(usize) -> Option<RGB8> + 'a {
        LaTeX { mesh: self.mesh,  edge_color: Some(Box::new(edge_color)),
                action: self.action, z: self.z,  levels: self.levels }
    }

    /// Specify that one wants to draw the level curves of `z`.
    pub fn level_curves<Z, L>(self, z: Z, levels: L) -> LaTeX<'a, M>
    where Z: P1 + 'a,
          L: IntoIterator<Item=(f64, RGB8)> {
        LaTeX { mesh: self.mesh,  edge_color: self.edge_color,
                action: Action::Levels,
                z: Some(Box::new(z)),  levels: valid_levels(levels) }
    }

    /// Specify that one wants to draw the super-levels of `z`.
    pub fn super_levels<Z, L>(self, z: Z, levels: L) -> LaTeX<'a, M>
        where Z: P1 + 'a,
              L: IntoIterator<Item=(f64, RGB8)> {
        LaTeX { mesh: self.mesh,  edge_color: self.edge_color,
                action: Action::SuperLevels,
                z: Some(Box::new(z)),  levels: valid_levels(levels) }
    }

    /// Specify that one wants to draw the sub-levels of `z`.
    pub fn sub_levels<Z, L>(self, z: Z, levels: L) -> LaTeX<'a, M>
    where Z: P1 + 'a,
          L: IntoIterator<Item=(f64, RGB8)> {
        LaTeX { mesh: self.mesh,  edge_color: self.edge_color,
                action: Action::SubLevels,
                z: Some(Box::new(z)),  levels: valid_levels(levels) }
    }

    /// Write the mesh `self` to the writer `w`.
    // Pass any `edge_color` so one can use the same value to both
    // `write` and `save` (because `self.edge` consumes `self`).
    fn write_mesh_begin<W, E>(&self, w: &mut W, edge_color: E) -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        write!(w, "{}", LATEX_BEGIN)?;
        // Write lines
        write!(w, "% {} triangles\n", m.n_triangles())?;
        for i in 0 .. m.n_edges() {
            match edge_color(i) {
                None => (),
                Some(c) => {
                    let (p1, p2) = m.edge(i);
                    let (x1, y1) = m.point(p1);
                    let (x2, y2) = m.point(p2);
                    write!(w, "  \\meshline{{{},{},{}}}{{{:.12}}}\
                                       {{{:.12}}}{{{:.12}}}{{{:.12}}}\n",
                           c.r, c.g, c.b, x1, y1, x2, y2)?;
                }
            }
        }
        // Write points
        write!(w, "  % {} points", m.n_points())?;
        for i in 0 .. m.n_points() {
            let (x, y) = m.point(i);
            write!(w, "  \\meshpoint{{{}}}{{{:.12}}}{{{:.12}}}\n",
                   i, x, y)?;
        }
        Ok(())
    }

    fn write_levels<W,E>(&self, w: &mut W, edge_color: E) -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        self.write_mesh_begin(w, edge_color)?;
        // Write level lines for each triangle.
        for t in 0 .. m.n_triangles() {
            let (i1, i2, i3) = m.triangle(t);
            let (x1, y1) = m.point(i1);
            let (x2, y2) = m.point(i2);
            let (x3, y3) = m.point(i3);
        }
        write!(w, "{}", LATEX_END)
    }

    fn write_superlevels<W,E>(&self, w: &mut W, edge_color: E) -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        self.write_mesh_begin(w, edge_color)?;

        write!(w, "{}", LATEX_END)
    }

    fn write_sublevels<W,E>(&self, w: &mut W, edge_color: E) -> io::Result<()>
    where W: Write,
          E: Fn(usize) -> Option<RGB8> {
        let m = self.mesh;
        self.write_mesh_begin(w, edge_color)?;

        write!(w, "{}", LATEX_END)
    }

    /// Write the mesh or the levels/superlevels/sublevels to `w`.
    pub fn write<W>(&self, w: &mut W) -> Result<(), io::Error>
    where W: Write {
        match self.action {
            Action::Mesh => match &self.edge_color {
                None => {
                    self.write_mesh_begin(w, default_mesh_color!(self))?;
                    write!(w, "{}", LATEX_END)
                }
                Some(e) => {
                    self.write_mesh_begin(w, e)?;
                    write!(w, "{}", LATEX_END)
                }
            }
            Action::Levels => match &self.edge_color {
                None => self.write_levels(w, default_level_color!(self)),
                Some(e) => self.write_levels(w, e)
            }
            Action::SuperLevels => match &self.edge_color {
                None => self.write_superlevels(w, default_level_color!(self)),
                Some(e) => self.write_levels(w, e)
            }
            Action::SubLevels => match &self.edge_color {
                None => self.write_sublevels(w, default_level_color!(self)),
                Some(e) => self.write_levels(w, e)
            }
        }
    }

    /// Same as [`write`] except that it writes to the given file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut f = fs::File::create(path)?;
        self.write(&mut f)?;
        f.flush()
    }
}

// We need to put paths in a scope otherwise one gets "TeX
// capacity exceeded".
const LATEX_BEGIN: &str =
    r#"\begin{pgfscope}
  % Written by Rust "mesh" crate.
  % \meshline{R,G,B}{x1}{y1}{x2}{y2}
  \providecommand{\meshline}[5]{%
    \begin{pgfscope}
      \definecolor{RustMesh}{RGB}{#1}
      \pgfsetcolor{RustMesh}
      \pgfpathmoveto{\pgfpointxy{#2}{#3}}
      \pgfpathlineto{\pgfpointxy{#4}{#5}}
      \pgfusepath{stroke}
    \end{pgfscope}}
  % \meshpoint{point number}{x}{y}
  \providecommand{\meshpoint}[3]{}\n";
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

/// Scilab Output.  Created by [`Mesh::scilab`].
pub struct Scilab<'a, M, Z>
where M: Mesh + ?Sized {
    mesh: &'a M,
    z: &'a Z,
    longitude: f64,
    azimuth: f64,
    mode: Mode,
    draw_box: DrawBox,
}

impl<'a, M, Z> Scilab<'a, M, Z>
where M: Mesh,
      Z: IntoIterator<Item=f64> {
    /// Set sets the longitude to `l` degrees of the observation point.
    pub fn longitude(self, l: f64) -> Self {
        Scilab { mesh: self.mesh,  z: self.z,
                 longitude: l,  azimuth: self.azimuth,
                 mode: self.mode,  draw_box: self.draw_box }
    }

    /// Sets the azimuth to `a` degrees of the observation point.
    pub fn azimuth(self, a: f64) -> Self {
        Scilab { mesh: self.mesh,  z: self.z,
                 longitude: self.longitude,  azimuth: a,
                 mode: self.mode,  draw_box: self.draw_box }
    }

    fn write<W>(&self, w: &mut W) -> Result<(), io::Error>
    where W: Write {
        write!(w, "")
    }

    /// Saves the mesh data and the function values `z` (see
    /// [`Mesh::scilab()`]) (i.e. ([fortran layout])) on that mesh so
    /// that when Scilab runs the created [file].sci script, the graph
    /// of the function is drawn.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut f = fs::File::create(path)?;
        self.write(&mut f)?;
        f.flush()
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
        impl Pslg for M {
            fn n_points(&self) -> usize { self.n_points }
            fn point(&self, _: usize) -> (f64, f64) { todo!() }
        }
        impl Mesh for M {
            fn n_triangles(&self) -> usize { todo!() }
            fn triangle(&self, _: usize) -> (usize, usize, usize) { todo!() }
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
