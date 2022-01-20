
use image::{io::Reader as ImageReader,
            imageops::flip_vertical_in_place,
            GenericImageView,
            Rgba, DynamicImage};
use density_mesh_core::{generator::DensityMeshGenerator,
                        mesh::settings::GenerateDensityMeshSettings};
use density_mesh_image::{
    *, settings::{GenerateDensityImageSettings, ImageDensitySource::*}};
use std::{env, fs::File, io, io::Write, path::Path};

use rgb::{RGB, RGB8};
use mesh2d::{Mesh2D, Mesh};

macro_rules! panic_on_err {
    ($e: expr, $loc: expr) => {
        match $e { Err(e) => { panic!("Error({}): {:?}", $loc, e) }
                   Ok(()) => () }}}

fn main() -> Result<(), GenericError> {
    let img_file = env::args().nth(1)
        .expect("Give the image file as the first argument");
    let mut img = ImageReader::open(&img_file)?.decode()?;
    flip_vertical_in_place(&mut img);
    let (w, h) = img.dimensions();
    println!("Read image {} of dim {}×{} and color {:?}.",
             img_file, w, h, img.color());

    let density = GenerateDensityImageSettings {
        density_source: Alpha,
        .. Default::default()
    };
    let map = generate_densitymap_from_image(
        img.clone(), &density).unwrap();
    let s = GenerateDensityMeshSettings {
        points_separation: 15.0.into(),
        keep_invisible_triangles: true,
        .. Default::default()
    };
    let mut g = DensityMeshGenerator::new(vec![], map, s);
    panic_on_err!(g.process_wait(), "DensityMeshGenerator");
    let m = g.mesh().expect("Get the density mesh");
    let m = Mesh2D::from(m);
    m.latex().save(Path::new(&img_file).with_extension("tex"))?;

    let u = eval(&m, |x,y| y - x);
    m.latex().super_levels(&u, [(0., RED), (-200., BLUE)])
        .save(Path::new(&img_file).with_extension("1.tex"))?;

    alpha_triangles(&m, &img, Path::new(&img_file).with_extension("2.tex"))?;
    Ok(())
}

const RED: RGB8 = RGB {r: 255, g: 0, b: 0};
const BLUE: RGB8 = RGB {r: 100, g: 100, b: 255};
const GREEN: RGB8 = RGB {r: 0, g: 200, b: 0};

fn eval(m: &impl Mesh, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut u = vec![0.; m.n_points()];
    for i in 0 .. m.n_points() {
        let (x, y) = m.point(i);
        u[i] = f(x,y);
    }
    u
}

fn alpha_triangles<P: AsRef<Path>>(
    m: &impl Mesh, img: &DynamicImage, path: P) -> io::Result<()> {
    let mut f = File::create(path)?;
    for t in 0 .. m.n_triangles() {
        let (p1, p2, p3) = m.triangle(t);
        let (x1, y1) = m.point(p1);
        let (x2, y2) = m.point(p2);
        let (x3, y3) = m.point(p3);
        let (x, y) = ((x1+x2+x3)/3., (y1+y2+y3)/3.);
        let Rgba(p) = img.get_pixel(x as u32, y as u32);
        if p[3] > 100 {
            let c = GREEN;
            write!(f, "\\meshtriangle{{{},{},{}}}{{{:.12}}}{{{:.12}}}\
                       {{{:.12}}}{{{:.12}}}{{{:.12}}}{{{:.12}}}\n",
                   c.r, c.g, c.b, x1, y1, x2, y2, x3, y3)?;
        }
    }
    Ok(())
}

type GenericError = Box<dyn std::error::Error + 'static>;
