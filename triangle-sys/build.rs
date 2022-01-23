
/// Select an appropriate directory for Github Actions.
fn in_appropriate_dir(file: &str) -> std::path::PathBuf {
    match std::env::var("OUT_DIR") {
        Err(_) => std::path::PathBuf::from(file),
        Ok(dir) => {
            use std::path::PathBuf;
            let mut path = PathBuf::new();
            path.push(dir);
            path.push(file);
            path
        }
    }
}

/// Return `Ok(true)` is the Triangle files are present on the system
/// (in standard development directories).
fn system_triangle() -> std::io::Result<bool> {
    let tmp_file = in_appropriate_dir("has_triangle.c");
    use std::fs::File;
    use std::io::Write;
    { let mut f = File::create(&tmp_file)?;
      write!(f, "#include <triangle.h>\n\
                 int main() {{\n\
                 return 0; }}\n")?;
      f.flush()? }
    let mut build = cc::Build::new();
    let triangle_exists = build.warnings(false)
                               .file(&tmp_file).try_expand().is_ok();
    std::fs::remove_file(tmp_file).expect("Cannot remove file.");
    Ok(triangle_exists)
}

fn main() -> std::io::Result<()> {
    if system_triangle()? {
        println!("cargo:rustc-link-lib=triangle");
    } else {
        let mut build = cc::Build::new();
        if cfg!(target_arch = "x86") || cfg!(target_arch = "x86_64") {
            if cfg!(unix) {
                build.define("LINUX", None);
            } else if cfg!(windows) {
                build.define("CPU86", None);
            }
        }
        build.define("TRILIBRARY", None)
            .define("EXTERNAL_TEST", None)
            .define("ANSI_DECLARATORS", None)
            .files(["Triangle/triangle.c", "Triangle/triangle.h"]);

        build.compile("Triangle");
    }
    Ok(())
}
