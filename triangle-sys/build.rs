
fn main() {
    // Since we want to use our own `triunsuitable` function, we
    // need to compile the library.
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
        .define("REAL", "double")
        .define("VOID", "int") // See triangle.c
        .files(["Triangle/triangle.c"]);
    build.warnings(false);
    build.compile("Triangle");
}
