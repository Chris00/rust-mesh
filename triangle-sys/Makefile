
build:
	cargo build

test:
	cargo test

doc:
	cargo doc

# With
# #define REAL double
# #define ANSI_DECLARATORS
# #define VOID void
generate_bindings:
	bindgen Triangle/triangle.h -o src/bindings.rs

clean:
	cargo clean

.PHONY: build test doc generate_bindings clean
