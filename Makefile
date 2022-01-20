FEATURES=--features density-mesh-core

build:
	cargo build $(FEATURES)

test:
	cargo test $(FEATURES)

doc:
	cargo doc $(FEATURES)

IMG = rust-logo.png
example:
	cp examples/$(IMG) examples/from_image.tex /tmp/
	cargo run $(FEATURES) --example from_image /tmp/$(IMG)
	cd /tmp/ && pdflatex from_image.tex

clean:
	cargo clean

.PHONY: build test doc clean
