FEATURES=--features ndarray,density-mesh-core

build:
	cargo $@ $(FEATURES)

test:
	cargo $@ $(FEATURES)

doc:
	cargo $@ $(FEATURES)

IMG = rust-logo.png
example:
	cp examples/$(IMG) examples/from_image.tex /tmp/
	cargo run $(FEATURES) --example from_image /tmp/$(IMG)
	cd /tmp/ && pdflatex from_image.tex

clean:
	cargo $@

.PHONY: build test doc clean
