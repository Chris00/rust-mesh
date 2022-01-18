#FEATURES=--features

build:
	cargo build $(FEATURES)

test:
	cargo test $(FEATURES)

doc:
	cargo doc $(FEATURES)

clean:
	cargo clean

.PHONY: build test doc clean
