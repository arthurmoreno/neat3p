# Include environment variables from .env if it exists
ifneq ("$(wildcard .env)","")
    include .env
    export
endif

# Default value for PREFIX_PATH if not set in .env (Right now it must be set in the .env)
PREFIX_PATH ?= /default/path/to/your/env

.PHONY: clean
clean:
	rm -rf build && \
	rm -rf neat3p.egg-info && \
	rm -rf dist

.PHONY: py-build
py-build:
	@echo "Building Python package..."
	python -m build

.PHONY: clean-py-build
clean-py-build:
	@echo "Cleaning and building Python package..."
	rm -rf build && \
	rm -rf neat3p.egg-info && \
	rm -rf dist && \
	python -m build
	@echo "Python package build complete."

clean-build-run:
	rm -rf build && mkdir build &&  \
	cmake -S . -B ./build -DCMAKE_PREFIX_PATH="$(PREFIX_PATH)" &&  \
	cd build && make &&  \
	cd .. &&  \
	cp build/lib/libmy_module.so my_module.so &&  \
	python test.py

build-test:
	rm -rf build && \
	mkdir build && \
	cd build && \
	cmake -DCMAKE_CXX_FLAGS="-fdiagnostics-color=always" -DBUILD_WORLD_TEST=OFF -G Ninja -DCMAKE_PREFIX_PATH="$(PREFIX_PATH)" .. && \
	stdbuf -oL ninja && \
	cd .. && \
	pytest tests

device-info:
	nvidia-smi

.PHONY: clang-format
clang-format:
	@echo "Formatting C++ files with clang-format..."
	find ./src -type f -regex '.*\.\(cpp\|hpp\|h\|cxx\|cc\)' | \
	xargs clang-format -i

.PHONY: style
style: clang-format
	@echo "All formatting complete."

.PHONY: generate-stubs
generate-stubs:
	@echo "Generating stubs for neat3p..."
	PYTHONPATH=build python -m nanobind.stubgen -m site-packages.neat3p -M py.typed -o neat3p.pyi

.PHONY: build-and-install
build-and-install: build-test generate-stubs install-package conda-install
	@echo "Build, stubs generated, and module installed."

.PHONY: format
format:
	@echo "Formatting and sorting imports with Ruff..."
	ruff format . && ruff check --fix .
