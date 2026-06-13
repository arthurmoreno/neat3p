# Include environment variables from .env if it exists
ifneq ("$(wildcard .env)","")
    include .env
    export
endif

# Default value for PREFIX_PATH if not set in .env (Right now it must be set in the .env)
PREFIX_PATH ?= /default/path/to/your/env

# Conda env for conda run targets (override: CONDA_ENV=myenv make build, or set in .env)
CONDA_ENV ?= neat3p

# Optional extras appended to the wheel on install, e.g. `make install EXTRAS='[bench]'`
# pulls in torch + gymnasium so the nn / benchmark tests can import them.
EXTRAS ?=


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

# --- aetherion-mirrored build / install -------------------------------------
# Builds the wheel into dist/ inside the conda env (toolchain lives there).
.PHONY: build
build:
	@echo "Building neat3p package with python -m build..."
	conda run --no-capture-output -n $(CONDA_ENV) python -m build

# Installs the most recent neat3p wheel from dist/ with --force-reinstall.
# Pass EXTRAS='[bench]' (or '[nn]') to also pull torch / gymnasium.
.PHONY: install
install:
	@echo "Installing latest neat3p wheel from dist/..."
	@WHEEL=$$(ls -t dist/neat3p-*.whl 2>/dev/null | head -n 1); \
	if [ -z "$$WHEEL" ]; then \
		echo "No wheel found in dist/. Run 'make build' first."; \
		exit 1; \
	fi; \
	echo "Using $$WHEEL$(EXTRAS)"; \
	conda run --no-capture-output -n $(CONDA_ENV) pip install --force-reinstall "$$WHEEL$(EXTRAS)"

.PHONY: test
test:
	@echo "Running neat3p tests with pytest..."
	conda run --no-capture-output -n $(CONDA_ENV) pytest tests

.PHONY: build-install-test
build-install-test: build install test
	@echo "Build, install, and test completed."

.PHONY: clang-format
clang-format:
	@echo "Formatting C++ files with clang-format..."
	find ./src -type f -regex '.*\.\(cpp\|hpp\|h\|cxx\|cc\)' | \
	xargs clang-format -i

.PHONY: clang-format-check
clang-format-check:
	@echo "Checking C++ formatting with clang-format..."
	find ./src -type f -regex '.*\.\(cpp\|hpp\|h\|cxx\|cc\)' | \
	xargs -r clang-format --dry-run --Werror

.PHONY: python-format
python-format:
	@echo "Formatting and sorting imports with Ruff..."
	conda run --no-capture-output -n $(CONDA_ENV) ruff format . && conda run --no-capture-output -n $(CONDA_ENV) ruff check --fix .

.PHONY: format
format: clang-format python-format
	@echo "All formatting complete."

.PHONY: generate-stubs
generate-stubs:
	@echo "Generating stubs for neat3p..."
	PYTHONPATH=build python -m nanobind.stubgen -m site-packages.neat3p -M py.typed -o neat3p.pyi

.PHONY: build-and-install
build-and-install: build-test generate-stubs install-package conda-install
	@echo "Build, stubs generated, and module installed."
