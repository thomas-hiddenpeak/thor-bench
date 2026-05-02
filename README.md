# thor-bench

C++20/CUDA hardware benchmark suite for NVIDIA Thor (SM110a, Blackwell).

Built on [thor-probe](https://github.com/thomas-hiddenpeak/thor-probe) for hardware detection and system probing.

## Prerequisites

- NVIDIA Jetson AGX Thor DevKit (aarch64)
- CUDA 13.0+
- GCC 13+
- thor-probe installed (`sudo make install` from thor-probe build)

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

## Usage

```bash
./build/thor_bench                    # run all suites, text output
./build/thor_bench --json             # JSON output
./build/thor_bench --suites memory,compute,tensor
./build/thor_bench --iterations 20    # more samples
```

## License

MIT
