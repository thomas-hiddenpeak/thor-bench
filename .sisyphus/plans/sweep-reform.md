# Thor-Bench Sweep Reform: Parameter Matrix Framework

## TL;DR

> **Quick Summary**: Transform thor-bench from single-optimal-value-per-suite to a sweep/matrix parameter testing framework. Each suite can define a grid of parameter combinations; the framework iterates them, collects per-point results with power measurements (Watts via sysfs INA3221), and produces JSON + CSV output showing performance surfaces (bandwidth utilization, TF/W efficiency).
>
> **Deliverables**:
> - New sweep framework: schema types, runner, CSV formatter, power monitor
> - `BENCH_REGISTER_SWEEP_SUITE` macro (parallel to existing `BENCH_REGISTER_SUITE`)
> - `--sweep` CLI flag with `--suites`, `--iterations`, `--warmup`, `--device` integration
> - Power monitor module (sysfs INA3221 polling, 100Hz, per-kernel integration)
> - Per-suite CSV output files + aggregate JSON
> - Refactor of 8 compute suites to use sweep registration (start with high-value suites)
>
> **Estimated Effort**: XL (34 suites touched, new modules, CUDA+C++ refactoring)
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Schema → Power Monitor → Sweep Runner → CSV Output → Suite Refactor

---

## Context

### Original Request
"扫描式检测，对于不同业务负载下有什么性能表现、带宽利用率、功耗、性能这将成为最重要的参考" — Sweep testing to measure performance surfaces, bandwidth utilization, power consumption (TF/W), and performance efficiency across parameter combinations.

### Interview Summary
**Key Discussions**:
- Transform from single-value-per-suite to parameter sweep framework
- Power measurement via sysfs INA3221 (not tegrastats popen — too high overhead)
- CSV output for machine-readable sweep surfaces
- Backward compatibility: `./thor_bench` (no --sweep) must work identically
- No YAML config, no visualization, no adaptive sweep, no chart generation
- Fix pre-existing device=0 hardcoding bug during refactor

**Research Findings**:
- **INA3221 sysfs**: `/sys/bus/i2c/devices/2-0040/hwmon/hwmon*/in_power1_input` (VDD_GPU, µW) — PRIMARY for benchmarks
- **INA238 sysfs**: `/sys/bus/i2c/devices/2-0044/hwmon/hwmon*/in_power1_input` (carrier board, may not exist)
- **tegrastats popen**: ~50-100ms overhead per call — unsuitable for per-kernel power polling
- **Reference implementations**: PowerLens (Python, 100Hz), Zeus (hwmon sysfs), ITU research (14ms interval)
- **Benchmark sweep patterns**: SHOC (discrete sizes, CSV) + MLPerf (JSON) is the model

### Metis Review
**Identified Gaps (addressed)**:
- "Minimal changes" assumption → Fixed: parallel `BENCH_REGISTER_SWEEP_SUITE` macro, no touching non-sweep suites
- Lambda signature ambiguity → Fixed: new `BenchSweepSuite` struct with separate registration path
- tegrastats power field uncertainty → Fixed: sysfs INA3221 direct read, no popen
- Combinatorial explosion → Fixed: each suite defines bounded parameter grid (4-8 points per axis)
- 3 pre-existing internal-sweep suites → Fixed: left as-is, not refactored in this plan
- CSV schema heterogeneity → Fixed: per-suite CSV files (one per suite), not aggregate
- --sweep + --cupti interaction → Fixed: mutually exclusive flags
- OOM mid-sweep → Fixed: per-point timeout + graceful error handling
- Power measurement failure → Fixed: optional field, null when unavailable, sweep continues

---

## Work Objectives

### Core Objective
Build a parameter sweep framework that allows each benchmark suite to define a grid of parameter combinations, execute each combination with power measurement, and output performance surfaces in JSON + CSV formats.

### Concrete Deliverables
- `src/include/sweep_schema.h` — SweepParams, SweepPoint, SweepResult structs
- `src/include/sweep_runner.h` — SweepRunner interface
- `src/sweep_runner.cpp` — Sweep execution engine
- `src/include/power_monitor.h` — PowerMonitor interface
- `src/power_monitor.cpp` — sysfs INA3221 polling thread
- `src/output/bench_csv_formatter.cpp` — CSV output formatter
- `src/include/bench_csv_formatter.h` — CSV formatter header
- Modified: `src/include/bench_schema.h` — Add `power_watts` optional field
- Modified: `src/include/bench_suites.h` — Add `BenchSweepSuite` struct + `BENCH_REGISTER_SWEEP_SUITE` macro
- Modified: `src/bench_suites.cpp` — Sweep suite registration in registry
- Modified: `src/benchmark_main.cpp` — `--sweep` CLI flag, sweep mode orchestration
- Modified: `src/output/bench_json_serializer.cpp` — SweepResult JSON serialization
- Modified: `src/output/bench_text_formatter.cpp` — SweepResult text table output
- Modified: `CMakeLists.txt` — New source files
- Refactored suites (Wave 3): 8 compute suites converted to sweep registration

### Definition of Done
- [ ] `./build/thor_bench --sweep --suites cublas --json` produces valid JSON with sweep surface
- [ ] `./build/thor_bench --sweep --suites cublas --csv` produces per-suite CSV file
- [ ] `./build/thor_bench` (no flags) produces identical output to pre-sweep version
- [ ] `power_watts` field populated when INA3221 sysfs available, null when not
- [ ] Single sweep-point failure does not abort sweep
- [ ] `--sweep` and `--cupti` are mutually exclusive
- [ ] `--sweep --device 1` uses device 1 (fixes pre-existing bug)
- [ ] Per-point timeout enforced

### Must Have
- New `BenchSweepSuite` struct + `BENCH_REGISTER_SWEEP_SUITE` macro
- Power monitor via sysfs INA3221 (no popen overhead)
- Per-suite CSV output (one file per suite in `sweep_results/` directory)
- SweepResult JSON schema with per-parameter-combination results
- `--sweep` CLI flag + integration with `--suites`, `--iterations`, `--warmup`, `--device`
- Graceful degradation when power measurement unavailable
- Per-point timeout + error isolation
- Backward compatibility: non-sweep mode unchanged

### Must NOT Have (Guardrails)
- NO YAML/JSON config file parsing for sweep params
- NO adaptive sweep, visualization, or chart generation
- NO separate binaries or executables
- NO changes to existing `BenchResult` fields (additive only: `power_watts`)
- NO removal or renaming of existing CLI flags
- NO concurrent suite execution
- NO refactoring of 3 pre-existing internal-sweep suites (shared_carveout, multi_stream, thermal_throttle)
- NO touching non-sweep suites in this plan (they stay registered via `BENCH_REGISTER_SUITE`)
- NO per-suite warmup/iterations override (global flags apply to all points)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO — No test framework in this project
- **Automated tests**: NONE
- **Framework**: NONE
- **Agent-Executed QA**: ALWAYS (mandatory for all tasks)

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Build verification**: `mkdir -p build && cd build && cmake .. && make` — Verify compilation succeeds
- **CLI verification**: Bash (execute binary with flags, check output)
- **JSON verification**: Bash (parse JSON with python/jq, validate schema)
- **CSV verification**: Bash (check CSV structure, column headers, row count)
- **Power measurement**: Bash (check sysfs paths exist, read values, verify watts range)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — types + infrastructure):
├── Task 1: Sweep schema types (sweep_schema.h) [quick]
├── Task 2: Power monitor module (power_monitor.h/.cpp) [quick]
├── Task 3: BenchResult power_watts field (bench_schema.h) [quick]
├── Task 4: CSV formatter (bench_csv_formatter.h/.cpp) [quick]
├── Task 5: Sweep suite registration (bench_suites.h/.cpp) [quick]
└── Task 6: CLI --sweep flag + main orchestration (benchmark_main.cpp) [quick]

Wave 2 (Core Engine — sweep runner + output):
├── Task 7: Sweep runner engine (sweep_runner.h/.cpp) [deep] (depends: 1, 5, 6)
├── Task 8: Sweep JSON serialization (bench_json_serializer.cpp) [quick] (depends: 1, 3)
├── Task 9: Sweep text table output (bench_text_formatter.cpp) [quick] (depends: 1, 3)
└── Task 10: CMakeLists.txt integration + build verification [quick] (depends: 1-9)

Wave 3 (Suite Refactor — 8 compute suites, PARALLEL):
├── Task 11: cublas_bench → sweep registration [deep] (depends: 10)
├── Task 12: memory_bench → sweep registration [deep] (depends: 10)
├── Task 13: sm_compute_bench → sweep registration [deep] (depends: 10)
├── Task 14: tensor_bench → sweep registration [deep] (depends: 10)
├── Task 15: sasp_bench → sweep registration [deep] (depends: 10)
├── Task 16: fp4_bench → sweep registration [deep] (depends: 10)
├── Task 17: tegra_memory → sweep registration [deep] (depends: 10)
└── Task 18: tma_copy → sweep registration [deep] (depends: 10)

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

### Dependency Matrix (FULL)

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | — | 7, 8, 9, 10 |
| 2 | — | 7, 10 |
| 3 | — | 7, 8, 9, 10 |
| 4 | — | 10 |
| 5 | — | 6, 7, 10 |
| 6 | 5 | 7, 10 |
| 7 | 1, 3, 5, 6 | 10 |
| 8 | 1, 3 | 10 |
| 9 | 1, 3 | 10 |
| 10 | 1-9 | 11-18 |
| 11 | 10 | — |
| 12 | 10 | — |
| 13 | 10 | — |
| 14 | 10 | — |
| 15 | 10 | — |
| 16 | 10 | — |
| 17 | 10 | — |
| 18 | 10 | — |
| F1 | ALL | — |
| F2 | ALL | — |
| F3 | ALL | — |
| F4 | ALL | — |

### Agent Dispatch Summary

- **Wave 1**: **6 tasks** — T1-T6 → `quick` (independent foundation)
- **Wave 2**: **4 tasks** — T7 → `deep`, T8-T9 → `quick`, T10 → `quick`
- **Wave 3**: **8 tasks** — T11-T18 → `deep` (parallel suite refactors)
- **Wave FINAL**: **4 tasks** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

**Critical Path**: T1/T5/T6 → T7 → T10 → T11 (or any T11-T18) → F1-F4 → user okay
**Parallel Speedup**: ~75% faster than sequential (8 parallel suite refactors in Wave 3)
**Max Concurrent**: 8 (Wave 3)

---

## TODOs

- [ ] 1. Sweep Schema Types (sweep_schema.h)

  **What to do**:
  - Create `src/include/sweep_schema.h` with:
    - `struct SweepParams`: name (string), values (vector of `std::variant<int, double, std::string>`)
    - `struct SweepPoint`: map<SweepParams index, value index> — represents one parameter combination
    - `struct SweepResult`: suite_name, test_name, params_json, BenchResult (nested), power_watts (optional<double>), error_message (optional<string>), timestamp
    - `struct SweepReport`: suite_name, description, param_names (vector<string>), points (vector<SweepResult>), sweep_timestamp, total_points, success_points, error_points
  - Include guards, namespace `deusridet::bench`
  - Follow `.h` extension convention

  **Must Not do**:
  - Do not modify existing BenchResult struct (that's Task 3)
  - Do not add YAML/JSON config parsing types

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward type definitions, no complex logic
  - **Skills**: []
    - No special skills needed, pure C++ header work

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-6)
  - **Blocks**: Tasks 7, 8, 9, 10
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/include/bench_schema.h` — Existing BenchResult/BenchReport schema for style reference
  - `src/include/bench_suites.h` — Existing BenchSuite struct for naming convention

  **Acceptance Criteria**:
  - [ ] Header compiles standalone (no dependencies beyond standard library)
  - [ ] All structs are in `deusridet::bench` namespace
  - [ ] SweepResult contains optional<double> power_watts and optional<string> error_message
  - [ ] SweepReport contains success_points/error_points counters

  **QA Scenarios**:
  ```
  Scenario: Header compiles standalone
    Tool: Bash (cmake)
    Steps:
      1. Create minimal test file: `#include "sweep_schema.h"` in empty .cpp
      2. Run `g++ -std=c++20 -c test.cpp -I src/include`
      3. Verify exit code 0
    Expected Result: Compilation succeeds with no errors
    Evidence: .sisyphus/evidence/task-1-compile-standalone.txt

  Scenario: Struct layout matches spec
    Tool: Bash (grep)
    Steps:
      1. grep for SweepParams, SweepPoint, SweepResult, SweepReport in sweep_schema.h
      2. Verify all 4 structs present
      3. Verify power_watts is std::optional<double>
    Expected Result: All structs found with correct field types
    Evidence: .sisyphus/evidence/task-1-struct-layout.txt
  ```

  **Evidence to Capture**:
  - [ ] Compilation output (exit code 0)
  - [ ] grep output showing struct definitions

  **Commit**: YES (groups with Wave 1)

- [ ] 2. Power Monitor Module (power_monitor.h/.cpp)

  **What to do**:
  - Create `src/include/power_monitor.h`:
    - `class PowerMonitor` with singleton pattern (`static PowerMonitor& instance()`)
    - `bool init()` — probe sysfs paths, start polling thread
    - `void shutdown()` — stop polling thread
    - `void markStart()` — mark power measurement start point
    - `std::optional<double> markEnd()` — return average power in Watts since last markStart
    - `std::optional<double> readInstant()` — return current power in Watts
    - `bool isAvailable()` — return whether power monitoring is active
  - Create `src/power_monitor.cpp`:
    - Sysfs path probing: iterate `/sys/bus/i2c/devices/2-0040/hwmon/hwmon*/in_power1_input` (INA3221 channel 1 = VDD_GPU)
    - Fallback: try `/sys/bus/i2c/devices/2-0044/hwmon/hwmon*/in_power1_input` (INA238, carrier board)
    - Background thread polling at 100Hz (10ms interval)
    - Ring buffer of 1000 samples (~10 seconds of history)
    - Trapezoidal integration between markStart/markEnd for per-kernel energy
    - Thread-safe access via std::mutex
    - Graceful degradation: if sysfs paths don't exist, isAvailable() returns false, all methods return nullopt
  - Use std::thread, std::atomic, std::mutex, std::condition_variable

  **Must Not do**:
  - Do NOT use popen("tegrastats") — too high overhead
  - Do NOT use NVML/CUPTI power APIs — not available on Tegra
  - Do NOT block benchmark execution on power monitor failure
  - Do NOT allocate GPU resources in power monitor

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Well-defined threading + sysfs I/O module, no complex algorithms
  - **Skills**: []
    - Standard C++ threading, no special domain knowledge needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-6)
  - **Blocks**: Task 7 (sweep runner uses it)
  - **Blocked By**: None

  **References**:
  - `src/thermal_throttle.cu:52-71` — Existing readTegrastats() pattern (popen, NOT to follow)
  - Research: PowerLens (Python, 100Hz sysfs polling), Zeus (hwmon sysfs)
  - Sysfs paths: `/sys/bus/i2c/devices/2-0040/hwmon/hwmon*/in_power1_input` (µW)

  **Acceptance Criteria**:
  - [ ] init() returns true when INA3221 sysfs path exists, false otherwise
  - [ ] markStart/markEnd returns average power in Watts (not µW)
  - [ ] isAvailable() returns false immediately when sysfs paths don't exist
  - [ ] shutdown() stops the polling thread cleanly
  - [ ] Concurrent markStart/markEnd calls don't race (mutex-protected)

  **QA Scenarios**:
  ```
  Scenario: Power monitor compiles and links
    Tool: Bash (cmake + make)
    Steps:
      1. Add power_monitor.cpp to CMakeLists.txt (temporarily)
      2. Run `mkdir -p build && cd build && cmake .. && make`
      3. Verify exit code 0
    Expected Result: Build succeeds
    Evidence: .sisyphus/evidence/task-2-build-success.txt

  Scenario: Power monitor handles missing sysfs gracefully
    Tool: Bash (execute test)
    Steps:
      1. On non-Thor machine, PowerMonitor::instance().init() should return false
      2. Verify isAvailable() returns false
      3. Verify readInstant() returns nullopt (no crash)
    Expected Result: No crash, graceful null returns
    Evidence: .sisyphus/evidence/task-2-graceful-degrade.txt
  ```

  **Evidence to Capture**:
  - [ ] Build output (exit code 0)
  - [ ] Test output showing graceful degradation

  **Commit**: YES (groups with Wave 1)

- [ ] 3. BenchResult power_watts Field (bench_schema.h)

  **What to do**:
  - Modify `src/include/bench_schema.h`:
    - Add `std::optional<double> power_watts;` to `BenchResult` struct (after `peak_pct`, before `sample_count`)
    - No other changes to BenchResult
  - Modify `src/output/bench_json_serializer.cpp`:
    - Add `power_watts` field to JSON serialization (only emit when non-null)
  - Modify `src/output/bench_text_formatter.cpp`:
    - Add power display to text output (e.g., `Power: XX.X W`) when available

  **Must Not do**:
  - Do NOT rename, remove, or reorder existing BenchResult fields
  - Do NOT change JSON schema for existing fields
  - Do NOT add any other new fields to BenchResult

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple additive field change, minimal risk
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4-6)
  - **Blocks**: Tasks 7, 8, 9
  - **Blocked By**: None

  **References**:
  - `src/include/bench_schema.h:19-40` — Existing BenchResult struct
  - `src/output/bench_json_serializer.cpp` — Existing JSON serialization
  - `src/output/bench_text_formatter.cpp` — Existing text formatting

  **Acceptance Criteria**:
  - [ ] BenchResult struct compiles with new field
  - [ ] JSON output includes `"power_watts": null` or `"power_watts": XX.X` (only when non-null)
  - [ ] Text output shows power when available, omits when null
  - [ ] Non-sweep mode JSON schema unchanged (all existing keys present)

  **QA Scenarios**:
  ```
  Scenario: power_watts field added to BenchResult
    Tool: Bash (grep)
    Steps:
      1. grep 'power_watts' src/include/bench_schema.h
      2. Verify std::optional<double> type
      3. Verify no other field changes (diff original vs new)
    Expected Result: power_watts field present, type correct, no other changes
    Evidence: .sisyphus/evidence/task-3-field-added.txt

  Scenario: JSON serialization includes power_watts
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --json --suites memory | python3 -c "import json,sys; d=json.load(sys.stdin); print('power_watts' in str(d))"`
    Expected Result: power_watts key present in JSON output (as null or value)
    Evidence: .sisyphus/evidence/task-3-json-serial.txt
  ```

  **Evidence to Capture**:
  - [ ] grep output showing field addition
  - [ ] JSON output snippet showing power_watts

  **Commit**: YES (groups with Wave 1)

- [ ] 4. CSV Formatter (bench_csv_formatter.h/.cpp)

  **What to do**:
  - Create `src/include/bench_csv_formatter.h`:
    - `std::string formatCsv(const SweepReport& report)` — formats a single sweep report as CSV string
    - `void writeCsvFile(const std::string& path, const SweepReport& report)` — writes CSV to file
  - Create `src/output/bench_csv_formatter.cpp`:
    - CSV columns (deterministic order): suite,test,param1_name,param1_value,param2_name,param2_value,...,median,stddev,p95,p99,peak_pct,power_watts,score,error
    - Dynamic column generation based on sweep parameter names
    - First row: header (column names)
    - Subsequent rows: one per sweep point
    - RFC 4180 compliant: quote fields containing commas, newlines
    - UTF-8 encoding, no BOM
    - No trailing comma on any line
    - Missing values: empty string (not "null" or "N/A")
    - Output directory: `sweep_results/` (create if not exists)
    - Filename: `{suite_name}.csv`

  **Must Not do**:
  - Do NOT create aggregate CSV (one file per suite)
  - Do NOT add visualization or chart generation
  - Do NOT use map-based column ordering (use deterministic vector)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward CSV formatting, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5-6)
  - **Blocks**: Task 10 (CMake integration)
  - **Blocked By**: None

  **References**:
  - `src/output/bench_json_serializer.cpp` — Existing serializer pattern to follow
  - `src/output/bench_text_formatter.cpp` — Existing formatter pattern to follow

  **Acceptance Criteria**:
  - [ ] CSV output has header row with deterministic column names
  - [ ] Each sweep point produces exactly one CSV row
  - [ ] Missing values are empty strings
  - [ ] File is valid RFC 4180 CSV (parseable by python csv module)

  **QA Scenarios**:
  ```
  Scenario: CSV output is valid RFC 4180
    Tool: Bash (python3)
    Steps:
      1. Generate test CSV with formatCsv()
      2. Run `python3 -c "import csv,io; r=csv.reader(io.StringIO(csv_str)); rows=list(r); print(len(rows), len(rows[0]))"`
      3. Verify row count = sweep_points + 1 (header)
    Expected Result: CSV parses successfully with correct dimensions
    Evidence: .sisyphus/evidence/task-4-csv-valid.txt

  Scenario: CSV has deterministic column order
    Tool: Bash (python3)
    Steps:
      1. Generate CSV twice with same data
      2. Compare header rows
    Expected Result: Headers are identical across runs
    Evidence: .sisyphus/evidence/task-4-csv-deterministic.txt
  ```

  **Evidence to Capture**:
  - [ ] Python CSV parsing output
  - [ ] Header comparison output

  **Commit**: YES (groups with Wave 1)

- [ ] 5. Sweep Suite Registration (bench_suites.h/.cpp)

  **What to do**:
  - Modify `src/include/bench_suites.h`:
    - Add `struct BenchSweepSuite`: name (string), description (string), runFn (function returning `std::vector<SweepReport>(BenchRunner&, int device)`)
    - Add `BENCH_REGISTER_SWEEP_SUITE(name_, desc_, fn_)` macro — parallel to BENCH_REGISTER_SUITE
    - Modify `BenchSuiteRegistry`: add `std::vector<BenchSweepSuite> sweepSuites_`, `registerSweepSuite()`, `allSweepSuites()`, `filteredSweepSuites()`
  - Modify `src/bench_suites.cpp`:
    - Implement `BenchSuiteRegistry::registerSweepSuite()`, `allSweepSuites()`, `filteredSweepSuites()`
    - Sweep suite registrar class (parallel to BenchSuiteRegistrar)

  **Must Not do**:
  - Do NOT modify the existing `BenchSuite` struct
  - Do NOT modify the existing `BENCH_REGISTER_SUITE` macro
  - Do NOT change `registerSuite()` or `allSuites()` behavior
  - Keep existing and sweep registrations completely separate

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Parallel structure to existing registration, straightforward implementation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4, 6)
  - **Blocks**: Task 6 (CLI needs registry), Task 7 (runner needs registry)
  - **Blocked By**: None

  **References**:
  - `src/include/bench_suites.h:1-39` — Existing BenchSuite, BenchSuiteRegistry, BENCH_REGISTER_SUITE
  - `src/bench_suites.cpp` — Existing implementation pattern

  **Acceptance Criteria**:
  - [ ] BenchSweepSuite struct compiles alongside BenchSuite
  - [ ] BENCH_REGISTER_SWEEP_SUITE macro works (test with dummy suite)
  - [ ] Registry returns empty vectors when no sweep suites registered
  - [ ] filteredSweepSuites() returns only matching names

  **QA Scenarios**:
  ```
  Scenario: Sweep suite registration compiles
    Tool: Bash (cmake + make)
    Steps:
      1. Build with new bench_suites.h
      2. Verify compilation succeeds
    Expected Result: Build succeeds
    Evidence: .sisyphus/evidence/task-5-build-success.txt

  Scenario: Sweep registry is separate from regular registry
    Tool: Bash (execute test)
    Steps:
      1. Check BenchSuiteRegistry has both suites_ and sweepSuites_
      2. Verify allSuites() returns regular suites only
      3. Verify allSweepSuites() returns sweep suites only
    Expected Result: Two independent registries
    Evidence: .sisyphus/evidence/task-5-registry-separate.txt
  ```

  **Evidence to Capture**:
  - [ ] Build output
  - [ ] Registry verification output

  **Commit**: YES (groups with Wave 1)

- [ ] 6. CLI --sweep Flag + Main Orchestration (benchmark_main.cpp)

  **What to do**:
  - Modify `src/benchmark_main.cpp`:
    - Add `bool sweep = false` to `CliArgs` struct
    - Add `--sweep` flag parsing (boolean flag, no argument)
    - Add `--csv` flag parsing (`bool csv = false`)
    - Add mutual exclusion check: `--sweep` and `--cupti` cannot be used together (error + usage)
    - In main(): when `args.sweep` is true:
      - Get sweep suites from registry (filtered by --suites if provided)
      - If no sweep suites found, print error
      - Print sweep banner to stderr
      - Create `BenchRunner` with args.warmup, args.iterations, args.timeout
      - For each sweep suite: call sweep runner (Task 7), collect results
      - Output results: JSON (--json), CSV (--csv), text (default)
      - Handle per-suite errors gracefully (don't abort sweep)
    - Update `print_usage()` with --sweep and --csv documentation

  **Must Not do**:
  - Do NOT modify existing CLI flag behavior
  - Do NOT change non-sweep execution path
  - Do NOT add new flag interactions beyond --sweep/--cupti mutual exclusion

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: CLI parsing + orchestration logic, no complex algorithms
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-5)
  - **Blocks**: Task 7 (runner needs main orchestration)
  - **Blocked By**: Task 5 (needs sweep registry)

  **References**:
  - `src/benchmark_main.cpp:41-168` — Existing CliArgs struct, parse_args()
  - `src/benchmark_main.cpp:192-375` — Existing main() orchestration

  **Acceptance Criteria**:
  - [ ] `--sweep` flag parsed correctly
  - [ ] `--csv` flag parsed correctly
  - [ ] `--sweep --cupti` produces error message and exits
  - [ ] `--sweep --suites cublas` filters to cublas sweep suite
  - [ ] Non-sweep mode (`./thor_bench` no flags) works identically

  **QA Scenarios**:
  ```
  Scenario: --sweep flag recognized
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep 2>&1`
      2. Verify it doesn't print "unrecognized option: --sweep"
    Expected Result: No error about unrecognized flag
    Evidence: .sisyphus/evidence/task-6-sweep-flag.txt

  Scenario: --sweep + --cupti mutual exclusion
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --cupti 2>&1`
      2. Verify exit code != 0
      3. Verify error message mentions mutual exclusion
    Expected Result: Error message, non-zero exit
    Evidence: .sisyphus/evidence/task-6-mutual-excl.txt

  Scenario: Non-sweep mode unchanged
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --suites memory --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['results']))"`
      2. Verify JSON is valid and has results
    Expected Result: Valid JSON output, non-zero results
    Evidence: .sisyphus/evidence/task-6-non-sweep-unchanged.txt
  ```

  **Evidence to Capture**:
  - [ ] --sweep flag output
  - [ ] --sweep --cupti error output
  - [ ] Non-sweep JSON output

  **Commit**: YES (groups with Wave 1)

- [ ] 7. Sweep Runner Engine (sweep_runner.h/.cpp)

  **What to do**:
  - Create `src/include/sweep_runner.h`:
    - `class SweepRunner` with methods:
      - `void setWarmup(int n)` / `void setIterations(int n)` / `void setTimeout(std::chrono::milliseconds ms)`
      - `SweepReport run(const BenchSweepSuite& suite, int device)` — execute full sweep for one suite
      - `std::vector<SweepReport> runMultiple(const std::vector<BenchSweepSuite>& suites, int device)` — batch execution
  - Create `src/sweep_runner.cpp`:
    - Per-suite execution: iterate all parameter combinations in suite's grid
    - For each combination: allocate resources, run kernel, collect BenchResult, capture power
    - Power measurement flow: `PowerMonitor::markStart()` → run kernel → `PowerMonitor::markEnd()` → store power_watts
    - Per-point timeout: if a single sweep point exceeds timeout, mark as error (error_message set), continue to next point
    - OOM handling: catch `std::bad_alloc` and CUDA allocation failures, mark point as error, continue
    - Error isolation: wrap each point in try-catch, never abort sweep on single-point failure
    - Deterministic ordering: points processed in grid-defined order, results stored in same order
    - Progress reporting: print `[sweep] suite: X/Y points completed` to stderr

  **Must Not do**:
  - Do NOT introduce concurrency (suites run sequentially)
  - Do NOT modify BenchRunner behavior
  - Do NOT skip points silently (errors must be recorded)
  - Do NOT retry failed points

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex orchestration logic with error handling, timeout management, power integration
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (with Tasks 8-10)
  - **Blocks**: Task 10 (CMake integration)
  - **Blocked By**: Tasks 1, 2, 3, 5, 6

  **References**:
  - `src/include/sweep_schema.h` — SweepResult, SweepReport types (Task 1)
  - `src/include/power_monitor.h` — PowerMonitor interface (Task 2)
  - `src/include/bench_suites.h` — BenchSweepSuite struct (Task 5)
  - `src/bench_runner.cpp` — Existing BenchRunner pattern for timing/stats

  **Acceptance Criteria**:
  - [ ] run() processes all parameter combinations in deterministic order
  - [ ] Single-point failure (OOM, exception, timeout) does not abort sweep
  - [ ] Failed points recorded with error_message, not silently dropped
  - [ ] power_watts populated for successful points (or null if PowerMonitor unavailable)
  - [ ] Per-point timeout enforced (not just per-suite)

  **QA Scenarios**:
  ```
  Scenario: Sweep runner processes all points
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites cublas --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'], d['sweepReports'][0]['success_points'])"`
    Expected Result: total_points matches expected grid size, success_points <= total_points
    Evidence: .sisyphus/evidence/task-7-all-points.txt

  Scenario: Single-point failure does not abort sweep
    Tool: Bash (execute binary)
    Steps:
      1. Run sweep with a suite that has an invalid parameter point (e.g., very large matrix)
      2. Verify sweep completes (non-zero success_points, non-zero error_points)
    Expected Result: Sweep completes with mixed success/error results
    Evidence: .sisyphus/evidence/task-7-error-isolation.txt
  ```

  **Evidence to Capture**:
  - [ ] JSON output showing total_points, success_points, error_points
  - [ ] JSON output showing error_message for failed points

  **Commit**: YES (groups with Wave 2)

- [ ] 8. Sweep JSON Serialization (bench_json_serializer.cpp)

  **What to do**:
  - Modify `src/output/bench_json_serializer.cpp`:
    - Add `serializeJson(const SweepReport& report)` overload
    - JSON schema for SweepReport: `{"suite_name": "...", "description": "...", "param_names": ["..."], "points": [...], "sweep_timestamp": "...", "total_points": N, "success_points": N, "error_points": N}`
    - JSON schema for SweepResult: `{"test_name": "...", "params_json": {...}, "result": {BenchResult fields...}, "power_watts": null|XX.X, "error_message": null|"..."}`
    - Add `serializeJson(const std::vector<SweepReport>& reports)` for multi-suite output
    - Multi-suite schema: `{"version": "0.1.0", "timestamp": "...", "hostname": "...", "mode": "sweep", "sweepReports": [...]}`
    - When `mode: "sweep"`, use `sweepReports` key. When `mode: "normal"`, use `results` key (existing)

  **Must Not do**:
  - Do NOT modify existing BenchResult serialization
  - Do NOT change BenchReport serialization for non-sweep mode
  - Do NOT add new top-level keys to the existing normal-mode JSON

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward JSON serialization, follows existing pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 9, 10)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 1, 3

  **References**:
  - `src/output/bench_json_serializer.cpp` — Existing serialization pattern
  - `src/include/sweep_schema.h` — SweepResult, SweepReport types

  **Acceptance Criteria**:
  - [ ] SweepReport serializes to valid JSON
  - [ ] SweepResult includes power_watts (null or value)
  - [ ] Multi-suite output has `mode: "sweep"` and `sweepReports` array
  - [ ] Normal-mode JSON unchanged (verify by comparing with pre-sweep output)

  **QA Scenarios**:
  ```
  Scenario: Sweep JSON is valid and parseable
    Tool: Bash (python3)
    Steps:
      1. `./build/thor_bench --sweep --suites cublas --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['mode'], len(d['sweepReports']))"`
    Expected Result: mode=sweep, sweepReports has entries
    Evidence: .sisyphus/evidence/task-8-json-valid.txt

  Scenario: Normal-mode JSON unchanged
    Tool: Bash (python3)
    Steps:
      1. `./build/thor_bench --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('results' in d, 'sweepReports' not in d)"`
    Expected Result: True True (has results, no sweepReports)
    Evidence: .sisyphus/evidence/task-8-normal-unchanged.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output snippet
  - [ ] Normal-mode JSON verification

  **Commit**: YES (groups with Wave 2)

- [ ] 9. Sweep Text Table Output (bench_text_formatter.cpp)

  **What to do**:
  - Modify `src/output/bench_text_formatter.cpp`:
    - Add `formatText(const SweepReport& report)` overload
    - Table format: columns = test_name, param1, param2, ..., median, stddev, peak_pct, power(W)
    - Dynamic column generation based on parameter names
    - ANSI-colored table (matching existing text formatter style)
    - Group by suite name, show suite description as header
    - Error points: show `[ERROR]` prefix with error_message truncated to 40 chars
    - Per-suite summary line: `X/Y points succeeded (Z errors)`
    - When power unavailable: show `-` instead of value

  **Must Not do**:
  - Do NOT modify existing BenchReport text formatting
  - Do NOT add chart/visualization generation
  - Do NOT use fixed-width columns (dynamic width based on data)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Formatting logic, follows existing text formatter patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 10)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 1, 3

  **References**:
  - `src/output/bench_text_formatter.cpp` — Existing text formatter pattern, color codes
  - `src/include/sweep_schema.h` — SweepResult, SweepReport types

  **Acceptance Criteria**:
  - [ ] Text output shows formatted table with columns matching sweep parameters
  - [ ] Error points marked with [ERROR] prefix
  - [ ] Per-suite summary line present
  - [ ] Power column shows `-` when unavailable

  **QA Scenarios**:
  ```
  Scenario: Sweep text table output is formatted
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites cublas --text 2>/dev/null`
      2. Verify output contains table with header row
      3. Verify [ERROR] markers for failed points
    Expected Result: Formatted table output
    Evidence: .sisyphus/evidence/task-9-text-table.txt

  Scenario: Normal-mode text output unchanged
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --suites memory --text 2>/dev/null`
      2. Verify output matches pre-sweep format
    Expected Result: Normal text output, no sweep tables
    Evidence: .sisyphus/evidence/task-9-normal-unchanged.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep text output (table format)
  - [ ] Normal-mode text output verification

  **Commit**: YES (groups with Wave 2)

- [ ] 10. CMakeLists.txt Integration + Build Verification

  **What to do**:
  - Modify `CMakeLists.txt`:
    - Add `src/power_monitor.cpp` to source list
    - Add `src/sweep_runner.cpp` to source list
    - Add `src/output/bench_csv_formatter.cpp` to source list
    - Ensure include paths are correct for new headers
    - Verify CUDA architecture settings unchanged (110a)
    - Verify thor-probe dependency unchanged
  - Run full build: `mkdir -p build && cd build && cmake .. && make`
  - Verify build succeeds with zero errors

  **Must Not do**:
  - Do NOT change CUDA architecture settings
  - Do NOT change compiler flags
  - Do NOT change thor-probe dependency

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple CMake source list additions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all Wave 1 + Wave 2 tasks)
  - **Parallel Group**: Wave 2 (final task)
  - **Blocks**: Tasks 11-18 (suite refactors)
  - **Blocked By**: Tasks 1-9

  **References**:
  - `CMakeLists.txt` — Existing build configuration

  **Acceptance Criteria**:
  - [ ] `cmake .. && make` succeeds with zero errors
  - [ ] All new source files compile and link
  - [ ] Binary runs with `--help` showing new flags

  **QA Scenarios**:
  ```
  Scenario: Full build succeeds
    Tool: Bash (cmake + make)
    Steps:
      1. `rm -rf build && mkdir build && cd build && cmake .. && make 2>&1`
      2. Verify exit code 0
      3. Verify `build/thor_bench` binary exists
    Expected Result: Build succeeds, binary exists
    Evidence: .sisyphus/evidence/task-10-build-success.txt

  Scenario: --help shows new flags
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --help 2>&1`
      2. Verify --sweep and --csv in output
    Expected Result: Help text includes new flags
    Evidence: .sisyphus/evidence/task-10-help-output.txt
  ```

  **Evidence to Capture**:
  - [ ] Build output (exit code 0)
  - [ ] --help output showing new flags

  **Commit**: YES (groups with Wave 2)

- [ ] 11. cublas_bench → Sweep Registration

  **What to do**:
  - Modify `bench/suites/compute/cublas_bench.h`:
    - Keep `runCublasBench(device, matDim, iterations)` as-is (non-sweep backward compat)
    - Add `runCublasSweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
  - Modify `bench/suites/compute/cublas_bench.cu`:
    - Define sweep grid: mat_sizes = {512, 1024, 2048, 4096} (4 points)
    - For each size: run SGEMM, DGEMM, SGEMM_strided_batched, DGEMM_strided_batched
    - Register via `BENCH_REGISTER_SWEEP_SUITE(cublas, "cuBLAS GEMM sweep", runCublasSweep)`
    - Keep existing `BENCH_REGISTER_SUITE(cublas, ...)` for backward compat (non-sweep mode)
    - Use `cudaDeviceSynchronize()` + `PowerMonitor::markStart/markEnd` per point

  **Must Not do**:
  - Do NOT remove existing BENCH_REGISTER_SUITE registration
  - Do NOT change runCublasBench() signature or behavior

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CUDA kernel work, cuBLAS API, power integration, sweep logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12-18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/compute/cublas_bench.cu:1-480` — Existing cuBLAS benchmark
  - `bench/suites/compute/cublas_bench.h` — Existing function declarations

  **Acceptance Criteria**:
  - [ ] Sweep mode: `./build/thor_bench --sweep --suites cublas --json` produces 16 sweep points (4 sizes × 4 tests)
  - [ ] Non-sweep mode: `./build/thor_bench --suites cublas --json` produces identical output to pre-sweep
  - [ ] power_watts populated for each point (or null if unavailable)

  **QA Scenarios**:
  ```
  Scenario: cublas sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites cublas --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 16 (4 sizes × 4 tests)
    Evidence: .sisyphus/evidence/task-11-cublas-points.txt

  Scenario: cublas non-sweep mode unchanged
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --suites cublas --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['results']))"`
    Expected Result: Non-zero results, no sweepReports key
    Evidence: .sisyphus/evidence/task-11-cublas-normal.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output showing total_points
  - [ ] Non-sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 12. memory_bench → Sweep Registration

  **What to do**:
  - Modify `bench/suites/compute/memory_bench.h`:
    - Keep existing `runMemoryBench()` as-is
    - Add `runMemorySweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
  - Modify `bench/suites/compute/memory_bench.cu`:
    - Define sweep grid: data_sizes = {64KB, 256KB, 1MB, 4MB, 16MB, 64MB} (6 points)
    - For each size: run read, write, copy tests
    - Register via `BENCH_REGISTER_SWEEP_SUITE(memory, "Memory bandwidth sweep", runMemorySweep)`
    - Keep existing BENCH_REGISTER_SUITE for backward compat

  **Must Not do**:
  - Do NOT remove existing registration
  - Do NOT change existing runMemoryBench() behavior

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CUDA memory operations, sweep logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 13-18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/compute/memory_bench.cu:1-340` — Existing memory benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 18 points (6 sizes × 3 tests)
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: memory sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites memory --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 18
    Evidence: .sisyphus/evidence/task-12-memory-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 13. sm_compute_bench → Sweep Registration

  **What to do**:
  - Modify `bench/suites/compute/sm_compute_bench.{cu,h}`:
    - Keep existing registration as-is
    - Add `runSmComputeSweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
    - Define sweep grid: register_counts = {16, 32, 64, 128, 255} (5 points)
    - For each count: run FP32 FMA test + register spill test
    - Register via `BENCH_REGISTER_SWEEP_SUITE(sm_compute, "SM compute sweep", runSmComputeSweep)`

  **Must Not do**:
  - Do NOT remove existing registration

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CUDA kernel work, register pressure manipulation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 14-18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/compute/sm_compute_bench.cu` — Existing SM compute benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 10 points (5 register counts × 2 tests)
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: sm_compute sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites sm_compute --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 10
    Evidence: .sisyphus/evidence/task-13-sm-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 14. tensor_bench → Sweep Registration

  **What to do**:
  - Modify `bench/suites/compute/tensor_bench.{cu,h}`:
    - Keep existing registration as-is
    - Add `runTensorSweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
    - Define sweep grid: m_sizes = {128, 256, 512, 1024} (4 points), n_sizes = {128, 256, 512, 1024} (4 points)
    - For each (m,n): run FP16 WMMA, BF16 WMMA
    - Register via `BENCH_REGISTER_SWEEP_SUITE(tensor, "Tensor Core sweep", runTensorSweep)`

  **Must Not do**:
  - Do NOT remove existing registration

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Tensor Core WMMA, complex CUDA kernel work
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11-13, 15-18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/compute/tensor_bench.cu` — Existing Tensor Core benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 32 points (4×4 sizes × 2 precisions)
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: tensor sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites tensor --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 32
    Evidence: .sisyphus/evidence/task-14-tensor-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 15. sasp_bench → Sweep Registration

  **What to do**:
  - Modify `bench/suites/compute/sasp_bench.{cu,h}`:
    - Keep existing registration as-is
    - Add `runSaspSweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
    - Define sweep grid: m_sizes = {128, 256, 512, 1024} (4 points)
    - For each size: run FP8 dense, FP8 sparse (2:4)
    - Register via `BENCH_REGISTER_SWEEP_SUITE(sasp, "FP8 dense/sparse sweep", runSaspSweep)`

  **Must Not do**:
  - Do NOT remove existing registration

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: FP8 Tensor Core, sparse formats, tcgen05 PTX
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11-14, 16-18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/compute/sasp_bench.cu` — Existing FP8 sparse benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 8 points (4 sizes × 2 modes)
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: sasp sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites sasp --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 8
    Evidence: .sisyphus/evidence/task-15-sasp-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 16. fp4_bench → Sweep Registration

  **What to do**:
  - Modify `bench/suites/compute/fp4_bench.{cu,h}`:
    - Keep existing registration as-is
    - Add `runFp4Sweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
    - Define sweep grid: m_sizes = {128, 256, 512, 1024} (4 points)
    - For each size: run FP4 dense, FP4 sparse (2:4)
    - Register via `BENCH_REGISTER_SWEEP_SUITE(fp4, "FP4 dense/sparse sweep", runFp4Sweep)`

  **Must Not do**:
  - Do NOT remove existing registration

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: FP4 Tensor Core, inline PTX, tcgen05
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11-15, 17-18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/compute/fp4_bench.cu` — Existing FP4 benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 8 points (4 sizes × 2 modes)
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: fp4 sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites fp4 --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 8
    Evidence: .sisyphus/evidence/task-16-fp4-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 17. tegra_memory → Sweep Registration

  **What to do**:
  - Modify `bench/suites/memory/tegra_memory.{cu,h}`:
    - Keep existing registration as-is
    - Add `runTegraMemorySweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
    - Define sweep grid: data_sizes = {64KB, 256KB, 1MB, 4MB, 16MB, 64MB} (6 points)
    - For each size: test Device, Pinned, Registered, Pageable memory types
    - Register via `BENCH_REGISTER_SWEEP_SUITE(tegra_memory, "Tegra memory type sweep", runTegraMemorySweep)`

  **Must Not do**:
  - Do NOT remove existing registration

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CUDA memory types, Tegra-specific memory architecture
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11-16, 18)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/memory/tegra_memory.cu` — Existing Tegra memory benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 24 points (6 sizes × 4 memory types)
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: tegra_memory sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites tegra_memory --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 24
    Evidence: .sisyphus/evidence/task-17-tegra-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

- [ ] 18. tma_copy → Sweep Registration

  **What to do**:
  - Modify `bench/suites/memory/tma_copy.{cu,h}`:
    - Keep existing registration as-is
    - Add `runTmaCopySweep(BenchRunner&, int device)` returning `std::vector<SweepReport>`
    - Define sweep grid: copy_sizes = {64KB, 256KB, 1MB, 4MB, 16MB} (5 points)
    - For each size: test TMA async copy bandwidth
    - Register via `BENCH_REGISTER_SWEEP_SUITE(tma_copy, "TMA async copy sweep", runTmaCopySweep)`

  **Must Not do**:
  - Do NOT remove existing registration

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: TMA async copy, CUDA mempool, tcgen05
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11-17)
  - **Blocks**: None
  - **Blocked By**: Task 10

  **References**:
  - `bench/suites/memory/tma_copy.cu` — Existing TMA copy benchmark

  **Acceptance Criteria**:
  - [ ] Sweep mode: 5 points
  - [ ] Non-sweep mode identical to pre-sweep

  **QA Scenarios**:
  ```
  Scenario: tma_copy sweep produces correct point count
    Tool: Bash (execute binary)
    Steps:
      1. `./build/thor_bench --sweep --suites tma_copy --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweepReports'][0]['total_points'])"`
    Expected Result: total_points = 5
    Evidence: .sisyphus/evidence/task-18-tma-points.txt
  ```

  **Evidence to Capture**:
  - [ ] Sweep JSON output

  **Commit**: YES (groups with Wave 3)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `mkdir -p build && cd build && cmake .. && make`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names. Verify C++20 compliance, CUDA 13.0 compliance.
  Output: `Build [PASS/FAIL] | Tests [N/A] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute: (1) `./build/thor_bench` — verify identical to pre-sweep output. (2) `./build/thor_bench --sweep --suites cublas --json` — verify valid JSON with sweep surface. (3) `./build/thor_bench --sweep --suites cublas --csv` — verify CSV file generated. (4) `./build/thor_bench --sweep --suites cublas --json` — check power_watts field populated/null. (5) `./build/thor_bench --sweep --cupti` — verify mutual exclusion error. Save evidence to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(bench): add sweep framework types, power monitor, CSV formatter` — sweep_schema.h, power_monitor.h/.cpp, bench_csv_formatter.h/.cpp, bench_schema.h (power_watts), bench_suites.h/.cpp (sweep registration), benchmark_main.cpp (--sweep flag)
- **Wave 2**: `feat(bench): add sweep runner engine and output serialization` — sweep_runner.h/.cpp, bench_json_serializer.cpp, bench_text_formatter.cpp, CMakeLists.txt
- **Wave 3**: `feat(bench): refactor 8 compute suites to sweep registration` — cublas_bench.cu/.h, memory_bench.cu/.h, sm_compute_bench.cu/.h, tensor_bench.cu/.h, sasp_bench.cu/.h, fp4_bench.cu/.h, tegra_memory.cu/.h, tma_copy.cu/.h
- **Each wave verified independently before committing**

---

## Success Criteria

### Verification Commands
```bash
# Non-sweep mode (backward compatibility)
./build/thor_bench --json  # Should produce valid JSON, identical schema to pre-sweep

# Sweep mode (JSON)
./build/thor_bench --sweep --suites cublas --json  # Should produce sweep surface with power_watts

# Sweep mode (CSV)
./build/thor_bench --sweep --suites cublas --csv  # Should produce sweep_results/cublas.csv

# Mutual exclusion
./build/thor_bench --sweep --cupti  # Should print error and exit

# Device flag fix
./build/thor_bench --sweep --suites cublas --device 0  # Should use device 0
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Build succeeds with `cmake .. && make`
- [ ] Non-sweep mode produces identical output to pre-sweep version
- [ ] Sweep mode produces valid JSON + CSV output
- [ ] Power measurement works on Thor (INA3221 sysfs available), degrades gracefully when not
- [ ] Single sweep-point failure does not abort sweep
- [ ] --sweep + --cupti mutual exclusion enforced
- [ ] --device flag honored in sweep mode
- [ ] Per-point timeout enforced

---

## Success Criteria

### Verification Commands
```bash
# Non-sweep mode (backward compatibility)
./build/thor_bench --json  # Should produce valid JSON, identical schema to pre-sweep

# Sweep mode (JSON)
./build/thor_bench --sweep --suites cublas --json  # Should produce sweep surface with power_watts

# Sweep mode (CSV)
./build/thor_bench --sweep --suites cublas --csv  # Should produce sweep_results/cublas.csv

# Mutual exclusion
./build/thor_bench --sweep --cupti  # Should print error and exit

# Device flag fix
./build/thor_bench --sweep --suites cublas --device 0  # Should use device 0
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Build succeeds with `cmake .. && make`
- [ ] Non-sweep mode produces identical output to pre-sweep version
- [ ] Sweep mode produces valid JSON + CSV output
- [ ] Power measurement works on Thor (INA3221 sysfs available), degrades gracefully when not
- [ ] Single sweep-point failure does not abort sweep
- [ ] --sweep + --cupti mutual exclusion enforced
- [ ] --device flag honored in sweep mode
- [ ] Per-point timeout enforced
