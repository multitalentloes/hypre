# Device level-set ILU0 regression tests

ILU type 60 supports CUDA/HIP device execution, zero fill, direct triangular
solves, and noniterative setup. CPU and SYCL execution are unsupported.
See the ILU user manual for the structural and pivot requirements.

Build with `HYPRE_BUILD_TESTS=ON`, then run from the CMake build directory:

```sh
cmake --build . --target test_ilu60
ctest --output-on-failure -R '^test_ilu60($|_)'
```

`test_ilu60` compares factors and corrections against an independent host
ILU0 oracle, exercises identity/RCM ordering, asymmetric and unsorted patterns,
invalid input and pivots, and repeated setup/destruction. `test_ilu60_stream`
checks stream/RHS changes, graph cache reuse, and cleanup. `test_ilu60_mpi`
uses three ranks to cover an empty rank, collective failure agreement, and
successful recovery after a rank-local failed setup. Stream tests require a
CUDA/HIP build; the MPI test additionally requires an MPI build and launcher.
Tests run serially to avoid competing GPU work. Exit status 77 means a skipped
test, including unavailable CUDA/HIP hardware; it is not correctness evidence.

The equivalent direct invocations, from the build's test directory, are:

```sh
./test_ilu60
./test_ilu60 --stream
mpirun -np 3 ./test_ilu60 --mpi
```

For the Makefile workflow, `make test_ilu60` (or `make tests`) builds the driver;
run the direct commands above to execute it. Exit status 77 indicates a
skip in this workflow as well. Select hardware supported by the build's
GPU architecture; a visible GPU alone does not guarantee compatibility.

Allocation accounting requires a separate build with
`HYPRE_ENABLE_MEMORY_TRACKER=ON`. The default test then checks live allocation
counts across repeated setups and destruction; without the tracker it reports
that accounting remains pending. Keep tracker overhead out of timing builds.
Debug builds exercise uncaptured graph fallback as well as numerical checks.
Passing these tests does not replace device memory/race sanitizer validation.

# Performance comparison protocol

Compare complete converged solves against both the original implementation
and type 0. Preserve the original library and executables before applying the
repair; build the compared executables with matching compiler, precision,
architecture, MPI, allocator, and profiling settings. Use an uninstrumented
Release build for timing, and run correctness tests separately.

The reference measurement used double precision, HIP Release, a Radeon PRO
W7900 (`gfx1100`), and one MPI rank. The companion hypredrive Laplacian driver
used BiCGSTAB, a seven-point stencil, device execution, RCM, and ILU0, with the
same solver options for all three configurations. Each process performed five
complete setup/solve repetitions. Two batches reversed configuration order:
original60/repaired60/type0, then type0/repaired60/original60. Repetition zero
from each process was retained separately and excluded from means; the means
use all eight remaining repetitions. Independent `b-Ax` residuals were computed
outside the solve timer. Profiling configuration was
`CALI_CONFIG=runtime-report,calc.inclusive` for every executable.

For each corresponding hypredrive executable and YAML configuration, the
primary workload command was:

```sh
CALI_CONFIG=runtime-report,calc.inclusive mpirun -np 1 ./laplacian \
    -i laplacian-ilu60.yml -n 112 112 112 -s 7 -ns 5
```

Use the original executable with its preserved type-60 library for original60;
use the repaired executable with `laplacian-ilu0.yml` for type0. Record raw
repetitions, first-use costs, per-batch means, standard deviations, iteration
counts, independent residuals, build settings, and binary/source hashes.
The measured configuration files and raw logs should accompany a performance
claim; this command alone does not specify every default of the driver.

Reference results before the PR cleanup (milliseconds, 112 cubed, 70 iterations
for every configuration):

| Configuration | Setup mean | Solve mean | Combined mean | Maximum relative residual |
| --- | ---: | ---: | ---: | ---: |
| Original type 60 | 33.811 | 553.887 | 587.697 | 7.90e-7 |
| Repaired type 60 | 64.115 | 554.835 | 618.950 | 7.90e-7 |
| Type 0 | 17.685 | 680.694 | 698.379 | 4.74e-7 |

The repair retained the solve-time benefit on this workload, while increasing
setup cost: combined time rose 5.3% relative to original type 60 and remained
11.4% below type 0. These are workload-specific measurements. The original
implementation fails the independent asymmetric-pattern regression, so its
timing does not establish correctness. Remeasure the final source and intended
applications before making broader performance claims. CUDA runtime, SYCL
build, and device sanitizer validation were not established by this HIP run.
