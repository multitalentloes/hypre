## Plan: Chain Optimization For GPU ILU0

Implement hybrid level-set triangular solves for `ilu_type == 60`: execute safe prefix and suffix chains of small level sets with one CUDA/HIP block and `__syncthreads()`, while retaining the graph-based multi-block solve for the middle containing oversized levels. Preserve existing full single-block behavior when every level set fits. Do not modify any file whose basename starts with `_`.

**Steps**
1. Reconcile the chain-metadata interface before wiring behavior. In `src/seq_mv/csr_matop.c`, fix the lower start/end counter updates and avoid extending the generated internal declaration. Keep the existing `hypre_CSRMatrixComputeLevelSetsHost` interface compatible with underscore-generated headers; add a separately named helper in a non-underscore source/header surface if cross-file access is required. The helper should derive chain lengths from level-set offsets/sizes and use the same `HYPRE_MAX_NTHREADS_BLOCK` threshold as the single-block decision, rather than a duplicated literal `1024`.
2. During `ilu_type == 60` setup in `src/parcsr_ls/par_ilu_setup_device.c`, compute chain lengths after host level sets are built and store all four scalar values in the already-existing `ilu_data` fields: lower start/end and upper start/end. Keep the host offsets and device row lists unchanged so chain ranges remain contiguous views into existing arrays. Define the semantics explicitly in solve order, especially for the upper level ordering.
3. Add ranged lower/upper level-set execution helpers in `src/seq_mv/csr_matop_device.c`. Helpers should accept a first level and number of levels, use `offsets + first_level`, and reuse the original row-list pointer with the offset-adjusted base. Provide single-block and graph-capable forms as needed; empty ranges must return without launching/capturing. Graph capture must describe only the middle range so its dependency schedule is stable and reusable.
4. Update `src/parcsr_ls/par_ilu_solve_device.c` to dispatch the hybrid solve. Preserve the existing all-level single-block path when `ls_use_single_block` is set. Otherwise, for each triangular solve execute three ordered regions: small prefix via single-block, middle via ranged graph helper, and small suffix via single-block. Apply the same sequence to upper solve, using the upper level-set order and reversing execution only where the current upper solve semantics require it. Keep residual, permutation, and solution update logic unchanged.
5. Wire the new scalar metadata through the solve caller in `src/parcsr_ls/par_ilu_solve.c` without changing public/generated underscore interfaces. Confirm setup/re-setup/destroy behavior: scalar fields need no allocation cleanup, while existing graph invalidation remains valid because captured graphs now represent the middle range.
6. Add or adapt focused regression coverage for a long-tail sparsity pattern with an oversized level and nonzero prefix/suffix chains. Keep the existing IJ ILU type-60 regression as a correctness check, and ensure tests cover lower and upper chain partitioning, empty prefix/suffix, oversized first/last level, and the all-small full single-block path.

**Relevant files**
- `src/seq_mv/csr_matop.c` — repair and/or isolate host chain-length computation; preserve compatibility with generated declarations.
- `src/seq_mv/protos.h` — optional non-generated declaration for a new helper; do not edit `_hypre_seq_mv.h`.
- `src/seq_mv/csr_matop_device.c` — add ranged single-block and graph level-set solve helpers for lower and upper directions.
- `src/parcsr_ls/par_ilu_setup_device.c` — compute and store chain lengths during type-60 setup; use the shared block-size threshold.
- `src/parcsr_ls/par_ilu_solve_device.c` — perform prefix/middle/suffix triangular solve dispatch and preserve existing permutation/update flow.
- `src/parcsr_ls/par_ilu_solve.c` — pass chain metadata into device solve dispatch if the local interface requires it.

**Verification**
1. Confirm no modified path starts with `_` and inspect the diff for interface drift.
2. Build the configured GPU target with bash build_hypre.sh --cmake --notests --hip

**Decisions**
- No files whose names begin with `_` may be modified, including generated internal headers. Therefore do not change generated declarations or metadata macros in place.
- The optimization targets CUDA/HIP single-block kernels; non-CUDA/HIP paths retain the existing behavior/error handling.
- The middle graph is reused; only its level range changes from the complete level sequence to the non-chain middle.
- The chain threshold must be tied to `HYPRE_MAX_NTHREADS_BLOCK`; the existing `1024` chain-breaking literal should not remain as an independent policy constant.
- Scalar chain lengths require no destroy-time allocation changes.

**Further Considerations**
1. Confirm where the current branch expects the four chain fields to be consumed. If the existing solve entry point cannot be extended without an underscore header change, add a non-generated local wrapper or retrieve the fields directly from `ilu_data` in the current device solve path.
2. Verify upper-chain indexing against the actual upper level-set order before implementation; “start” means start of solve order, not necessarily lowest row index.
3. Prefer a generated test matrix or deterministic synthetic CSR fixture that guarantees a long tail, since the existing IJ saved case may not trigger the hybrid branch.
