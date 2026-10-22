/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ILU_DEVICE_HEADER
#define hypre_ILU_DEVICE_HEADER

#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
/* ILU owns all captured identities; shared graph utilities own no solver state.
 * A successful setup retires this cache before changing schedules or factors. */
struct hypre_ILU60GraphCache
{
   hypre_GPUGraphHandler handler;
   hypre_DeviceStream stream;
   const void *ia, *ja, *values, *rows, *offsets, *device_offsets, *rhs;
   HYPRE_Int num_levels;
   /* Private one-shot fault injection for lifecycle tests, never a public API:
    * 1 invalidates owned capture, 2 rejects instantiate, 3 rejects launch. */
   HYPRE_Int test_failure;
};

static inline hypre_GraphError
hypre_ILU60LastError()
{
#if defined(HYPRE_USING_CUDA)
   return cudaGetLastError();
#else
   return hipGetLastError();
#endif
}

static inline hypre_GraphError
hypre_ILU60DeviceSynchronize()
{
#if defined(HYPRE_USING_CUDA)
   return cudaDeviceSynchronize();
#else
   return hipDeviceSynchronize();
#endif
}

static inline hypre_GraphError
hypre_ILU60InvalidateCapture(hypre_DeviceStream stream)
{
#if defined(HYPRE_USING_CUDA)
   return cudaStreamSynchronize(stream);
#else
   return hipStreamSynchronize(stream);
#endif
}

/* Caller synchronizes work first. Count only native handles actually retired;
 * failed destruction retains handles and their referenced storage for retry. */
static inline hypre_GraphError
hypre_ILU60CacheReset(hypre_ILU60GraphCache *cache)
{
   HYPRE_Int before = (cache->handler.graph != NULL) + (cache->handler.graph_exec != NULL);
   hypre_GraphError error = hypre_GPUGraphReset(&cache->handler);
   HYPRE_Int after = (cache->handler.graph != NULL) + (cache->handler.graph_exec != NULL);
   hypre_ILU60GraphCount(after - before);
   return error;
}
#endif
#endif
