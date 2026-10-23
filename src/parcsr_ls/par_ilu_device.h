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
/* ILU owns all captured identities; graph macros own no solver state.
 * A successful setup retires this cache before changing schedules or factors. */
struct hypre_ILU60GraphCache
{
   hypre_DeviceGraphData handler;
   hypre_DeviceStream stream;
   const void *ia, *ja, *values, *rows, *offsets, *device_offsets, *rhs;
   HYPRE_Int num_levels;
};

#if defined(HYPRE_USING_CUDA)
static inline cudaError_t
#else
static inline hipError_t
#endif
hypre_ILU60DeviceSynchronize()
{
#if defined(HYPRE_USING_CUDA)
   return cudaDeviceSynchronize();
#else
   return hipDeviceSynchronize();
#endif
}

/* Caller synchronizes work first. Native graph failures terminate immediately. */
static inline void
hypre_ILU60CacheReset(hypre_ILU60GraphCache *cache)
{
   if (cache->handler.graph_exec)
   {
      hypre_DeviceGraphExecDestroy(cache->handler.graph_exec);
      cache->handler.graph_exec = NULL;
      hypre_ILU60GraphCount(-1);
   }
   if (cache->handler.graph)
   {
      hypre_DeviceGraphDestroy(cache->handler.graph);
      cache->handler.graph = NULL;
      hypre_ILU60GraphCount(-1);
   }
   cache->handler.is_ready = 0;
}
#endif
#endif
