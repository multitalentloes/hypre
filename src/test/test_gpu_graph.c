/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#include "_hypre_utilities.hpp"
#if defined(HYPRE_USING_CUDA)
#define gpu(call) cuda##call
#else
#define gpu(call) hip##call
#endif
#define CHECK_GPU(call) do { auto error = (call); if (error != gpu(Success)) { \
   fprintf(stderr, "GPU operation %s failed: %s (%d)\n", #call, \
           gpu(GetErrorString)(error), (int) error); abort(); } } while (0)
#define CHECK(value, expected) do { if ((value) != (expected)) { \
   fprintf(stderr, "graph replay got %d; expected %d at line %d\n", \
           (int) (value), (int) (expected), __LINE__); return 1; } } while (0)

__global__ void increment(int *value) { ++*value; }

int main(void)
{
   int devices = 0;
   auto device_error = gpu(GetDeviceCount)(&devices);
   if (device_error == gpu(ErrorNoDevice) || (device_error == gpu(Success) && !devices))
   {
      puts("SKIP: CUDA/HIP device unavailable");
      return 77;
   }
   CHECK_GPU(device_error);
   gpu(Stream_t) streams[2];
   int *values[2];
   hypre_DeviceGraphData graphs[2] = {};
   for (int i = 0; i < 2; i++)
   {
      CHECK_GPU(gpu(StreamCreateWithFlags)(&streams[i], gpu(StreamNonBlocking)));
      CHECK_GPU(gpu(Malloc)((void **) &values[i], sizeof(int)));
      CHECK_GPU(gpu(MemsetAsync)(values[i], 0, sizeof(int), streams[i]));
      hypre_DeviceGraphStreamBeginCapture(streams[i]);
      increment <<<1, 1, 0, streams[i]>>>(values[i]);
      hypre_DeviceGraphStreamEndCapture(streams[i], &graphs[i].graph);
      hypre_DeviceGraphInstantiate(&graphs[i].graph_exec, graphs[i].graph);
      graphs[i].is_ready = 1;
   }
   for (int rep = 1; rep <= 4; rep++)
   {
      for (int i = 0; i < 2; i++)
      {
         CHECK(graphs[i].is_ready, 1);
         hypre_DeviceGraphLaunch(graphs[i].graph_exec, streams[i]);
      }
      for (int i = 0; i < 2; i++)
      {
         int value = -1;
         CHECK_GPU(gpu(StreamSynchronize)(streams[i]));
         CHECK_GPU(gpu(Memcpy)(&value, values[i], sizeof(int), gpu(MemcpyDeviceToHost)));
         CHECK(value, rep);
      }
   }
   for (int i = 0; i < 2; i++)
   {
      hypre_DeviceGraphExecDestroy(graphs[i].graph_exec);
      hypre_DeviceGraphDestroy(graphs[i].graph);
      CHECK_GPU(gpu(Free)(values[i]));
      CHECK_GPU(gpu(StreamDestroy)(streams[i]));
   }
   puts("GPU graph capture and repeated launch PASS");
   return 0;
}
#else
int main(void) { puts("SKIP: graph tests require CUDA or HIP"); return 77; }
#endif
