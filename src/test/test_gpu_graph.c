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

static int failures;
#define CHECK(condition) do { if (!(condition)) { \
   fprintf(stderr, "graph check failed at line %d: %s\n", __LINE__, #condition); \
   failures++; } } while (0)
/* Stop before using handles or allocations from a failed prerequisite. */
#define OK(call) do { hypre_GraphError error = (call); \
   if (error != hypre_GraphSuccess) { \
      fprintf(stderr, "graph operation failed at line %d: %s: %s (%d)\n", \
              __LINE__, #call, gpu(GetErrorString)(error), (int) error); \
      exit(EXIT_FAILURE); } } while (0)

__global__ void increment(int *value)
{
   ++*value;
}

static void capture(hypre_GPUGraphHandler *handler, gpu(Stream_t) stream, int *value)
{
   OK(hypre_GraphTryStreamBeginCapture(stream));
   increment <<< 1, 1, 0, stream>>>(value);
   OK(hypre_GraphTryStreamEndCapture(stream, &handler->graph));
   OK(hypre_GraphTryInstantiate(&handler->graph_exec, handler->graph));
   handler->is_ready = handler->graph_exec != NULL;
}

int main(void)
{
   int devices = 0, i, rep, value, initial_error = HYPRE_GetError();
   gpu(Stream_t) streams[2];
   int *values[2];
   hypre_GPUGraphHandler handlers[2] = {};
   hypre_GPUGraphHandler empty = {};
   hypre_GraphError device_error = gpu(GetDeviceCount)(&devices);
   if (device_error == gpu(ErrorNoDevice) ||
       (device_error == hypre_GraphSuccess && !devices))
   {
      printf("SKIP: CUDA/HIP device unavailable: %s (%d), devices=%d\n",
             gpu(GetErrorString)(device_error), (int) device_error, devices);
      return 77;
   }
   OK(device_error);
   OK(hypre_GPUGraphReset(NULL));
   OK(hypre_GPUGraphReset(&empty));
   OK(hypre_GPUGraphReset(&empty));
   for (i = 0; i < 2; i++)
   {
      OK(gpu(StreamCreateWithFlags)(&streams[i], gpu(StreamNonBlocking)));
      OK(gpu(Malloc)((void **) &values[i], sizeof(int)));
      OK(gpu(MemsetAsync)(values[i], 0, sizeof(int), streams[i]));
      capture(&handlers[i], streams[i], values[i]);
   }
   for (rep = 0; rep < 3; rep++)
   {
      for (i = 0; i < 2; i++)
      {
         OK(hypre_GraphTryLaunch(handlers[i].graph_exec, streams[i]));
      }
   }
   for (i = 0; i < 2; i++)
   {
      OK(gpu(StreamSynchronize)(streams[i]));
      OK(gpu(Memcpy)(&value, values[i], sizeof(int), gpu(MemcpyDeviceToHost)));
      CHECK(value == 3);
      /* Change data, retaining every captured address. */
      value = 10 + i;
      OK(gpu(Memcpy)(values[i], &value, sizeof(int), gpu(MemcpyHostToDevice)));
      OK(hypre_GraphTryLaunch(handlers[i].graph_exec, streams[i]));
      OK(gpu(StreamSynchronize)(streams[i]));
      OK(gpu(Memcpy)(&value, values[i], sizeof(int), gpu(MemcpyDeviceToHost)));
      CHECK(value == 11 + i);
      OK(hypre_GPUGraphReset(&handlers[i]));
      CHECK(!handlers[i].is_ready && !handlers[i].graph_exec && !handlers[i].graph);
      OK(hypre_GPUGraphReset(&handlers[i]));
   }
   /* A forbidden synchronization invalidates this thread's owned capture.
    * End it explicitly, even after the checked operation reports failure. */
   OK(hypre_GraphTryStreamBeginCapture(streams[0]));
   CHECK(gpu(StreamSynchronize)(streams[0]) != hypre_GraphSuccess);
   CHECK(hypre_GraphTryStreamEndCapture(streams[0], &handlers[0].graph) !=
         hypre_GraphSuccess);
   CHECK(HYPRE_GetError() == initial_error);
   OK(hypre_GPUGraphReset(&handlers[0]));
   /* Runtime error consumption is an explicit recovery decision of this test. */
   (void) gpu(GetLastError)();
   capture(&handlers[0], streams[0], values[0]);
   OK(hypre_GraphTryLaunch(handlers[0].graph_exec, streams[0]));
   OK(gpu(StreamSynchronize)(streams[0]));
   OK(gpu(Memcpy)(&value, values[0], sizeof(int), gpu(MemcpyDeviceToHost)));
   CHECK(value == 12);
   OK(hypre_GPUGraphReset(&handlers[0]));

   /* Keep the existing diagnostic macro interface covered as well. */
   hypre_DeviceGraphStreamBeginCapture(streams[0]);
   increment <<< 1, 1, 0, streams[0]>>>(values[0]);
   hypre_DeviceGraphStreamEndCapture(streams[0], &handlers[0].graph);
   hypre_DeviceGraphInstantiate(&handlers[0].graph_exec, handlers[0].graph);
   hypre_DeviceGraphLaunch(handlers[0].graph_exec, streams[0]);
   OK(gpu(StreamSynchronize)(streams[0]));
   OK(gpu(Memcpy)(&value, values[0], sizeof(int), gpu(MemcpyDeviceToHost)));
   CHECK(value == 13);
   hypre_DeviceGraphExecDestroy(handlers[0].graph_exec);
   hypre_DeviceGraphDestroy(handlers[0].graph);
   handlers[0] = {};
   for (i = 0; i < 2; i++)
   {
      OK(gpu(Free)(values[i]));
      OK(gpu(StreamDestroy)(streams[i]));
   }
   CHECK(HYPRE_GetError() == initial_error);
   printf("GPU graph checks: %d failures\n", failures);
   return failures ? 1 : 0;
}
#else
int main(void)
{
   puts("SKIP: graph tests require CUDA or HIP");
   return 77;
}
#endif
