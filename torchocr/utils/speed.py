# coding=utf-8  
# @Time   : 2021/2/26 12:19
# @Auto   : zzf-jeff
import torch
import time


def speed_evaluation(model, dummy_input, iters=100):
    with torch.no_grad():
        # warm up
        for _ in range(10):
            model(dummy_input)

        # throughput evaluate
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy_input)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        throughput = int(1.0 * iters / (t1 - t0))

        # latency evaluate
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy_input)
            torch.cuda.current_stream().synchronize()
        t1 = time.time()
        latency = round(1000.0*(t1 - t0) / iters, 2)

    return throughput, latency
