"""GPU health check: correctness + sustained matmul stress.

Usage:
    python scripts/gpu_healthcheck.py                  # all visible GPUs
    python scripts/gpu_healthcheck.py --gpu 0          # one GPU
    python scripts/gpu_healthcheck.py --gpu 0 --duration 120 --size 8192
"""
import argparse
import time
import torch


def correctness_check(device, size, iters, thresh):
    # fp32 matmul vs CPU fp64 reference. Run in fp32 (not the bf16 stress dtype) so
    # tolerance reflects hardware faults, not bf16 rounding. Gate on the COUNT of
    # elements exceeding `thresh`, not mean error: silent bit-flips corrupt only a
    # handful of the millions of output elements and get averaged away otherwise.
    # Legit fp32 matmul rounding here is ~1e-3, so thresh=0.05 is pure noise margin.
    # Repeat `iters` times because the fault is intermittent (different elements
    # corrupted each run on a bad GPU).
    torch.manual_seed(0)
    a_cpu = torch.randn(size, size, dtype=torch.float64)
    b_cpu = torch.randn(size, size, dtype=torch.float64)
    ref = (a_cpu @ b_cpu).to(torch.float32)
    a = a_cpu.to(device=device, dtype=torch.float32)
    b = b_cpu.to(device=device, dtype=torch.float32)

    total_bad = 0
    max_err = 0.0
    for _ in range(iters):
        out = (a @ b).cpu()
        err = (out - ref).abs()
        total_bad += int((err > thresh).sum().item())
        max_err = max(max_err, err.max().item())
        if not torch.isfinite(out).all().item():
            total_bad += 1
    ok = total_bad == 0
    return ok, max_err, total_bad


def stress_loop(device, size, dtype, duration):
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    # Warmup
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize(device)

    flops_per_matmul = 2.0 * size ** 3
    start = time.time()
    iters = 0
    last_print = start
    nan_inf = False
    while time.time() - start < duration:
        c = a @ b
        # Cheap drift check every ~50 iters without forcing a sync each step.
        if iters % 50 == 0:
            torch.cuda.synchronize(device)
            if not torch.isfinite(c.view(-1)[:1024]).all().item():
                nan_inf = True
                break
        iters += 1
        now = time.time()
        if now - last_print > 5.0:
            torch.cuda.synchronize(device)
            elapsed = now - start
            tflops = iters * flops_per_matmul / elapsed / 1e12
            mem = torch.cuda.memory_allocated(device) / 1e9
            print(f"  [{device}] {elapsed:5.1f}s  iters={iters}  {tflops:6.1f} TFLOPS  mem={mem:.1f} GB")
            last_print = now

    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    tflops = iters * flops_per_matmul / elapsed / 1e12
    return iters, tflops, nan_inf


def memory_cycle(device, gb_per_alloc, cycles):
    # Alloc/free large tensors repeatedly to surface allocator/ECC issues.
    elems = int(gb_per_alloc * 1e9 / 4)
    for i in range(cycles):
        t = torch.empty(elems, device=device, dtype=torch.float32)
        t.fill_(float(i))
        # Read back a slice to force memory traffic.
        val = t[::elems // 1024].sum().item()
        if not (val == val):  # NaN
            return False
        del t
        torch.cuda.empty_cache()
    return True


def check_gpu(idx, size, duration, dtype):
    device = torch.device(f"cuda:{idx}")
    name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"\n=== cuda:{idx}  {name}  ({total_mem:.1f} GB) ===")

    print("[1/3] correctness check (fp32 vs CPU fp64 ref, 8x)...")
    ok, max_err, n_bad = correctness_check(device, size=2048, iters=8, thresh=0.05)
    print(f"      ok={ok}  max_err={max_err:.3e}  corrupted_elems={n_bad}")

    print(f"[2/3] memory cycle (4 x 8 GB allocs)...")
    mem_ok = memory_cycle(device, gb_per_alloc=8.0, cycles=4)
    print(f"      ok={mem_ok}")

    print(f"[3/3] sustained matmul stress ({size}x{size} {dtype}, {duration}s)...")
    iters, tflops, nan_inf = stress_loop(device, size, dtype, duration)
    print(f"      iters={iters}  avg={tflops:.1f} TFLOPS  nan/inf={nan_inf}")

    healthy = ok and mem_ok and not nan_inf
    print(f"==> cuda:{idx} {'HEALTHY' if healthy else 'FAIL'}")
    return healthy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=None, help="GPU index; default = all visible")
    p.add_argument("--size", type=int, default=4096, help="matmul side length for stress")
    p.add_argument("--duration", type=float, default=30.0, help="stress seconds per GPU")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    indices = [args.gpu] if args.gpu is not None else list(range(torch.cuda.device_count()))

    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  devices={indices}")
    results = {i: check_gpu(i, args.size, args.duration, dtype) for i in indices}

    print("\n=== summary ===")
    for i, ok in results.items():
        print(f"  cuda:{i}  {'HEALTHY' if ok else 'FAIL'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
