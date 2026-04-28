# save as profile_1_CPU_Script.py and run: python3 profile_1_CPU_Script.py
import torch
import torchvision.models as models
import time
import json
import platform

ENV_LABEL     = "mac_cpu"
BATCH_SIZE    = 32
WARMUP_ITERS  = 3    # reduced for CPU since it's slow
MEASURE_ITERS = 10   # reduced for same reason
DEVICE        = torch.device("cpu")

print(f"Environment : {ENV_LABEL}")
print(f"Device      : {DEVICE}")
print(f"Platform    : {platform.platform()}")
print(f"Processor   : {platform.processor()}")
print(f"PyTorch     : {torch.__version__}\n")

def count_flops(model_name):
    return {
        "ResNet18": 1_820_000_000,
        "ResNet50": 4_100_000_000,
        "VGG16":    15_500_000_000,
    }.get(model_name)

def measure_latency(model, dummy_input):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(dummy_input)
        for _ in range(MEASURE_ITERS):
            t0 = time.perf_counter()
            model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    mean_ms = sum(times) / len(times)
    std_ms  = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
    return round(mean_ms, 3), round(std_ms, 3)

def profile_model(model_name, model, dummy_input):
    print(f"Profiling {model_name}...")
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_MB    = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    flops       = count_flops(model_name)
    mean_ms, std_ms = measure_latency(model, dummy_input)
    throughput  = round((BATCH_SIZE / mean_ms) * 1000, 1)
    bw_bytes    = param_MB * 1024**2 * 2 + dummy_input[:1].numel() * dummy_input.element_size()
    ai          = round(flops / bw_bytes, 4)
    attainable  = round((flops / 1e9) / (mean_ms / 1000), 2)

    print(f"  Params    : {param_count:,} ({param_MB:.1f} MB)")
    print(f"  FLOPs     : {flops:,}")
    print(f"  Latency   : {mean_ms} ± {std_ms} ms")
    print(f"  Throughput: {throughput} samples/sec")
    print(f"  AI        : {ai} FLOPs/byte")
    print(f"  GFLOP/s   : {attainable}\n")

    return {
        "model":                   model_name,
        "param_count":             param_count,
        "param_MB":                round(param_MB, 2),
        "flops_per_sample":        flops,
        "latency_mean_ms":         mean_ms,
        "latency_std_ms":          std_ms,
        "throughput_samples_sec":  throughput,
        "gpu_memory_allocated_MB": 0,
        "bandwidth_bytes":         int(bw_bytes),
        "arithmetic_intensity":    ai,
        "attainable_gflops":       attainable,
    }

def main():
    print("Warning: CPU runs are slow — VGG16 may take several minutes.\n")
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)

    results = {
        "env_label":  ENV_LABEL,
        "hardware": {
            "gpu_name":      "N/A",
            "platform":      platform.platform(),
            "processor":     platform.processor(),
            "torch_version": torch.__version__,
        },
        "batch_size": BATCH_SIZE,
        "models": {}
    }

    for name, model in {
        "ResNet18": models.resnet18(weights=None),
        "ResNet50": models.resnet50(weights=None),
        "VGG16":    models.vgg16(weights=None),
    }.items():
        results["models"][name] = profile_model(name, model, dummy_input)

    with open("results_mac_cpu.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results_mac_cpu.json")

if __name__ == "__main__":
    main()