import json
import glob
import matplotlib.pyplot as plt
import numpy as np

ENV_LABELS = {
    "mac_cpu":  "Mac CPU",
    "colab_t4": "Colab T4",
    "colab_l4": "Colab L4",
}
MODEL_COLORS = {
    "ResNet18": "#378ADD",
    "ResNet50": "#D85A30",
    "VGG16":    "#1D9E75",
}

def load_results():
    data = {}
    for f in sorted(glob.glob("results_*.json")):
        r = json.load(open(f))
        data[r["env_label"]] = r
    if not data:
        print("No results_*.json found.")
    return data

def extract_metric(data, metric_key):
    envs   = list(data.keys())
    models = list(next(iter(data.values()))["models"].keys())
    values = {m: [] for m in models}
    for env in envs:
        for m in models:
            val = data[env]["models"][m].get(metric_key, 0) or 0
            values[m].append(val)
    return envs, models, values

def grouped_bar(envs, models, values, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    x       = np.arange(len(envs))
    n       = len(models)
    width   = 0.25
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, model in enumerate(models):
        bars = ax.bar(
            x + offsets[i],
            values[model],
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, "#7F77DD"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 1.02,
                    f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in envs], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()

def main():
    data = load_results()
    if not data:
        return

    # first we check the Latency
    envs, models, values = extract_metric(data, "latency_mean_ms")
    grouped_bar(envs, models, values,
                ylabel="Latency (ms)",
                title="Inference Latency: ResNet18 vs ResNet50 vs VGG16",
                filename="latency_chart.png")

    # then we check the Throughput
    envs, models, values = extract_metric(data, "throughput_samples_sec")
    grouped_bar(envs, models, values,
                ylabel="Throughput (samples/sec)",
                title="Throughput: ResNet18 vs ResNet50 vs VGG16",
                filename="throughput_chart.png")

    # then we do the GFLOP/s
    envs, models, values = extract_metric(data, "attainable_gflops")
    grouped_bar(envs, models, values,
                ylabel="Attainable GFLOP/s",
                title="Attainable GFLOP/s: ResNet18 vs ResNet50 vs VGG16",
                filename="gflops_chart.png")

    print("\nAll charts saved:")
    print("  roofline_all.png    — from plot_roofline.py")
    print("  latency_chart.png")
    print("  throughput_chart.png")
    print("  gflops_chart.png")

if __name__ == "__main__":
    main()
