import json, glob, numpy as np, matplotlib.pyplot as plt

PEAKS = {
    "mac_cpu":  {"gflops":    800, "bw":   50,  "label": "Mac CPU (M1 Max)"},
    "colab_t4": {"gflops":   8100, "bw":  300,  "label": "Colab T4"},
    "colab_l4": {"gflops":  31300, "bw":  864,  "label": "Colab L4"},
}

COLORS  = {
    "ResNet18": "#378ADD",
    "ResNet50": "#D85A30",
    "VGG16":    "#1D9E75",
}
MARKERS = {
    "ResNet18": "o",
    "ResNet50": "s",
    "VGG16":    "^",
}
ENV_LS = {
    "mac_cpu":  ":",
    "colab_t4": "--",
    "colab_l4": "-",
}

def roofline_curve(peak_gflops, peak_bw):
    ai = np.logspace(-2, 4, 500)
    return ai, np.minimum(peak_bw * ai, peak_gflops)

def load_results():
    data = {}
    for f in sorted(glob.glob("results_*.json")):
        r = json.load(open(f))
        data[r["env_label"]] = r
    if not data:
        print("No results_*.json found. Run profile_models.py first on each environment.")
    return data

def make_plots(data):
    envs = list(data.keys())
    fig, axes = plt.subplots(1, len(envs) + 1,
                             figsize=(6 * (len(envs) + 1), 5),
                             constrained_layout=True)

    for idx, env in enumerate(envs):
        ax = axes[idx]
        hw = PEAKS.get(env)
        if not hw:
            print(f"No peak specs for '{env}': add to PEAKS dict.")
            continue

        ai_curve, roof = roofline_curve(hw["gflops"], hw["bw"])
        ax.plot(ai_curve, roof, color="#888780", lw=2, label="Roofline")
        ax.axhline(hw["gflops"], color="#D85A30", lw=1, ls="--", alpha=0.6)
        ax.text(1e-2 * 1.2, hw["gflops"] * 1.08,
                f"Peak {hw['gflops']} GFLOP/s", fontsize=8, color="#D85A30")

        ridge = hw["gflops"] / hw["bw"]
        ax.axvline(ridge, color="#aaa", lw=0.8, ls=":")
        ax.text(ridge * 1.05, hw["gflops"] * 0.4,
                f"ridge\n{ridge:.1f}", fontsize=7, color="#888")

        for mname, mdata in data[env]["models"].items():
            ai = mdata.get("arithmetic_intensity")
            gf = mdata.get("attainable_gflops")
            if ai and gf:
                ax.scatter(ai, gf,
                           color=COLORS.get(mname, "#7F77DD"),
                           marker=MARKERS.get(mname, "D"),
                           s=120, zorder=5, label=mname)
                ax.annotate(mname, (ai, gf),
                            textcoords="offset points",
                            xytext=(6, 4), fontsize=8,
                            color=COLORS.get(mname, "#7F77DD"))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)", fontsize=10)
        ax.set_ylabel("Attainable GFLOP/s", fontsize=10)
        ax.set_title(hw["label"], fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

    ax = axes[-1]
    ai_range = np.logspace(-2, 4, 500)
    for env, env_data in data.items():
        hw = PEAKS.get(env)
        if not hw:
            continue
        roof = np.minimum(hw["bw"] * ai_range, hw["gflops"])
        ax.plot(ai_range, roof, lw=2,
                ls=ENV_LS.get(env, "-"), label=hw["label"])
        for mname, mdata in env_data["models"].items():
            ai = mdata.get("arithmetic_intensity")
            gf = mdata.get("attainable_gflops")
            if ai and gf:
                ax.scatter(ai, gf,
                           color=COLORS.get(mname, "#7F77DD"),
                           marker=MARKERS.get(mname, "D"),
                           s=100, zorder=5)

    # Model legend
    from matplotlib.lines import Line2D
    model_legend = [
        Line2D([0], [0], marker=MARKERS[m], color="w",
               markerfacecolor=COLORS[m], markersize=9, label=m)
        for m in COLORS
    ]
    env_legend = [
        Line2D([0], [0], color="#888780", lw=2,
               ls=ENV_LS.get(e, "-"), label=PEAKS[e]["label"])
        for e in data if e in PEAKS
    ]
    ax.legend(handles=model_legend + env_legend, fontsize=7, loc="upper left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)", fontsize=10)
    ax.set_ylabel("Attainable GFLOP/s", fontsize=10)
    ax.set_title("All Environments", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2)

    plt.savefig("roofline_all.png", dpi=150, bbox_inches="tight")
    print("Saved: roofline_all.png")
    plt.show()

def print_table(data):
    print(f"\n{'Env':<12} {'Model':<10} {'AI':>8} {'GFLOP/s':>10} {'ms':>8} {'samples/s':>12}")
    print("-" * 60)
    for env, env_data in data.items():
        for mname, m in env_data["models"].items():
            print(f"{env:<12} {mname:<10} "
                  f"{m.get('arithmetic_intensity') or 0:>8.3f} "
                  f"{m.get('attainable_gflops') or 0:>10.1f} "
                  f"{m.get('latency_mean_ms') or 0:>8.1f} "
                  f"{m.get('throughput_samples_sec') or 0:>12.1f}")
    print()

if __name__ == "__main__":
    data = load_results()
    if data:
        print_table(data)
        make_plots(data)
