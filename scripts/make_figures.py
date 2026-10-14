"""Make every figure in figures/ from the images in data/.

Run from the repository root:  python scripts/make_figures.py
(run scripts/evaluate.py first; figure 4 reads results/controlled_test.csv)
"""

import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sharpening import filters as F                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

# Fixed categorical order: one colour per method, the same in every chart
COLOURS = {"laplacian": "#2a78d6", "unsharp": "#eb6834", "highpass": "#1baf7a"}
LABELS = {"laplacian": "Laplacian", "unsharp": "Unsharp masking",
          "highpass": "High-frequency emphasis"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"

# Zoom window on the mid-foot bones (rows, cols) in xray_foot.png
ZOOM = (slice(170, 360), slice(250, 520))

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def load(name):
    img = cv2.imread(os.path.join(DATA, name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"data/{name} not found")
    return img


def grid(images, titles, ncols, path, size):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, title in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig1_gaussian(scan):
    imgs = [scan] + [F.gaussian_blur(scan, k, s) for k, s in ((3, 0.5), (5, 1.0), (9, 2.0))]
    titles = ["Scan as received", "Gaussian 3x3, σ=0.5", "Gaussian 5x5, σ=1.0", "Gaussian 9x9, σ=2.0"]
    grid(imgs, titles, 2, os.path.join(FIG, "fig1_gaussian_blur.png"), (10, 7))


def fig2_methods(scan):
    outs = [scan, F.laplacian_sharpen(scan, c=1.0),
            F.unsharp_mask(scan, 5, 1.0, 1.5), F.high_frequency_emphasis(scan, 60, 2, 1.0)]
    titles = ["Scan as received", "Laplacian (c=1.0)",
              "Unsharp masking (5x5, σ=1, k=1.5)", "High-frequency emphasis (D0=60, k=1)"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4),
                             gridspec_kw={"height_ratios": [1.45, 1]})
    for i, (img, t) in enumerate(zip(outs, titles)):
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(t)
        r, c = ZOOM
        axes[0, i].add_patch(plt.Rectangle((c.start, r.start), c.stop - c.start,
                                           r.stop - r.start, fill=False, ec="#eda100", lw=1.2))
        axes[1, i].imshow(img[ZOOM], cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title("Zoom: mid-foot", color=MUTED)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_methods_compared.png"), dpi=130)
    plt.close(fig)


def fig3_mechanics():
    """1-D illustration of unsharp masking on a ramp edge (own drawing)."""
    x = np.linspace(0, 10, 1000)
    signal = np.clip((x - 4) / 2, 0, 1)
    sigma = 0.35 / (x[1] - x[0])
    kernel = np.exp(-0.5 * (np.arange(-300, 301) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(signal, 300, mode="edge")
    blurred = np.convolve(padded, kernel, mode="same")[300:-300]
    mask = signal - blurred
    sharpened = signal + 1.0 * mask
    panels = [(signal, "1. Original edge", None), (blurred, "2. Blurred copy B = I * G", signal),
              (mask, "3. Mask M = I - B", None), (sharpened, "4. Sharpened I' = I + kM", signal)]
    fig, axes = plt.subplots(4, 1, figsize=(6, 7), sharex=True)
    for ax, (y, title, ghost) in zip(axes, panels):
        if ghost is not None:
            ax.plot(x, ghost, color=MUTED, lw=1, ls="--")
        ax.plot(x, y, color=COLOURS["unsharp"], lw=2)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xticks([])
    axes[-1].set_xlabel("Position across the edge")
    fig.text(0.99, 0.005, "Dashed line: original edge", ha="right", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_unsharp_mechanics.png"), dpi=130)
    plt.close(fig)


def fig4_controlled():
    path = os.path.join(ROOT, "results", "controlled_test.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    blurs = list(dict.fromkeys(r["blur"] for r in rows))
    fig, axes = plt.subplots(1, len(blurs), figsize=(11, 4), sharey=True)
    for ax, blur in zip(axes, blurs):
        sub = [r for r in rows if r["blur"] == blur]
        base = float(sub[0]["ssim"])
        for m in ("laplacian", "unsharp", "highpass"):
            pts = [r for r in sub if r["method"] == m]
            # for each strength, the best of that method's other settings
            best = {}
            for r in pts:
                s = float(r["strength"])
                best[s] = max(best.get(s, 0), float(r["ssim"]))
            xs = sorted(best)
            ax.plot(xs, [best[s] for s in xs], marker="o", ms=5, lw=2,
                    color=COLOURS[m], label=LABELS[m])
        ax.axhline(base, color=MUTED, lw=1, ls="--")
        ax.text(3.05, base, "blurred input ", va="bottom", ha="right", color=MUTED, fontsize=8)
        ax.set_title(f"Simulated blur {blur.replace('sigma=', 'σ=')}", loc="left")
        ax.set_xlabel("Sharpening strength (c or k)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("SSIM vs reference (anatomy only)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_controlled_test.png"), dpi=130)
    plt.close(fig)


def fig5_test_print(chart):
    settings = [(3, 0.5, 1.0), (9, 2.0, 2.0), (13, 3.0, 3.0)]
    outs = [chart] + [F.unsharp_mask(chart, k, s, a) for k, s, a in settings]
    titles = ["Test print as received"] + [f"Unsharp {k}x{k}, k={a}" for k, _, a in settings]
    zoom = (slice(440, 540), slice(250, 620))   # the "Black and White test print" banner
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), gridspec_kw={"height_ratios": [1.6, 1]})
    for i, (img, t) in enumerate(zip(outs, titles)):
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(t)
        axes[1, i].imshow(img[zoom], cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title("Zoom: banner text", color=MUTED)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_test_print_oversharpening.png"), dpi=130)
    plt.close(fig)


def fig6_laplacian_fix(scan):
    """Original coursework version (absolute Laplacian added) vs signed version."""
    f = scan.astype(np.float64)
    lap = cv2.Laplacian(f, cv2.CV_64F, ksize=3)
    abs_lap = np.uint8(np.clip(1.5 * np.abs(lap), 0, 255))
    original = cv2.addWeighted(scan, 1.0, abs_lap, 0.5, 0)
    fixed = F.laplacian_sharpen(scan, c=1.0)
    imgs = [scan[ZOOM], original[ZOOM], fixed[ZOOM]]
    titles = ["Scan as received", "Coursework code: |Laplacian| added",
              "Corrected: f - c·Laplacian"]
    grid(imgs, titles, 3, os.path.join(FIG, "fig6_laplacian_fix.png"), (13, 3.6))


def main():
    os.makedirs(FIG, exist_ok=True)
    scan = load("xray_foot.png")
    chart = load("test_print.png")
    fig1_gaussian(scan)
    fig2_methods(scan)
    fig3_mechanics()
    fig4_controlled()
    fig5_test_print(chart)
    fig6_laplacian_fix(scan)
    print("Figures written to figures/")


if __name__ == "__main__":
    main()
