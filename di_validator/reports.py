"""Standalone scientific figures included in reproducibility bundles."""

from pathlib import Path


def classification_figures(result, folder):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    selected = next(m for m in result["models"] if m["selected"])
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    figure.suptitle(result["name"] + "\n" + result["interpretation"], fontsize=11)
    axes[0].plot(selected["curves"]["fpr"], selected["curves"]["tpr"], color="#087d71")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Holdout ROC")
    axes[1].plot(selected["curves"]["recall"], selected["curves"]["precision"], color="#087d71")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Holdout precision–recall", ylim=(0, 1.05))
    axes[2].imshow(selected["confusion"], cmap="Greens")
    axes[2].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted label",
        ylabel="Evidence label",
        title="Holdout confusion matrix",
    )
    for row in range(2):
        for col in range(2):
            axes[2].text(
                col,
                row,
                str(selected["confusion"][row][col]),
                ha="center",
                va="center",
                color="#1a3027",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
    for extension in ["png", "svg"]:
        figure.savefig(Path(folder) / f"holdout-figures.{extension}", dpi=160)
    plt.close(figure)
