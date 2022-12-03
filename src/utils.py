import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from paths import *


def plot_calibration(tracks, predictions, filename):
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax = [ax]
    ax[0].boxplot(predictions, vert=1)
    ax[0].xaxis.set_major_locator(ticker.FixedLocator(
        np.arange(1, len(tracks) + 1)
    ))
    ax[0].plot()
    ax[0].xaxis.set_major_formatter(ticker.FixedFormatter(tracks))
    ax[0].set_xlabel("BPM")
    ax[0].set_ylabel("Model output")
    ax[0].grid(True)
    # ax[0].title.set_text(f"Prediction with fixed shift. a = {np.round(a_fixed, 2)}, b = {np.round(b_fixed, 2)}")

    fig.suptitle(model_name, fontsize=16)

    if savefig:
        plt.savefig(f"../data/{filename}_synthetic.png")

    return fig, ax


def plot_reconstructions(bpm_tracks, bpm_dict, main_file, theta):
    fig, ax = plt.subplots(len(bpm_tracks), 1, figsize=(5, 45))
    idx = 0

    for i, key in enumerate(bpm_tracks):
        ax[idx].plot(bpm_dict[key]["slice"], label="input")
        ax[idx].plot(bpm_dict[key]["estimation"], label="estimation")
        shift = bpm_dict[key]["shift"][0]
        ax[idx].title.set_text(
            f"BPM = {key}, shift {shift} estimation: {np.median(bpm_dict[key]['predictions']):.4f}")

        ax[idx].set_xticks(np.arange(0, 128, 20))
        ax[idx].set_xticklabels(np.round(theta[0 + shift:128 + shift:20], 2))
        ax[idx].legend()
        idx += 1

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(
        os.path.join(
            DATA_FOLDER,
            f"imgs/{main_file}_reconstructions.png"))
    return


def plot_tempogram(T, t, freqs, title=None):
    figsize = (10, 5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    kwargs = _tempogram_kwargs(t, freqs)

    ax.imshow(T, **kwargs)

    xlim = (t[0], t[-1])
    ylim = (freqs[0], freqs[-1])

    plt.setp(ax, xlim=xlim, ylim=ylim)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    return fig, ax


def plot_comparison(
        T,
        t,
        freqs,
        reference_tempo,
        predicted_tempo,
        share_plot=False,
        xlim=None,
        ylim=None,
        title=None):
    """
    Plot tempogram comparison

    Parameters
    ---------
    T :  np.ndarray(t, freqs)
        tempogram matrix
    t : np.ndarray
        array with time values
    freqs : np.ndarray
        array with bpm covered
    reference_tempo : float64
        ground truth tempo annotation
    predicted_tempo : np.ndarray
        array with tempo predictions
    share_plot : bool, optional
        if True, ground truth and predictions are plotted together. otherwise,
        each one is in a different plot.
    xlim : tuple, optional
        limit for xaxis. if None, xlim is defined by t values
    ylim : tuple, optional
        limit for yaxis. if None, ylim is defined by freqs values *or* by the
        reference_tempo value + 10 BPM for visualization purposes.

    Return
    -----
    fig, ax
    """
    fig = None
    figsize = (10, 5)

    if share_plot:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = [ax]
        prediction_plot_idx = 0
    else:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        prediction_plot_idx = 1

    kwargs = _tempogram_kwargs(t, freqs)

    # plot tempogram and ground_truth tempo
    ax[0].imshow(T, **kwargs)
    ax[0].hlines(reference_tempo,
                 xmin=t[0],
                 xmax=t[-1],
                 label=f"ground_truth: {reference_tempo} bpm",
                 color="r",
                 linestyle="-.")
    ax[0].legend()

    # plot tempogram and tempo prediction
    median_prediction = np.median(predicted_tempo)
    ax[prediction_plot_idx].imshow(T, **kwargs)
    ax[prediction_plot_idx].hlines(
        median_prediction,
        xmin=t[0], xmax=t[-1],
        label=f"median(predictions): {median_prediction:.2f} BPM",
        color="r"
    )
    ax[prediction_plot_idx].plot(t, predicted_tempo, color="orange", alpha=0.4)
    ax[prediction_plot_idx].scatter(
        t,
        predicted_tempo,
        label="predictions",
        s=6,
        color="orange",
        alpha=0.7)
    ax[prediction_plot_idx].legend()

    if xlim is None:
        xlim = (t[0], t[-1])
    if ylim is None:
        ylim = (freqs[0], freqs[-1])

    plt.setp(ax, xlim=xlim, ylim=ylim)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    return fig, ax


def plot_experiment_results(
    results,
    n_plots=10,
    ylim=None,
    theta=np.arange(
        30,
        301,
        1)):
    """
    Plot experiment results

    Parameters
    ---------
    results : dict
        Dictionary with track_id, tempogram, times, tempi range, reference_tempo
        and predicted_tempo values. baseline_tempo is an optional key for the
        dictionary.
    n_plots : int
        Number of plots one desires to check
    """
    if n_plots > len(results):
        raise ValueError(
            f"You're trying to plot {n_plots} samples, but there are {len(results)} available")

    if ylim is None:
        ylim = (30, 300)

    n_rows = int(n_plots // 2)
    n_cols = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 25))

    row_idx, col_idx = 0, 0
    plots = 0

    for track_id, values in results.items():
        if plots >= n_plots:
            break

        T = values["T"]
        t = values["t"]
        freqs = values["freqs"]
        reference_tempo = values["reference_tempo"]
        predicted_tempo = values["predicted_tempo"]
        baseline_tempo = values.get("baseline_tempo", None)

        kwargs = _tempogram_kwargs(t, freqs)

        ax[row_idx][col_idx].imshow(T, **kwargs)
        ax[row_idx][col_idx].hlines(
            reference_tempo,
            xmin=t[0], xmax=t[-1],
            label=f"ground_truth: {reference_tempo} bpm",
            color="r",
            linestyle="-."
        )

        if baseline_tempo is not None:
            median_baseline = np.median(baseline_tempo)
            ax[row_idx][col_idx].hlines(
                median_baseline,
                xmin=t[0], xmax=t[-1],
                label=f"median(baseline): {median_baseline:.2f} BPM",
                color="b",
                linestyle="--"
            )
            ax[row_idx][col_idx].scatter(
                t,
                baseline_tempo,
                label="prediction by frame",
                s=6,
                color="blue",
                alpha=0.7)

        median_prediction = np.median(predicted_tempo)
        ax[row_idx][col_idx].hlines(
            median_prediction,
            xmin=t[0], xmax=t[-1],
            label=f"median(predictions): {median_prediction:.2f} BPM",
            color="green"
        )
        # ax[row_idx][col_idx].plot(t, predicted_tempo, color="orange", alpha=0.4)
        ax[row_idx][col_idx].scatter(
            t,
            predicted_tempo,
            label="baseline by frame",
            s=6,
            color="green",
            alpha=0.7)

        ax[row_idx][col_idx].title.set_text(track_id)
        ax[row_idx][col_idx].legend()

        col_idx = (col_idx + 1) % n_cols
        if col_idx == 0:
            row_idx += 1 % n_rows

        plots += 1

        plt.setp(ax, ylim=ylim)
        plt.tight_layout()

    return fig, ax


def _tempogram_kwargs(t, freqs):
    kwargs = {}
    x_ext1 = (t[1] - t[0]) / 2
    x_ext2 = (t[-1] - t[-2]) / 2
    y_ext1 = (freqs[1] - freqs[0]) / 2
    y_ext2 = (freqs[-1] - freqs[-2]) / 2

    kwargs["extent"] = [t[0] - x_ext1, t[-1] +
                        x_ext2, freqs[0] - y_ext1, freqs[-1] + y_ext2]
    kwargs["cmap"] = "gray_r"
    kwargs["aspect"] = "auto"
    kwargs["origin"] = "lower"
    kwargs["interpolation"] = "nearest"

    return kwargs


def plot_slice(tempo_slice, freqs, tempo, harmonics=False):
    ylim = tempo_slice.max() + 1
    plt.vlines(
        tempo,
        ymin=0,
        ymax=ylim,
        linestyle="--",
        colors="r",
        label=f"{tempo} bpm")

    if harmonics:
        plt.vlines(tempo * 2, ymin=0, ymax=ylim, linestyle="--", colors="r")
        plt.vlines(tempo * 4, ymin=0, ymax=ylim, linestyle="--", colors="r",
                   alpha=0.6)

    plt.plot(freqs, tempo_slice, label="tempogram slice")
    plt.xlim(30, 300)
    plt.legend()


def get_slope(model_output, frequencies):
    """
    Return values to translate model output to BPM frequencies
    """
    if not isinstance(model_output, np.ndarray):
        raise TypeError("head_output should be an array")

    x = model_output
    y = frequencies
    n = np.size(x)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    Sxy = np.sum(x * y) - n * x_mean * y_mean
    Sxx = np.sum(x * x) - n * x_mean * x_mean

    a = Sxy / Sxx
    b = y_mean - a * x_mean

    return a, b
