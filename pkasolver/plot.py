import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

def plot_results(prediction, true_vals, name: str):
    """Plot the prediction results in three subgraphs, showing the regression of y_hat against y."""

    a = {"y": true_vals, "y_hat": prediction}
    df = pd.DataFrame(data=a)

    r2 = (stats.pearsonr(df["y"], df["y_hat"])[0]) ** 2
    d = df["y"] - df["y_hat"]
    mse_f = np.mean(d ** 2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1 - (sum(d ** 2) / sum((df["y"] - np.mean(df["y"])) ** 2))

    kl_div = compute_kl_divergence(df["y"], df["y_hat"], n_bins=20)
    js_div = compute_js_divergence(df["y"], df["y_hat"], n_bins=20)

    stat_info = f"""
    $r^2$ = {r2_f:.2} 
    MAE = {mae_f:.2}
    MASE = {mse_f:.2}
    RMSE = {rmse_f:.2}
    """

    dist_info = f"""
    kl divergence = {kl_div:.2}
    js divergence = {js_div:.2}
    """

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(name)
    plt.subplot(221)

    ax = sns.regplot(x="y", y="y_hat", data=df)
    ax.text(
        0,
        1,
        stat_info,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        linespacing=1,
    )
    ax.set_xlabel("$\mathrm{pKa}_{exp}$")
    ax.set_ylabel("$\mathrm{pKa}_{calc}$")

    plt.subplot(222)
    ax = sns.distplot(df["y"], bins=20, label="$\mathrm{pKa}_{exp}$")
    sns.distplot(df["y_hat"], bins=20, label="$\mathrm{pKa}_{calc}$")
    ax.set_xlabel("pKa")
    ax.text(
        0,
        1,
        dist_info,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        linespacing=1,
    )
    ax.legend()

    plt.subplot(223)
    ax = sns.distplot(df["y"] - df["y_hat"], bins=10)
    ax.set_xlabel("$\mathrm{pKa}_{exp} - \mathrm{pKa}_{calc}$")

    plt.show()
    plt.close()