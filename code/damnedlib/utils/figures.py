import matplotlib.pyplot as plt
import tikzplotlib


def save_fig(
    fig: plt.figure,
    filename: str,
    save_png: bool = True,
    save_tikz: bool = False,
    save_pdf: bool = False,
):
    """Saves a matplotlib figure to the specified location.

    Args:
        fig (plt.figure): The figure you want to save
        filename (str): The name of the figure (with all path information, excluding any file extensions)
        save_png (bool, optional): Wether or not to save a png copy. Defaults to True.
        save_tikz (bool, optional): Wether or not to save a tikz copy. Defaults to False.
    """
    if save_png:
        fig.savefig(filename + ".png", facecolor="w", bbox_inches="tight", pad_inches=0)
    if save_pdf:
        fig.savefig(filename + ".pdf", facecolor="w", bbox_inches="tight", pad_inches=0)
    if save_tikz:
        tikzplotlib.clean_figure(fig=fig)
        size = fig.get_size_inches()
        tikzplotlib.save(
            figure=fig,
            filepath=filename + ".tikz",
            axis_width=f"{size[0]}in",
            axis_height=f"{size[1]}in",
            standalone=False,
            encoding="utf-8",
            strict=True,
        )
