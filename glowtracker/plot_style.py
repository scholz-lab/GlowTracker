"""Dark matplotlib figures for the app's dark windows (DAQ, calibration), so a plot sits in its
card without a white box. Draw the figure as usual, then call `darken(fig)` before rendering."""
from __future__ import annotations

from matplotlib.colors import to_rgba

BACKGROUND = '#212226'      # the popup card
TEXT = '#d8d8dc'
EDGE = '#55565c'

# line colours that read on the dark background
BLUE = '#4ea1ff'
PINK = '#ff6b9a'
GREEN = '#3ddc97'
VIOLET = '#c792ea'
WHITE = '#f2f2f4'


def darken(fig, grid: bool = False) -> None:
    """Restyle a finished figure like a journal figure on the dark background: inward ticks on
    all four sides with minor ticks, thin axes, no grid (unless asked), frameless legends."""
    fig.patch.set_facecolor(BACKGROUND)
    for ax in fig.axes:
        isColorbar = getattr(ax, '_colorbar', None) is not None
        ax.set_facecolor(BACKGROUND)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
            spine.set_linewidth(0.8)
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', colors=TEXT, labelcolor=TEXT,
                       top=not isColorbar, right=not isColorbar)
        ax.tick_params(which='major', length=5, width=0.8)
        ax.tick_params(which='minor', length=2.5, width=0.6)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        if grid:
            ax.grid(True, which='major', color='#ffffff', alpha=0.08, linewidth=0.6)
        else:
            ax.grid(False, which='both')
        for text in ax.texts:
            if to_rgba(text.get_color())[:3] == (0, 0, 0):
                text.set_color(TEXT)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_color(TEXT)


def panelLabel(ax, letter: str) -> None:
    """A bold panel letter (a, b, ...) at the axes' top-left corner, outside the plot."""
    ax.text(-0.1, 1.04, letter, transform=ax.transAxes, fontsize=16, fontweight='bold',
            color=TEXT, va='bottom', ha='left')
