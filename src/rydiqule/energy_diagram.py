# MIT License
#
# Original Copyright (c) 2017 Giacomo Marchioro
# Modified by Duy Nguyen and David Meyer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Energy profile diagram

This is a simple script to plot energy profile diagram using matplotlib.

.. code-block:: text

    E|          4__
    n|   2__    /  \\
    e|1__/  \__/5   \\
    r|  3\__/       6\__
    g|
    y|

Original author is Giacomo Marchioro.
The following is modified from https://github.com/giacomomarchioro/PyEnergyDiagrams
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np

from collections.abc import Collection
from typing import Tuple, List, Dict, Optional, Union

COLORS = [
    '#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc',
    '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9'
]
"Seaborn colorblind palette, for use with coupling arrows."

COLORS_W = [
    '#006374', '#b8850a', '#3c3c3c', '#a23582', '#592f0d',
    '#591e71', '#8c0800', '#12711c', '#b1400d', '#001c7f'
]
"""Reversed Seaborn colorblind palette, for use with wavy arrows.

Reversing the order helps limit color overlap when using default ordering.
"""


def draw_wiggly_arrow(ax: plt.Axes, start: Collection, stop: Collection,
                      amp: float = 0.07, nhalfwaves: int = 8, arrow_size: float = 0.06,
                      linestyle: str = 'solid', color: str = 'k', alpha: float = 1.0) -> None:
    """
    Helper funtion that draws a wavy arrow between two points on a plot.

    Parameters
    ----------
    ax: :class:`matplotlib:matplotlib.axes.Axes`
        Axes to add the wiggly arrow to.
    start: tuple
        Arrow start point in axes units
    stop: tuple
        Arrow stop point in axes units
    amp: float, optional
        Amplitude of the wave in axes units.
        Default is 0.07.
    nhalfwaves: int, optional
        Number of half-waves to wave the arrow.
        Default is 8.
    arrow_size: float, optional
        Size of the arrow in axes units.
        Default is 0.06.
    linestyle: str, optional
        Matplotlib linestyle definition.
        Default is `'solid'`.
    color: str, optional
        Matplotlib color specification.
        Default is `'k'`.
    alpha: float, optional
        Matplotlib alpha specification.
        Default is 1.
    """
    start = np.array(start)
    stop = np.array(stop)
    vec = stop-start
    dist = np.sqrt(vec.dot(vec)) - 2*arrow_size  # accounts for arrow head
    ang = np.arctan2(*vec[::-1])

    # define wiggly line
    omega = np.pi*nhalfwaves
    phi = 0
    x0 = np.linspace(0,dist,151) + start[0]
    y0 = amp*np.sin(omega*x0 + phi) + start[1]
    line = mpl.lines.Line2D(x0,y0,color=color,alpha=alpha,linestyle=linestyle)
    # rotate it to the correct angle
    line.set_transform(mpl.transforms.Affine2D().rotate_around(*start,ang) + ax.transData)

    # define the arrowhead
    verts = np.array([[-2,1],[-2,-1],[0,0],[-2,1]]).astype(float) * arrow_size
    verts[:,1] += stop[1]
    verts[:,0] += stop[0]
    path = mpl.path.Path(verts)
    patch = mpl.patches.PathPatch(path,fc=color,ec=color,alpha=alpha)
    # rotate it
    patch.set_transform(mpl.transforms.Affine2D().rotate_around(*stop, ang) + ax.transData)
    # plot them
    ax.add_line(line)
    ax.add_patch(patch)


class ED(object):
    "Energy diagram class"

    def __init__(self, aspect: str = 'equal') -> None:
        """
        Constructor for an energy diagram.

        This class contains methods to add energy levels and connections between them.
        It uses matplotlib to generate the figure, so standard manipulations via
        matplotlib function calls (including saving of the figure) can be done.

        Call :meth:`~.plot` to actually create the plot.

        Parameters
        ----------
        aspect: str,optional
            Kwarg passed to `fig.add_subplot()`. Default is `'equal'`.

        Note
        ----
        Calling :meth:`~.plot` in a jupyter notebook will automatically show
        the generated figure thanks to jupyter notebook magic in handling
        matplotlib figures.
        If you wish to see the plot outside a jupyter notebook, you will need
        to call `plt.show()` just like any other matplotlib figure.
        """
        # plot parameters
        self.ratio = 1.6181
        self.auto_adjust = ('dimension', 'space', 'offset')
        self.dimension: float
        self.space: float
        self.offset: float
        self.offset_ratio = 0.02
        self.color_bottom_text = 'blue'
        self.aspect = aspect
        self.round_energies_at_digit: Union[str,int] = "keep all digits"
        # data
        self.pos_number = 0
        self.energies: List[float] = []
        self.positions: List[float] = []
        self.colors: List[str] = []
        self.top_texts: List[str] = []
        self.bottom_texts: List[str] = []
        self.left_texts: List[str] = []
        self.right_texts: List[str] = []
        self.links: List = []
        self.arrows: List = []
        self.wiggly_arrows: List = []
        self.arrows_linestyles: Dict = {}
        self.level_linestyles: List[str] = []
        self.wiggly_linestyles: Dict = {}
        # matplotlib fiugre handlers
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def add_level(self, energy: float, bottom_text: str = '',
                  position: Optional[Union[str, float]] = None, color: str = 'k',
                  top_text: str = 'Energy', right_text: str = '',
                  left_text: str = '', linestyle: str = 'solid') -> None:
        '''
        This method add a new energy level to the plot.

        Parameters
        ----------
        energy : int
                 The energy of the level in Kcal mol-1
        bottom_text  : str
                The text on the bottom of the level (label of the level)
                (default '')
        position  : str
                The position of the level in the plot. Keep it empty to add
                the level on the right of the previous level use 'last' as
                argument for adding the level to the last position used
                for the level before.
                An integer can be used for adding the level to an arbitrary
                position.
                (default  None)
        color  : str
                matplotlib color specification of the level  (default  'k')
        top_text  : str
                Text on the top of the level. By default it will print the
                energy of the level. (default  'Energy')
        right_text  : str
                Text at the right of the level. (default  '')
        left_text  : str
                Text at the left of the level. (default  '')
        linestyle  : str
                The linestyle of the level, one of the following values:
                'solid', 'dashed', 'dashdot', 'dotted' (default  'solid')
        '''

        if position is None:
            position = self.pos_number + 1
            self.pos_number += 1
        elif isinstance(position, (int, float)):
            pass
        elif position == 'last' or position == 'l':
            position = self.pos_number
        else:
            raise ValueError(("Position must be None or 'last' (abrv. 'l') or "
                              "in case an integer or float specifing the position. "
                              "It was: %s" % position))
        if top_text == 'Energy':
            if self.round_energies_at_digit == "keep all digits":
                top_text = str(energy)
            else:
                assert isinstance(self.round_energies_at_digit, int)
                top_text = str(round(energy,self.round_energies_at_digit))

        link: List[Tuple[int, str, float, str]] = []
        self.colors.append(color)
        self.energies.append(energy)
        self.positions.append(position)
        self.top_texts.append(top_text)
        self.bottom_texts.append(bottom_text)
        self.left_texts.append(left_text)
        self.right_texts.append(right_text)
        self.links.append(link)
        self.level_linestyles.append(linestyle)
        self.arrows.append([])
        self.wiggly_arrows.append([])

    def add_arrow(self, start_level_id: int, end_level_id: int, linestyle: str) -> None:
        '''
        Add a arrow between two energy levels using IDs of the level.

        Use self.plot(show_index=True) to show the IDs of the levels.

        Parameters
        ----------
        start_level_id : int
                 Starting level ID
        end_level_id : int
                 Ending level ID
        linestyle: str
                 matplotlib linestyle string
        '''
        self.arrows[start_level_id].append(end_level_id)
        self.arrows_linestyles[(start_level_id,end_level_id)] = linestyle

    def add_wiggly_arrow(self, start_level_id: int, end_level_id: int, ls_dict: dict) -> None:
        '''
        Add a wiggly arrow between two energy levels using IDs of the level.

        Use self.plot(show_index=True) to show the IDs of the levels.

        Parameters
        ----------
        start_level_id : int
                 Starting level ID
        end_level_id : int
                 Ending level ID
        ls_dict: dict
                 Dictionary of linestyle parameters.
                 Passed as kwargs to :func:`~.draw_wiggly_arrow`.
        '''
        self.wiggly_arrows[start_level_id].append(end_level_id)
        self.wiggly_linestyles[(start_level_id,end_level_id)] = ls_dict

    def add_link(self, start_level_id: int, end_level_id: int,
                 color: str = 'k',
                 ls: str = '--',
                 linewidth: float = 1,
                 ) -> None:
        '''
        Add a link between two energy levels using IDs of the level.

        Use self.plot(show_index=True) to show the IDs of the levels.

        Parameters
        ----------
        start_level_id : int
                 Starting level ID
        end_level_id : int
                 Ending level ID
        color : str
                matplotlib color specification of the line
        ls : str
                matplotlib line style e.g. -- , ..
        linewidth : float
                line width
        '''
        self.links[start_level_id].append((end_level_id, ls, linewidth, color))

    def plot(self, show_IDs: bool = False, ylabel: str = "Energy / $kcal$ $mol^{-1}$",
             ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        '''
        Plot the energy diagram.

        Use show_IDs=True for showing the IDs of the
        energy levels and allowing an easy linking.

        .. code-block:: text

            E|          4__
            n|   2__    /  \\
            e|1__/  \__/5   \\
            r|  3\__/       6\__
            g|
            y|

        Parameters
        ----------
        show_IDs : bool
            show the IDs of the energy levels
        ylabel : str
            The label to use on the left-side axis. "Energy / $kcal$
            $mol^{-1}$" by default.
        ax : :class:`matplotlib:matplotlib.axes.Axes`
            The axes to plot onto. If not specified, a Figure and Axes will be
            created for you.

        Returns
        -------
        fig: :class:`matplotlib:matplotlib.figure.Figure`
            Figure handle for the generated figure.
        ax: :class:`matplotlib:matplotlib.axes.Axes`
            Axes handle for the generated figure.
        '''

        # Create a figure and axis if the user didn't specify them.
        if ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, aspect=self.aspect)
        # Otherwise register the axes and figure the user passed.
        else:
            self.ax = ax
            self.fig = ax.figure
            # Constrain the target axis to have the proper aspect ratio
            self.ax.set_aspect(self.aspect)
        self.ax.set_ylabel(ylabel)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.axis('off')
        self.__auto_adjust()

        data = list(
            zip(
                self.energies,  # 0
                self.positions,  # 1
                self.bottom_texts,  # 2
                self.top_texts,  # 3
                self.colors,  # 4
                self.right_texts,  # 5
                self.left_texts,  # 6
                self.level_linestyles
            )
        )
        # plot energy levels
        level_coords = {}
        for level in data:
            start = level[1]*(self.dimension+self.space)
            self.ax.hlines(
                level[0],
                start,
                start + self.dimension,
                color=level[4],
                linestyles=level[7]
            )
            level_coords[level[0]] = start
            self.ax.text(
                start + self.dimension,  # X
                level[0] - self.offset,  # Y
                level[2],  # self.bottom_text
                horizontalalignment='left',
                verticalalignment='top',
                color=self.color_bottom_text
            )
        self.level_counter = len(level_coords)
        if show_IDs:
            # for showing the ID allowing the user to identify the level
            for ind, level in enumerate(data):
                start = level[1]*(self.dimension+self.space)
                self.ax.text(
                    start,
                    level[0]+self.offset,
                    str(ind),
                    horizontalalignment='right',
                    color='black'
                )

        # draw coupling arrows between levels
        for idx, arrow in enumerate(self.arrows):
            diff = 0.0
            coupling_color = COLORS[idx]
            for i in arrow:
                linestyle = self.arrows_linestyles.get((idx, i))
                start = self.positions[idx]*(self.dimension+self.space)
                x1 = start + 0.5*self.dimension  # arrow base
                x2 = start + 0.5*self.dimension
                y1 = self.energies[idx]
                y2 = self.energies[i]
                self.ax.annotate(
                    "",
                    xy=(level_coords[i] + diff, y2),  # arrow tip
                    xytext=(x1, y1),  # arrow base
                    arrowprops=dict(
                        color=coupling_color,
                        linewidth=2.5,
                        arrowstyle='->',
                        linestyle=linestyle,
                        shrinkA=0,
                        shrinkB=0,
                    )
                )
                diff += 0.4

        # draw wiggly coupling arrows between levels
        for idx, warrow in enumerate(self.wiggly_arrows):
            diff = 0.0
            coupling_color = COLORS_W[idx]
            for i in warrow:
                ls_dict = self.wiggly_linestyles.get((idx, i),{})
                start = self.positions[idx]*(self.dimension+self.space)
                x1 = start + 0.9*self.dimension  # arrow base
                x2 = level_coords[i] + 0.9*self.dimension - diff*np.abs(idx-i)
                y1 = self.energies[idx]
                y2 = self.energies[i]
                draw_wiggly_arrow(self.ax, (x1,y1),
                                  (x2, y2),
                                  color=coupling_color,
                                  **ls_dict)
                diff += 0.1

        # draw links between levels
        for idx, link in enumerate(self.links):
            # here we connect the levels with the links
            # x1, x2   y1, y2
            for i in link:
                start = self.positions[idx]*(self.dimension+self.space)
                x1 = start + self.dimension
                x2 = self.positions[i[0]]*(self.dimension+self.space)
                y1 = self.energies[idx]
                y2 = self.energies[i[0]]
                line = Line2D(
                    [x1, x2],
                    [y1, y2],
                    ls=i[1],
                    linewidth=i[2],
                    color=i[3]
                )
                self.ax.add_line(line)

        return self.fig, self.ax

    def __auto_adjust(self) -> None:
        '''
        Sets the ratio to the best dimension and space between
        the levels.
        '''
        # Max range between the energy
        Energy_variation = abs(max(self.energies) - min(self.energies))
        if 'dimension' in self.auto_adjust or 'space' in self.auto_adjust:
            # Unique positions of the levels
            unique_positions = float(len(set(self.positions)))
            space_for_level = Energy_variation*self.ratio/unique_positions
            self.dimension = space_for_level*0.7
            self.space = space_for_level*0.3
        if 'offset' in self.auto_adjust:
            self.offset = Energy_variation*self.offset_ratio
