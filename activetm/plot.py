"""Plotting code"""
import colorsys
from itertools import cycle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

import GPy

LINES = ['-', '--', '-.', ':']
DEFAULT_COLORS = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

def get_separate_colors(count):
    """Get a number of colors equal to count"""
    num_div = np.linspace(0.0, 1.0, num=count, endpoint=False)
    return [[a, b, c] for (a, b, c) in [colorsys.hsv_to_rgb(d, 1.0, 1.0) \
            for d in num_div]]


def get_stats(mat):
    """Compute various statistics along the columns of mat

    Assumes that every row corresponds to the successive values of a given
    experiment.  Thus, a column is the spread of results for every experiment at
    a given step in the succession.
    """
    # compute the medians along the columns
    mat_medians = np.median(mat, axis=0)
    # compute the means along the columns
    mat_means = np.mean(mat, axis=0)
    # compute standard deviation
    mat_stddev = np.std(mat, axis=0)
    return mat_medians, mat_means, mat_stddev, mat_stddev


def _build_gpx(xdata):
    """Transform data so that it is suitable for Gaussian process

    The returned matrix will have the flattened xdata along its first column and
    other information in the other columns.
    """
    xdim0, xdim1 = xdata.shape
    result = np.zeros((xdata.size, 2))
    result[:, 0] = np.reshape(xdata, xdata.size)
    for i in range(xdim0):
        for j in range(xdim1):
            result[(i*xdim0)+j, 1] = j
    return result


class Plotter(object):
    """Class to encapsulate simple plotting"""

    def __init__(self, colors=None):
        self.fig, self.axis = plt.subplots(1, 1)
        self.linesplotted_count = 0
        self.linecycler = cycle(LINES)
        if colors is None:
            colors = DEFAULT_COLORS
        self.colorcycler = cycle(colors)
        self.ymax = float('-inf')

    def _line_plot(self, xdata, ydata, color, label):
        """Make a line plot; update maximum y"""
        cur_ymax = np.max(ydata)
        if cur_ymax > self.ymax:
            self.ymax = cur_ymax
        self.axis.plot(
            xdata,
            ydata,
            color=color,
            label=label,
            linewidth=3)

    def _error_plot(self, xdata, ydata, color, label):
        """Make a line plot with error bars; update maximum y"""
        ys_medians, ys_means, ys_errs_minus, ys_errs_plus = get_stats(ydata)
        cur_ymax = np.max(ys_medians + ys_errs_plus)
        if cur_ymax > self.ymax:
            self.ymax = cur_ymax
        self.axis.errorbar(
            xdata,
            ys_means,
            yerr=[ys_errs_minus, ys_errs_plus],
            color=color,
            label=label,
            linewidth=3)
        self.axis.errorbar(
            xdata,
            ys_medians,
            fmt='o',
            color=color)

    def _spread_plot(self, xdata, ydata, color, label):
        """Make a plot of the spread; update maximum y"""
        assert xdata.size == ydata.size
        #pylint:disable-msg=no-member
        choices = np.random.choice(xdata.size, 1000, replace=False)
        # sampling seems to address the MemoryError
        gpx = xdata.reshape((xdata.size, 1))[choices]
        gpy = ydata.reshape((ydata.size, 1))[choices]
        kernel = GPy.kern.RBF(input_dim=1)
        gpr = GPy.models.GPRegression(gpx, gpy, kernel)
        gpr.optimize()
        plotx = np.linspace(0, xdata.max(), 200)
        plotx = plotx.reshape((plotx.size, 1))
        pred_mean, _ = gpr.predict(plotx)
        pred_quants = gpr.predict_quantiles(plotx, quantiles=(25., 75.))
        self.axis.plot(
            plotx,
            pred_mean,
            alpha=0.75,
            color=color,
            linestyle='solid',
            label=label,
            linewidth=3)
        self.axis.plot(
            plotx,
            pred_quants[0],
            alpha=0.75,
            color=color,
            linestyle='dashed',
            linewidth=1.5)
        self.axis.plot(
            plotx,
            pred_quants[1],
            alpha=0.75,
            color=color,
            linestyle='dashed',
            linewidth=1.5)
        cur_ymax = np.max(pred_quants[1])
        if cur_ymax > self.ymax:
            self.ymax = cur_ymax

    def plot(self, xdata, ydata, label):
        """Add data to plot

        If xdata and ydata are 1D numpy arrays, then the plot will be a simple
        line plot.

        If xdata is a 1D numpy array and ydata is a 2D numpy array, then the
        plot will be a line plot with error bars, where the error bars are
        calculated per column of ydata.

        If xdata and ydata are 2D numpy arrays, then the plot will be drawn with
        the help of a Gaussian process fit to corresponding pairs in xdata and
        ydata (i.e., (xdata[x, y], ydata[x, y])).  These pairs will also be
        plotted as a scatter plot.

        In the course of plotting, the maximum y value is recorded.
        """
        color = next(self.colorcycler)
        xdimscount = len(xdata.shape)
        ydimscount = len(ydata.shape)
        if xdimscount == 1 and ydimscount == 1:
            self._line_plot(xdata, ydata, color, label)
        elif xdimscount == 1 and ydimscount == 2:
            self._error_plot(xdata, ydata, color, label)
        elif xdimscount == 2 and ydimscount == 2:
            self._spread_plot(xdata, ydata, color, label)
        else:
            raise Exception(
                'Bad arguments: xdata shape=' + str(xdata.shape) +\
                ' ydata shape=' + str(ydata.shape))

    def savefig(self, name):
        """Save plot"""
        self.axis.relim()
        self.axis.autoscale_view()
        minx, maxx = self.axis.get_xlim()
        deltax = (maxx - minx) * 0.05
        self.axis.set_xlim(minx-deltax, maxx+deltax)
        miny, _ = self.axis.get_ylim()
        deltay = (self.ymax - miny) * 0.05
        self.axis.set_ylim(miny-deltay, self.ymax+deltay)
        self.axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        self.fig.savefig(name, bbox_inches='tight')

    def set_title(self, title):
        """Set title"""
        self.axis.set_title(title)

    def set_xlabel(self, label):
        """Set x label"""
        self.axis.set_xlabel(label)

    def set_ylabel(self, label):
        """Set y label"""
        self.axis.set_ylabel(label)

    def set_ylim(self, lims):
        """Set y limits"""
        self.axis.set_ylim(lims)

    def close(self):
        """Free resources"""
        plt.close(self.fig)


def _run():
    """Example of how to use Plotter"""
    line_count = 4
    colors = get_separate_colors(line_count)
    plotter = Plotter(colors)
    xdata = np.linspace(0.0, 1.0, 10)
    for i in range(line_count):
        ymeans = (xdata * xdata / 2) - i
        perx_ycount = 50
        #pylint:disable-msg=no-member
        ydata = np.random.normal(size=len(xdata)*perx_ycount)
        ydata = ydata.reshape((perx_ycount, len(xdata)))
        # np.add adds ymeans to every column of ydata; transpose to get effect
        # of having added ymeans to every row of ydata
        ydata = np.add(ydata, ymeans).T
        plotter.plot(
            xdata,
            ydata,
            'line {:d}'.format(i))
    plotter.set_title('Example')
    plotter.set_xlabel('x label')
    plotter.set_ylabel('y label')
    plotter.savefig('example.pdf')


if __name__ == '__main__':
    _run()
