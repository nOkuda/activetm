import colorsys
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

lines = ['-', '--', '-.', ':']
default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def get_separate_colors(count):
    num_div = np.linspace(0.0,1.0,num=count,endpoint=False)
    return [[a,b,c] for (a,b,c) in [colorsys.hsv_to_rgb(d,1.0,1.0) \
            for d in num_div]]

class Plotter(object):
    def __init__(self, colors=default_colors):
        self.fig, self.axis = plt.subplots(1,1)
        self.linesplotted_count = 0
        self.linecycler = cycle(lines)
        self.colorcycler = cycle(colors)

    def plot(self, xmeans, ymeans, label, ymedians, yerr=None,
            xmedians=None, xerr=None):
        color = next(self.colorcycler)
        self.axis.errorbar(xmeans, ymeans, xerr=xerr, yerr=yerr,
                fmt=next(self.linecycler), color=color, label=label,
                linewidth=3)
        if xmedians is not None:
            self.axis.errorbar(xmedians, ymedians, fmt='o', color=color)
        else:
            self.axis.errorbar(xmeans, ymedians, fmt='o', color=color)

    def savefig(self, name):
        self.axis.relim()
        self.axis.autoscale_view()
        minx, maxx = self.axis.get_xlim()
        deltax = (maxx - minx) * 0.05
        self.axis.set_xlim(minx-deltax, maxx+deltax)
        miny, maxy = self.axis.get_ylim()
        deltay = (maxy - miny) * 0.05
        self.axis.set_ylim(miny-deltay, maxy+deltay)
        self.axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        self.fig.savefig(name, bbox_inches='tight')

    def set_title(self, title):
        self.axis.set_title(title)

    def set_xlabel(self, label):
        self.axis.set_xlabel(label)

    def set_ylabel(self, label):
        self.axis.set_ylabel(label)

    def set_ylim(self, lims):
        self.axis.set_ylim(lims)

if __name__ == '__main__':
    # example of how to use Plotter
    line_count = 4
    colors = get_separate_colors(line_count)
    plotter = Plotter(colors)
    xmeans = np.linspace(0.0,1.0,10)
    for i in range(line_count):
        ymeans = (xmeans * xmeans / 2) - i
        yerr = np.vstack((np.array([0.1]*len(xmeans)),
            np.array([0.1]*len(xmeans))))
        plotter.plot(xmeans, ymeans, 'line {:d}'.format(i), ymeans,
                yerr=yerr)
    plotter.set_title('Example')
    plotter.set_xlabel('x label')
    plotter.set_ylabel('y label')
    plotter.savefig('example.pdf')

