#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np

import matplotlib.pyplot as plt


import units as unit

# plotting

def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab: ax.xaxis.set_label_text(xlab)
    if ylab: ax.yaxis.set_label_text(ylab)
    if title: ax.set_title(title, {'fontsize': 20 })
    if legend != None: ax.legend(legend, loc='best')
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if min_yspan != None: ensure_yspan(ax, min_yspan)

def ensure_yspan(ax, yspan):
    ymin, ymax = ax.get_ylim()
    if ymax-ymin < yspan:
        ym =  (ymin+ymax)/2
        ax.set_ylim(ym-yspan/2, ym+yspan/2)

def prepare_fig(fig=None, window_title=None, figsize=(20.48, 10.24), margins=None, suptitle=None):
    if fig == None: 
        fig = plt.figure(figsize=figsize)
    else:
        plt.figure(fig.number)
    if margins:
        left, bottom, right, top, wspace, hspace = margins
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            hspace=hspace, wspace=wspace)
    if window_title:
         fig.canvas.manager.set_window_title(window_title)
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig
 


# text formating
def format_trim(trim_results):
    return f"aoa {np.rad2deg(trim_results['aoa'][0]): .2f}deg throttle {trim_results['dthr'][0]*100:.2f}% phr {np.rad2deg(trim_results['dtrim'][0]): .2f}deg"




def get_data(this_dict):
    """Retrieve the content of function output dict as a list of values
    """
    return [v[0] for v in this_dict.values()]


def nice_print(this_dict):
    """Nice print of function output dict
    Key is used as the label
    The number of digits can be added after the unit name, 2 digits is the default
    """
    for key in this_dict.keys():
        if len(this_dict[key])<3:
            print(key, " = ", "%.2f" % unit.convert_to(this_dict[key][1], this_dict[key][0]), " # ", this_dict[key][1])
        else:
            fmt = "%." + str(this_dict[key][2]) + "f"
            print(key, " = ", fmt % unit.convert_to(this_dict[key][1], this_dict[key][0]), " # ", this_dict[key][1])
    print("")


def draw_polars(polar_list, mode="drag"):
    """Print aerodynamic polar curves
    """
    plot_title = "Aerodynamic polar"
    window_title = "QDV analytic"

    fig, axes = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(plot_title)
    fig.suptitle(window_title, fontsize=14)

    if mode=="lift":
        for polar in polar_list:
            plt.plot(np.degrees(polar["aoa"]), polar["cz"], linewidth=1, label=polar["title"])

        plt.grid(True)

        plt.ylabel('Lift (no_dim)')
        plt.xlabel('AoA (deg)')
        plt.legend(loc="lower right")

    elif mode == "drag":
        for polar in polar_list:
            plt.plot(polar["cx"], polar["cz"], linewidth=1, label=polar["title"])

        plt.grid(True)

        plt.ylabel('Lift (no_dim)')
        plt.xlabel('Drag (no_dim)')
        plt.legend(loc="lower right")


def show_time_simulation(title, data):
    """Draw time simulation delivered in data
    """
    label = list(data.keys())
    ncurve = len(label) - 1
    abscissa = label[0]

    fig, axis = plt.subplots(ncurve, sharex=True)
    fig.canvas.manager.set_window_title("Time Simulation")
    fig.suptitle(title, fontsize=16)

    for i, key in enumerate(label[1:]):
        data[key][0] = unit.convert_to(data[key][1], data[key][0])
        axis[i].plot(data[abscissa][0], data[key][0])
        axis[i].set_ylabel(key+" ("+data[key][1]+")", rotation=0, fontsize=14, labelpad=40)
        axis[i].grid(True)

    plt.xlabel(abscissa + " (s)", fontsize=14)
    plt.tight_layout()



