import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import find_clouds as fc
import numpy.ma as ma
import numpy as np

cycle  = ('turquoise blue', 'blue7', 'red7', 'gray4')


def plot_clouds(start, end, allclouds_, time,  height, h0, cbh, plot_path):

    cloudtype = np.array([[255, 255, 255],
                          [200,200,200],
                          [230, 152, 63],
                          [89, 115, 255],
                         ]) / 255.
    ac = matplotlib.colors.ListedColormap(cloudtype)

    xmin, xmax = mdates.date2num(start), mdates.date2num(end)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 4.5, 1), cloudtype.shape[0])

    fig = plt.figure(figsize=(15, 4))
    cats = [
               "Clear sky",
               "Cold & mixed \nphase clouds",
               "Warm clouds",
               "Warm clouds\n with CBH < 1km",]

    ax0 = fig.add_subplot(1,1,1)     # 3 rows, 4 columns, 4rd cell
    ax0.axhline(y=1, linestyle = ':', alpha = 0.5, linewidth = 2, color = 'black')


    ax0.plot(time, np.array(h0)/1000., color = 'red', linewidth = 2, label = '0°C')
    ax0.annotate('0°C', xy=(0.23, 0.31), bbox=dict(ec = 'white', fc="white"), #(facecolor='none', edgecolor='red'))

                 color = 'red', xycoords='axes fraction',fontsize=14, ha='right', va='top')
    ax0.set_title(f'BCO: {start} UTC')
    cbh_4 = ma.masked_where(cbh   >4000, cbh)
    cbh_3 = ma.masked_where(cbh_4 >3000, cbh_4)
    cbh_2 = ma.masked_where(cbh_3 >2000, cbh_3)
    cbh_1 = ma.masked_where(cbh_2 >1000, cbh_2)

    cbh_4 = ma.masked_where(cbh_4 <= 3000, cbh_4)
    cbh_3 = ma.masked_where(cbh_3 <= 2000, cbh_3)
    cbh_2 = ma.masked_where(cbh_2 <= 1000, cbh_2)
    cbh_4 = ma.masked_where(cbh_4 > np.nanmin(np.array(h0)), cbh_4)

    ax0.step(time, cbh_4/1000., color = 'black', linewidth = 2, label = 'cloud base height')
    ax0.step(time, cbh_3/1000., color = 'black', linewidth = 1)
    ax0.step(time, cbh_2/1000., color = 'black', linewidth = 1)
    ax0.step(time, cbh_1/1000., color = 'black', linewidth = 1)
    #
    cp = ax0.pcolormesh(time[:], height[:]/1000., allclouds_ , cmap = ac, norm =norm )
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="3%", pad=0.25)
    cbar = plt.colorbar(cp, cax=cax0, ax=ax0, fraction=0.13, pad=0.025)
    cbar.set_ticks(np.arange(0,len(cats),1))
    cbar.ax.set_yticklabels(cats, fontsize =14)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=4)
    #cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)


    custom_legend_item = matplotlib.lines.Line2D([0], [0], color='black', linewidth=2, label='cloud base height (CBH)')

    # Add the custom legend item to the legend
    fig.legend(loc = (0.64, 0.78), handles=[custom_legend_item],fontsize = 16)


    ax = ax0

    #ax.set_title(sdate.strftime('%d/%m/%Y'))
    ax.set_ylabel('Height (km)')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylim(0,13)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    if (end-start).seconds /3600 > 2:
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval = 1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval = 15))
    if (end-start).seconds /3600 < 2:
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval = 30))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval = 5))

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major',  width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    ax.set_xlim(xmin, xmax)


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(14)

    plt.tight_layout()

    ymd = 10000*start.year + 100*start.month + start.day
    h1 = start.hour
    h2 = end.hour
    fig.savefig(f"{plot_path}/{ymd}_{h1}-{h2}UTC_cloudtypes.png", dpi=300)

    return fig, ax

