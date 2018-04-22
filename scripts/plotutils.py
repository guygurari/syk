import matplotlib.pyplot as plt

def prepare_axes(df, title='', logx=False, logy=False):
    axes = plt.gca()
    axes.set_title(title)
    axes.set_xlabel(df.columns[0])
    axes.set_xlim(df[df.columns[0]].min(), df[df.columns[0]].max())
    axes.set_ylim(df[df.columns[1]].min(), df[df.columns[1]].max())
    try:
        axes.legend_.remove()
    except AttributeError: 
        pass
    if logx: axes.set_xscale('log')
    if logy: axes.set_yscale('log')

def plot_dataframe(df, title='', logx=False, logy=False):
    df.plot(df.columns[0], df.columns[1])
    prepare_axes(df, title, logx, logy)
    plt.show()

def plot_dataframe_scatter(df, title='', logx=False, logy=False):
    df.plot.scatter(df.columns[0], df.columns[1])
    prepare_axes(df, title, logx, logy)
    plt.show()

def plot_two_dataframes(
        df1, df2,
        title='', ylabel1='', ylabel2='',
        logx=False, logy=False,
        save_to_file=None):
    fig, ax1 = plt.subplots()
    ax1.plot(df1[df1.columns[0]], df1[df1.columns[1]], 'b-')

    ax1.set_title(title)
    ax1.set_xlabel(df1.columns[0])
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel(ylabel1, color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    if logx: ax1.set_xscale('log')
    if logy: ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(df2[df2.columns[0]], df2[df2.columns[1]], 'r-')
    ax2.set_ylabel(ylabel2, color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    if logx: ax2.set_xscale('log')
    if logy: ax2.set_yscale('log')
    plt.show()

    if save_to_file != None:
        plt.savefig(save_to_file)
