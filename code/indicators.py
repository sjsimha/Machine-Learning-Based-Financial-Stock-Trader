import datetime as dt
from util import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates


def author():
    return 'ssimha31'


def run(symbols, sd, ed, file_name='', plot_symbol='JPM'):
    get_rsi(symbols, sd, ed, do_plot=True, file_name=file_name, plot_symbol=plot_symbol)
    get_bbp(symbols, sd, ed, do_plot=True, file_name=file_name, plot_symbol=plot_symbol)
    get_sco(symbols, sd, ed, do_plot=True, file_name=file_name, plot_symbol=plot_symbol)
    get_roc(symbols, sd, ed, do_plot=True, file_name=file_name, plot_symbol=plot_symbol)
    get_wma(symbols, sd, ed, do_plot=True, file_name=file_name, plot_symbol=plot_symbol)


def get_rsi(symbols, sd, ed, do_plot=False, file_name='', plot_symbol='JPM'):
    lookback = 14
    factor_of_safety = 11  # account for non-trading days

    # get validated price data
    df_prices = get_cleansed_data(symbols, sd, ed, lookback + factor_of_safety)

    # first compute daily returns
    daily_returns = df_prices - df_prices.shift(1)

    # seggregate up and down returns; note that zero-return rows go to both
    up = daily_returns[daily_returns >= 0]
    down = daily_returns[daily_returns <= 0]

    # replace Nan with zero
    up = up.fillna(0)
    down = down.fillna(0)

    # Heart of vectorization; sum of any particular day going backwards to lookback window
    # achieved by calculating difference in cumulative sums of start and end ends of lookback window
    # till end of array; finally compute average over the lookback period. Need to access underlying numpy array
    # since similar calculation with dataframe doesn't work as expected. Convert down values to absolute per forumula.
    up_gains = up.cumsum()
    up_gains.values[lookback:, :] = (up_gains.values[lookback:, :] - up_gains.values[0:-lookback, :]) / lookback

    down_losses = down.cumsum()
    down_losses.values[lookback:, ] = (-1 * (
            down_losses.values[lookback:, ] - down_losses.values[0:-lookback, ])) / lookback

    # Step 1. Calculate Relative Strength
    # if there were no down days, down_losses will be zero, but numpy/pandas marks those as np.inf, which we can
    # correct later
    rs = up_gains / down_losses

    # Step 2. Now calculate Relative Strength Index and fix infinity results to 100
    rsi = 100 - (100 / (1 + rs))
    rsi[rsi == np.inf] = 100

    # RSI values are only valid from the day for which 14 (or lookback) days of previous trading data was available,
    # so only return from there. Note that there will likely be a few extra days of valid RSI data at the beginning.
    rsi = rsi.iloc[lookback:]

    if do_plot:
        plot_rsi(rsi, plot_symbol, df_prices, show_plot=False, save_fig=True, fig_name='indicator_rsi.png')

    return rsi


def plot_rsi(df_rsi, plot_symbol, df_prices, show_plot=False, save_fig=False, fig_name=''):
    pd.plotting.deregister_matplotlib_converters()
    # determine the first day rsi results are available
    sd = df_rsi.index[0]

    # extract prices only from that range
    index_loc = df_prices.index.get_indexer([sd])[0]
    df_prices = df_prices.iloc[index_loc:]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    x = df_prices.index.values
    y1 = df_prices[plot_symbol]
    y2 = df_rsi[plot_symbol]

    y3 = y2[y2 < 30]
    y4 = y2[y2 > 70]

    ax1.plot(x, y1, color='#A020F0', linewidth=0.5)
    ax2.plot(x, y2, color='#7393B3', linewidth=0.5)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, fontsize=8)
    plt.tick_params(bottom=False)
    fig.suptitle('Relative Strength Index and JPM Prices \n14 day window', weight='bold')
    ax1.set_ylabel('Stock Price', weight='bold')
    ax2.set_ylabel('RSI', weight='bold')
    ax2.axhline(y=30, c="green", linewidth=1)
    ax2.axhline(y=70, c="red", linewidth=1)
    ax2.scatter(x=y3.index.values, y=y3, marker='^', color='green')
    ax2.scatter(x=y4.index.values, y=y4, marker='v', color='red')

    fig.set_facecolor('#D3D3D3')
    ax1.set_facecolor('#D8D8D8')
    ax2.set_facecolor('#D8D8D8')

    ax1.grid()
    ax2.grid()

    if show_plot:
        plt.show()

    if save_fig:
        save_figure(fig_name)
    plt.close()


def get_bbp(symbols, sd, ed, do_plot=False, file_name='', plot_symbol='JPM'):
    lookback = 20
    factor_of_safety = 20
    df_prices = get_cleansed_data(symbols, sd, ed, lookback + factor_of_safety)
    rolling_mean = df_prices.rolling(window=lookback).mean()
    rolling_std = df_prices.rolling(window=lookback).std(ddof=0)
    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    bbp = (df_prices - bb_lower) / (bb_upper - bb_lower)

    if do_plot:
        plot_bbp(rolling_mean, bb_lower, bb_upper, bbp, plot_symbol, df_prices, show_plot=False, save_fig=True,
                 fig_name='indicator_bbp.png')
    return bbp


def plot_bbp(df_sma, df_lower, df_upper, df_bbp, plot_symbol, df_prices, show_plot=False, save_fig=False, fig_name=''):
    pd.plotting.deregister_matplotlib_converters()

    bbp = df_bbp[plot_symbol].dropna(inplace=False)
    sd = bbp.index[0]
    ix = df_bbp.index.get_indexer([sd])[0]

    sma = df_sma.iloc[ix:][plot_symbol]
    lower = df_lower.iloc[ix:][plot_symbol]
    upper = df_upper.iloc[ix:][plot_symbol]
    prices = df_prices.iloc[ix:][plot_symbol]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    x = bbp.index.values

    y3 = bbp[bbp < 0]
    y4 = bbp[bbp > 1]

    ax1.plot(x, lower.values, color='#800080', linewidth=1)
    ax1.plot(x, sma.values, color='blue', linewidth=0.5, linestyle='--')
    ax1.plot(x, prices.values, color='red', linewidth=0.5, linestyle='--')
    ax1.plot(x, upper.values, color='#A020F0', linewidth=1)
    ax1.set_ylabel('Price', weight='bold')
    ax1.legend(['Lower', 'SMA', 'JPM', 'Upper'], fontsize='x-small', facecolor='#D8D8D8')

    fig.suptitle('Bollinger Band Percent and Bands (JPM) \n20 day +/- 2 Std', weight='bold')
    ax2.plot(x, bbp.values, color='black', linewidth=1)
    ax2.axhline(y=0, c="green", linewidth=1, linestyle='--')
    ax2.axhline(y=1, c="red", linewidth=1, linestyle='--')
    ax2.set_ylabel('BBP', weight='bold')

    ax2.scatter(x=y3.index.values, y=y3, marker='^', color='green')
    ax2.scatter(x=y4.index.values, y=y4, marker='v', color='red')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, fontsize=8)
    plt.tick_params(bottom=False)
    ax1.set_facecolor('#ECECEC')
    ax2.set_facecolor('#ECECEC')
    fig.set_facecolor('#D3D3D3')

    ax1.grid()
    ax2.grid()

    if show_plot:
        plt.show()

    if save_fig:
        save_figure(fig_name)

    plt.close()
    return


def get_sco(symbols, sd, ed, do_plot=False, file_name='', plot_symbol='JPM'):
    lookback = 14
    df_adj, df_close, df_high, df_low = get_sco_data(symbols, sd, ed, lookback)

    low = df_low.rolling(window=lookback).min()
    high = df_high.rolling(window=lookback).max()
    sco_fast_k = (df_adj - low) / (high - low) * 100
    sco_fast_d = sco_fast_k.rolling(window=3).mean()
    sco_slow_k = sco_fast_d.rolling(window=3).mean()
    sco_slow_d = sco_slow_k.rolling(window=3).mean()

    if do_plot:
        plot_sco(sco_slow_k, sco_slow_d, df_adj, plot_symbol, save_fig=True, fig_name='indicator_sco.png', show_plot=False)

    return sco_slow_d


def get_sco_data(symbols, sd, ed, lookback):
    factor_of_safety = 25
    buffer_days = lookback + factor_of_safety

    df_adj = get_cleansed_data(symbols, sd, ed, buffer_days)
    df_close = get_cleansed_data(symbols, sd, ed, buffer_days, 'Close')
    df_high = get_cleansed_data(symbols, sd, ed, buffer_days, 'High')
    df_low = get_cleansed_data(symbols, sd, ed, buffer_days, 'Low')

    df_ratio = df_adj / df_close
    df_close = df_close * df_ratio
    df_high = df_high * df_ratio
    df_low = df_low * df_ratio

    return df_adj, df_close, df_high, df_low


def plot_sco(sco_k, sco_d, df_adj, plot_symbol, show_plot=False, save_fig=False, fig_name=''):
    pd.plotting.deregister_matplotlib_converters()

    # get sco_d for plot_symbol
    sco_d_ps = sco_d[plot_symbol]
    start_dt = sco_d_ps.dropna(inplace=False).index.values[0]
    ix = sco_d_ps.index.get_indexer([start_dt])[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.suptitle('Stochastic Oscillator (JPM) \nSlow(14, 3)', weight='bold')
    x = sco_d_ps.iloc[ix:].index.values

    ax1.plot(x, df_adj.iloc[ix:][plot_symbol], color='#800080', linewidth=1)
    ax1.set_ylabel('Adjusted Close', weight='bold')
    ax2.plot(x, sco_d_ps.iloc[ix:], color='red', linewidth=1)
    ax2.plot(x, sco_k.iloc[ix:][plot_symbol], color='black', linewidth=1)
    ax2.axhline(y=20, c="green", linewidth=1, linestyle='--')
    ax2.axhline(y=80, c="red", linewidth=1, linestyle='--')
    ax2.set_ylabel('Stochastic Score', weight='bold')
    ax2.legend(['%D', '%K'], fontsize='x-small', facecolor='#D8D8D8')

    y3 = sco_d_ps[sco_d_ps < 20]
    y4 = sco_d_ps[sco_d_ps > 80]
    ax2.scatter(x=y3.index.values, y=y3, marker='^', color='green')
    ax2.scatter(x=y4.index.values, y=y4, marker='v', color='red')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, fontsize=8)
    plt.tick_params(bottom=False)
    ax1.set_facecolor('#D8D8D8')
    ax2.set_facecolor('#D8D8D8')
    fig.set_facecolor('#D3D3D3')

    ax2.set_facecolor('#D8D8D8')
    ax1.grid()
    ax2.grid()

    if show_plot:
        plt.show()

    if save_fig:
        save_figure(fig_name)
    plt.close()

    return


def get_roc(symbols, sd, ed, do_plot=False, file_name='', plot_symbol='JPM'):
    lookback = 14
    factor_of_safety = 25
    df_prices = get_cleansed_data(symbols, sd, ed, lookback + factor_of_safety)
    roc = 100 * (df_prices - df_prices.shift(lookback)) / df_prices.shift(lookback)

    if do_plot:
        plot_roc(roc, df_prices, plot_symbol, show_plot=False, save_fig=True, fig_name='indicator_roc.png')

    return roc


def plot_roc(roc, df_prices, plot_symbol, show_plot=False, save_fig=False, fig_name=''):
    pd.plotting.deregister_matplotlib_converters()

    roc_ps = roc[plot_symbol]
    start_dt = roc_ps.dropna(inplace=False).index.values[0]
    ix = roc_ps.index.get_indexer([start_dt])[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.suptitle('Rate of Change (JPM) \n12 day lookback', weight='bold')
    x = roc_ps.iloc[ix:].index.values

    ax1.plot(x, df_prices.iloc[ix:][plot_symbol], color='#800080', linewidth=1)
    ax1.set_ylabel('Adjusted Close', weight='bold')

    ax2.plot(x, roc_ps.iloc[ix:], color='#7393B3', linewidth=1)
    ax2.axhline(y=0, c="black", linewidth=1, linestyle='--')
    ax2.set_ylabel('Rate of Change', weight='bold')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, fontsize=8)
    plt.tick_params(bottom=False)
    ax1.set_facecolor('#D8D8D8')
    ax2.set_facecolor('#D8D8D8')
    fig.set_facecolor('#D3D3D3')

    ax1.grid()
    ax2.grid()

    if show_plot:
        plt.show()

    if save_fig:
        save_figure(fig_name)

    plt.close()
    return


def save_figure(fig_name):
    plt.savefig(f'images/{fig_name}')


def calc_wma(data, lookback):
    w = np.arange(1, lookback + 1)
    return (data * w).sum() / w.sum()


def get_wma(symbols, sd, ed, do_plot=False, file_name='', plot_symbol='JPM'):
    lookback = 200
    factor_of_safety = 100

    df_prices = get_cleansed_data(symbols, sd, ed, lookback + factor_of_safety)
    wma = df_prices.rolling(window=lookback).apply(calc_wma, raw=True, args=(lookback,))

    if do_plot:
        plot_wma(wma, df_prices, 'JPM', show_plot=False, save_fig=True, fig_name='indicator_wma.png')

    return wma


def plot_wma(wma, df_prices, plot_symbol, show_plot=False, save_fig=False, fig_name=''):
    pd.plotting.deregister_matplotlib_converters()
    ix = get_first_non_Nan_index(wma, plot_symbol)

    fig, ax = plt.subplots()
    x = wma[plot_symbol].iloc[ix:].index.values
    ax.plot(x, df_prices.iloc[ix:][plot_symbol], color='black', linewidth=1, label='JPM')
    ax.plot(x, wma.iloc[ix:][plot_symbol], color='#800080', linewidth=1, linestyle='--', label='200 day')

    fig_txt = fig.suptitle('Weighted Moving Average\n JPM 10 and 200 day')
    fig_txt.set(fontweight='bold')
    ax.set_xlabel('Date', weight='bold')
    ax.set_ylabel('Adjusted Close', weight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax.tick_params(bottom=False)
    ax.set_facecolor('#D8D8D8')
    ax.legend(fontsize='x-small', facecolor='#D8D8D8')
    fig.set_facecolor('#D3D3D3')
    ax.grid()

    if show_plot:
        plt.show()

    if save_fig:
        save_figure(fig_name)

    plt.close()
    return


def get_first_non_Nan_index(df, symbol):
    first_dt = df[symbol].dropna(inplace=False).index[0]
    ix = df.index.get_indexer([first_dt])[0]

    return ix


def get_cleansed_data(symbols, sd, ed, buffer_days, use_buffer_days=True, col='Adj Close'):
    if use_buffer_days:
        # start_dt = dt.datetime.strptime(sd, '%m-%d-%Y') - dt.timedelta(days=buffer_days)
        start_dt = sd - dt.timedelta(days=buffer_days)
    else:
        start_dt = sd

    df_prices = get_data(symbols, pd.date_range(start_dt, ed), colname=col)
    df_prices.fillna(method='ffill')
    df_prices.fillna(method='bfill')

    return df_prices


if __name__ == "__main__":
    m_file_name = 'p6_results.txt'
    m_symbol, m_start_date, m_end_date = 'JPM', '01-01-2008', '12-31-2009'
    run(['JPM', 'IBM'], m_start_date, m_end_date, m_file_name, plot_symbol=m_symbol)
