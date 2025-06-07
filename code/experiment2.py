import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import StrategyLearner as sl
import marketsimcode as msc
import datetime as dt


def author():
    return 'ssimha31'


def run(file):
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    sample_display_string = 'In-Sample'
    file_name_1 = 'images/exp2_chart1.png'
    file_name_2 = 'images/exp2_chart2.png'

    symbol = 'JPM'
    sv = 100000
    commission = 0.0

    stats = []
    portfolios = {}
    impacts = []
    trades = []
    crs = []
    for impact in np.arange(0, 0.015, 0.005):
        stl = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
        stl.add_evidence(symbol, sd, ed, sv)
        df_trades = stl.testPolicy(symbol, sd, ed, sv)

        portfolio = msc.compute_portvals(sd, ed, df_trades, sv, commission, impact)
        cr, adr, std, sr = msc.compute_portfolio_stats(portfolio.values)
        cr = cr[0]

        impacts.append('{:.3f}'.format(impact))
        trades.append(df_trades.shape[0])
        crs.append('{:.6f}'.format(cr))

        portfolios['{:.3f}'.format(impact)] = portfolio / portfolio.iloc[0]

    write_results(file, sd, ed, symbol, sample_display_string, impacts, crs, trades)
    plot_exp2_chart1(sd, ed, symbol, portfolios, file_name_1, sample_display_string, False)
    plot_exp2_chart2(sd, ed, symbol, impacts, trades, crs, file_name_2, sample_display_string, False)


def write_results(file, sd, ed, symbol, sample_type_str, impacts, crs, trades):
    with open(file, 'a') as sf:
        sf.write(f'\n****Experiment 2 Impact Variation Results {sd.strftime("%Y-%m-%d")} to {ed.strftime("%Y-%m-%d")} '
                 f'{sample_type_str} {symbol} '
                 f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        sf.write('Impact,Cumulative Return,Trades\n')

        for i in range(len(impacts)):
            s = ','.join([impacts[i], crs[i], str(trades[i])])
            sf.write(s + '\n')


def plot_exp2_chart2(sd, ed, symbol, impacts, trades, crs, file_name, sample_type_str, show_plot=False):
    plt.bar(impacts, trades)
    plt.xlabel('Impact')
    plt.ylabel('Number of Trades')
    plt.title(f'Effect of Impact on Number of Trades - {sample_type_str}\n{symbol} '
              f'{sd.strftime("%m/%d/%Y")} - {ed.strftime("%m/%d/%Y")}')

    for i in range(len(impacts)):
        plt.text(i, trades[i], trades[i], ha='center')

    if show_plot:
        plt.show()

    plt.savefig(file_name, format='png', bbox_inches='tight', pad_inches=0.050)
    plt.close()


def plot_exp2_chart1(sd, ed, symbol, portfolios, file_name, sample_type_str, show_plot=False):
    df = pd.DataFrame()

    for key, val in portfolios.items():
        df[key] = val.iloc[:, 0]

    df.index = list(portfolios.values())[0].index

    ax = df.plot(fontsize=8, style={'Manual Strategy': 'r', 'Benchmark': '#A020F0', 'Strategy Learner': 'b'})
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Cumulative Value')

    plt.xlim(sd, ed)
    plt.title(f'Effect of Impact on Strategy Learner Returns - {sample_type_str}\n{symbol} '
              f'{sd.strftime("%m/%d/%Y")} - {ed.strftime("%m/%d/%Y")}')
    plt.grid()

    if show_plot:
        plt.show()
    plt.savefig(file_name, format='png', bbox_inches='tight', pad_inches=0.050)
    plt.close()


def run_experiment2(file):
    run(file)
