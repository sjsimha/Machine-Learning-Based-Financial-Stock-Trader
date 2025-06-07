import pandas as pd
import matplotlib.pyplot as plt
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as msc
import datetime as dt


def author():
    return 'ssimha31'


def run(exp_type, file):
    sd_train = dt.datetime(2008, 1, 1)
    ed_train = dt.datetime(2009, 12, 31)

    if exp_type == 'IN_SAMPLE':
        sd_test = sd_train
        ed_test = ed_train
        sample_display_string = 'In-Sample'
        file_name = 'images/exp1_in_sample.png'
    else:  # OUT_OF_SAMPLE
        sd_test = dt.datetime(2010, 1, 1)
        ed_test = dt.datetime(2011, 12, 31)
        sample_display_string = 'Out-Of-Sample'
        file_name = 'images/exp1_out_of_sample.png'

    symbol = 'JPM'
    sv = 100000
    commission = 9.95
    impact = 0.005

    man = ms.ManualStrategy()
    df_manual_trades = man.testPolicy(symbol, sd_test, ed_test, sv)

    stl = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    stl.add_evidence(symbol, sd_train, ed_train, sv)
    df_stl_trades = stl.testPolicy(symbol, sd_test, ed_test, sv)

    df_benchmark_trades = man.get_benchmark(symbol, sd_test, ed_test, sv)

    mp = msc.compute_portvals(sd_test, ed_test, df_manual_trades, sv, commission, impact)
    sp = msc.compute_portvals(sd_test, ed_test, df_stl_trades, sv, commission, impact)
    bp = msc.compute_portvals(sd_test, ed_test, df_benchmark_trades, sv, commission, impact)

    plot_exp1(sd_test, ed_test, symbol, mp / mp.iloc[0], sp / sp.iloc[0], bp / bp.iloc[0], file_name,
              sample_display_string)

    write_results(file, sd_test, ed_test, symbol, sample_display_string, mp, sp, bp)


def write_results(file, sd, ed, symbol, sample_type_str, mp, sp, bp):
    # Portfolio Stats
    m_cr, m_adr, m_std, m_sr = msc.compute_portfolio_stats(mp.values)
    s_cr, s_adr, s_std, s_sr = msc.compute_portfolio_stats(sp.values)
    b_cr, b_adr, b_std, b_sr = msc.compute_portfolio_stats(bp.values)

    m_cr = m_cr[0]
    b_cr = b_cr[0]
    s_cr = s_cr[0]

    with open(file, 'a') as sf:
        sf.write(f'\n****Experiment 1 Manual Strategy Results {sd.strftime("%Y-%m-%d")} to {ed.strftime("%Y-%m-%d")} '
                 f'{sample_type_str} {symbol} '
                 f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        sf.write('Cumulative Return,Standard Deviation,Mean Daily Return,Sharpe Ratio\n')
        s = ','.join(['{:.6f}'.format(m_cr), '{:.6f}'.format(m_std), '{:.6f}'.format(m_adr), '{:.6f}'.format(m_sr)])
        sf.write(s + '\n')

        sf.write(f'\n****Experiment 1  Strategy Learner Results {sd.strftime("%Y-%m-%d")} to {ed.strftime("%Y-%m-%d")} '
                 f'{sample_type_str} {symbol} '
                 f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        sf.write('Cumulative Return,Standard Deviation,Mean Daily Return,Sharpe Ratio\n')
        s = ','.join(['{:.6f}'.format(s_cr), '{:.6f}'.format(s_std), '{:.6f}'.format(s_adr), '{:.6f}'.format(s_sr)])
        sf.write(s + '\n')

        sf.write(f'\n****Experiment 1 Benchmark Results {sd.strftime("%Y-%m-%d")} to {ed.strftime("%Y-%m-%d")}  '
                 f'{sample_type_str} {symbol} '
                 f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        sf.write('Cumulative Return,Standard Deviation,Mean Daily Return,Sharpe Ratio\n')
        s = ','.join(['{:.6f}'.format(b_cr), '{:.6f}'.format(b_std), '{:.6f}'.format(b_adr),  '{:.6f}'.format(b_sr)])
        sf.write(s + '\n')


def plot_exp1(sd, ed, symbol, df_m, df_s, df_b, file_name, sample_type_str, show_plot=False):
    df = pd.DataFrame()
    df['Manual Strategy'] = df_m.iloc[:, 0]
    df['Strategy Learner'] = df_s.iloc[:, 0]
    df['Benchmark'] = df_b.iloc[:, 0]
    df.index = df_m.index

    ax = df.plot(fontsize=8, style={'Manual Strategy': 'r', 'Benchmark': '#A020F0', 'Strategy Learner': 'b'})
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Cumulative Value')

    plt.xlim(sd, ed)
    plt.title(f'Manual and Strategy Learner v/s Benchmark {sample_type_str}\n{symbol} '
              f'{sd.strftime("%m/%d/%Y")} - {ed.strftime("%m/%d/%Y")}')
    plt.grid()
    plt.margins(x=0, y=0)

    if show_plot:
        plt.show()
    plt.savefig(file_name, format='png', bbox_inches='tight', pad_inches=0.050)
    plt.close()


def run_experiment1(file):
    run('IN_SAMPLE', file)
    run('OUT_OF_SAMPLE', file)
