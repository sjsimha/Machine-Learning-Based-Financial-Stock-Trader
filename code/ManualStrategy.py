import datetime as dt
from util import get_data
import pandas as pd
import marketsimcode as msc
import matplotlib.pyplot as plt
import indicators as ind


class ManualStrategy(object):

    def __init__(self):
        self.user_id = 'ssimha31'
        self.entry_trades_long = []
        self.entry_trades_short = []

        # Tuning parameters for RSI
        self.rsi_lower = 30
        self.rsi_upper = 70

        # Tuning parameters for STO
        self.sco_lower = 20
        self.sco_upper = 80

        self.exit_strategy = 'E'  # E for close positions; H for hold positions

    def author(self):
        return self.user_id

    def testPolicy(self, symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        df_prices = get_data([symbol], dates)
        df_prices.fillna(method='ffill')
        df_prices.fillna(method='bfill')

        # Indicators Used:
        #   Relative Strength Index (RSI - 14 period)
        #   %Bollinger Band (%BB with SMA +/- 2 Std.)
        #   Stoichastic Oscillator (SCO - Slow %K)
        # Majority voting system - enter position based on signals from any two indicators
        # Close positions - based on %BB indicator crossing 0.5 in the appropriate direction
        df_rsi = ind.get_rsi([symbol], sd, ed, do_plot=False, file_name='rsi.png', plot_symbol=symbol)
        df_bbp = ind.get_bbp([symbol], sd, ed, do_plot=False, file_name='bbp.png', plot_symbol=symbol)
        df_sco = ind.get_sco([symbol], sd, ed, do_plot=False, file_name='sco.png', plot_symbol=symbol)

        holdings = 0
        df_trades = pd.DataFrame()
        for date, row in df_prices.iterrows():
            rsi_action = self.get_rsi_action(symbol, date, df_rsi)
            sco_action = self.get_sco_action(symbol, date, df_sco)
            bbp_action = self.get_bbp_action(symbol, date, df_bbp, holdings)

            action = self.get_action(rsi_action, sco_action, bbp_action)
            if action == 1:
                df_trades, holdings = self.go_long(df_trades, date, symbol, holdings)
            elif action == -1:
                df_trades, holdings = self.go_short(df_trades, date, symbol, holdings)
            elif self.exit_strategy == 'E' and (action == 2 or action == -2):
                df_trades, holdings = self.close_position(df_trades, date, symbol, holdings)

        return df_trades

    def get_action(self, rsi_action, sco_action, bbp_action):
        # use a majority voting scheme for entering positions
        # use bbp midline crossover mark to exit positions
        if rsi_action == 1 and (sco_action == 1 or bbp_action == 1):
            return 1
        elif sco_action == 1 and (rsi_action == 1 or bbp_action == 1):
            return 1
        elif bbp_action == 1 and (rsi_action == 1 or sco_action == 1):
            return 1
        elif rsi_action == -1 and (sco_action == -1 or bbp_action == -1):
            return -1
        elif sco_action == -1 and (rsi_action == -1 or bbp_action == -1):
            return -1
        elif bbp_action == -1 and (rsi_action == -1 or sco_action == -1):
            return -1
        elif bbp_action == 2 or bbp_action == -2:
            return bbp_action
        else:
            return 0

    def get_rsi_action(self, symbol, date, df_rsi):
        rsi_ix = df_rsi.index.get_indexer([date])
        rsi = df_rsi.iloc[rsi_ix][symbol][0]

        if rsi < self.rsi_lower:
            return 1
        elif rsi > self.rsi_upper:
            return -1
        else:
            return 0

    def get_sco_action(self, symbol, date, df_sco):
        sco_ix = df_sco.index.get_indexer([date])
        prior_sco = df_sco.iloc[sco_ix - 1][symbol][0]
        cur_sco = df_sco.iloc[sco_ix][symbol][0]

        if prior_sco < self.sco_lower <= cur_sco:
            return 1
        elif prior_sco > self.sco_upper >= cur_sco:
            return -1
        else:
            return 0

    def get_bbp_action(self, symbol, date, df_bbp, holdings):
        bbp_ix = df_bbp.index.get_indexer([date])
        prior_bbp = df_bbp.iloc[bbp_ix - 1][symbol][0]
        cur_bbp = df_bbp.iloc[bbp_ix][symbol][0]

        if prior_bbp < 0 <= cur_bbp <= 0.5:
            return 1
        elif prior_bbp > 1 > cur_bbp >= 0.5:
            return -1
        elif prior_bbp < 0.5 <= cur_bbp and holdings > 0:
            # exit long position
            return 2
        elif prior_bbp > 0.5 >= cur_bbp and holdings < 0:
            # return short position
            return -2
        else:
            return 0

    def add_trade(self, df_trades, trade_dt, symbol, shares):
        df_temp = pd.DataFrame(index=[trade_dt], data=[shares], columns=[symbol])
        df_trades = pd.concat([df_trades, df_temp])
        df_trades.index.name = 'Date'

        return df_trades

    def close_position(self, df_trades, trade_dt, symbol, holdings):
        if holdings > 0:
            shares = -1000
        elif holdings < 0:
            shares = 1000
        else:
            return df_trades, holdings

        df_trades = self.add_trade(df_trades, trade_dt, symbol, shares)
        return df_trades, holdings + shares

    def go_long(self, df_trades, trade_dt, symbol, holdings):
        if holdings < 0:
            shares = 2000
        elif holdings == 0:
            shares = 1000
        else:
            return df_trades, holdings

        df_trades = self.add_trade(df_trades, trade_dt, symbol, shares)
        self.entry_trades_long.append(trade_dt)
        return df_trades, holdings + shares

    def go_short(self, df_trades, trade_dt, symbol, holdings):
        if holdings > 0:
            shares = -2000
        elif holdings == 0:
            shares = -1000
        else:
            return df_trades, holdings

        df_trades = self.add_trade(df_trades, trade_dt, symbol, shares)
        self.entry_trades_short.append(trade_dt)
        return df_trades, holdings + shares

    def get_benchmark(self, p_symbol, p_sd, p_ed, p_sv):
        df_prices = get_data([p_symbol], pd.date_range(start=p_sd, end=p_ed))
        first_trading_dt = df_prices.index.values[0]

        benchmark_trades = pd.DataFrame()
        benchmark_trades = self.add_trade(benchmark_trades, first_trading_dt, p_symbol, 1000)

        return benchmark_trades

    def plot_ms(self, sd, ed, symbol, df_tos, df_benchmark, file, sample_type_str, show_plot=False):
        df = pd.DataFrame()
        df['Manual Strategy'] = df_tos.iloc[:, 0]
        df['Benchmark'] = df_benchmark.iloc[:, 0]
        df.index = df_tos.index

        ax = df.plot(fontsize=8, style={'Manual Strategy': 'r', 'Benchmark': '#A020F0'})
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Cumulative Value')

        i = 0
        for long_trade in self.entry_trades_long:
            i += 1
            if i == 1:
                plt.axvline(long_trade, color='blue', linewidth=1.0, linestyle='--', label='Long Entry')
            else:
                plt.axvline(long_trade, color='blue', linewidth=1.0, linestyle='--')

        i = 0
        for short_trade in self.entry_trades_short:
            i += 1
            if i == 1:
                plt.axvline(short_trade, color='black', linewidth=1.0, linestyle='--', label='Short Entry')
            else:
                plt.axvline(short_trade, color='black', linewidth=1.0, linestyle='--')

        plt.xlim(sd, ed)
        plt.title(
            f'Manual Strategy v/s Benchmark {sample_type_str}\n{symbol} {sd.strftime("%m/%d/%Y")} - {ed.strftime("%m/%d/%Y")}')
        plt.legend(loc='lower left', fontsize='8')

        if show_plot:
            plt.show()
        plt.savefig(file, format='png', bbox_inches='tight', pad_inches=0.050)
        plt.close()


def run(symbol, sd, ed, sv, impact, commission, chart_file_name, results_file_name, sample_type_str='In-Sample'):
    ms = ManualStrategy()

    df_manual_trades = ms.testPolicy(symbol, sd, ed, sv)
    df_benchmark_trades = ms.get_benchmark(symbol, sd, ed, sv)

    df_manual_portfolio = msc.compute_portvals(sd, ed, df_manual_trades, sv, commission, impact)
    df_benchmark_portfolio = msc.compute_portvals(sd, ed, df_benchmark_trades, sv, commission, impact)

    # Portfolio Stats
    m_cr, m_adr, m_std, m_sr = msc.compute_portfolio_stats(df_manual_portfolio.values)
    b_cr, b_adr, b_std, b_sr = msc.compute_portfolio_stats(df_benchmark_portfolio.values)
    m_cr = m_cr[0]
    b_cr = b_cr[0]

    with open(results_file_name, 'a') as sf:
        sf.write(f'\n****Manual Strategy Results {sample_type_str} {symbol} '
                 f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        sf.write('Cumulative Return,Standard Deviation,Mean Daily Return,Sharpe Ratio\n')
        s = ','.join(['{:.6f}'.format(m_cr), '{:.6f}'.format(m_std), '{:.6f}'.format(m_adr), '{:.6f}'.format(m_sr)])
        sf.write(s + '\n')

        sf.write(f'\n****Benchmark Results {sample_type_str} {symbol} '
                 f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        sf.write('Cumulative Return,Standard Deviation,Mean Daily Return,Sharpe Ratio\n')
        s = ','.join(['{:.6f}'.format(b_cr), '{:.6f}'.format(b_std), '{:.6f}'.format(b_adr), '{:.6f}'.format(b_sr)])
        sf.write(s + '\n')

    ms.plot_ms(sd, ed, symbol, df_manual_portfolio / df_manual_portfolio.iloc[0],
               df_benchmark_portfolio / df_benchmark_portfolio.iloc[0],
               chart_file_name, sample_type_str, show_plot=False)

    return


def run_ms(results_file_name):
    run('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000, 0.005, 9.95, 'images/manual_in_sample.png',
        results_file_name, 'In-Sample')
    run('JPM', dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000, 0.005, 9.95,
        'images/manual_out_of_sample.png',
        results_file_name, 'Out-Of-Sample')


if __name__ == "__main__":
    run_ms('p8_results.txt')
