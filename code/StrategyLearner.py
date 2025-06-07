""""""
import numpy as np

"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: ssimha31 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903845103 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""

import datetime as dt
import pandas as pd
import util as ut
import indicators as ind
import RTLearner as rtl
import BagLearner as bgl


class StrategyLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type impact: float  		  	   		  		 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type commission: float  		  	   		  		 			  		 			     			  	 
    """

    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
        Constructor method
        """
        self.user_id = 'ssimha31'
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        # Tuning Parameters
        self.exit_strategy = 'H'  # Hold instead of cash out when no long/short decision
        self.N = 5
        self.YBUY = 0.030
        self.YSELL = -0.030
        self.bags = 40
        self.leaf_size = 10

        self.bl = bgl.BagLearner(rtl.RTLearner, kwargs={'leaf_size': self.leaf_size, 'verbose': self.verbose},
                                 bags=self.bags, boost=False, verbose=self.verbose)

        self.entry_trades_long = []
        self.entry_trades_short = []

    def author(self):
        return self.user_id

    # this method should create a QLearner, and train it for trading
    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        """
        df_prices = ut.get_data([symbol], pd.date_range(sd, ed))
        df_prices.fillna(method='ffill')
        df_prices.fillna(method='bfill')

        p = df_prices[symbol].values
        x = self.make_x(symbol, sd, ed, df_prices)
        y = np.zeros((x.shape[0]))
        y_long = np.zeros((x.shape[0]))
        y_short = np.zeros((x.shape[0]))

        # compute N day returns
        y[0:-self.N] = p[self.N:] / p[0:-self.N] - 1

        # compute both long and short returns including impact; we'll only use one of them as appropriate (if at all)
        y_long[0:-self.N] = (p[self.N:] * (1 - self.impact)) / (p[0:-self.N] * (1 + self.impact)) - 1
        y_short[0:-self.N] = (p[self.N:] * (1 + self.impact)) / (p[0:-self.N] * (1 - self.impact)) - 1

        # use long if ndr is positive, and short if ndr is negative (if applicable)
        y[y > 0] = y_long[y > 0]
        y[y < 0] = y_short[y < 0]

        # drop last N rows
        x = x[0:-self.N, :]
        y = y[0:-self.N]

        # classify
        y[y >= self.YBUY] = 1
        y[y <= self.YSELL] = -1
        y[(y < self.YBUY) & (y > self.YSELL)] = 0

        # add evidence
        self.bl.add_evidence(x, y)

    def make_x(self, symbol, sd, ed, df_prices):
        df_rsi = ind.get_rsi([symbol], sd, ed, do_plot=False, file_name='rsi.png', plot_symbol=symbol)
        df_rsi.drop(columns=['SPY'], inplace=True)

        df_bbp = ind.get_bbp([symbol], sd, ed, do_plot=False, file_name='bbp.png', plot_symbol=symbol)
        df_bbp.drop(columns=['SPY'], inplace=True)

        df_sco = ind.get_sco([symbol], sd, ed, do_plot=False, file_name='sco.png', plot_symbol=symbol)
        df_sco.drop(columns=['SPY'], inplace=True)

        df_x = df_prices[[symbol]].join(df_rsi.join(df_bbp, lsuffix='_r', rsuffix='_b').join(df_sco), rsuffix='_s')
        return df_x.values[:, 1:]

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			     			  	 

        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			     			  	 
        :rtype: pandas.DataFrame
        """
        dates = pd.date_range(sd, ed)
        df_prices = ut.get_data([symbol], dates)
        df_prices.fillna(method='ffill')
        df_prices.fillna(method='bfill')
        df_prices.sort_index(inplace=True)

        x = self.make_x(symbol, sd, ed, df_prices)
        y = self.bl.query(x)

        holdings = 0
        df_trades = pd.DataFrame()
        i = 0
        for date, row in df_prices.iterrows():
            action = y[i]
            if action == 1:
                df_trades, holdings = self.go_long(df_trades, date, symbol, holdings)
            elif action == -1:
                df_trades, holdings = self.go_short(df_trades, date, symbol, holdings)
            elif self.exit_strategy == 'C':
                df_trades, holdings = self.close_position(df_trades, date, symbol, holdings)

            i += 1

        return df_trades

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


if __name__ == "__main__":
    print("One does not simply think up a strategy")
