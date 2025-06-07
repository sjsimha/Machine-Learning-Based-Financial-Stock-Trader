import pandas as pd
from util import get_data
import numpy as np


def author():
    return 'ssimha31'


def compute_portvals(
        sd,
        ed,
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """  		  	   		  		 			  		 			     			  	 
    Computes the portfolio values.  		  	   		  		 			  		 			     			  	 
    :param orders_df: DataFrame with a single column representing the trade to execute
    :type orders_df: DataFrame
    :param start_val: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
    :type start_val: int  		  	   		  		 			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 			  		 			     			  	 
    :type commission: float  		  	   		  		 			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 			  		 			     			  	 
    :type impact: float  		  	   		  		 			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		  		 			  		 			     			  	 
    """

    orders_df = orders_df.sort_index(axis=0)
    symbol = orders_df.columns[0]

    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices.fillna(method='ffill', inplace=True)
    df_prices.fillna(method='bfill', inplace=True)
    df_prices['Cash'] = 1.0

    df_trades = df_prices.copy(deep=True)
    df_trades.iloc[:, :] = 0.0

    for trade_dt, trade in orders_df.iterrows():
        execute_trade(trade_dt, trade, df_trades, df_prices, commission, impact)

    df_holdings = df_trades.copy(deep=True)
    df_holdings.iloc[0]['Cash'] += start_val
    df_holdings = df_holdings.cumsum()

    df_value = df_prices * df_holdings
    port_val = df_value.sum(axis=1).to_frame()

    return port_val


def execute_trade(trade_dt, trade, df_trades, df_prices, commission, impact):
    symbol = trade.index[0]
    shares = trade[0]
    if shares > 0:
        order_type = 'BUY'
    elif shares < 0:
        order_type = 'SELL'
    else:
        return

    shares = abs(shares)
    try:
        trade_row = df_prices.loc[trade_dt]
    except KeyError as ke:
        # do not execute trade on a non-trading day
        return

    base_stock_price = trade_row[symbol]
    if order_type == 'BUY':
        cost_to_acquire = (base_stock_price * (1 + impact) * shares) + commission
        df_trades.loc[trade_dt][symbol] += shares
        df_trades.loc[trade_dt]['Cash'] -= cost_to_acquire
    else:
        proceeds = (base_stock_price * (1 - impact) * shares) - commission
        df_trades.loc[trade_dt][symbol] -= shares
        df_trades.loc[trade_dt]['Cash'] += proceeds

    return


def compute_portfolio_stats(nd_portvals):
    daily_returns = compute_daily_returns(nd_portvals)
    cr = compute_cumulative_returns(nd_portvals)
    adr = compute_avg_daily_return(daily_returns)
    std = compute_std_daily_return(daily_returns)
    sr = compute_sharpe_ratio(adr, std)

    return cr, adr, std, sr


def compute_daily_returns(nd):
    return nd[1:] / nd[0:-1] - 1


def compute_avg_daily_return(nd):
    return np.mean(nd)


def compute_std_daily_return(nd):
    return np.std(nd, ddof=1)


def compute_cumulative_returns(nd):
    return nd[-1] / nd[0] - 1


def compute_sharpe_ratio(avg_daily_rate, std_daily_rate):
    return (252 ** 0.5) * avg_daily_rate / std_daily_rate
