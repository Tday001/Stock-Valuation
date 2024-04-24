import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


def fetch_and_preprocess_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.ffill(inplace=True)  # Forward fill to handle missing values
    return data


def fetch_financial_metrics(symbol):
    stock = yf.Ticker(symbol)
    cash_flow = stock.cashflow.transpose().get("Free Cash Flow", pd.Series()).dropna()

    # 
    if len(cash_flow) > 1 and (cash_flow > 0).all():
        years = cash_flow.index.year.astype(float)
        values = cash_flow.values.astype(float)
        log_years = np.log(years)
        log_values = np.log(values)
        rates = np.polyfit(log_years, log_values, 1)
        fcf_growth_rate = np.exp(rates[0]) - 1
        latest_fcf = cash_flow.iloc[-1]
    else:
        fcf_growth_rate = np.nan
        latest_fcf = np.nan

    # Fetch and sum dividends
    end_year = pd.to_datetime('today').year
    start_year = end_year - 5
    dividends = stock.dividends.loc[f'{start_year}':f'{end_year}'].sum()

    return {
        'dividend_rate': dividends,
        'fcf_growth_rate': fcf_growth_rate,
        'latest_fcf': latest_fcf
    }


def calculate_wacc(market_cap, total_debt, interest_expense, tax_rate):
    cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.04
    cost_of_equity = 0.08
    equity_ratio = market_cap / (market_cap + total_debt)
    debt_ratio = total_debt / (market_cap + total_debt)
    wacc = (cost_of_equity * equity_ratio) + (cost_of_debt * debt_ratio * (1 - tax_rate))
    return wacc


def calculate_dcf(latest_fcf, fcf_growth_rate, wacc, years=10):
    try:
        cash_flows = [latest_fcf * ((1 + fcf_growth_rate) ** min(i, 20)) for i in range(1, years + 1)]  # Cap growth compounding
        terminal_value = cash_flows[-1] * (1 + 0.02) / (wacc - 0.02)
        discount_factors = [(1 / (1 + wacc)) ** i for i in range(1, years + 2)]
        dcf_valuation = sum(cf * df for cf, df in zip(cash_flows, discount_factors[:-1])) + terminal_value * discount_factors[-1]
    except OverflowError:
        print("Overflow error encountered in DCF calculation")
        dcf_valuation = np.nan
    return dcf_valuation


def build_and_train_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X, y = [], scaled_data[1:]
    for i in range(len(scaled_data) - 1):
        X.append(scaled_data[i:i + 1, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(1, 1)),  # Define the input shape directly here
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
        ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32)
    return model


def visualize_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def generate_financial_report(financial_metrics, dcf_valuation):
    report = f"Financial Analysis Report:\n"
    report += f"DCF Valuation: ${dcf_valuation:.2f}\n"
    report += f"WACC: {financial_metrics['wacc']:.2%}\n"
    report += f"FCF Growth Rate: {financial_metrics['fcf_growth_rate']:.2%}\n"
    report += f"Total Dividends (Last 5 Years): ${financial_metrics['dividend_rate']}\n"
    print(report)


def main():
    symbol = 'LULU'
    data = fetch_and_preprocess_data(symbol, '2017-01-01', '2024-01-01')
    financial_metrics = fetch_financial_metrics(symbol)

    visualize_data(data)
    if 'latest_fcf' in financial_metrics and financial_metrics['latest_fcf'] is not np.nan:
        financial_metrics['wacc'] = calculate_wacc(2.2e12, 1.1e11, 5e9, 0.21)  # Example values for Apple
        dcf_valuation = calculate_dcf(financial_metrics['latest_fcf'], financial_metrics['fcf_growth_rate'],
                                      financial_metrics['wacc'])
        generate_financial_report(financial_metrics, dcf_valuation)
    else:
        print("Insufficient data to calculate DCF.")

    lstm_model = build_and_train_lstm_model(data)
    print("LSTM model trained.")


if __name__ == "__main__":
    main()
