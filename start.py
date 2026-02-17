def fetch_binance_bid_ask(symbol):
    # Assume we have a way to fetch this data from the Binance API
    response = get_binance_data(symbol)
    ask = response['ask']
    bid = response['bid']
    timestamp = response['timestamp']  # Get the timestamp
    return bid, ask, timestamp  # Return timestamp as third value


def generate_signals(data):
    entry_time = data['timestamp']  # Use the timestamp
    # ... Implement logic for generating signals ...
    return signals, entry_time  # Return signals and entry_time


def portfolio_live_runner(signals):
    for signal in signals:
        entry_time = signal['entry_time']  # Use entry_time from generate_signals
        # ... Implement the rest of the function logic ...