import os
import json
import time
import requests
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bisect import bisect_left
from coinbase.rest import RESTClient

# === Referal Link ===
"https://advanced.coinbase.com/join/TPKTGY7"

# === Tip Jar ===
"https://www.paypal.com/paypalme/chancevandyke"

# === Adjustable Settings ===
CRYPTO_SYMBOL = "ETH"                         # Crypto you want to trade, e.g., "BTC", "ETH", "LTC"
MAX_BUY_USD = 100                             # Max amount in 1 trade
FEE_BUFFER_PERCENT = 0.0005                   # Buffer for fees (0.5%)
PERCENT_THRESHOLD = 0.0065                    # % Threshold for trade
COOLDOWN_MINUTES = 2                          # - Minutes Cooldown between trades
MA_PERIOD = 10                                # Moving average window
TRAILING_STOP_PERCENT = 0.005                 # --% trailing stop
POLL_INTERVAL_SECONDS = 1                     # How often to fetch new price
RUN_DURATION_hours = 8                        # Run duration in hours
LOCAL_TIMEZONE = ZoneInfo("America/Chicago")  # Time Zone: Change as needed
ENABLE_TRADING = 1                            # Set to 1 to enable trading

# === Coinbase Client ===
client = RESTClient(key_file="file_path")

# Fetch actual balances from Coinbase
def get_balance_value(balance):
    # balance might be an object or dict
    if hasattr(balance, "value"):
        return float(balance.value)
    elif isinstance(balance, dict):
        return float(balance.get("value", 0))
    else:
        return 0.0

try:
    accounts_response = client.get_accounts()
    accounts = accounts_response.accounts
    usd_balance = 0.0
    crypto_balance = 0.0
    for account in accounts:
        currency = account.currency if hasattr(account, "currency") else account.get("currency", "")
        balance = account.available_balance if hasattr(account, "available_balance") else account.get("available_balance", {})
        bal_value = get_balance_value(balance)
        if currency == "USD":
            usd_balance = bal_value
        elif currency == CRYPTO_SYMBOL:
            crypto_balance = bal_value

    print("✅ Fetched balances:")
    print(f"   USD Balance: {usd_balance}")
    print(f"   {CRYPTO_SYMBOL} Balance: {crypto_balance}")

except Exception as e:
    print(f"❌ Failed to fetch account balances: {e}")
    usd_balance = 0.0
    crypto_balance = 0.0

def fetch_current_balances():
    try:
        accounts_response = client.get_accounts()
        accounts = accounts_response.accounts
        usd_balance = 0.0
        crypto_balance = 0.0
        for account in accounts:
            currency = getattr(account, "currency", "") or account.get("currency", "")
            balance = getattr(account, "available_balance", None) or account.get("available_balance", {})
            bal_value = get_balance_value(balance)
            if currency == "USD":
                usd_balance = bal_value
            elif currency == CRYPTO_SYMBOL:
                crypto_balance = bal_value
        return usd_balance, crypto_balance
    except Exception as e:
        print(f"❌ Failed to fetch account balances: {e}")
        # Optionally fallback to last known balances or zero
        return 0.0, 0.0

# === Tracking Variables ===
price_history = []
time_history = []
trade_history = []
balance_times = []
balance_values = []
recent_low = None
recent_high = None
peak_price_after_buy = None
last_trade_time = None
last_heartbeat_time = None

start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(hours=RUN_DURATION_hours)
local_start = start_time.astimezone(LOCAL_TIMEZONE)
local_end = end_time.astimezone(LOCAL_TIMEZONE)

print(
    f"🤖 Starting trading...\n"
    f"   Start Time: {local_start.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
    f"   Run Duration: {RUN_DURATION_hours} hours\n"
    f"   End Time:   {local_end.strftime('%Y-%m-%d %H:%M:%S %Z')}"
)

# === Definitions/Concrete Variables ===

def floor_8_decimals(amount):
    return math.floor(amount * 1e8) / 1e8

crypto_base_size = str(floor_8_decimals(crypto_balance))
trade_usd = min(usd_balance, MAX_BUY_USD) * (1 - FEE_BUFFER_PERCENT)  # buffer for fees
trade_usd = math.floor(trade_usd)  # for whole dollar amounts
PRODUCT_ID = f"{CRYPTO_SYMBOL}-USD"  # Coinbase product ID for the trading pair

def moving_average(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def get_latest_price():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(f"https://api.coinbase.com/v2/prices/{PRODUCT_ID}/spot", headers=headers)
        data = response.json()
        price = float(data["data"]["amount"])
        timestamp = datetime.now(timezone.utc)
        return price, timestamp
    except Exception as e:
        print(f"❌ Failed to fetch latest price: {e}")
        return None, None

def safe_place_order(order_func, **kwargs):
    try:
        order_response = order_func(**kwargs)
        order_dict = order_response.to_dict() if hasattr(order_response, "to_dict") else order_response
        return {"success": True, "order": order_dict}
    except Exception as e:
        return {"success": False, "error_response": {"message": str(e)}}

# === Starting Trading ===

try:
    if ENABLE_TRADING == 0:
        print("🚫 Trading is disabled. Build your own Model 😈🖕")
        raise SystemExit  # clean exit
    while datetime.now(timezone.utc) < end_time:
        price, timestamp = get_latest_price()
        if price is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        local_time = timestamp.astimezone(LOCAL_TIMEZONE)
        price_history.append(price)
        time_history.append(timestamp)
        total_value = usd_balance + crypto_balance * price
        balance_times.append(local_time.strftime('%Y-%m-%d %H:%M:%S'))
        balance_values.append(total_value)

        ma = moving_average(price_history, MA_PERIOD)
        minutes_since_last_trade = ((timestamp - last_trade_time).total_seconds() / 60) if last_trade_time else None

        if last_heartbeat_time is None or (timestamp - last_heartbeat_time) >= timedelta(minutes=30):
            ma_display = f"{ma:.2f}" if ma is not None else "N/A"
            print(f"💓 Heartbeat at {local_time.strftime('%Y-%m-%d %H:%M:%S')} | {CRYPTO_SYMBOL}=${price:.2f} | MA({MA_PERIOD})={ma_display} | Portfolio=${total_value:.2f}")
            last_heartbeat_time = timestamp

        if recent_low is None or price < recent_low:
            recent_low = price
        if recent_high is None or price > recent_high:
            recent_high = price

        if float(crypto_base_size) > 0 and peak_price_after_buy is not None:
            if price > peak_price_after_buy:
                peak_price_after_buy = price  

            if price <= peak_price_after_buy * (1 - TRAILING_STOP_PERCENT):
                try:
                    order = safe_place_order(
                        client.market_order_sell,
                        client_order_id=f"trailstop-sell-{local_time.strftime('%Y-%m-%d-%H-%M-%S')}",
                        product_id=PRODUCT_ID,
                        base_size=crypto_base_size

                    )
                    if order.get("success") is True:
                        print(f"⚠️ Trailing stop SELL order at {local_time} for {crypto_balance:.6f} {CRYPTO_SYMBOL}")
                        success_response = order.get("order", {}).get("success_response", {})
                        market_conf = order.get("order", {}).get("order_configuration", {}).get("market_market_ioc", {})
                        print(f"📦 Order Response: {success_response.get('order_id')} | Side: {success_response.get('side')} | Size: {market_conf.get('base_size')}")
                        print(f"✅ Status: {order.get('success')}")
                        time.sleep(1)
                        usd_balance, crypto_balance = fetch_current_balances()
                        usd_gained = float(crypto_base_size) * price
                        trade_history.append((local_time.strftime('%Y-%m-%d %H:%M:%S'), "SELL (Trailing Stop)", price, usd_gained))
                        peak_price_after_buy = None
                        last_trade_time = timestamp
                        recent_low = price
                        recent_high = price
                        crypto_base_size = str(floor_8_decimals(crypto_balance))
                        time.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    else:
                        print(f"❌ Trailing stop sell failed at {local_time}: {order.get('error_response', {}).get('message', 'Unknown error')}")
                except Exception as e:
                    print(f"❌ Trailing stop sell failed at {local_time}: {e}")

        if (float(crypto_base_size) == 0 
            and ma 
            and price >= recent_low * (1 + PERCENT_THRESHOLD)
            and price >= ma 
            and (minutes_since_last_trade is None or minutes_since_last_trade >= COOLDOWN_MINUTES)
            and usd_balance >= 2
            ):
            trade_usd = min(usd_balance, MAX_BUY_USD)
            try:
                order = safe_place_order(
                    client.market_order_buy,
                    client_order_id=f"buy-{local_time.strftime('%Y-%m-%d-%H-%M-%S')}",
                    product_id=PRODUCT_ID,
                    quote_size=str(math.floor(trade_usd))
                )
                if order.get("success") is True:
                    print(f"✅ Placed BUY order at {local_time} for ${trade_usd}")
                    success_response = order.get("order", {}).get("success_response", {})
                    market_conf = order.get("order", {}).get("order_configuration", {}).get("market_market_ioc", {})
                    print(f"📦 Order Response: {success_response.get('order_id')} | Side: {success_response.get('side')} | Size: {market_conf.get('base_size')}")
                    print(f"✅ Status: {order.get('success')}")
                    time.sleep(1)
                    usd_balance, crypto_balance = fetch_current_balances()
                    print(f"🔄 Post-BUY Balances - USD: {usd_balance:.2f}, ETH: {crypto_balance:.8f}")
                    crypto_base_size = str(floor_8_decimals(crypto_balance))
                    crypto_bought = trade_usd / price
                    trade_history.append((local_time.strftime('%Y-%m-%d %H:%M:%S'), "BUY", price, trade_usd))
                    last_trade_time = timestamp
                    peak_price_after_buy = price
                    recent_low = price
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                else:
                    print(f"❌ Buy order failed at {local_time}: {order.get('error_response', {}).get('message', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Buy order failed at {local_time}: {e}")

        if (
    crypto_balance > 0
    and ma
    and price <= recent_high * (1 - PERCENT_THRESHOLD)
    and float(crypto_base_size) > 0
    and price <= ma
    and (minutes_since_last_trade is None or minutes_since_last_trade >= COOLDOWN_MINUTES)
):
            try:
                order = safe_place_order(
                    client.market_order_sell,
                    client_order_id=f"sell-{local_time.strftime('%Y-%m-%d-%H-%M-%S')}",
                    product_id=PRODUCT_ID,
                    base_size=crypto_base_size
                )
                if order.get("success") is True:
                    print(f"✅ Placed SELL order at {local_time} for {crypto_balance:.6f} {CRYPTO_SYMBOL}")
                    success_response = order.get("order", {}).get("success_response", {})
                    market_conf = order.get("order", {}).get("order_configuration", {}).get("market_market_ioc", {})
                    print(f"📦 Order Response: {success_response.get('order_id')} | Side: {success_response.get('side')} | Size: {market_conf.get('base_size')}")
                    print(f"✅ Status: {order.get('success')}")
                    time.sleep(1)
                    usd_balance, crypto_balance = fetch_current_balances()
                    print(f"🔄 Post-SELL Balances - USD: {usd_balance:.2f}, ETH: {crypto_balance:.8f}")
                    usd_gained = crypto_balance * price
                    trade_history.append((local_time.strftime('%Y-%m-%d %H:%M:%S'), "SELL", price, usd_gained))
                    last_trade_time = timestamp
                    recent_low = price
                    recent_high = price
                    peak_price_after_buy = price
                    crypto_base_size = str(floor_8_decimals(crypto_balance))
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                else:
                    print(f"❌ Sell order failed at {local_time}: {order.get('error_response', {}).get('message', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Sell order failed at {local_time}: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\n🛑 Script interrupted by user. Closing position...")

print("🕒 Trading window ended. Wrapping up...")

# Final forced sell if Crypto remains
if (crypto_balance > 0 
    and price_history 
    and float(crypto_base_size) > 0):
    last_price = price_history[-1]
    try:
        final_sell_time = datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        order = safe_place_order(
            client.market_order_sell,
            client_order_id="final-sell",
            product_id=PRODUCT_ID,
            base_size=crypto_base_size
        )
        if order.get("success") is True:
            usd_gained = crypto_balance * last_price
            trade_history.append((final_sell_time, "SELL", last_price, usd_gained))
            usd_balance, crypto_balance = fetch_current_balances()
            print(f"✔️ Sold remaining {CRYPTO_SYMBOL} for ${usd_gained:.2f}")
            print("📦 Order Response:", json.dumps(order, indent=2))
        else:
            print(f"❌ Final sell failed: {order.get('error_response', {}).get('message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Final sell failed: {e}")

final_price, _ = get_latest_price()
if final_price is None:
    final_price = price_history[-1] if price_history else 0
final_value = usd_balance + crypto_balance * final_price
time.sleep(2)
print(f"\n🤑 Final USD Values:")
print(f"💰 USD Balance: ${usd_balance:.2f}")
print(f"🏦 {CRYPTO_SYMBOL} Balance: {crypto_balance:.6f} (worth ${crypto_balance * final_price:.2f})")
print(f"📊 Trades Executed: {len(trade_history)}")
print("\nLast 5 Trades:")
for t in trade_history[-5:]:
    print(t)

# === Plot Price and Portfolio Value with Trade Markers ===
if balance_times and balance_values:
    price_values = price_history[-len(balance_times):]
    balance_dt = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in balance_times]

    def find_closest_index(dt_list, target_dt):
        pos = bisect_left(dt_list, target_dt)
        if pos == 0:
            return 0
        if pos == len(dt_list):
            return len(dt_list) - 1
        before = dt_list[pos - 1]
        after = dt_list[pos]
        if abs((after - target_dt).total_seconds()) < abs((target_dt - before).total_seconds()):
            return pos
        else:
            return pos - 1

    # Extract trade points
    buy_times = [t[0] for t in trade_history if "BUY" in t[1]]
    buy_prices = [float(t[2]) for t in trade_history if "BUY" in t[1]]
    sell_times = [t[0] for t in trade_history if "SELL" in t[1]]
    sell_prices = [float(t[2]) for t in trade_history if "SELL" in t[1]]

    buy_values = []
    sell_values = []
    for t_str in buy_times:
        t_dt = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
        idx = find_closest_index(balance_dt, t_dt)
        buy_values.append(balance_values[idx])
    for t_str in sell_times:
        t_dt = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
        idx = find_closest_index(balance_dt, t_dt)
        sell_values.append(balance_values[idx])

    # Compute separate USD and crypto values over time
    usd_values = []
    crypto_values = []
    for i in range(len(balance_times)):
        price = price_values[i]
        total_value = balance_values[i]
        # Estimate crypto value as: (crypto_balance * price)
        # Estimate USD as: total - crypto_value
        # Assumes price is correct and holdings don't change between data points
        if i == 0:
            usd_values.append(total_value)
            crypto_values.append(0)
        else:
            prev_price = price_values[i - 1]
            delta_price = price - prev_price
            # Approximate crypto value as diff from total
            est_crypto_val = total_value - usd_balance  # crude est
            crypto_values.append(est_crypto_val)
            usd_values.append(total_value - est_crypto_val)

    # Convert timestamps to datetime for plotting
    x_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in balance_times]
    x_buy = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in buy_times]
    x_sell = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in sell_times]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: Price
    ax1.plot(x_times, price_values, label=f"{CRYPTO_SYMBOL} Price", color="gray")
    ax1.scatter(x_buy, buy_prices, marker='^', color='green', label='Buy', zorder=5)
    ax1.scatter(x_sell, sell_prices, marker='v', color='red', label='Sell', zorder=5)
    ax1.set_ylabel(f"{CRYPTO_SYMBOL} Price (USD)")
    ax1.set_title(f"{CRYPTO_SYMBOL} Price Over Time with Trades")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Portfolio Components
    ax2.plot(x_times, balance_values, label="Total Portfolio Value", color="blue")
    ax2.plot(x_times, usd_values, label="USD Value", color="green", linestyle="--")
    ax2.plot(x_times, crypto_values, label=f"{CRYPTO_SYMBOL} Value", color="orange", linestyle="--")
    ax2.scatter(x_buy, buy_values, marker='^', color='green', label='Buy Marker', zorder=5)
    ax2.scatter(x_sell, sell_values, marker='v', color='red', label='Sell Marker', zorder=5)
    ax2.set_ylabel("Value (USD)")
    ax2.set_xlabel("Time")
    ax2.set_title("Portfolio Value Breakdown Over Time")
    ax2.legend()
    ax2.grid(True)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ Not enough balance data to plot results.")
