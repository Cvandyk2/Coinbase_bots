# === This version finds the strong coin and holds it until the sell is triggered. then buys the new top trader. ===
# === If the coin it buys has a gradual decrease in value it will hold and not trade for a new coin trading better until the sell conditrions are met ===

# === Dynamic Coinbase Top Gainer Trader (Improved & Cleaned) ===
import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from coinbase.rest import RESTClient
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
import tempfile

# === Configurable Settings ===
MAX_BUY_USD = 100                                    # 💰 Maximum USD to spend per buy order (pre-fee)
FEE_BUFFER_PERCENT = 0.004                           # 🧾 Buffer to account for trading fees (0.5%)
COOLDOWN_MINUTES = 20                                # 🕒 Minimum wait time between buys of the same asset
TOP_WINDOW_MINUTES = 10                              # 📈 Lookback window to calculate top gainer performance
TRAILING_STOP_BASE = 0.005                           # 🛑 Base trailing stop percentage (0.5%) to lock in gains
POLL_INTERVAL_SECONDS = 15                           # 🔁 Time to wait between each market check
MIN_VOLUME_USD = 100000                              # 📊 Minimum 24h USD volume required to consider a pair
RUN_DURATION_HOURS = 24                              # ⏱️ Total runtime duration for the trading bot
LOCAL_TIMEZONE = ZoneInfo("America/Chicago")         # 🌍 Local timezone for logging and plotting
DEBUG_MODE = False                                    # 🐞 Enables verbose logging if True (recommended during testing)
MAX_VOLATILITY = 0.025                               # ⚡ Max acceptable price volatility (3%) to avoid unstable assets
GRANULARITY = 60                                     # ⏳ Candle granularity in seconds (1 min = 60); must be one of [60, 300, 900, 3600, 21600, 86400]
PAIR_CACHE_REFRESH_HOURS = 24                        # 🔁 How often to refresh the list of tradable USD pairs
MAX_WORKERS = 2                                      # ⚙️ Max threads used when fetching candles in parallel (watch: high numbers can cause API Limit errors)
ENABLE_TRADING = 1                                   # ✅ Set to 1 to enable live trading, 0 for dry-run

# === API KEY ===
API_KEY_PATH = "/Users/chancevandyke/.investments/coinbase/API_Key/cdp_api_key.json" 

# === Cache filenames ===
CACHE_FILENAME = "/Users/chancevandyke/.investments/coinbase/Valid_Pairs/usd_cache.json"
VALID_PAIRS_LIST_PATH = "/Users/chancevandyke/.investments/coinbase/Valid_Pairs/Readable_list.txt"
cache_lock = threading.Lock()

# === Setup logging ===
logger.remove()
logger.add(sys.stderr, level="DEBUG" if DEBUG_MODE else "INFO")

# === Initialize Coinbase Client ===
client = RESTClient(key_file=API_KEY_PATH)

# === State Tracking ===
holdings = None
last_trade_time_per_symbol = {}
peak_price_after_buy = None
start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(hours=RUN_DURATION_HOURS)

price_history = []
time_history = []
balance_history = []
trade_history = []
symbol_drops = {}

# === Load/Save Cache ===
def load_cache():
    if os.path.exists(CACHE_FILENAME):
        with open(CACHE_FILENAME, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted, starting fresh.")
                return {}
    return {}

def save_cache(cache, filename=CACHE_FILENAME):
    with cache_lock:
        # Write to a temp file in the same directory
        dir_name = os.path.dirname(filename)
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
            temp_name = tf.name
            json.dump(cache, tf)
            tf.flush()
            os.fsync(tf.fileno())  # ensure write is flushed to disk
        
        # Atomically replace the original file with the temp file
        os.replace(temp_name, filename)
    # Also update the clean valid pairs list if valid_pairs_cache is available

cache = load_cache()

def write_valid_pairs_list(pairs, path=VALID_PAIRS_LIST_PATH):
    try:
        with open(path, "w") as f:
            for product in pairs:
                f.write(f"{product.product_id}\n")
        logger.info(f"📝 Valid pairs list written to {path} ({len(pairs)} pairs)")
    except Exception as e:
        logger.error(f"❌ Failed to write valid pairs list: {e}")

# === Utilities ===
def floor_8_decimals(amount):
    return math.floor(amount * 1e8) / 1e8

def get_balance(retries=2, delay=2):
    for i in range(retries):
        try:
            accounts = client.get_accounts().accounts
            balances = {}
            for a in accounts:
                currency = getattr(a, "currency", None)
                available = getattr(a, "available_balance", None) or {}
                balances[currency] = get_balance_value(available)
            return balances
        except Exception as e:
            logger.warning(f"🔁 get_balance failed (attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
    raise Exception("❌ get_balance failed after retries")

def get_balance_value(balance):
    if hasattr(balance, "value"):
        return float(balance.value)
    elif isinstance(balance, dict):
        return float(balance.get("value", 0))
    else:
        return 0.0

# === Log Starting USD Balance ===
try:
    starting_balances = get_balance()
    starting_usd = starting_balances.get("USD", 0.0)
    logger.info(f"💵 Starting USD Balance: ${starting_usd:,.2f}")
except Exception as e:
    logger.error(f"💥 Failed to fetch starting balance: {e}")
    starting_usd = 0.0

def safe_place_order(order_func, **kwargs):
    retries = 3
    for i in range(retries):
        try:
            response = order_func(**kwargs)
            return response.to_dict() if hasattr(response, "to_dict") else response
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 300  # wait 60 seconds or check Retry-After header if available
                logger.warning(f"Rate limit hit. Sleeping for {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Order failed: {e}")
                break
    return None

# === Fetch candles directly via Coinbase public API ===
def fetch_candles_direct(product_id, start_iso, end_iso, granularity=GRANULARITY):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "start": start_iso,
        "end": end_iso,
        "granularity": granularity,
    }
    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Error fetching candles directly for {product_id}: {e}")
        return None

# === Fetch candles with cache and expiration (10 min validity) ===
def fetch_candles_cached(product_id, start_iso, end_iso, granularity=GRANULARITY):
    key = f"{product_id}_{start_iso}_{end_iso}_{granularity}"
    if key in cache:
        cached_entry = cache[key]
        cached_time = datetime.fromisoformat(cached_entry["cached_at"])
        if datetime.utcnow() - cached_time < timedelta(minutes=10):
            if DEBUG_MODE:
                logger.debug(f"Cache hit for {product_id}")
            return cached_entry["candles"]
    # Cache miss or expired
    candles = fetch_candles_direct(product_id, start_iso, end_iso, granularity)
    if candles:
        with cache_lock:
            cache[key] = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "candles": candles,
            }
        # Save cache only once per fetch cycle or after updates to reduce IO
        save_cache(cache)
    return candles

# === Global cache for valid trading pairs ===
valid_pairs_cache = []
last_cache_update = None

def refresh_valid_pairs_cache():
    global valid_pairs_cache, last_cache_update
    logger.info("🔄 Refreshing valid pairs cache...")
    products = client.get_products().products
    usd_pairs = [
        p for p in products
        if getattr(p, "quote_currency_id", "").upper() == "USD"
        and not getattr(p, "trading_disabled", False)
        and float(getattr(p, "approximate_quote_24h_volume", 0)) >= MIN_VOLUME_USD
    ]

    now = datetime.now(timezone.utc)
    start_iso = (now - timedelta(minutes=TOP_WINDOW_MINUTES)).isoformat()
    end_iso = now.isoformat()

    valid_pairs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_candles_cached, p.product_id, start_iso, end_iso, GRANULARITY): p for p in usd_pairs}
        for future in as_completed(futures):
            product = futures[future]
            try:
                candles = future.result()
                if candles and len(candles) >= 2:
                    valid_pairs.append(product)
                elif DEBUG_MODE:
                    logger.debug(f"Pair {product.product_id} excluded: insufficient candle data")
            except Exception as e:
                logger.warning(f"Error fetching candles for {product.product_id}: {e}")

    valid_pairs_cache = valid_pairs
    last_cache_update = now
    logger.info(f"✅ Valid pairs cache updated: {len(valid_pairs_cache)} pairs.")
    write_valid_pairs_list(valid_pairs)

def fetch_candle_for_product(product):
    pid = product.product_id
    now = datetime.now(timezone.utc)
    start_iso = (now - timedelta(minutes=TOP_WINDOW_MINUTES)).isoformat()
    end_iso = now.isoformat()
    candles = fetch_candles_cached(pid, start_iso, end_iso, GRANULARITY)
    if candles and len(candles) >= TOP_WINDOW_MINUTES:
        return (product, candles)
    return None

def get_top_gainer():
    global last_cache_update
    now = datetime.now(timezone.utc)
    if last_cache_update is None or (now - last_cache_update) > timedelta(hours=PAIR_CACHE_REFRESH_HOURS):
        refresh_valid_pairs_cache()

    growth_rates = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_candle_for_product, p): p for p in valid_pairs_cache}
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            product, candles = result
            pid = product.product_id
            candles_sorted = sorted(candles, key=lambda c: c[0])  # sort by time ascending
            prices = [c[4] for c in candles_sorted]  # closing prices
            old_price = prices[0]
            new_price = prices[-1]
            if old_price == 0:
                continue
            growth = (new_price - old_price) / old_price
            volatility = np.std(prices) / old_price if old_price != 0 else 0

            if volatility > MAX_VOLATILITY:
                if DEBUG_MODE:
                    logger.debug(f"Skipping {pid} due to high volatility: {volatility:.4f}")
                continue

            # 🚫 Skip coins that have too little growth
            if growth < 0.005:
                if DEBUG_MODE:
                    logger.debug(f"Skipping {pid} due to low growth: {growth:.4f}")
                continue

            growth_rates.append((growth, product.base_currency_id, new_price, pid, volatility, len(candles)))

    top = max(growth_rates, key=lambda x: x[0], default=(None, None, None, None, None, 0))
    if DEBUG_MODE:
        if top[0] is not None:
            logger.debug(f"Top gainer: {top[1]} growth={top[0]*100:.2f}% volatility={top[4]:.4f} candles={top[5]}")
        else:
            logger.debug("No top gainer found this cycle.")
    return top

def liquidate_non_usd(balances):
    for symbol, amount in balances.items():
        if symbol != "USD" and amount > 0:
            product_id = f"{symbol}-USD"
            logger.info(f"💸 Selling {amount:.6f} {symbol}...")
            result = safe_place_order(client.market_order_sell,
                                      client_order_id=f"liquidate-{symbol}-{datetime.now().strftime('%H%M%S')}",
                                      product_id=product_id,
                                      base_size=str(floor_8_decimals(amount)))
            if result:
                logger.success(f"✅ Sold {amount:.6f} {symbol}")
            time.sleep(1)

def should_buy(last_time):
    if not last_time:
        return True
    return (datetime.now(timezone.utc) - last_time).total_seconds() / 60 >= COOLDOWN_MINUTES

def plot_portfolio():
    if not balance_history:
        print("⚠️ No balance data to plot yet. Wait for a full trading cycle.")
        return
    times, values = zip(*balance_history)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, values, label="Total Portfolio Value", color="blue")
    ma = pd.Series(values).rolling(window=10).mean()
    ax.plot(times, ma, label="Moving Average", linestyle="--")

    buy_times = [datetime.strptime(t[0], '%Y-%m-%d %H:%M:%S') for t in trade_history if "BUY" in t[1]]
    buy_vals = [t[3] for t in trade_history if "BUY" in t[1]]
    sell_times = [datetime.strptime(t[0], '%Y-%m-%d %H:%M:%S') for t in trade_history if "SELL" in t[1]]
    sell_vals = [t[3] for t in trade_history if "SELL" in t[1]]

    ax.scatter(buy_times, buy_vals, color="green", marker="^", label="Buy")
    ax.scatter(sell_times, sell_vals, color="red", marker="v", label="Sell")
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("USD Value")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === Main Trading Loop ===
logger.info("🤖 Starting Dynamic Gainer Bot — please wait at least one loop (~10 minutes) for meaningful activity...")
try:
    while datetime.now(timezone.utc) < end_time:
        if ENABLE_TRADING == 0:
            logger.warning("Trading disabled via config.")
            break

        growth, symbol, price, product_id, volatility, candle_count = get_top_gainer()

        if symbol:
            if candle_count < TOP_WINDOW_MINUTES:
                logger.info(f"⏳ Waiting for full data window: have {candle_count} min, need {TOP_WINDOW_MINUTES} min.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            logger.info(f"🔍 Current top gainer: {symbol} | Growth: {growth*100:.2f}% | Price: ${price:.4f}")
        else:
            runtime_minutes = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
            if runtime_minutes < 10:
                if int(runtime_minutes * 60) % 120 == 0:  # every 2 minutes
                    logger.info("📊 Gathering market data… please wait 10 minutes for sufficient history.")
            else:
                logger.warning("⚠️ No valid gainer found, retrying...")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        now = datetime.now(timezone.utc)
        local_now = now.astimezone(LOCAL_TIMEZONE)

        balances = {}
        try:
            balances = get_balance()
        except Exception as e:
            logger.error("💥 Balance fetch failed: " + str(e))
            time.sleep(POLL_INTERVAL_SECONDS)
            continue  # skip loop to avoid processing with bad balance
        
        if not balances:
            logger.warning("⚠️ Skipping cycle due to missing balances.")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        usd_balance = balances.get("USD", 0.0)
        crypto_balance = balances.get(symbol, 0.0)
        total_value = usd_balance + crypto_balance * price

        price_history.append(price)
        time_history.append(local_now)
        balance_history.append((local_now, total_value))

        logger.info(f"🚀 Top Gainer: {symbol} at ${price:.2f} ({growth*100:.2f}%) | USD=${usd_balance:.2f} | Volatility={volatility:.2%}")

        # Buy condition
        if holdings != symbol and usd_balance >= 2 and should_buy(last_trade_time_per_symbol.get(symbol)):
            liquidate_non_usd(balances)
            trade_usd = min(usd_balance, MAX_BUY_USD) * (1 - FEE_BUFFER_PERCENT)
            # Keep 2 decimals in USD amount for buying
            quote_size = str(round(trade_usd, 2))
            result = safe_place_order(client.market_order_buy,
                                      client_order_id=f"buy-{symbol}-{datetime.now().strftime('%H%M%S')}",
                                      product_id=product_id,
                                      quote_size=quote_size)
            if result:
                holdings = symbol
                peak_price_after_buy = price
                last_trade_time_per_symbol[symbol] = now
                logger.success(f"✅ Bought {symbol} for ~${quote_size}")
                trade_history.append((local_now.strftime('%Y-%m-%d %H:%M:%S'), "BUY", price, float(quote_size)))

                # 💰 Log total USD worth after BUY
                updated_balances = get_balance()
                updated_crypto = updated_balances.get(symbol, 0.0)
                updated_usd = updated_balances.get("USD", 0.0)
                logger.info(f"💰 Post-BUY Total USD Value: ${updated_usd + updated_crypto * price:.2f}")

            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        # Sell condition - trailing stop
        if holdings == symbol:
            if price > peak_price_after_buy:
                peak_price_after_buy = price

            stop_price = peak_price_after_buy * (1 - max(TRAILING_STOP_BASE, volatility * 1.5))
            if price <= stop_price:
                crypto_amt = balances.get(symbol, 0.0)
                if crypto_amt > 0:
                    result = safe_place_order(client.market_order_sell,
                                              client_order_id=f"sell-{symbol}-{datetime.now().strftime('%H%M%S')}",
                                              product_id=product_id,
                                              base_size=str(floor_8_decimals(crypto_amt)))
                    if result:
                        logger.success(f"⚠️ Trailing Stop SELL {symbol} @ ${price:.2f}")
                        holdings = None
                        usd_gained = crypto_amt * price
                        trade_history.append((local_now.strftime('%Y-%m-%d %H:%M:%S'), "SELL", price, usd_gained))
                        symbol_drops[symbol] = symbol_drops.get(symbol, 0) + 1

                        # 💰 Log total USD worth after SELL
                        updated_balances = get_balance()
                        updated_usd = updated_balances.get("USD", 0.0)
                        logger.info(f"💰 Post-SELL Total USD Value: ${updated_usd:.2f}")

                time.sleep(POLL_INTERVAL_SECONDS)
                continue

        time.sleep(POLL_INTERVAL_SECONDS)

except KeyboardInterrupt:
    logger.warning("👋 Interrupted by user. Liquidating...")

# === Final Liquidation ===
final_balances = get_balance()
for symbol, amount in final_balances.items():
    if symbol != "USD" and amount > 0:
        product_id = f"{symbol}-USD"
        logger.info(f"🔚 Final Sell: {amount:.6f} {symbol}")
        result = safe_place_order(client.market_order_sell,
                                  client_order_id=f"final-sell-{symbol}-{datetime.now().strftime('%H%M%S')}",
                                  product_id=product_id,
                                  base_size=str(floor_8_decimals(amount)))
        if result:
            price = float(result.get("price", 0.0))
            usd_gained = amount * price
            trade_history.append((datetime.now().astimezone(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
                                  "SELL (Final)", price, usd_gained))
            logger.success(f"✅ Final sell of {symbol} complete.")

# === Plot portfolio value over time ===
plot_portfolio()
