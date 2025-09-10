# Trades the strongest coins and switches if another outperforms.

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
from collections import deque

# HTTP session with pooling and retries
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=3
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# === Referral Link ===
# "https://advanced.coinbase.com/join/TPKTGY7"

# === Time Zone ===
LOCAL_TIMEZONE = ZoneInfo("America/Chicago")         # Local timezone for logs/plots

# Stablecoins set (treated as cash equivalents; avoid querying nonexistent <stable>-USD orderbooks)
STABLECOINS = {"USDC", "USDT", "DAI", "PYUSD", "TUSD", "USDS"}

###############################
# === CONFIGURABLE SETTINGS ===
###############################

# General Bot Timing
RUN_DURATION_HOURS = 20          # ⏱️ Total runtime duration for the trading bot
COOLDOWN_MINUTES = 20            # 🕒 Minimum wait time between buys of the same asset
MIN_SWITCH_MINUTES = 10          # 🧊 Minimum minutes between switches to reduce churn
PAIR_CACHE_REFRESH_HOURS = 24    # 🔁 How often to refresh the list of tradable USD pairs

# Data Windows & Candle Settings
TOP_WINDOW_MINUTES = 30              # 📈 Lookback window to calculate top gainer performance
MIN_CANDLES_REQUIRED = 5             # ⛳ Minimum number of candles required for analysis
MAX_WORKERS = 5                      # ⚙️ Max threads used when fetching candles in parallel (modest parallelism)
SUSTAINED_ROLLING_WINDOW = 4         # 📏 Window size for rolling outperformance check
SUSTAINED_REQUIRED_BEATS = 2         # ✅ Required number of beats in the rolling window
SUSTAINED_OUTPERFORMANCE_CYCLES = 2  # 🔂 Number of consecutive cycles required before switching
TIME_WEIGHT_BASE = 1.14               # ⏱️ Exponential weight base for recent-vs-old growth (>=1.0)
GRANULARITY = 60                     # ⏳ Candle granularity in seconds (1 min = 60)
POLL_INTERVAL_SECONDS = int(GRANULARITY) + 2  # 🔁 Poll on closed candles: candle size + 2s buffer

# Entry Filters (Market & Signals)
MIN_VOLUME_USD = 100000          # 📊 Minimum 24h USD volume required to consider a pair
MIN_GROWTH = .025                 # 🚀 Minimum growth rate required for buy
RSI_MIN = 40                     # 🔻 Minimum acceptable RSI for entry
RSI_MAX = 80                     # 🔺 Maximum acceptable RSI for entry
MAX_VOLATILITY = 0.75            # ⚡ Max price volatility % to avoid unstable assets
SIGNIFICANT_DELTA = .01          # 🔀 New crypto must outperform current by this delta to swap

# Execution Constraints & Cost Buffers
MAX_SPREAD_PCT = 0.006           # 🔄 Skip entries when L1 spread > 0.6% to reduce slippage
FEE_BUFFER_PERCENT = 0.006       # 🧾 Buffer to account for trading fees (approx taker 0.6%)

# Exit Settings
TRAILING_STOP_BASE = 0.002            # 🛑 Base trailing stop percentage to lock in gains
EXIT_MOMENTUM_THRESHOLD = -0.003      # 🔻 Exit if recent momentum turns negative by this much
EXIT_VOLUME_THRESHOLD = 0.6           # 📉 Exit if volume drops below 60% of recent average
EXIT_ACCELERATION_THRESHOLD = 0.3     # 🐌 Exit if growth acceleration slows to 30% of previous
EXIT_COMPOSITE_THRESHOLD = 0.6        # 🚪 Exit when composite exit score exceeds this (60%)
EXIT_GRACE_SECONDS = 10              # 🕊️ Post-entry grace period (seconds) before evaluating exits

# === Profit Optimization (Quick Adjustments) ===
ENABLE_PARTIAL_PROFITS = True         # Enable partial profit taking & ratcheting trailing stops
PARTIAL_PROFIT_1_PCT = 0.03           # First profit threshold (5%)
PARTIAL_PROFIT_1_SELL_PCT = 0.5       # Sell 50% at first threshold
RATCHET_THRESHOLDS = [0.02, 0.04, 0.06]      # Unrealized gain levels to tighten trailing stop
RATCHET_TRAIL_PCTS = [0.025, 0.02, 0.015]    # Corresponding trailing stop ceilings
MIN_SELL_NOTIONAL_USD = 3.0           # Minimum notional for partial sells

# Direct swap toggle
ENABLE_DIRECT_SWAP = True             # Attempt direct (non-USD) base/base swaps when possible
STABLE_BRIDGE_SYMBOLS = ["USDT", "USDC"]  # Stable symbols to try as emergency bridge if direct USD sell fails repeatedly
MAX_SELL_FAILURES_BEFORE_BRIDGE = 8    # After this many failures attempt stable bridge
MAX_SELL_FAILURES_BEFORE_CHUNK = 5     # After this many failures try chunked sells
CHUNK_SELL_SPLIT = 2                   # Split into this many chunks when chunk selling
MAX_SELL_FAILURES_FORCE_SKIP = 120     # Safety ceiling to abandon further attempts (prevents infinite loop)

# === Enhanced Negative Trend Exit Settings ===
ENABLE_NEG_CONSEC_EXIT = True          # Enable exit on consecutive negative closes (micro-trend reversal)
NEG_CONSEC_CANDLES_EXIT = 3            # Exit if this many consecutive lower closes occur
NEG_MIN_AVG_DROP_PCT = 0.002           # And the average % drop per candle exceeds this (0.2%)
ENABLE_EMA_TREND_EXIT = True           # Enable EMA fast/slow bearish crossover exit
EMA_FAST_PERIOD = 5                    # Fast EMA period (short-term momentum)
EMA_SLOW_PERIOD = 13                   # Slow EMA period (context trend)
EMA_EXIT_MIN_GAIN_PCT = 0.004          # Require at least this unrealized gain (0.4%) before EMA bearish cross can trigger exit
NEG_EXIT_MIN_SECONDS_SINCE_ENTRY = 30  # Don't trigger negative trend exits within this many seconds of entry

# Position Sizing Caps
MAX_BUY_USD = 100                # 💵 Maximum USD to spend per buy order (pre-fee)
MAX_POSITION_PCT = 1.0           # 📈 Maximum percent of USD balance to use per trade

# Dynamic Position Sizing (Signal-based)
RISK_BASE_PCT = 0.5              # 🎯 Base fraction of MAX_BUY_USD when signals are average
RISK_MAX_MULTIPLIER = 2.0        # 🚀 Max multiplier over base when signals are strong
RISK_MIN_MULTIPLIER = 0.5        # 🛡️ Min multiplier under base when signals are weak
DRAWDOWN_DAMPENER = 0.5          # 📉 Reduce size by up to 50% if recent drawdown observed

# Maker-First Entries
MAKER_FIRST_ENABLED = True        # 🧩 Try post-only maker buy before market when eligible
MAKER_FIRST_TIMEOUT_SEC = 5       # ⏳ How long to wait for maker fill before fallback
MAKER_FIRST_BID_OFFSET_TICKS = 1  # 🔧 Place price = best_bid - N*tick

# Toggles & Debug
ENABLE_TRADING = 1                 # ✅ Set to 1 to enable live trading, 0 for dry-run
ENABLE_PLOTTING = True             # 📊 Show final portfolio value chart
USE_ROLLING_OUTPERFORMANCE = True  # 🔄 Use rolling window logic instead of strict consecutive
SHOW_FILTER_FAILURES = False       # 🔍 Display how many candidates fail each filter
DEBUG_MODE = False                 # 🐞 Enables verbose logging

# Black List
CUSTOM_BLACKLIST = "USDC"  # e.g. "USDC,USDT,DAI" (leave empty string to use env/file/defaults)

# === API KEY ===
API_KEY_PATH = "/Users/chancevandyke/.investments/coinbase/API_Key/cdp_api_key.json" 

# === Cache filenames ===
FEATHER_CACHE_PATH = "/Users/chancevandyke/.investments/coinbase/Valid_Pairs/usd_cache.feather"
VALID_PAIRS_LIST_PATH = "/Users/chancevandyke/.investments/coinbase/Valid_Pairs/Readable_list.txt"

cache_lock = threading.Lock()

# Logging
logger.remove()
logger.add(sys.stderr, level="DEBUG" if DEBUG_MODE else "INFO")

# Coinbase client (re-init on auth fail)
client_lock = threading.Lock()

def init_client():
    global client
    with client_lock:
        client = RESTClient(key_file=API_KEY_PATH)
    logger.info("🔑 Coinbase client (re)initialized.")

init_client()

# Time bounds
start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(hours=RUN_DURATION_HOURS)
local_start = start_time.astimezone(LOCAL_TIMEZONE)
local_end = end_time.astimezone(LOCAL_TIMEZONE)
logger.info(f"🟢 Bot start time (local): {local_start.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"🛑 Scheduled end time (local): {local_end.strftime('%Y-%m-%d %H:%M:%S')}")

# State
holdings = None
last_trade_time_per_symbol = {}
peak_price_after_buy = None
entry_price_at_buy = None  # price at entry for current holding
partial_profit_taken = False  # whether partial profit already executed
balance_history = []  # entries: (timestamp, total_value, usd_cash, crypto_value)
trade_history = []
last_growth = {}
last_switch_time = None
sustained_beats = {}
last_candidate_symbol = None
outperformance_history = {}  # symbol -> deque of 1/0 indicating beat occurrences
no_gainer_streak = 0  # count consecutive cycles with no valid gainer
last_market_best = None  # (growth, symbol, price, product_id, volatility, candle_count)
sell_failures: dict[str, int] = {}

# Standardized trade recorder: action labels (BUY <sym>, SELL, SWAP A->B)
def record_trade(action: str, price: float | None, notional_usd: float | None):
    try:
        trade_history.append((datetime.now().astimezone(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'), action, price, notional_usd))
    except Exception:
        # Fallback minimal
        trade_history.append((datetime.now().astimezone(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'), action))

BLACKLISTED_SYMBOLS: set[str] = set()
_BLACKLIST_FILE_MTIME = None

def load_blacklist(force: bool = False):
    """(Re)load blacklist from env + optional file. Returns current set.
    Automatically merges sources; ignores blanks; uppercases symbols.
    """
    global BLACKLISTED_SYMBOLS, _BLACKLIST_FILE_MTIME
    changed = False
    # Start from custom override if provided
    if CUSTOM_BLACKLIST and CUSTOM_BLACKLIST.strip():
        base_source = CUSTOM_BLACKLIST
    else:
        base_source = os.environ.get("SYMBOL_BLACKLIST", "USDC,USDT,DAI,PYUSD,TUSD,USDS")
    env_syms = {s.strip().upper() for s in base_source.split(",") if s.strip()}
    symbols = set(env_syms)
    file_path = os.environ.get("SYMBOL_BLACKLIST_FILE")
    if file_path and os.path.isfile(file_path):
        try:
            mtime = os.path.getmtime(file_path)
            if force or _BLACKLIST_FILE_MTIME is None or mtime > _BLACKLIST_FILE_MTIME:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                # Support comma or newline separated
                parts = [p for chunk in raw.replace("\n", ",").split(",") for p in [chunk.strip()] if p]
                file_syms = {p.upper() for p in parts}
                symbols.update(file_syms)
                _BLACKLIST_FILE_MTIME = mtime
                changed = True
        except Exception as e:
            logger.warning(f"⚠️ Failed reading blacklist file {file_path}: {e}")
    # Always enforce USD (cash) exclusion
    symbols.add("USD")
    if symbols != BLACKLISTED_SYMBOLS:
        BLACKLISTED_SYMBOLS = symbols
        changed = True
    if changed and DEBUG_MODE:
        logger.debug(f"🔒 Blacklist updated ({len(BLACKLISTED_SYMBOLS)} symbols): {sorted(BLACKLISTED_SYMBOLS)}")
    return BLACKLISTED_SYMBOLS

# Initial load
load_blacklist(force=True)

# Cache I/O
def load_cache_feather():
    if os.path.exists(FEATHER_CACHE_PATH):
        try:
            # If the file is too small, treat as corrupt and rebuild
            try:
                if os.path.getsize(FEATHER_CACHE_PATH) < 128:
                    logger.warning("🧹 Feather cache too small; rebuilding cache file")
                    os.remove(FEATHER_CACHE_PATH)
                    return {}
            except Exception:
                pass
            df = pd.read_feather(FEATHER_CACHE_PATH)
            cache = {
                row["key"]: {
                    "cached_at": row["cached_at"],
                    "candles": json.loads(row["candles"])
                }
                for _, row in df.iterrows()
            }
            return cache
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Feather cache: {e}")
    return {}

def save_cache_feather(cache):
    with cache_lock:
        try:
            rows = []
            for key, value in cache.items():
                rows.append({
                    "key": key,
                    "cached_at": value["cached_at"],
                    "candles": json.dumps(value["candles"])
                })
            df = pd.DataFrame(rows)
            # Atomic write: write to temp then replace
            tmp_path = FEATHER_CACHE_PATH + ".tmp"
            try:
                df.to_feather(tmp_path)
                os.replace(tmp_path, FEATHER_CACHE_PATH)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"⚠️ Failed to save Feather cache: {e}")

# --- Helper utilities added/restored ---
def floor_8_decimals(amount: float) -> float:
    try:
        return math.floor(float(amount) * 1e8) / 1e8
    except Exception:
        return 0.0

def floor_to_increment(amount: float, increment: float | str | None) -> float:
    try:
        inc = float(increment) if increment is not None else 0.0
        if inc > 0:
            return math.floor(float(amount) / inc) * inc
    except Exception:
        pass
    return floor_8_decimals(amount)

def _parse_float_attr(obj, attr: str, default: float = 0.0) -> float:
    try:
        if obj is None:
            return default
        if isinstance(obj, dict):
            val = obj.get(attr)
        else:
            val = getattr(obj, attr, None)
        if val is None:
            return default
        return float(val)
    except Exception:
        return default

def _get_product_safely(pid: str):
    try:
        # Use cached if available
        if 'all_products_by_id' in globals():
            p = all_products_by_id.get(pid)
            if p:
                return p
    except Exception:
        pass
    try:
        return client.get_product(pid)
    except Exception:
        return None

def _is_tradable_product(p) -> bool:
    try:
        if not p:
            return False
        td = getattr(p, 'trading_disabled', False) or (p.get('trading_disabled') if isinstance(p, dict) else False)
        lo = getattr(p, 'limit_only', False) or (p.get('limit_only') if isinstance(p, dict) else False)
        po = getattr(p, 'post_only', False) or (p.get('post_only') if isinstance(p, dict) else False)
        return (not td) and (not lo) and (not po)
    except Exception:
        return False

def write_valid_pairs_list(pairs, path: str = VALID_PAIRS_LIST_PATH):
    try:
        with open(path, 'w') as f:
            for product in pairs:
                pid = getattr(product, 'product_id', None) or (product.get('product_id') if isinstance(product, dict) else None)
                if pid:
                    f.write(f"{pid}\n")
        logger.info(f"📝 Valid pairs list written to {path} ({len(pairs)})")
    except Exception as e:
        logger.warning(f"⚠️ Failed to write valid pairs list: {e}")

# Initialize on import
cache = load_cache_feather()

def get_mid_price(product_id: str) -> float:
    """Return L1 orderbook mid price or 0.0 if unavailable."""
    try:
        bid, ask = get_orderbook_bid_ask(product_id)
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    return 0.0

def get_symbol_usd_price(symbol: str) -> float:
    """Convenience for mid price of SYMBOL-USD."""
    try:
        if symbol.upper() in STABLECOINS:
            return 1.0  # treat stablecoin ≈ $1 to avoid unnecessary API calls
        return float(get_mid_price(f"{symbol}-USD"))
    except Exception:
        return 0.0

# Lightweight price cache (symbol -> (price, ts)) to smooth transient misses
_price_cache: dict[str, tuple[float, float]] = {}
PRICE_CACHE_TTL = 15  # seconds

def get_reliable_usd_price(symbol: str, allow_stale: bool = True) -> float:
    """Robust USD price lookup with fallbacks and short-term caching.
    Order: cached (fresh) -> orderbook mid -> ticker -> recent candle close -> stale cache (if allowed) -> 0.0.
    """
    now_ts = time.time()
    if symbol.upper() in STABLECOINS:
        return 1.0
    # Fresh cache hit
    cached = _price_cache.get(symbol)
    if cached and (now_ts - cached[1]) <= PRICE_CACHE_TTL:
        return cached[0]
    pid = f"{symbol}-USD"
    price = 0.0
    # 1. Orderbook mid
    try:
        price = get_mid_price(pid)
    except Exception:
        price = 0.0
    # 2. Ticker last trade
    if not price or price <= 0:
        try:
            if hasattr(client, 'get_product_ticker'):
                t = client.get_product_ticker(product_id=pid)
                # Coinbase SDK may expose 'price' attr or dict
                tp = getattr(t, 'price', None) or (t.get('price') if isinstance(t, dict) else None)
                if tp:
                    price = float(tp)
        except Exception:
            pass
    # 3. Recent candle close
    if not price or price <= 0:
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=TOP_WINDOW_MINUTES)
            candles = fetch_candles_cached(pid, start.isoformat(), end.isoformat(), GRANULARITY) or []
            if candles:
                last = max(candles, key=lambda c: c[0])
                if isinstance(last, (list, tuple)) and len(last) >= 5:
                    price = float(last[4])
        except Exception:
            pass
    # 4. Use stale cache if allowed
    if (not price or price <= 0) and allow_stale and cached:
        price = cached[0]
    if price and price > 0:
        _price_cache[symbol] = (price, now_ts)
    return price or 0.0

def perform_direct_swap(current_base: str, target_base: str, amount: float) -> bool:
    if not ENABLE_DIRECT_SWAP or amount <= 0 or current_base == target_base:
        return False

    def _live_product(pid: str):
        try:
            # Try cached first
            prod = all_products_by_id.get(pid) if 'all_products_by_id' in globals() else None
            if prod:
                return prod
        except Exception:
            pass
        try:
            prod = client.get_product(pid)
            # Cache it for later
            try:
                all_products_by_id[pid] = prod
            except Exception:
                pass
            return prod
        except Exception:
            return None

    def _tradability_reason(prod) -> str:
        if not prod:
            return "missing"
        try:
            flags = []
            if getattr(prod, 'trading_disabled', False):
                flags.append('trading_disabled')
            if getattr(prod, 'limit_only', False):
                flags.append('limit_only')
            if getattr(prod, 'post_only', False):
                flags.append('post_only')
            return ",".join(flags) if flags else "ok"
        except Exception:
            return "unknown"

    direct_pid = f"{current_base}-{target_base}"
    reverse_pid = f"{target_base}-{current_base}"

    # First attempt: use existing cache/live fetch
    p_dir = _live_product(direct_pid)
    p_rev = _live_product(reverse_pid)

    # Refresh global product list if neither appears tradable
    if not _is_tradable_product(p_dir) and not _is_tradable_product(p_rev):
        try:
            prods = client.get_products().products
            try:
                for p in prods:
                    all_products_by_id[p.product_id] = p
            except Exception:
                pass
            p_dir = _live_product(direct_pid)
            p_rev = _live_product(reverse_pid)
        except Exception:
            pass

    # Attempt direct (sell current_base for target_base)
    if _is_tradable_product(p_dir):
        base_inc = _parse_float_attr(p_dir, "base_increment", 1e-8)
        base_min = _parse_float_attr(p_dir, "base_min_size", 0.0)
        base_sz = floor_to_increment(amount, base_inc)
        if base_sz >= (base_min or 0):
            logger.info(f"🔁 Direct swap path {direct_pid} (size={base_sz:.8f}, min={base_min})")
            res = safe_place_order(
                client.market_order_sell,
                client_order_id=f"direct-swap-sell-{direct_pid}-{datetime.now().strftime('%H%M%S')}",
                product_id=direct_pid,
                base_size=str(base_sz),
            )
            if res:
                logger.success(f"✅ Direct swap filled via {direct_pid}")
                return True
            else:
                logger.warning(f"❌ Direct swap order failed on {direct_pid}; evaluating reverse path…")
        else:
            logger.debug(f"Direct path size below min: size={base_sz} < min={base_min}")
    else:
        logger.debug(f"Direct pair {direct_pid} not tradable (reason={_tradability_reason(p_dir)})")

    # Attempt reverse (buy target_base spending current_base as quote)
    if _is_tradable_product(p_rev):
        quote_inc = _parse_float_attr(p_rev, "quote_increment", 1e-8)
        quote_min = _parse_float_attr(p_rev, "quote_min_size", 0.0)
        quote_sz = floor_to_increment(amount, quote_inc)
        if quote_sz >= (quote_min or 0):
            logger.info(f"🔁 Reverse swap path {reverse_pid} (spend={quote_sz:.8f} {current_base}, min_quote={quote_min})")
            res = safe_place_order(
                client.market_order_buy,
                client_order_id=f"direct-swap-buy-{reverse_pid}-{datetime.now().strftime('%H%M%S')}",
                product_id=reverse_pid,
                quote_size=str(quote_sz),
            )
            if res:
                logger.success(f"✅ Reverse direct swap filled via {reverse_pid}")
                return True
            else:
                logger.warning(f"❌ Reverse direct swap order failed on {reverse_pid}")
        else:
            logger.debug(f"Reverse path quote size below min: size={quote_sz} < min={quote_min}")
    else:
        logger.debug(f"Reverse pair {reverse_pid} not tradable (reason={_tradability_reason(p_rev)})")

    logger.info(f"🔍 No viable direct swap route between {current_base} and {target_base}; will fallback to USD bridge if enabled.")
    return False

def get_orderbook_bid_ask(product_id: str):
    """Return (best_bid, best_ask) or (None, None) if unavailable."""
    try:
        book = client.get_product_book(product_id=product_id, level=1)
        bids = getattr(book, 'bids', None)
        asks = getattr(book, 'asks', None)
        if bids is None and isinstance(book, dict):
            bids = book.get('bids')
        if asks is None and isinstance(book, dict):
            asks = book.get('asks')
        bid = None
        ask = None
        if bids and len(bids) > 0:
            top_bid = bids[0]
            try:
                bid = float(top_bid[0]) if isinstance(top_bid, (list, tuple)) else float(top_bid.get('price'))
            except Exception:
                bid = None
        if asks and len(asks) > 0:
            top_ask = asks[0]
            try:
                ask = float(top_ask[0]) if isinstance(top_ask, (list, tuple)) else float(top_ask.get('price'))
            except Exception:
                ask = None
        return bid, ask
    except Exception as e:
        if DEBUG_MODE:
            logger.debug(f"Orderbook fetch failed for {product_id}: {e}")
        return None, None

def estimate_pair_spread_pct(product_id: str, fallback: float = 0.003) -> float:
    """Estimate spread %. Use fallback if book missing."""
    bid, ask = get_orderbook_bid_ask(product_id)
    try:
        if bid is not None and ask is not None and bid > 0 and ask > bid:
            mid = (bid + ask) / 2.0
            return max(0.0, (ask - bid) / mid)
    except Exception:
        pass
    return fallback

def estimate_switch_cost_pct(from_symbol: str, to_symbol: str) -> float:
    """Estimate total switch cost % via cheaper of USD bridge or direct pair."""
    try:
        fee = float(FEE_BUFFER_PERCENT)
    except Exception:
        fee = 0.008

    # Route 1: USD bridge (two taker trades)
    spread_from_usd = estimate_pair_spread_pct(f"{from_symbol}-USD")
    spread_to_usd = estimate_pair_spread_pct(f"{to_symbol}-USD")
    cost_usd_route = spread_from_usd + spread_to_usd + (2 * fee)

    # Route 2: Direct/Reverse single trade if tradable
    direct_costs = []
    for pid in (f"{from_symbol}-{to_symbol}", f"{to_symbol}-{from_symbol}"):
        p = _get_product_safely(pid)
        if _is_tradable_product(p):
            spread = estimate_pair_spread_pct(pid)
            direct_costs.append(spread + fee)  # single trade

    est_cost = min(direct_costs) if direct_costs else cost_usd_route
    return max(0.0, min(est_cost, 0.05))  # clamp to 0–5%

def try_maker_first_buy(product_id: str, quote_size: float) -> bool:
    """Attempt a post-only limit buy slightly below best bid and wait for fill.
    Returns True if filled, False otherwise. Cancels on timeout.
    """
    if not MAKER_FIRST_ENABLED:
        return False
    try:
        # Get tick, increments, and min sizes
        prod = client.get_product(product_id)
        tick = _parse_float_attr(prod, "quote_increment", 0.01)
        base_inc = _parse_float_attr(prod, "base_increment", 1e-8)
        base_min = _parse_float_attr(prod, "base_min_size", 0.0)
    except Exception:
        tick = 0.01
        base_inc = 1e-8
        base_min = 0.0
    # Get best bid
    bid, ask = get_orderbook_bid_ask(product_id)
    if bid is None or bid <= 0:
        return False
    # Price slightly below best bid
    price = max(0.0, bid - tick * max(1, int(MAKER_FIRST_BID_OFFSET_TICKS)))
    # Compute base_size from quote_size/price
    prelim = float(quote_size) / max(price, 1e-8)
    base_size = floor_to_increment(prelim, base_inc)
    if base_min and base_size < base_min:
        return False
    client_order_id = f"maker-first-buy-{product_id}-{datetime.now().strftime('%H%M%S')}"
    try:
        res = client.limit_order_buy(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=str(base_size),
            price=str(price),
            post_only=True,
        )
    except Exception as e:
        logger.debug(f"Maker-first limit buy failed to place: {e}")
        return False
    # Poll for fill
    try:
        # Extract order ids
        order_ids = []
        if hasattr(res, 'order_id'):
            order_ids = [getattr(res, 'order_id')]
        if not order_ids and isinstance(res, dict) and 'order_id' in res:
            order_ids = [res['order_id']]
        deadline = time.time() + max(3, int(MAKER_FIRST_TIMEOUT_SEC))
        terminal = {"FILLED", "DONE", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED"}
        while time.time() < deadline and order_ids:
            all_terminal = True
            filled = False
            for oid in order_ids:
                try:
                    oi = client.get_order(order_id=oid)
                    status = getattr(oi, 'status', None) or (oi.get('status') if isinstance(oi, dict) else None)
                    if status in ("FILLED", "DONE", "PARTIALLY_FILLED"):
                        filled = True
                    if status is None or status not in terminal:
                        all_terminal = False
                except Exception:
                    pass
            if filled:
                logger.success(f"Maker-first BUY filled for {product_id} at {price}")
                return True
            if all_terminal:
                break
            time.sleep(1.5)
        # Timeout: cancel
        try:
            if hasattr(client, 'cancel_orders'):
                client.cancel_orders(order_ids=order_ids)
            elif hasattr(client, 'cancel_order'):
                for oid in order_ids:
                    client.cancel_order(order_id=oid)
        except Exception:
            pass
        logger.info(f"Maker-first BUY timed out for {product_id}, canceled; will fallback")
    except Exception:
        pass
    return False

# Profit helpers
def calculate_rsi(prices, period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _ema(arr: list[float] | np.ndarray, period: int) -> np.ndarray:
    try:
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or period <= 1:
            return arr
        alpha = 2 / (period + 1)
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, arr.size):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out
    except Exception:
        return np.asarray(arr, dtype=float) if not isinstance(arr, np.ndarray) else arr

def detect_consecutive_negative(closes: list[float] | np.ndarray, lookback: int, min_avg_drop: float) -> tuple[bool, float]:
    """Return (trigger, avg_drop_pct_per_candle) for last 'lookback' closes.
    trigger True iff there are strictly descending closes over 'lookback' period and
    average absolute % drop per step >= min_avg_drop.
    """
    try:
        if not isinstance(closes, np.ndarray):
            closes = np.asarray(closes, dtype=float)
        if closes.size < lookback + 1:
            return False, 0.0
        recent = closes[-(lookback+1):]
        diffs = np.diff(recent)
        if not np.all(diffs < 0):
            return False, 0.0
        pct_moves = -diffs / recent[:-1]
        avg_drop = float(np.mean(pct_moves)) if pct_moves.size else 0.0
        return (avg_drop >= min_avg_drop), avg_drop
    except Exception:
        return False, 0.0

def detect_ema_bearish_cross(closes: list[float] | np.ndarray, fast_p: int, slow_p: int) -> bool:
    """Return True if fast EMA crosses below slow EMA on latest candle (classic bearish signal)."""
    try:
        if slow_p <= fast_p:
            slow_p = fast_p + 1
        if not isinstance(closes, np.ndarray):
            closes = np.asarray(closes, dtype=float)
        if closes.size < slow_p + 2:  # need enough data for meaningful EMAs and previous comparison
            return False
        fast = _ema(closes, fast_p)
        slow = _ema(closes, slow_p)
        # Cross occurred if previously fast >= slow and now fast < slow
        if fast.size < 2 or slow.size < 2:
            return False
        return fast[-2] >= slow[-2] and fast[-1] < slow[-1]
    except Exception:
        return False

def calculate_dynamic_exit_signal(closes, volumes, current_price, peak_price):
    """Compute exit signal from momentum, volume, and price action."""
    
    if len(closes) < 6 or len(volumes) < 4:
        return False, 0.0, "insufficient data"
    
    # 1. Momentum deterioration using time-weighted growth on recent data
    recent_momentum = calculate_time_weighted_growth(closes[-5:])
    
    # 2. Volume confirmation (declining volume on recent moves)
    recent_volume_avg = (volumes[-2] + volumes[-1]) / 2
    older_volume_avg = (volumes[-4] + volumes[-3]) / 2
    volume_trend = recent_volume_avg / older_volume_avg if older_volume_avg > 0 else 1.0
    
    # 3. Price acceleration (is growth slowing?)
    recent_accel = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] > 0 else 0
    older_accel = (closes[-3] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0
    acceleration_ratio = recent_accel / older_accel if older_accel != 0 else 1.0
    
    # 4. Peak decline check (how far from peak?)
    if peak_price > 0:
        peak_decline = (peak_price - current_price) / peak_price
    else:
        peak_decline = 0
    
    # Composite exit score calculation (0.0 to 1.0+)
    exit_score = 0.0
    exit_reasons = []
    
    # Momentum factor (40% weight)
    if recent_momentum < EXIT_MOMENTUM_THRESHOLD:
        momentum_score = min(0.4, abs(recent_momentum) / abs(EXIT_MOMENTUM_THRESHOLD) * 0.4)
        exit_score += momentum_score
        exit_reasons.append(f"momentum_decline({recent_momentum*100:.2f}%)")
    
    # Volume factor (30% weight)
    if volume_trend < EXIT_VOLUME_THRESHOLD:
        volume_score = min(0.3, (EXIT_VOLUME_THRESHOLD - volume_trend) / EXIT_VOLUME_THRESHOLD * 0.3)
        exit_score += volume_score
        exit_reasons.append(f"volume_decline({volume_trend:.2f})")
    
    # Acceleration factor (20% weight)
    if acceleration_ratio < EXIT_ACCELERATION_THRESHOLD:
        accel_score = min(0.2, (EXIT_ACCELERATION_THRESHOLD - acceleration_ratio) / EXIT_ACCELERATION_THRESHOLD * 0.2)
        exit_score += accel_score
        exit_reasons.append(f"deceleration({acceleration_ratio:.2f})")
    
    # Peak decline factor (10% weight) - only if significant decline from peak
    if peak_decline > 0.02:  # 2% decline from peak
        peak_score = min(0.1, peak_decline * 5)  # Scale peak decline
        exit_score += peak_score
        exit_reasons.append(f"peak_decline({peak_decline*100:.1f}%)")
    
    should_exit = exit_score >= EXIT_COMPOSITE_THRESHOLD
    reason_str = ", ".join(exit_reasons) if exit_reasons else "no_issues"
    
    return should_exit, exit_score, reason_str

def calculate_time_weighted_growth(closes):
    """Vectorized time-weighted growth identical to previous logic (no accuracy loss)."""
    if closes is None:
        return 0
    # Accept list or numpy array
    if not isinstance(closes, np.ndarray):
        try:
            closes = np.asarray(closes, dtype=float)
        except Exception:
            closes = np.array(closes)
    n = closes.size
    if n < 3:
        return (closes[-1] - closes[0]) / closes[0] if n > 1 and closes[0] > 0 else 0
    try:
        base = float(TIME_WEIGHT_BASE)
    except Exception:
        base = 1.5
    if base < 1.0:
        base = 1.0
    # Weight cache keyed by (base,n)
    global _WEIGHT_CACHE
    try:
        _WEIGHT_CACHE
    except NameError:
        _WEIGHT_CACHE = {}
    key = (base, n)
    weights = _WEIGHT_CACHE.get(key)
    if weights is None:
        # Same order as old list comprehension base**i
        weights = base ** np.arange(n, dtype=float)
        _WEIGHT_CACHE[key] = weights
    mid_point = n // 2
    first_idx = slice(0, mid_point + 1)  # include overlap index mid_point
    second_idx = slice(mid_point, n)
    w_first = weights[first_idx]
    w_second = weights[second_idx]
    first_vals = closes[first_idx]
    second_vals = closes[second_idx]
    # Weighted means
    denom_first = w_first.sum()
    denom_second = w_second.sum()
    if denom_first == 0 or denom_second == 0:
        return 0
    weighted_first = (first_vals * w_first).sum() / denom_first
    weighted_second = (second_vals * w_second).sum() / denom_second
    if weighted_first <= 0:
        return 0
    return (weighted_second - weighted_first) / weighted_first

def fast_momentum_from_arrays(closes_arr: np.ndarray, vols_arr: np.ndarray) -> float:
    """Momentum score using last 5 closes/volumes; mirrors calculate_momentum_score logic."""
    if closes_arr.size < 5:
        return 0.0
    last5_close = closes_arr[-5:]
    last5_vol = vols_arr[-5:] if vols_arr.size >= closes_arr.size else vols_arr[-5:]
    # Price momentum
    if last5_close.size >= 3 and last5_close[-3] > 0:
        recent_change = (last5_close[-1] - last5_close[-3]) / last5_close[-3]
        momentum_strength = recent_change * 100
    else:
        momentum_strength = 0
    # Volume confirmation
    if last5_vol.size >= 3:
        recent_vol_avg = last5_vol[-2:].mean()
        older_vol_avg = last5_vol[:-2].mean()
        volume_ratio = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0
        volume_score = min(2.0, volume_ratio)
    else:
        volume_score = 1.0
    return momentum_strength * volume_score

def calculate_momentum_score(candles, period=None):
    """Momentum score from recent price and volume."""
    if len(candles) < 5:
        return 0.0
    
    # Sort by timestamp
    sorted_candles = sorted(candles, key=lambda c: c[0])
    closes = [float(c[4]) for c in sorted_candles[-5:]]  # Last 5 closes
    volumes = [float(c[5]) for c in sorted_candles[-5:]]  # Last 5 volumes
    
    # Price momentum (acceleration)
    if len(closes) >= 3:
        recent_change = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] > 0 else 0
        momentum_strength = recent_change * 100  # Convert to percentage
    else:
        momentum_strength = 0
    
    # Volume confirmation (higher volume = more conviction)
    if len(volumes) >= 3:
        recent_vol_avg = np.mean(volumes[-2:])
        older_vol_avg = np.mean(volumes[:-2])
        volume_ratio = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0
        volume_score = min(2.0, volume_ratio)  # Cap at 2x
    else:
        volume_score = 1.0
    
    # Combined momentum score
    momentum_score = momentum_strength * volume_score
    return momentum_score

def estimate_recent_drawdown(closes, lookback: int = 20) -> float:
    """Recent max drawdown (0..1) over last N closes."""
    if not closes:
        return 0.0
    window = closes[-min(len(closes), max(3, lookback)) :]
    peak = max(window)
    last = window[-1]
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - last) / peak)

def dynamic_position_multiplier(growth: float, volatility: float, momentum_score: float, drawdown_pct: float) -> float:
    """Return sizing multiplier based on growth, vol, momentum, drawdown."""
    # Normalize signals
    growth_norm = np.tanh((growth - MIN_GROWTH) / max(1e-6, MIN_GROWTH))  # around 0 near threshold
    vol_penalty = np.clip(volatility / max(1e-6, MAX_VOLATILITY), 0, 1)
    momentum_norm = np.tanh(momentum_score / 50.0)  # heuristic scaling

    quality = 0.5 * growth_norm + 0.5 * momentum_norm - 0.4 * vol_penalty
    # Map quality ~[-1,1] to multiplier range
    quality = np.clip(quality, -1, 1)
    mult = RISK_BASE_PCT * (1 + quality)
    mult = np.clip(mult, RISK_MIN_MULTIPLIER, RISK_MAX_MULTIPLIER)

    # Apply drawdown dampener
    if drawdown_pct > 0:
        mult *= (1 - min(DRAWDOWN_DAMPENER, drawdown_pct))
    return float(mult)

def calculate_atr(candles, period=None):
    # Available candles from settings: TOP_WINDOW_MINUTES / (GRANULARITY / 60)
    max_available_candles = TOP_WINDOW_MINUTES // (GRANULARITY // 60)
    
    if period is None:
        period = max(2, min(14, int(max_available_candles * 0.7)))
    
    if DEBUG_MODE and len(candles) > 0:
        logger.debug(f"ATR Debug: Max available candles: {max_available_candles}, using period: {period}")
    
    if len(candles) < period + 1:
        if DEBUG_MODE:
            logger.debug(f"ATR Debug: Not enough candles - have {len(candles)}, need {period + 1}")
        return None
    
    true_ranges = []
    recent_candles = candles[-period:]  # Use last 'period' candles
    
    if DEBUG_MODE:
        logger.debug(f"ATR Debug: Using {len(recent_candles)} recent candles for ATR calculation")
    
    for i in range(1, len(recent_candles)):
        try:
            current = recent_candles[i]
            previous = recent_candles[i-1]
            
            # Coinbase candle: [time, low, high, open, close, volume]
            if isinstance(current, (list, tuple)) and len(current) >= 6:
                high = float(current[2])
                low = float(current[1])
                prev_close = float(previous[4])
            else:
                if DEBUG_MODE:
                    logger.debug(f"ATR Debug: Unexpected candle format: {current}")
                return None
            
            # TR = max(high-low, |high-prev_close|, |low-prev_close|)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            
        except (IndexError, ValueError, TypeError) as e:
            if DEBUG_MODE:
                logger.debug(f"ATR Debug: Error processing candle {i}: {e}")
            return None
    
    result = np.mean(true_ranges) if true_ranges else None
    if DEBUG_MODE and result:
        logger.debug(f"ATR Debug: Final ATR = {result:.6f} from {len(true_ranges)} true ranges")
    
    return result

def get_balance_value(account):
    BALANCE_KEYS = os.environ.get("COINBASE_BALANCE_KEYS", "available_balance,balance,available,value,amount,free").split(",")
    try:
        # Try dict-style access
        if isinstance(account, dict):
            for key in BALANCE_KEYS:
                val = account.get(key)
                if val is None:
                    continue
                # Nested dict with 'value'
                if isinstance(val, dict) and "value" in val:
                    return float(val["value"])
                elif isinstance(val, (int, float, str)):
                    try:
                        return float(val)
                    except Exception:
                        continue
        # Try attribute-style access
        for key in BALANCE_KEYS:
            val = getattr(account, key, None)
            if val is None:
                continue
            if hasattr(val, "value"):
                try:
                    return float(val.value)
                except Exception:
                    continue
            elif isinstance(val, dict) and "value" in val:
                try:
                    return float(val["value"])
                except Exception:
                    continue
            elif isinstance(val, (int, float, str)):
                try:
                    return float(val)
                except Exception:
                    continue
    # Try nested dict with a 'value'
        if isinstance(account, dict):
            for v in account.values():
                if isinstance(v, dict) and "value" in v:
                    try:
                        return float(v["value"])
                    except Exception:
                        continue
    # Unknown structure
        logger.debug(f"[get_balance_value] Unknown account structure: {account}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract balance value: {e}")
    return 0.0

def get_balance(retries=2, delay=2):
    CURRENCY_KEYS = os.environ.get("COINBASE_CURRENCY_KEYS", "currency,currency_id,asset,coin").split(",")
    for i in range(retries):
        try:
            # Fetch all account pages
            all_accounts = []
            cursor = None
            
            while True:
                if cursor:
                    response = client.get_accounts(cursor=cursor)
                else:
                    response = client.get_accounts()
                
                accounts = response.accounts
                all_accounts.extend(accounts)
                
                # Handle pagination
                if hasattr(response, 'has_next') and response.has_next and hasattr(response, 'cursor'):
                    cursor = response.cursor
                    if DEBUG_MODE:
                        logger.debug(f"🔄 Fetching next page of accounts (cursor: {cursor})")
                else:
                    break
            
            logger.debug(f"📊 Total accounts fetched: {len(all_accounts)}")
            
            balances = {}
            for a in all_accounts:
                currency = None
                # Find currency field
                if isinstance(a, dict):
                    for k in CURRENCY_KEYS:
                        currency = a.get(k)
                        if currency:
                            break
                else:
                    for k in CURRENCY_KEYS:
                        currency = getattr(a, k, None)
                        if currency:
                            break
                if not currency:
                    continue
                value = get_balance_value(a)
                if value is None or (isinstance(value, float) and math.isnan(value)) or value <= 0:
                    continue
                balances[currency] = value
            if not balances:
                logger.debug("No non-zero balances found.")
            else:
                logger.debug(f"Balances: {balances}")
            return balances
        except requests.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            # New: auth error handling – reinitialize client and retry immediately
            if status in (401, 403, 404) and ("auth" in msg or "unauthorized" in msg):
                logger.warning(f"🔁 get_balance auth error (status={status}). Reinitializing client and retrying (attempt {i+1}/{retries})…")
                init_client()
                time.sleep(2)
                continue
            logger.warning(f"🔁 get_balance failed (attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
        except Exception as e:
            logger.warning(f"🔁 get_balance failed (attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
    raise Exception("❌ get_balance failed after retries")

def get_symbol_balance(symbol: str) -> float:
    """Return current balance for a symbol (0.0 if missing)."""
    try:
        return float(get_balance().get(symbol, 0.0))
    except Exception:
        return 0.0

def wait_for_symbol_increase(symbol: str, baseline: float, timeout_s: int = 30, interval_s: int = 2) -> tuple[bool, float]:
    """Wait until symbol balance exceeds baseline. Return (True,new_value) if increased."""
    start = time.time()
    last = baseline
    while time.time() - start < timeout_s:
        try:
            cur = get_symbol_balance(symbol)
            last = cur
            if cur > baseline:
                return True, cur
        except Exception:
            pass
        time.sleep(interval_s)
    return False, last

def wait_for_symbol_decrease(symbol: str, baseline: float, timeout_s: int = 30, interval_s: int = 2) -> tuple[bool, float]:
    """Wait until symbol balance goes below baseline. Return (True,new_value) if decreased."""
    start = time.time()
    last = baseline
    while time.time() - start < timeout_s:
        try:
            cur = get_symbol_balance(symbol)
            last = cur
            if cur < baseline:
                return True, cur
        except Exception:
            pass
        time.sleep(interval_s)
    return False, last

def wait_for_buy_confirmation(symbol: str, baseline_symbol: float, cash_symbol: str, baseline_cash: float,
                              timeout_s: int = 90, interval_s: int = 3,
                              cash_drop_min: float = 0.10) -> tuple[bool, float, float, str | None]:
    """Confirm a buy by detecting either target symbol increase or cash decrease.
    Returns (confirmed, new_symbol_bal, new_cash_bal, mode) where mode is 'symbol' or 'cash'.
    """
    start = time.time()
    last_sym = baseline_symbol
    last_cash = baseline_cash
    while time.time() - start < timeout_s:
        try:
            bals = get_balance()
            cur_sym = float(bals.get(symbol, 0.0))
            cur_cash = float(bals.get(cash_symbol, last_cash))
            last_sym, last_cash = cur_sym, cur_cash
            if cur_sym > baseline_symbol:
                return True, cur_sym, cur_cash, "symbol"
            if (baseline_cash - cur_cash) >= cash_drop_min:
                return True, cur_sym, cur_cash, "cash"
        except Exception:
            pass
        time.sleep(interval_s)
    return False, last_sym, last_cash, None

def determine_primary_holding(balances: dict, price_map: dict[str, float] | None = None, min_usd: float = 1.0) -> tuple[str | None, float]:
    """Return (symbol, usd_value) of the largest non-USD holding above min_usd; else (None, 0.0).
    price_map: optional dict of {symbol: usd_price} to avoid refetching.
    """
    best_symbol = None
    best_usd = 0.0
    price_map = price_map or {}
    for coin, amt in balances.items():
        if coin == "USD" or amt <= 0 or coin.upper() in STABLECOINS:
            continue
        price = 0.0
        try:
            price = float(price_map.get(coin, 0.0))
        except Exception:
            price = 0.0
        if not price:
            price = get_reliable_usd_price(coin)
        usd_val = amt * price
        if usd_val > best_usd:
            best_usd = usd_val
            best_symbol = coin
    if best_usd >= float(min_usd):
        return best_symbol, best_usd
    return None, 0.0

def has_non_dust_crypto(balances: dict, price_map: dict[str, float] | None = None, dust_usd: float = 1.0) -> bool:
    """Return True only if there exists a non-USD asset with explicit USD value >= dust_usd.
    Changes from previous conservative version:
      * Unknown/failed price lookups NO LONGER block trading (ignored as dust).
      * Assets < dust_usd are ignored (treated as if they don't exist).
      * Only tradable USD pairs are considered.
    """
    price_map = price_map or {}
    threshold = float(dust_usd)
    for coin, amt in balances.items():
        if coin == "USD" or amt <= 0 or coin.upper() in STABLECOINS:
            continue
        pid = f"{coin}-USD"
        try:
            if not is_product_tradable(pid):
                continue
        except Exception:
            # If tradability can't be confirmed, skip (do not block)
            continue
        # Resolve price (prefer cached map, then live)
        try:
            price = float(price_map.get(coin, 0.0))
        except Exception:
            price = 0.0
        if price <= 0:
            price = get_reliable_usd_price(coin)
        if price <= 0:  # still unknown -> skip (treat as dust)
            continue
        usd_val = amt * price
        if usd_val >= threshold:
            return True
    return False

def is_product_tradable(product_id: str) -> bool:
    p = _get_product_safely(product_id)
    return _is_tradable_product(p)

def choose_quote_for_symbol(symbol: str) -> tuple[str | None, str | None]:
    """USD-only trading: return ("USD", symbol-USD) if tradable, else (None, None)."""
    pid_usd = f"{symbol}-USD"
    if is_product_tradable(pid_usd):
        return "USD", pid_usd
    return None, None

def choose_bridge_quote(base_symbol: str, target_symbol: str) -> str | None:
    """USD-only bridge: return "USD" if both base-USD and target-USD tradable; else None."""
    if is_product_tradable(f"{base_symbol}-USD") and is_product_tradable(f"{target_symbol}-USD"):
        return "USD"
    return None

# === Log Starting USD Balance ===
try:
    starting_balances = get_balance()
    starting_usd = starting_balances.get("USD", 0.0)
    logger.info(f"💵 Starting USD Balance: ${starting_usd:,.2f}")
except Exception as e:
    logger.error(f"💥 Failed to fetch starting balance: {e}")
    starting_usd = 0.0

def safe_place_order(order_func, **kwargs):
    def _extract_order_ids(resp):
        ids = []
        try:
            if resp is None:
                return ids
            # Object
            if hasattr(resp, 'order_id'):
                ids.append(getattr(resp, 'order_id'))
            # Object with 'orders'
            if hasattr(resp, 'orders') and isinstance(getattr(resp, 'orders'), (list, tuple)):
                for o in getattr(resp, 'orders'):
                    if hasattr(o, 'order_id'):
                        ids.append(getattr(o, 'order_id'))
                    elif isinstance(o, dict) and 'order_id' in o:
                        ids.append(o['order_id'])
            # Dict
            if isinstance(resp, dict):
                if 'order_id' in resp:
                    ids.append(resp['order_id'])
                if 'orders' in resp and isinstance(resp['orders'], (list, tuple)):
                    for o in resp['orders']:
                        if isinstance(o, dict) and 'order_id' in o:
                            ids.append(o['order_id'])
            # Deduplicate
            ids = [x for x in dict.fromkeys(ids) if x]
        except Exception:
            pass
        return ids

    def _get_status(order_id):
        try:
            oi = client.get_order(order_id=order_id)
            status = getattr(oi, 'status', None) or (oi.get('status') if isinstance(oi, dict) else None)
            return status, oi
        except Exception as e:
            logger.warning(f"Polling order {order_id} failed: {e}")
            return None, None

    def _cancel_order(order_id):
        try:
            # Try bulk cancel else single
            if hasattr(client, 'cancel_orders'):
                return client.cancel_orders(order_ids=[order_id])
            # Fallback single cancel
            if hasattr(client, 'cancel_order'):
                return client.cancel_order(order_id=order_id)
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
        return None

    retries = 3
    for i in range(retries):
        try:
            response = order_func(**kwargs)

            # Poll order status until terminal
            client_order_id = kwargs.get("client_order_id")
            order_ids = _extract_order_ids(response)
            if order_ids:
                poll_timeout = 60
                poll_interval = 2
                elapsed = 0
                terminal_statuses = {"FILLED", "PARTIALLY_FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED", "DONE"}
                success_statuses = {"FILLED", "PARTIALLY_FILLED", "DONE"}
                while elapsed < poll_timeout:
                    all_terminal = True
                    combined_infos = []
                    combined_statuses = []
                    for oid in order_ids:
                        status, info = _get_status(oid)
                        combined_infos.append(info)
                        if status:
                            combined_statuses.append(status)
                        if status is None or status not in terminal_statuses:
                            all_terminal = False
                    if all_terminal:
                        # Only treat as success if any order reached a success status
                        first = combined_infos[0] if combined_infos else response
                        if not any(s in success_statuses for s in combined_statuses):
                            try:
                                info_dict = first.to_dict() if hasattr(first, "to_dict") else (first if isinstance(first, dict) else {})
                                status_val = info_dict.get('status') if isinstance(info_dict, dict) else None
                                reason = None
                                if isinstance(info_dict, dict):
                                    for k in ("reject_reason", "failure_reason", "reason", "message", "error"):
                                        if k in info_dict and info_dict[k]:
                                            reason = info_dict[k]
                                            break
                                logger.warning(f"Order terminal status without fill: {status_val or combined_statuses} — {reason or 'no reason'} (client_order_id={client_order_id})")
                            except Exception:
                                logger.warning(f"Order terminal status without fill: {combined_statuses} (client_order_id={client_order_id})")
                            return None
                        return first.to_dict() if hasattr(first, "to_dict") else first
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                logger.warning(f"⏱️ Order status polling timed out for order_ids={order_ids} (client_order_id={client_order_id})")
            else:
                # No order id – inspect response and treat as success only if clearly filled
                resp_obj = response.to_dict() if hasattr(response, "to_dict") else response
                status = None
                if isinstance(resp_obj, dict):
                    status = resp_obj.get('status')
                if status in {"FILLED", "PARTIALLY_FILLED", "DONE"}:
                    return resp_obj
                logger.debug(f"No order_id in response or ambiguous status ({status}); treating as failure (client_order_id={client_order_id}).")
                return None
            # Fallback return (should not generally reach here)
            return response.to_dict() if hasattr(response, "to_dict") else response
        except requests.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e).lower()
            # New: auth error handling – reinitialize client and retry
            if status in (401, 403, 404) and ("auth" in msg or "unauthorized" in msg):
                logger.warning(f"Auth error from Coinbase (status={status}). Reinitializing client and retrying (attempt {i+1}/{retries})…")
                init_client()
                time.sleep(2)
                continue
            if status == 429:
                wait_time = 240 + np.random.randint(30, 90)
                logger.warning(f"Rate limit hit. Sleeping for {wait_time} seconds before retry (attempt {i+1}/{retries})...")
                time.sleep(wait_time)
            elif status == 400 and any(k in msg for k in ("limit only mode", "only limit orders", "market orders are not supported", "market orders are not allowed")):
                product_id = kwargs.get('product_id', 'unknown')
                logger.warning(f"Market order failed for {product_id} - limit only mode, attempting limit order fallback...")
                try:
                    # Get current L1 order book
                    book = client.get_product_book(product_id=product_id, level=1)
                    bids = getattr(book, 'bids', []) or (book.get('bids') if isinstance(book, dict) else [])
                    asks = getattr(book, 'asks', []) or (book.get('asks') if isinstance(book, dict) else [])
                    # Tick size for pricing
                    try:
                        prod = client.get_product(product_id)
                        tick = _parse_float_attr(prod, "quote_increment", 0.01)
                        base_inc = _parse_float_attr(prod, "base_increment", 1e-8)
                        base_min = _parse_float_attr(prod, "base_min_size", 0.0)
                    except Exception:
                        tick = 0.01
                        base_inc = 1e-8
                        base_min = 0.0

                    is_buy = 'quote_size' in kwargs  # buys use quote_size; sells use base_size

                    if is_buy:
                        # Maker-only Limit BUY slightly below best bid (post_only)
                        if bids and len(bids[0]) >= 1:
                            bid_price = float(bids[0][0])
                            price = max(0.0, bid_price - tick)
                            try:
                                quote_sz = float(kwargs.get('quote_size'))
                            except Exception:
                                logger.error(f"No valid quote_size provided for limit buy fallback on {product_id}")
                                return None
                            prelim = quote_sz / max(price, 1e-8)
                            base_size = floor_to_increment(prelim, base_inc)
                            if base_size <= 0:
                                logger.error(f"Computed base_size <= 0 for limit buy fallback on {product_id}")
                                return None
                            if base_min and base_size < base_min:
                                logger.warning(f"Limit BUY size below min ({base_size} < {base_min}) for {product_id}")
                                return None
                            limit_order_id = f"limit-fallback-buy-{product_id}-{datetime.now().strftime('%H%M%S')}"
                            limit_result = client.limit_order_buy(
                                client_order_id=limit_order_id,
                                product_id=product_id,
                                base_size=str(base_size),
                                price=str(price),
                                post_only=True
                            )
                            logger.warning(f"Limit BUY (post-only) placed for {product_id} at price {price}")
                            # Poll status
                            order_ids = _extract_order_ids(limit_result)
                            poll_timeout = 60
                            poll_interval = 3
                            elapsed = 0
                            terminal_statuses = {"FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED", "DONE"}
                            while elapsed < poll_timeout and order_ids:
                                all_terminal = True
                                combined_infos = []
                                for oid in order_ids:
                                    status, info = _get_status(oid)
                                    combined_infos.append(info)
                                    if status is None or status not in terminal_statuses:
                                        all_terminal = False
                                if all_terminal:
                                    first = combined_infos[0] if combined_infos else limit_result
                                    logger.success(f"Limit BUY for {product_id} reached terminal state.")
                                    return first.to_dict() if hasattr(first, "to_dict") else first
                                time.sleep(poll_interval)
                                elapsed += poll_interval
                            # If not filled, cancel
                            for oid in order_ids:
                                _cancel_order(oid)
                            logger.warning(f"Limit BUY for {product_id} canceled or timed out. Trying crossing limit to fill…")
                            # Second-phase: crossing limit to fill immediately (still a limit order)
                            if asks and len(asks[0]) >= 1:
                                ask_price = float(asks[0][0])
                                prelim2 = quote_sz / max(ask_price, 1e-8)
                                base_size2 = floor_to_increment(prelim2, base_inc)
                                if base_min and base_size2 < base_min:
                                    logger.warning(f"Crossing Limit BUY size below min ({base_size2} < {base_min}) for {product_id}")
                                    return None
                                limit_result2 = client.limit_order_buy(
                                    client_order_id=f"limit-fallback2-buy-{product_id}-{datetime.now().strftime('%H%M%S')}",
                                    product_id=product_id,
                                    base_size=str(base_size2),
                                    price=str(ask_price),
                                    post_only=False
                                )
                                order_ids2 = _extract_order_ids(limit_result2)
                                elapsed2 = 0
                                while elapsed2 < poll_timeout and order_ids2:
                                    all_terminal2 = True
                                    for oid in order_ids2:
                                        status2, _ = _get_status(oid)
                                        if status2 is None or status2 not in terminal_statuses:
                                            all_terminal2 = False
                                    if all_terminal2:
                                        first2 = limit_result2
                                        logger.success(f"Crossing Limit BUY for {product_id} reached terminal state.")
                                        return first2.to_dict() if hasattr(first2, "to_dict") else first2
                                    time.sleep(poll_interval)
                                    elapsed2 += poll_interval
                                for oid in order_ids2:
                                    _cancel_order(oid)
                                logger.warning(f"Crossing Limit BUY for {product_id} canceled or timed out.")
                            return None
                        else:
                            logger.error(f"No bid price available for {product_id}, cannot place limit BUY fallback.")
                            return None
                    else:
                        # Maker-only Limit SELL slightly above best ask (post_only)
                        if asks and len(asks[0]) >= 1:
                            ask_price = float(asks[0][0])
                            price = ask_price + tick
                            base_size = kwargs.get('base_size')
                            if not base_size:
                                logger.error(f"No base_size provided for limit sell fallback on {product_id}")
                                return None
                            try:
                                base_size = str(floor_to_increment(float(base_size), base_inc))
                                if base_min and float(base_size) < base_min:
                                    logger.warning(f"Limit SELL size below min ({base_size} < {base_min}) for {product_id}")
                                    return None
                            except Exception:
                                pass
                            limit_order_id = f"limit-fallback-sell-{product_id}-{datetime.now().strftime('%H%M%S')}"
                            limit_result = client.limit_order_sell(
                                client_order_id=limit_order_id,
                                product_id=product_id,
                                base_size=base_size,
                                price=str(price),
                                post_only=True
                            )
                            logger.warning(f"Limit SELL (post-only) placed for {product_id} at price {price}")
                            # Poll status
                            order_ids = _extract_order_ids(limit_result)
                            poll_timeout = 60
                            poll_interval = 3
                            elapsed = 0
                            terminal_statuses = {"FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "FAILED", "DONE"}
                            while elapsed < poll_timeout and order_ids:
                                all_terminal = True
                                combined_infos = []
                                for oid in order_ids:
                                    status, info = _get_status(oid)
                                    combined_infos.append(info)
                                    if status is None or status not in terminal_statuses:
                                        all_terminal = False
                                if all_terminal:
                                    first = combined_infos[0] if combined_infos else limit_result
                                    logger.success(f"Limit SELL for {product_id} reached terminal state.")
                                    return first.to_dict() if hasattr(first, "to_dict") else first
                                time.sleep(poll_interval)
                                elapsed += poll_interval
                            # If not filled, cancel
                            for oid in order_ids:
                                _cancel_order(oid)
                            logger.warning(f"Limit SELL for {product_id} canceled or timed out. Trying crossing limit to fill…")
                            # Second-phase: crossing limit to fill immediately (still a limit order)
                            if bids and len(bids[0]) >= 1:
                                bid_price = float(bids[0][0])
                                limit_result2 = client.limit_order_sell(
                                    client_order_id=f"limit-fallback2-sell-{product_id}-{datetime.now().strftime('%H%M%S')}",
                                    product_id=product_id,
                                    base_size=base_size,
                                    price=str(bid_price),
                                    post_only=False
                                )
                                order_ids2 = _extract_order_ids(limit_result2)
                                elapsed2 = 0
                                while elapsed2 < poll_timeout and order_ids2:
                                    all_terminal2 = True
                                    for oid in order_ids2:
                                        status2, _ = _get_status(oid)
                                        if status2 is None or status2 not in terminal_statuses:
                                            all_terminal2 = False
                                    if all_terminal2:
                                        first2 = limit_result2
                                        logger.success(f"Crossing Limit SELL for {product_id} reached terminal state.")
                                        return first2.to_dict() if hasattr(first2, "to_dict") else first2
                                    time.sleep(poll_interval)
                                    elapsed2 += poll_interval
                                for oid in order_ids2:
                                    _cancel_order(oid)
                                logger.warning(f"Crossing Limit SELL for {product_id} canceled or timed out.")
                            return None
                        else:
                            logger.error(f"No ask price available for {product_id}, cannot place limit SELL fallback.")
                            return None
                except Exception as le:
                    logger.error(f"Limit order fallback failed for {product_id}: {le}")
                    return None
            else:
                logger.error(f"Order failed (attempt {i+1}/{retries}): {e}")
                break
        except Exception as e:
            logger.error(f"Order failed with exception (attempt {i+1}/{retries}): {e}")
            break
    return None

def place_market_sell_sized(product_id: str, amount: float, order_tag: str) -> dict | None:
    """Place a market sell sized to base_increment and respecting base_min_size.
    Returns order response dict on success, else None.
    """
    try:
        prod = client.get_product(product_id)
        base_inc = _parse_float_attr(prod, "base_increment", 1e-8)
        base_min = _parse_float_attr(prod, "base_min_size", 0.0)
    except Exception:
        base_inc, base_min = 1e-8, 0.0
    sz = floor_to_increment(amount, base_inc)
    if base_min and sz < base_min:
        logger.warning(f"⚠️ SELL skipped for {product_id}: size {sz:.8f} below min {base_min}")
        return None
    return safe_place_order(
        client.market_order_sell,
        client_order_id=f"{order_tag}-{product_id}-{datetime.now().strftime('%H%M%S')}",
        product_id=product_id,
        base_size=str(sz),
    )

# Fetch candles via Coinbase public API
def fetch_candles_direct(product_id, start_iso, end_iso, granularity=GRANULARITY):
    # Guard: skip known unsupported / unnecessary stable pairs to avoid 404 spam
    STABLE_PRODUCT_IDS = {"USDC-USD","USDT-USD","DAI-USD","PYUSD-USD","TUSD-USD","USDS-USD"}
    if product_id in STABLE_PRODUCT_IDS or product_id.split('-')[0].upper() in BLACKLISTED_SYMBOLS:
        if DEBUG_MODE:
            logger.debug(f"Skipping candle fetch for stable/blacklisted product {product_id}")
        return None
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "start": start_iso,
        "end": end_iso,
        "granularity": granularity,
    }
    headers = {"Accept": "application/json"}
    
    # Retry on 429 with backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Delay to avoid rate limits
            if attempt > 0:
                wait_time = (2 ** attempt) + np.random.randint(1, 5)
                logger.warning(f"Rate limit retry {attempt} for {product_id}, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                time.sleep(0.2 + np.random.uniform(0, 0.3))
            
            resp = session.get(url, params=params, headers=headers, timeout=30)  # Use session with timeout
            resp.raise_for_status()
            return resp.json()
            
        except requests.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    continue  # Retry with exponential backoff
                else:
                    logger.error(f"Rate limit exceeded for {product_id} after {max_retries} attempts")
                    return None
            else:
                logger.error(f"HTTP error fetching candles for {product_id}: {e}")
                return None
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection/timeout error for {product_id}, retrying... ({e})")
                continue
            else:
                logger.error(f"Connection/timeout error for {product_id} after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching candles directly for {product_id}: {e}")
            return None
    
    return None

# Cached candles with 10 min TTL
def fetch_candles_cached(product_id, start_iso, end_iso, granularity=GRANULARITY):
    key = f"{product_id}_{start_iso}_{end_iso}_{granularity}"
    if key in cache:
        cached_entry = cache[key]
        cached_time = datetime.fromisoformat(cached_entry["cached_at"])  # tz-aware
        now_utc = datetime.now(timezone.utc)
        if now_utc - cached_time < timedelta(minutes=10):
            if DEBUG_MODE:
                logger.debug(f"Cache hit for {product_id}")
            return cached_entry["candles"]
    # Miss or expired
    candles = fetch_candles_direct(product_id, start_iso, end_iso, granularity)
    if candles:
        with cache_lock:
            cache[key] = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "candles": candles,
            }
        save_cache_feather(cache)
    return candles

# Global caches
valid_pairs_cache = []
last_cache_update = None
# Cache all products for direct pair checks
all_products_by_id = {}
tradable_product_ids = set()

# --- Metrics array cache (per product) ---
# Stores numpy arrays derived from candle list to avoid rebuilding per metric.
metrics_arrays: dict[str, dict[str, np.ndarray]] = {}
last_metrics_build: dict[str, float] = {}
METRICS_ARRAY_TTL = 30  # seconds; rebuild if older than this (short so we include newest candle soon after close)

def get_metrics_arrays(pid: str, candles: list):
    """Return dict with keys: 't','open','high','low','close','vol'. Rebuild only if stale."""
    now_t = time.time()
    if pid in metrics_arrays:
        if (now_t - last_metrics_build.get(pid, 0)) < METRICS_ARRAY_TTL and len(metrics_arrays[pid]['close']) == len(candles):
            return metrics_arrays[pid]
    # Build arrays
    try:
        # Expect candle format [t, low, high, open, close, volume]
        t = np.fromiter((c[0] for c in candles), dtype='float64')
        low = np.fromiter((float(c[1]) for c in candles), dtype='float64')
        high = np.fromiter((float(c[2]) for c in candles), dtype='float64')
        open_ = np.fromiter((float(c[3]) for c in candles), dtype='float64')
        close = np.fromiter((float(c[4]) for c in candles), dtype='float64')
        vol = np.fromiter((float(c[5]) for c in candles), dtype='float64') if len(candles[0]) > 5 else np.zeros_like(close)
        arrs = {'t': t, 'open': open_, 'high': high, 'low': low, 'close': close, 'vol': vol}
        metrics_arrays[pid] = arrs
        last_metrics_build[pid] = now_t
        return arrs
    except Exception:
        # Fallback: build minimal close array only
        try:
            close = np.array([float(c[4]) for c in candles], dtype='float64')
            arrs = {'t': np.array([c[0] for c in candles], dtype='float64'), 'open': close, 'high': close, 'low': close, 'close': close, 'vol': np.zeros_like(close)}
            metrics_arrays[pid] = arrs
            last_metrics_build[pid] = now_t
            return arrs
        except Exception:
            return {'t': np.array([]), 'open': np.array([]), 'high': np.array([]), 'low': np.array([]), 'close': np.array([]), 'vol': np.array([])}

# --- Incremental Candle Store (sliding window) ---
# Stores recent candles per product_id to avoid full-window refetch every loop.
# Format: product_id -> list of candles [t, low, high, open, close, volume] sorted ascending by t
incremental_candles: dict[str, list] = {}
last_incremental_fetch: dict[str, float] = {}
CANDLE_REFRESH_SEC = 65  # do not refetch within this interval (slightly > 1m so candle closes)

def _trim_window(pid: str):
    lst = incremental_candles.get(pid)
    if not lst:
        return
    try:
        cutoff = time.time() - (TOP_WINDOW_MINUTES * 60) - 5
        # candles are ascending; find first index >= cutoff
        i = 0
        while i < len(lst) and lst[i][0] < cutoff:
            i += 1
        if i > 0:
            del lst[:i]
        # Safety: cap max length
        max_len = int((TOP_WINDOW_MINUTES * 60) / GRANULARITY) + 4
        if len(lst) > max_len:
            del lst[:-max_len]
    except Exception:
        pass

def _initial_full_fetch(pid: str):
    now = datetime.now(timezone.utc)
    start_iso = (now - timedelta(minutes=TOP_WINDOW_MINUTES)).isoformat()
    end_iso = now.isoformat()
    candles = fetch_candles_cached(pid, start_iso, end_iso, GRANULARITY) or []
    if not candles:
        return False
    # Sort ascending by timestamp (API may return any order)
    try:
        candles_sorted = sorted(candles, key=lambda c: c[0])
    except Exception:
        candles_sorted = candles
    incremental_candles[pid] = candles_sorted
    last_incremental_fetch[pid] = time.time()
    _trim_window(pid)
    return True

def _fetch_new_candles(pid: str):
    # Fetch only from last known timestamp onward (with small overlap for robustness)
    existing = incremental_candles.get(pid, [])
    if not existing:
        return _initial_full_fetch(pid)
    try:
        last_ts = existing[-1][0]
    except Exception:
        return _initial_full_fetch(pid)
    # Overlap one granularity backwards to re-sync last candle
    start_dt = datetime.fromtimestamp(last_ts - GRANULARITY, timezone.utc)
    end_dt = datetime.now(timezone.utc)
    candles = fetch_candles_direct(
        pid,
        start_dt.isoformat(),
        end_dt.isoformat(),
        GRANULARITY,
    ) or []
    if not candles:
        return True  # treat as non-fatal; keep old
    try:
        new_sorted = sorted(candles, key=lambda c: c[0])
    except Exception:
        new_sorted = candles
    # Append only strictly newer timestamps
    seen_last = existing[-1][0]
    for c in new_sorted:
        try:
            if c[0] > seen_last:
                existing.append(c)
        except Exception:
            continue
    last_incremental_fetch[pid] = time.time()
    _trim_window(pid)
    return True

def get_incremental_candles(pid: str):
    """Return sliding-window candles for pid, updating incrementally respecting TTL."""
    now_t = time.time()
    last_t = last_incremental_fetch.get(pid, 0)
    if (now_t - last_t) >= CANDLE_REFRESH_SEC:
        try:
            _fetch_new_candles(pid)
        except Exception as e:
            if DEBUG_MODE:
                logger.debug(f"Incremental fetch failed for {pid}: {e}")
    elif pid not in incremental_candles:
        _initial_full_fetch(pid)
    return incremental_candles.get(pid, [])

# Dynamic spread limit based on product liquidity (approx 24h quote volume)
def spread_limit_for_pid(pid: str) -> float:
    """Return a spread percentage limit for a given product_id.
    Uses MAX_SPREAD_PCT by default and tightens for very low-liquidity pairs.
    """
    limit = float(MAX_SPREAD_PCT)
    try:
        p = all_products_by_id.get(pid) if 'all_products_by_id' in globals() else None
    except Exception:
        p = None
    if not p:
        return limit
    try:
        vol_str = getattr(p, 'approximate_quote_24h_volume', '0') or '0'
        vol = float(vol_str)
        # Tighten the allowed spread for illiquid products
        if vol < 500_000:
            limit *= 0.70
        elif vol < 2_000_000:
            limit *= 0.90
    except Exception:
        pass
    return limit

def refresh_valid_pairs_cache():
    global valid_pairs_cache, last_cache_update
    # Update product caches too
    global all_products_by_id, tradable_product_ids
    # Hot-reload blacklist (if file updated) each refresh
    load_blacklist()
    logger.info("🔄 Refreshing valid pairs cache...")
    try:
        products = client.get_products().products
    # Build product caches (all products)
        try:
            all_products_by_id = {p.product_id: p for p in products}
            tradable_product_ids = {
                p.product_id
                for p in products
                if not getattr(p, "trading_disabled", False)
                and not getattr(p, "limit_only", False)
                and not getattr(p, "post_only", False)
            }
        except Exception as e:
            if DEBUG_MODE:
                logger.debug(f"Failed building product caches: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch products: {e}")
        return

    usd_pairs = []
    for p in products:
        if getattr(p, "quote_currency_id", "").upper() != "USD":
            continue
        if getattr(p, "trading_disabled", False):
            continue
    # Skip limit/post-only
        if getattr(p, "limit_only", False) or getattr(p, "post_only", False):
            if DEBUG_MODE:
                logger.debug(f"Skipping {p.product_id} - limit/post only")
            continue
        try:
            base_sym = getattr(p, "base_currency_id", "").upper()
            if base_sym in BLACKLISTED_SYMBOLS:
                if DEBUG_MODE:
                    logger.debug(f"Blacklist exclude {p.product_id} ({base_sym})")
                continue
        except Exception:
            pass
        vol_str = getattr(p, "approximate_quote_24h_volume", "0") or "0"
        try:
            vol = float(vol_str)
        except Exception:
            vol = 0.0
        if vol >= MIN_VOLUME_USD:
            usd_pairs.append(p)

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
                if candles and len(candles) >= MIN_CANDLES_REQUIRED:
                    valid_pairs.append(product)
                elif DEBUG_MODE:
                    logger.debug(f"Excluded {product.product_id}: not enough candles")
            except Exception as e:
                logger.warning(f"Error fetching candles for {product.product_id}: {e}")

    valid_pairs_cache = valid_pairs
    last_cache_update = now
    write_valid_pairs_list(valid_pairs)

def fetch_candle_for_product(product):
    """Return (product, candles) using incremental sliding window store.
    Falls back to initial full fetch if store empty.
    """
    pid = product.product_id
    candles = get_incremental_candles(pid)
    if candles and len(candles) >= MIN_CANDLES_REQUIRED:
        return (product, candles)
    return None

def get_top_gainer() -> tuple:
    global no_gainer_streak
    global last_market_best
    growth_fail = 0
    volatility_fail = 0
    rsi_fail = 0
    atr_fail = 0
    data_fail = 0
    atr_values = []
    global last_cache_update

    now = datetime.now(timezone.utc)
    if last_cache_update is None or (now - last_cache_update) > timedelta(hours=PAIR_CACHE_REFRESH_HOURS):
        refresh_valid_pairs_cache()

    growth_rates = []
    processed_count = 0

    logger.debug(f"🔍 Analyzing {len(valid_pairs_cache)} cached pairs (blacklist={sorted(BLACKLISTED_SYMBOLS)})...")
    # Enable a temporary relaxed mode if we've had repeated empty scans
    relaxed_mode = no_gainer_streak >= 3

    best_overall = None  # (growth, symbol, price, pid, volatility, candle_count)
    # Track why the best growth candidate (unfiltered) was excluded
    best_candidate_debug = None  # dict with keys: symbol, pid, growth, price, reason

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_candle_for_product, p): p for p in valid_pairs_cache}
        completed_count = 0
        for fut in as_completed(futures):
            product = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                data_fail += 1
                logger.warning(f"Error fetching candles for {product.product_id}: {e}")
                continue
            pid = product.product_id
            completed_count += 1
            if completed_count % 10 == 0:
                # Add tiny jitter to avoid synchronized bursts
                jitter = 0.05 + (time.time() % 0.1)  # ~0.05–0.15s
                time.sleep(0.2 + jitter)
            if not result:
                data_fail += 1
                if DEBUG_MODE:
                    logger.debug(f"❌ {pid} data failure: insufficient candle data")
                continue

            processed_count += 1
            product, candles = result
            # Defensive blacklist check
            try:
                base_sym = getattr(product, 'base_currency_id', '').upper()
                if base_sym in BLACKLISTED_SYMBOLS:
                    if DEBUG_MODE:
                        logger.debug(f"Skip blacklisted {product.product_id}")
                    continue
            except Exception:
                pass
            # Use metrics arrays (vectorized) for speed
            arrs = get_metrics_arrays(pid, candles)
            closes = arrs['close']
            # Build a one-time sorted candle list if downstream helpers expect list form
            try:
                candles_sorted = sorted(candles, key=lambda c: c[0])
            except Exception:
                candles_sorted = candles
            if closes.size == 0 or closes[0] == 0:
                continue
            old_price = closes[0]
            new_price = closes[-1]

            # Time-weighted growth
            # Adapt calculate_time_weighted_growth to accept numpy array transparently
            try:
                growth = calculate_time_weighted_growth(closes.tolist() if isinstance(closes, np.ndarray) else closes)
            except Exception:
                growth = None
            # Track best overall regardless of filters
            try:
                vola_tmp = (closes.std() / old_price) if old_price else 0
            except Exception:
                vola_tmp = 0.0
            overall_tuple = (growth, product.base_currency_id, closes[-1], pid, vola_tmp, len(candles))
            # Safely compare even if growth is None
            if best_overall is None:
                best_overall = overall_tuple
            else:
                try:
                    cur_g = float(growth) if growth is not None else float('-inf')
                    prev_g = float(best_overall[0]) if best_overall[0] is not None else float('-inf')
                    if cur_g > prev_g:
                        best_overall = overall_tuple
                except Exception:
                    pass
            if closes.size >= 3:
                mid = closes.size // 2
                mid_price = closes[mid]
                second_half_growth = (new_price - mid_price) / mid_price if mid_price > 0 else 0
                if growth and growth > 0 and second_half_growth < -0.008:
                    growth *= 0.6

            volatility = (closes.std() / old_price) if old_price else 0

            # Relaxed mode lowers growth requirement a bit more
            min_growth_req = MIN_GROWTH * (0.4 if relaxed_mode else 1.0)
            min_growth_req = max(0.004, min_growth_req) if relaxed_mode else MIN_GROWTH
            if growth is None or growth < min_growth_req:
                # Capture reason for best candidate if applicable
                if (best_candidate_debug is None) or (growth is not None and growth > best_candidate_debug.get('growth', -1)):
                    if growth is None:
                        rsn = "growth unavailable"
                    else:
                        rsn = f"growth {growth*100:.2f}% < threshold {min_growth_req*100:.2f}%"
                    best_candidate_debug = {
                        'symbol': product.base_currency_id,
                        'pid': pid,
                        'growth': growth,
                        'price': closes[-1],
                        'reason': rsn
                    }
                growth_fail += 1
                continue
            if volatility > MAX_VOLATILITY:
                if (best_candidate_debug is None) or (growth is not None and growth > best_candidate_debug.get('growth', -1)):
                    best_candidate_debug = {
                        'symbol': product.base_currency_id,
                        'pid': pid,
                        'growth': growth,
                        'price': closes[-1],
                        'reason': f"volatility {volatility*100:.2f}% > max {MAX_VOLATILITY*100:.2f}%"
                    }
                volatility_fail += 1
                continue

            # RSI filter (skip or relax in relaxed_mode)
            # Clamp denominator to avoid division by zero for sub-minute granularities
            denom = max(1, (GRANULARITY // 60) if isinstance(GRANULARITY, int) else int(GRANULARITY) // 60)
            max_available_candles = TOP_WINDOW_MINUTES // denom
            rsi_period = max(2, min(14, int(max_available_candles * 0.6)))
            if closes.size >= rsi_period + 1:
                try:
                    rsi = calculate_rsi(closes.tolist(), period=rsi_period)
                except Exception:
                    rsi = None
                if not relaxed_mode:
                    if rsi is None or not (RSI_MIN <= rsi <= RSI_MAX):
                        if (best_candidate_debug is None) or (growth is not None and growth > best_candidate_debug.get('growth', -1)):
                            if rsi is None:
                                r = "RSI unavailable"
                            else:
                                r = f"RSI {rsi:.1f} outside [{RSI_MIN},{RSI_MAX}]"
                            best_candidate_debug = {
                                'symbol': product.base_currency_id,
                                'pid': pid,
                                'growth': growth,
                                'price': closes[-1],
                                'reason': r,
                            }
                        rsi_fail += 1
                        continue
                else:
                    # Wider band in relaxed mode (a bit more permissive)
                    rsi_min_relaxed, rsi_max_relaxed = min(25, RSI_MIN), max(85, RSI_MAX)
                    if rsi is None or not (rsi_min_relaxed <= rsi <= rsi_max_relaxed):
                        if (best_candidate_debug is None) or (growth is not None and growth > best_candidate_debug.get('growth', -1)):
                            if rsi is None:
                                r = "RSI unavailable (relaxed)"
                            else:
                                r = f"RSI {rsi:.1f} outside relaxed [{rsi_min_relaxed},{rsi_max_relaxed}]"
                            best_candidate_debug = {
                                'symbol': product.base_currency_id,
                                'pid': pid,
                                'growth': growth,
                                'price': closes[-1],
                                'reason': r,
                            }
                        rsi_fail += 1
                        continue

            atr = calculate_atr(candles_sorted)
            atr_threshold = old_price * MAX_VOLATILITY
            if atr is not None and old_price > 0:
                atr_values.append((atr / old_price) * 100)
                if atr > atr_threshold:
                    if (best_candidate_debug is None) or (growth is not None and growth > best_candidate_debug.get('growth', -1)):
                        best_candidate_debug = {
                            'symbol': product.base_currency_id,
                            'pid': pid,
                            'growth': growth,
                            'price': closes[-1],
                            'reason': f"ATR {(atr/old_price)*100:.2f}% > max {(MAX_VOLATILITY*100):.2f}%",
                        }
                    atr_fail += 1
                    continue

            momentum = fast_momentum_from_arrays(closes, arrs['vol'])
            growth_rates.append((growth, product.base_currency_id, closes[-1], pid, volatility, len(candles), momentum, candles_sorted))

    logger.debug(f"📊 Processed {processed_count} pairs with sufficient candle data")
    logger.debug(f"📊 Found {len(growth_rates)} qualifying gainers after filtering")

    if atr_values and DEBUG_MODE:
        atr_avg = np.mean(atr_values)
        atr_median = np.median(atr_values)
        atr_75th = np.percentile(atr_values, 75)
        atr_90th = np.percentile(atr_values, 90)
        logger.info(f"📈 ATR Analysis: avg={atr_avg:.2f}%, median={atr_median:.2f}%, 75th={atr_75th:.2f}%, 90th={atr_90th:.2f}% (threshold={MAX_VOLATILITY*100:.1f}%)")

    if SHOW_FILTER_FAILURES:
        logger.info(f"🔍 Filter Failures — Growth: {growth_fail}, Volatility: {volatility_fail}, RSI: {rsi_fail}, ATR: {atr_fail}, Data: {data_fail}")

    # Update market-best snapshot for external logging
    last_market_best = best_overall

    if growth_rates:
        no_gainer_streak = 0
        def composite(item):
            growth, symbol, price, pid, volatility, candles_count, momentum, candle_data = item
            momentum_factor = max(0.5, 1 + momentum / 5)
            volatility_penalty = max(0.3, 1 - (volatility / MAX_VOLATILITY) * 0.5)
            trend_bonus = 1.0
            try:
                recent = [float(c[4]) for c in candle_data[-2:]]
                if len(recent) == 2 and recent[1] > recent[0]:
                    trend_bonus = 1.2
            except Exception:
                pass
            return growth * momentum_factor * volatility_penalty * trend_bonus
        best = max(growth_rates, key=composite)
        return best[:6]
    else:
        no_gainer_streak += 1
        # Always provide a short diagnostic line when empty to aid debugging
        logger.warning(
            f"⚠️ No valid gainer (streak={no_gainer_streak}). Failures — growth:{growth_fail}, vol:{volatility_fail}, rsi:{rsi_fail}, atr:{atr_fail}, data:{data_fail}"
        )
        # Also show market's best performer even if not tradable
        if last_market_best:
            mg, ms, mp, mpid, mv, mc = last_market_best
            try:
                logger.info(f"📈 Market Best (unfiltered): {ms} at ${float(mp):.4f} | Growth={mg*100:.2f}% | SpreadGuard={MAX_SPREAD_PCT*100:.2f}%")
            except Exception:
                logger.info(f"📈 Market Best (unfiltered): {ms} | Growth={mg*100:.2f}%")
        if best_candidate_debug and best_candidate_debug.get('reason'):
            try:
                g = best_candidate_debug.get('growth', None)
                g_str = (f"{float(g)*100:.2f}%" if isinstance(g, (int, float)) else "n/a")
                price_val = best_candidate_debug.get('price', None)
                p_str = (f"${float(price_val):.4f}" if isinstance(price_val, (int, float)) else "n/a")
                logger.info(
                    f"🧭 Best excluded: {best_candidate_debug.get('symbol')} at {p_str} | Growth={g_str} — {best_candidate_debug.get('reason')}"
                )
            except Exception:
                logger.info(
                    f"🧭 Best excluded: {best_candidate_debug.get('symbol')} — {best_candidate_debug.get('reason')}"
                )
        if relaxed_mode:
            logger.info("🧪 Relaxed mode active: lowered growth threshold and widened RSI band.")
        return (None, None, None, None, None, 0)

# Final liquidation

def robust_final_liquidation(max_attempts=1, min_wait_seconds=3, min_remaining_usd=1.0):
    try:
        min_wait_seconds = max(3, int(min_wait_seconds))
    except Exception:
        min_wait_seconds = 3

    def snapshot_value():
        balances = get_balance()
        holdings_map = {s: a for s, a in balances.items() if s != "USD" and a > 0}
        total_usd = 0.0
        prices = {}
        for s, a in holdings_map.items():
            p = get_symbol_usd_price(s)
            prices[s] = p
            total_usd += a * p
        return balances, holdings_map, total_usd, prices

    logger.info("🔚 Starting final liquidation…")
    _, holdings_map, initial_value, _ = snapshot_value()
    recorded_sells: set[str] = set()
    if not holdings_map:
        logger.success("✅ No crypto holdings to liquidate.")
        return

    logger.info(f"📊 Initial non-USD value ≈ ${initial_value:.2f} across {len(holdings_map)} assets")

    def _sell_symbol_usd_confirm(symbol: str, amount: float, confirm_timeout_s: int = 45, confirm_interval_s: int = 3) -> tuple[bool, float, float]:
        """Attempt to sell SYMBOL-USD and confirm via balance decrease.
        Returns (confirmed, old_amt, new_amt).
        """
        if amount <= 0:
            return True, 0.0, 0.0
        before_amt = get_symbol_balance(symbol)
        pid = f"{symbol}-USD"
        logger.info(f"💸 Selling {amount:.8f} {symbol} → USD…")
        _ = place_market_sell_sized(product_id=pid, amount=amount, order_tag="final-sell")
        ok, new_amt = wait_for_symbol_decrease(symbol, before_amt, timeout_s=confirm_timeout_s, interval_s=confirm_interval_s)
        # One slow recheck in case of race
        if not ok:
            try:
                bals = get_balance()
                new_amt = float(bals.get(symbol, before_amt))
                ok = new_amt < before_amt
            except Exception:
                ok = False
        return ok, before_amt, new_amt

    for attempt in range(1, max_attempts + 1):
        logger.info(f"🧾 Liquidation attempt {attempt}/{max_attempts}")
        # Keep a snapshot of amounts before this attempt
        pre_attempt_holdings = holdings_map.copy()
        # Try selling all
        for symbol, amount in list(holdings_map.items()):
            if amount <= 0:
                continue
            try:
                confirmed, old_amt, new_amt = _sell_symbol_usd_confirm(symbol, amount)
                if confirmed:
                    delta = max(0.0, old_amt - new_amt)
                    # Record immediately for plotting
                    try:
                        px = get_symbol_usd_price(symbol)
                    except Exception:
                        px = 0.0
                    local_now = datetime.now().astimezone(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                    trade_history.append((local_now, "SELL (Final)", px, delta * (px or 0.0)))
                    recorded_sells.add(symbol)
                    logger.success(f"✅ Final sell confirmed for {symbol}: {old_amt:.8f} → {new_amt:.8f}")
                else:
                    logger.info(f"⏳ {symbol} sale not yet confirmed; will re-check after global wait.")
            except Exception as e:
                logger.warning(f"❌ Error during {symbol} liquidation attempt: {e}")
            # Small pause between asset sells to avoid rate spikes and allow processing
            time.sleep(0.5)

    # Wait for balances to update
        logger.info(f"⏳ Waiting {min_wait_seconds}s for balances to update…")
        time.sleep(min_wait_seconds)

        _, holdings_map, remaining_value, _ = snapshot_value()
        # Record any detected reductions as final sells (for plotting) if not already recorded
        try:
            for sym, prev_amt in pre_attempt_holdings.items():
                new_amt = float(holdings_map.get(sym, 0.0))
                if new_amt < prev_amt and sym not in recorded_sells:
                    try:
                        px = get_symbol_usd_price(sym)
                    except Exception:
                        px = 0.0
                    delta = prev_amt - new_amt
                    local_now = datetime.now().astimezone(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                    trade_history.append((local_now, "SELL (Final)", px, delta * (px or 0.0)))
                    recorded_sells.add(sym)
        except Exception:
            pass
        logger.info(f"📉 Remaining non-USD value ≈ ${remaining_value:.2f}")

        if remaining_value <= float(min_remaining_usd):
            logger.success("🎉 Final liquidation success: remaining < $1")
            # Append final portfolio value point for plotting
            try:
                _, _, final_total, _ = snapshot_value()
                balance_history.append((datetime.now().astimezone(LOCAL_TIMEZONE), final_total, None, None))
            except Exception:
                pass
            return

    # After all attempts
    _, holdings_map, remaining_value, _ = snapshot_value()
    if remaining_value <= float(min_remaining_usd):
        logger.success("🎉 Final liquidation success after retries: remaining < $1")
    else:
        logger.error(f"⚠️ Final liquidation incomplete: remaining ≈ ${remaining_value:.2f}, holdings: {holdings_map}")
    # Append final portfolio value point for plotting regardless
    try:
        _, _, final_total, _ = snapshot_value()
        balance_history.append((datetime.now().astimezone(LOCAL_TIMEZONE), final_total, None, None))
    except Exception:
        pass

def should_buy(last_time, extended=False):
    if not last_time:
        return True
    minutes_passed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
    cooldown = COOLDOWN_MINUTES * (2 if extended else 1)
    return minutes_passed >= cooldown

def plot_portfolio():
    if not balance_history:
        print("⚠️ No balance data to plot yet. Wait for a full trading cycle.")
        return
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    def parse_local(dt_str: str):
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=LOCAL_TIMEZONE)

    def value_at(ts):
        try:
            times_l = [e[0] for e in balance_history]
            vals_l = [e[1] for e in balance_history]
            idx = min(range(len(times_l)), key=lambda i: abs(times_l[i] - ts))
            return vals_l[idx]
        except Exception:
            return None

    times = [e[0] for e in balance_history]
    totals = [e[1] for e in balance_history]
    usd_vals = [e[2] if len(e) > 2 else None for e in balance_history]
    crypto_vals = []
    for i, row in enumerate(balance_history):
        if len(row) > 3:
            crypto_vals.append(row[3])
        else:
            u = usd_vals[i]
            if u is not None:
                try:
                    crypto_vals.append(max(0.0, totals[i] - u))
                except Exception:
                    crypto_vals.append(None)
            else:
                crypto_vals.append(None)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(times, totals, label="Total Value", color="#1976D2", linewidth=2.2, zorder=3)
    if any(v is not None for v in usd_vals):
        usd_line = [v if v is not None else float('nan') for v in usd_vals]
        ax.plot(times, usd_line, label="USD Cash", color="#43A047", linewidth=1.4, alpha=0.9, zorder=2)
        try:
            ax.fill_between(times, 0, usd_line, color="#43A047", alpha=0.07, linewidth=0)
        except Exception:
            pass
    if any(v is not None for v in crypto_vals):
        crypto_line = [v if v is not None else float('nan') for v in crypto_vals]
        if any(v is not None for v in usd_vals):
            stacked = []
            for c, u in zip(crypto_line, usd_vals):
                if c is None or u is None:
                    stacked.append(float('nan'))
                else:
                    stacked.append(c + u)
            ax.plot(times, stacked, label="USD+Crypto", color="#EF6C00", linewidth=0.9, alpha=0.6, zorder=1)
            try:
                ax.fill_between(times, usd_line, stacked, color="#EF6C00", alpha=0.05, linewidth=0)
            except Exception:
                pass
        else:
            ax.plot(times, crypto_line, label="Crypto Value", color="#EF6C00", linewidth=1.2, alpha=0.8, zorder=1)
            try:
                ax.fill_between(times, 0, crypto_line, color="#EF6C00", alpha=0.05, linewidth=0)
            except Exception:
                pass
    ma = pd.Series(totals).rolling(window=10, min_periods=1).mean()
    ax.plot(times, ma, label="10-Period MA (Total)", linestyle="--", color="#90CAF9", linewidth=1.5, zorder=4)

    # Build holding segments from trade history
    segments = []  # list of (start_dt, end_dt, symbol)
    current = None
    cur_start = None
    for t in sorted(trade_history, key=lambda x: x[0]):
        dt = parse_local(t[0]) if isinstance(t[0], str) else t[0]
        action = t[1]
        if action.startswith("BUY "):
            sym = action.replace("BUY", "").strip()
            # close any previous
            if current and cur_start:
                segments.append((cur_start, dt, current))
            current = sym
            cur_start = dt
        elif action.startswith("SWAP ") and "->" in action:
            try:
                right = action.split("->", 1)[1].strip()
                # close previous
                if current and cur_start:
                    segments.append((cur_start, dt, current))
                current = right
                cur_start = dt
            except Exception:
                pass
        elif action.startswith("SELL"):
            # close current
            if current and cur_start:
                segments.append((cur_start, dt, current))
            current = None
            cur_start = None
    # Close last open segment at final timestamp
    if current and cur_start:
        segments.append((cur_start, times[-1], current))

    # Color map for symbols
    cmap = plt.get_cmap('tab20')
    sym_colors = {}
    def color_for(sym):
        if sym not in sym_colors:
            sym_colors[sym] = cmap(len(sym_colors) % 20)
        return sym_colors[sym]

    # Shade holding periods and label
    for start, end, sym in segments:
        ax.axvspan(start, end, color=color_for(sym), alpha=0.07, zorder=0)
        # place label near start
        try:
            y = value_at(start) or totals[0]
            ax.text(start, y, sym, fontsize=8, color=color_for(sym), alpha=0.9, va='bottom', ha='left')
        except Exception:
            pass

    # Buy/Sell markers with vertical lines and annotations
    for t in trade_history:
        dt = parse_local(t[0]) if isinstance(t[0], str) else t[0]
        action = t[1]
        price = t[2] if len(t) > 2 else None
        yval = value_at(dt) or (t[3] if len(t) > 3 else None) or totals[0]
        if action.startswith("BUY "):
            sym = action.replace("BUY", "").strip()
            ax.axvline(dt, color="#43A047", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.scatter([dt], [yval], color="#43A047", marker="^", s=90, edgecolor='white', linewidth=0.6, zorder=5, label=None)
            label = f"BUY {sym}"
            if price is not None:
                try:
                    label += f" @ ${float(price):.4f}"
                except Exception:
                    pass
            ax.annotate(label, (dt, yval), textcoords="offset points", xytext=(8,10), ha='left', fontsize=9, color="#2E7D32")
        elif action.startswith("SELL"):
            ax.axvline(dt, color="#E53935", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.scatter([dt], [yval], color="#E53935", marker="v", s=90, edgecolor='white', linewidth=0.6, zorder=5, label=None)
            label = "SELL"
            if price is not None:
                try:
                    label += f" @ ${float(price):.4f}"
                except Exception:
                    pass
            ax.annotate(label, (dt, yval), textcoords="offset points", xytext=(8,-14), ha='left', fontsize=9, color="#B71C1C")
        elif action.startswith("SWAP ") and "->" in action:
            ax.axvline(dt, color="#6D4C41", linestyle=":", linewidth=1.0, alpha=0.6)
            ax.scatter([dt], [yval], color="#6D4C41", marker="s", s=75, edgecolor='white', linewidth=0.6, zorder=5, label=None)
            ax.annotate(action, (dt, yval), textcoords="offset points", xytext=(8,10), ha='left', fontsize=9, color="#5D4037")

    # Summary stats box
    start_val = totals[0]
    end_val = totals[-1]
    ret_pct = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
    # Simple max drawdown
    peak = totals[0]
    max_dd = 0
    for v in totals:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    usd_start = usd_vals[0] if usd_vals and usd_vals[0] is not None else None
    usd_end = usd_vals[-1] if usd_vals and usd_vals[-1] is not None else None
    crypto_end = crypto_vals[-1] if crypto_vals and crypto_vals[-1] is not None else None
    extras = []
    if usd_start is not None:
        extras.append(f"Start USD: ${usd_start:,.2f}")
    if usd_end is not None:
        extras.append(f"End USD: ${usd_end:,.2f}")
    if crypto_end is not None:
        extras.append(f"End Crypto: ${crypto_end:,.2f}")
    extra_txt = ("\n" + " | ".join(extras)) if extras else ""
    text = f"Start: ${start_val:,.2f}\nEnd: ${end_val:,.2f}\nReturn: {ret_pct:.2f}%\nMax DD: {max_dd*100:.2f}%\nTrades: {len(trade_history)}{extra_txt}"
    ax.text(0.01, 0.97, text, transform=ax.transAxes, va='top', ha='left', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='#BDBDBD', boxstyle='round,pad=0.4', alpha=0.9))

    # Formatting
    ax.set_title("Portfolio Value Breakdown Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("USD Value (Total & Components)", fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# Main trading loop
logger.info("🤖 Starting Dynamic Gainer Bot — please wait at least one loop (~10 minutes) for meaningful activity...")

def main_trading_loop():
    global holdings, peak_price_after_buy, last_switch_time
    try:
        while datetime.now(timezone.utc) < end_time:
            try:
                if ENABLE_TRADING == 0:
                    logger.warning("Trading disabled via config.")
                    break

                growth, symbol, price, product_id, volatility, candle_count = get_top_gainer()
                if not symbol:
                    runtime_minutes = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
                    if runtime_minutes < 10:
                        if int(runtime_minutes * 60) % 120 == 0:
                            logger.info("📊 Gathering market data… please wait 10 minutes for sufficient history.")
                    else:
                        logger.warning("⚠️ No valid gainer found, retrying...")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                if candle_count < MIN_CANDLES_REQUIRED:
                    logger.info(f"⏳ Waiting: have {candle_count} candles, need {MIN_CANDLES_REQUIRED}")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                now = datetime.now(timezone.utc)
                local_now = now.astimezone(LOCAL_TIMEZONE)

                try:
                    balances = get_balance()
                except Exception as e:
                    logger.error("💥 Balance fetch failed: " + str(e))
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                if not balances:
                    logger.warning("⚠️ Skipping cycle due to missing balances.")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                usd_balance = float(balances.get("USD", 0.0))
                cash_total = usd_balance
                total_value = cash_total
                crypto_value = 0.0
                # Global risk usage: how much USD is deployed vs starting USD
                try:
                    risk_used = max(0.0, float(starting_usd) - float(usd_balance))
                except Exception:
                    risk_used = 0.0
                try:
                    risk_left = max(0.0, float(MAX_BUY_USD) - risk_used)
                except Exception:
                    risk_left = float(MAX_BUY_USD)
                # USD-only mode
                live_prices = {}
                for coin, amount in balances.items():
                    if coin != "USD" and amount > 0 and coin.upper() not in STABLECOINS:
                        try:
                            coin_price = get_symbol_usd_price(coin)
                            live_prices[coin] = coin_price
                            val = amount * coin_price
                            crypto_value += val
                            total_value += val
                        except Exception as e:
                            if DEBUG_MODE:
                                logger.debug(f"Failed to fetch price for {coin}-USD: {e}")

                # Track history
                # Track total portfolio value (USD + crypto) for plotting
                balance_history.append((local_now, total_value, usd_balance, crypto_value))
                logger.debug(f"📊 Portfolio snapshot — USD=${usd_balance:.2f}, Crypto=${crypto_value:.2f}, Total=${total_value:.2f}")
                if DEBUG_MODE:
                    logger.debug(f"🛡️ Risk usage — used=${risk_used:.2f}, left=${risk_left:.2f} (cap=${MAX_BUY_USD:.2f})")
                # Reconcile actual holding from balances
                actual_holding, holding_usd = determine_primary_holding(balances, live_prices, min_usd=1.0)
                if actual_holding and holdings != actual_holding:
                    logger.debug(f"Sync holdings from balances: {holdings or 'USD'} -> {actual_holding} (~${holding_usd:.2f})")
                    holdings = actual_holding
                elif (not actual_holding or holding_usd < 1.0) and holdings is not None:
                    # Only treat as flat if we can confirm no non-dust crypto remains; otherwise keep current holding
                    try:
                        if not has_non_dust_crypto(balances, live_prices, dust_usd=1.0):
                            logger.debug("No active crypto position detected (all < $1). Treating as USD.")
                            holdings = None
                            peak_price_after_buy = None
                            entry_price_at_buy = None
                            partial_profit_taken = False
                        else:
                            if DEBUG_MODE:
                                logger.debug("Retaining previous holding due to unpriced/non-dust balance; skipping flat state.")
                    except Exception:
                        # On errors, be conservative and keep holding
                        pass

                current_hold_growth = None  # decimal (e.g., 0.0123 = 1.23%)
                hold_growth_pct = None
                if holdings:
                    try:
                        h_pid = f"{holdings}-USD"
                        h_start_iso = (now - timedelta(minutes=TOP_WINDOW_MINUTES)).isoformat()
                        h_end_iso = now.isoformat()
                        h_candles = fetch_candles_cached(h_pid, h_start_iso, h_end_iso, GRANULARITY) or []
                        h_closes = [float(c[4]) for c in sorted(h_candles, key=lambda c: c[0])]
                        if len(h_closes) >= 2:
                            current_hold_growth = calculate_time_weighted_growth(h_closes)
                            # Apply second-half penalty
                            if len(h_closes) >= 3:
                                mid = len(h_closes) // 2
                                mid_price = h_closes[mid]
                                second_half_growth = (h_closes[-1] - mid_price) / mid_price if mid_price > 0 else 0
                                if current_hold_growth > 0 and second_half_growth < -0.008:
                                    current_hold_growth *= 0.6
                            hold_growth_pct = current_hold_growth * 100
                            last_growth[holdings] = current_hold_growth
                    except Exception:
                        current_hold_growth = None
                        hold_growth_pct = None

                msg = f"🚀 Top Gainer: {symbol} at ${price:.4f} | Growth={growth*100:.2f}% | Cash=${cash_total:.2f} (USD=${usd_balance:.2f})"
                if holdings:
                    hg_str = f"{hold_growth_pct:.2f}%" if hold_growth_pct is not None else "n/a"
                    hold_amt = balances.get(holdings, 0.0)
                    msg += f" | Hold({holdings}) {hold_amt:.6f} | Growth={hg_str}"
                logger.info(msg)

                # Decide to switch or buy
                should_switch = False
                if holdings and symbol != holdings:
                    # Use current measurement or last known; don't default to 0
                    prev_hold_growth = current_hold_growth if current_hold_growth is not None else last_growth.get(holdings, None)
                    if prev_hold_growth is not None:
                        # Require new outperformance > fees + spreads + threshold
                        dynamic_cost = estimate_switch_cost_pct(holdings, symbol)
                        base_threshold = SIGNIFICANT_DELTA
                        delta_required = base_threshold + dynamic_cost
                        delta = growth - prev_hold_growth

                        # Update sustained outperformance
                        candidate = symbol
                        # Reset if candidate changes
                        global last_candidate_symbol
                        if last_candidate_symbol != candidate:
                            # Reset tracking when candidate changes
                            sustained_beats.clear()
                            outperformance_history.clear()
                            last_candidate_symbol = candidate

                        beat_this_cycle = 1 if delta >= delta_required else 0

                        if USE_ROLLING_OUTPERFORMANCE:
                            dq = outperformance_history.get(candidate)
                            if dq is None:
                                dq = deque(maxlen=SUSTAINED_ROLLING_WINDOW)
                                outperformance_history[candidate] = dq
                            dq.append(beat_this_cycle)
                            beats = sum(dq)
                            total = len(dq)
                            sustained_ok = (total >= SUSTAINED_REQUIRED_BEATS and beats >= SUSTAINED_REQUIRED_BEATS)
                            sustained_display = f"{beats}/{total} (need {SUSTAINED_REQUIRED_BEATS}/{SUSTAINED_ROLLING_WINDOW})"
                        else:
                            if beat_this_cycle:
                                sustained_beats[candidate] = sustained_beats.get(candidate, 0) + 1
                            else:
                                sustained_beats[candidate] = 0
                            beats = sustained_beats.get(candidate, 0)
                            sustained_ok = beats >= SUSTAINED_OUTPERFORMANCE_CYCLES
                            sustained_display = f"{beats} consecutive (need {SUSTAINED_OUTPERFORMANCE_CYCLES})"

                        can_switch_time = (last_switch_time is None) or ((now - last_switch_time).total_seconds() / 60 >= MIN_SWITCH_MINUTES)
                        if DEBUG_MODE:
                            logger.debug(
                                "🔁 Switch eval — new={n:.4%}, hold={h:.4%}, delta={d:.4%}, dyn_cost={c:.4%}, need>={req:.4%}, sustained={sust}, cooldown_ok={cd}".format(
                                    n=growth,
                                    h=prev_hold_growth,
                                    d=delta,
                                    c=dynamic_cost,
                                    req=delta_required,
                                    sust=sustained_display,
                                    cd=can_switch_time
                                )
                            )
                        if prev_hold_growth is not None:
                            try:
                                logger.info(
                                    "📈 Compare — hold={hold} {hg:.2f}% vs top={sym} {tg:.2f}% | Δ={d:.2f}pp | need>={req:.2f}% | sustained={sust}".format(
                                        hold=holdings,
                                        hg=(prev_hold_growth or 0.0) * 100,
                                        sym=symbol,
                                        tg=growth * 100,
                                        d=(growth - (prev_hold_growth or 0.0)) * 100,
                                        req=delta_required * 100,
                                        sust=sustained_display
                                    )
                                )
                            except Exception:
                                pass
                        if sustained_ok and can_switch_time:
                            should_switch = True
                        else:
                            # Summarize why switch is gated
                            reasons = []
                            try:
                                if delta < delta_required:
                                    reasons.append("Δ below required ({:.2f}% < {:.2f}%)".format(delta*100, delta_required*100))
                            except Exception:
                                pass
                            if not sustained_ok:
                                reasons.append(f"sustained not met ({sustained_display})")
                            if not can_switch_time:
                                try:
                                    mins_since = (now - last_switch_time).total_seconds() / 60 if last_switch_time else 0
                                    mins_left = max(0.0, MIN_SWITCH_MINUTES - mins_since)
                                    reasons.append(f"cooldown {mins_left:.1f}m remaining")
                                except Exception:
                                    reasons.append("cooldown active")
                            if reasons:
                                logger.info("🔒 Switch gated — " + "; ".join(reasons))
                    else:
                        if DEBUG_MODE:
                            logger.debug("🔁 Switch eval skipped — unable to compute holding growth this cycle")

                # Treat flat as: no non-cash crypto with USD value >= $1 (robust against transient price lookup failures)
                is_flat = (holdings is None) and (not has_non_dust_crypto(balances, live_prices, dust_usd=1.0))
                should_fresh_buy = (is_flat and cash_total >= 1 and should_buy(last_trade_time_per_symbol.get(symbol)))
                if not should_fresh_buy:
                    reasons = []
                    if not is_flat:
                        reasons.append("not flat (non-dust crypto detected)")
                    if cash_total < 1:
                        reasons.append(f"USD ${cash_total:.2f} < $1")
                    try:
                        lt = last_trade_time_per_symbol.get(symbol)
                        if not should_buy(lt):
                            if lt:
                                mins_passed = (now - lt).total_seconds() / 60
                                mins_req = COOLDOWN_MINUTES
                                mins_left = max(0.0, mins_req - mins_passed)
                                reasons.append(f"cooldown {mins_left:.1f}m remaining for {symbol}")
                            else:
                                reasons.append("cooldown active")
                    except Exception:
                        pass
                    if reasons:
                        logger.info("🧯 Fresh buy gated: " + "; ".join(reasons))

                # Track if we already purchased (swap or USD bridge)
                completed_purchase = False
                did_direct_swap = False

                # Switching flow (only if current holding >= $1 to avoid dust)
                hold_amt = balances.get(holdings, 0.0) if holdings else 0.0
                hold_price = 0.0
                if holdings:
                    hold_price = live_prices.get(holdings, 0.0)
                    if not hold_price:
                        try:
                            hold_price = get_symbol_usd_price(holdings)
                        except Exception:
                            hold_price = 0.0
                hold_usd = (hold_amt or 0.0) * (hold_price or 0.0)
                if should_switch and hold_usd >= 1.0:
                    amt = balances.get(holdings, 0.0)
                    base_before = holdings
                    # Safety: if there are other non-dust cryptos besides base_before, skip switching to avoid multiple positions
                    try:
                        others_exist = False
                        for c, a in balances.items():
                            if c == "USD" or a <= 0 or c == base_before:
                                continue
                            p = live_prices.get(c, 0.0) or get_symbol_usd_price(c)
                            # Treat unknown/zero price as dust; only block if clearly >= $1
                            if p > 0 and a * p >= 1.0:
                                others_exist = True
                                break
                        if others_exist:
                            logger.warning("Skipping switch: another non-dust crypto is present; enforcing single-position rule.")
                            time.sleep(POLL_INTERVAL_SECONDS)
                            continue
                    except Exception:
                        pass
                    # Spread guard: skip switching if target USD spread is too wide
                    try:
                        target_pid = f"{symbol}-USD"
                        target_spread = estimate_pair_spread_pct(target_pid)
                        spread_cap = spread_limit_for_pid(target_pid)
                        if target_spread > spread_cap:
                            logger.info(f"⏳ Switch deferred: {target_pid} spread {target_spread*100:.2f}% > {spread_cap*100:.2f}%")
                            time.sleep(POLL_INTERVAL_SECONDS)
                            continue
                    except Exception:
                        pass

                    # Try direct swap first, then confirm balance increased
                    before_target_bal = balances.get(symbol, 0.0)
                    if perform_direct_swap(base_before, symbol, amt):
                        ok, new_bal = wait_for_symbol_increase(symbol, before_target_bal, timeout_s=30, interval_s=3)
                        if ok:
                            did_direct_swap = True
                            completed_purchase = True
                            # Update state after swap
                            holdings = symbol
                            last_trade_time_per_symbol[symbol] = datetime.now(timezone.utc)
                            last_switch_time = datetime.now(timezone.utc)
                            peak_price_after_buy = price
                            entry_price_at_buy = price
                            partial_profit_taken = False
                            try:
                                swap_notional = (price or 0.0) * (amt or 0.0)
                            except Exception:
                                swap_notional = None
                            record_trade(f"SWAP {base_before}->{symbol}", price, swap_notional)
                            logger.info(f"🏁 Direct swap confirmed {base_before}->{symbol} (bal {before_target_bal:.8f}→{new_bal:.8f}).")
                            logger.info(f"Balances: {symbol}={new_bal:.6f}")
                        else:
                            logger.warning("Swap placed but target balance did not increase; will try USD bridge.")

                    # If swap failed, use USD bridge only
                    if not did_direct_swap:
                        bridge_cash = choose_bridge_quote(base_before, symbol)
                        if bridge_cash != "USD":
                            logger.warning("Skipping switch: USD bridge not available for both symbols.")
                        else:
                            # USD bridge: sell current holding to USD, then buy target
                            before_base_amt = balances.get(base_before, 0.0)
                            if before_base_amt <= 0:
                                logger.warning("No base asset to sell for switch; skipping.")
                            else:
                                logger.info(f"🔄 Switching via USD bridge: selling {before_base_amt:.6f} {base_before} → USD…")
                                sell_res = place_market_sell_sized(
                                    product_id=f"{base_before}-USD",
                                    amount=before_base_amt,
                                    order_tag="switch-sell",
                                )
                                ok_dec, new_base_bal = wait_for_symbol_decrease(base_before, before_base_amt, timeout_s=45, interval_s=3)
                                if not ok_dec:
                                    # Slow-path recheck
                                    try:
                                        re_bal = get_balance()
                                        new_base_bal = float(re_bal.get(base_before, before_base_amt))
                                    except Exception:
                                        new_base_bal = before_base_amt
                                if new_base_bal < before_base_amt:
                                    logger.success(f"✅ Sold {base_before}. Proceeding to buy {symbol}.")
                                    # Record SELL leg of USD bridge (approx notional using last known hold price)
                                    try:
                                        sell_notional = hold_price * before_base_amt if hold_price else None
                                    except Exception:
                                        sell_notional = None
                                    record_trade("SELL", hold_price, sell_notional)
                                    last_trade_time_per_symbol[base_before] = datetime.now(timezone.utc)
                                    # Refresh balances
                                    time.sleep(2)
                                    try:
                                        balances = get_balance()
                                    except Exception as e:
                                        logger.warning(f"⚠️ Failed to refresh balances after sell: {e}")
                                        balances = {}
                                    usd_balance = float(balances.get("USD", 0.0))
                                    logger.info(f"💵 Cash after sell: USD=${usd_balance:.2f}")
                                    # Recompute risk after sell
                                    try:
                                        risk_used = max(0.0, float(starting_usd) - float(usd_balance))
                                        risk_left = max(0.0, float(MAX_BUY_USD) - risk_used)
                                    except Exception:
                                        risk_left = float(MAX_BUY_USD)

                                    # Buy target with refreshed USD only
                                    cash_symbol, buy_pid = choose_quote_for_symbol(symbol)
                                    if cash_symbol != "USD" or not buy_pid:
                                        logger.warning(f"Skipping switch-buy: {symbol}-USD not tradable.")
                                    else:
                                        available_cash = usd_balance
                                        # USD-only mode
                                        if available_cash < 1:
                                            logger.info("💤 Not enough USD after sell to switch-buy (< $1)")
                                        else:
                                            if risk_left < 1.0:
                                                logger.info("🛡️ Global risk cap reached; skipping switch-buy")
                                                continue
                                            # Budget based on available USD with hard $MAX_BUY_USD cap
                                            quote_budget = min(MAX_BUY_USD, available_cash * MAX_POSITION_PCT)
                                            quote_budget *= (1 - FEE_BUFFER_PERCENT)
                                            if quote_budget <= 0:
                                                logger.warning("⚠️ Computed quote_budget <= 0, skipping switch buy")
                                            else:
                                                # Enforce product quote_min_size and quote_increment
                                                try:
                                                    prod = client.get_product(buy_pid)
                                                    qmin = _parse_float_attr(prod, "quote_min_size", 0.0)
                                                    qinc = _parse_float_attr(prod, "quote_increment", 0.0)
                                                except Exception:
                                                    qmin, qinc = 0.0, 0.0
                                                adj_budget = quote_budget
                                                if qinc and qinc > 0:
                                                    adj_budget = math.floor(adj_budget / qinc) * qinc
                                                adj_budget = floor_8_decimals(adj_budget)
                                                # Constrain by remaining global risk
                                                adj_budget = min(adj_budget, risk_left)
                                                # Final hard cap safety
                                                adj_budget = min(MAX_BUY_USD, adj_budget)
                                                if qmin and adj_budget < qmin:
                                                    logger.warning(f"⚠️ Switch BUY skipped: budget ${adj_budget:.2f} below min quote ${qmin:.2f} for {buy_pid}")
                                                else:
                                                    # USD-only mode
                                                    logger.info(f"🛒 Switching: buying {symbol} with USD ${adj_budget:.2f}")
                                                    before_target_bal = balances.get(symbol, 0.0)
                                                    # Try maker-first path before market
                                                    did_maker = False
                                                    if MAKER_FIRST_ENABLED:
                                                        try:
                                                            did_maker = try_maker_first_buy(buy_pid, adj_budget)
                                                        except Exception:
                                                            did_maker = False
                                                    if not did_maker:
                                                        buy_result = safe_place_order(
                                                            client.market_order_buy,
                                                            client_order_id=f"switch-buy-{symbol}-{datetime.now().strftime('%H%M%S')}",
                                                            product_id=buy_pid,
                                                            quote_size=str(adj_budget),
                                                        )
                                                    else:
                                                        buy_result = {"status": "FILLED"}
                                                    if did_maker and DEBUG_MODE:
                                                        logger.info(f"🏷️ Maker-first switch-buy submitted for {buy_pid}")
                                                    # Confirm via symbol increase or USD decrease
                                                    confirmed, new_bal, new_cash, mode = wait_for_buy_confirmation(
                                                        symbol, before_target_bal, "USD", available_cash, timeout_s=60, interval_s=3
                                                    )
                                                    if confirmed:
                                                        holdings = symbol
                                                        last_trade_time_per_symbol[symbol] = datetime.now(timezone.utc)
                                                        last_switch_time = datetime.now(timezone.utc)
                                                        peak_price_after_buy = price
                                                        try:
                                                            usd_spent = available_cash - new_cash
                                                            if usd_spent <= 0:
                                                                usd_spent = None
                                                        except Exception:
                                                            usd_spent = None
                                                        record_trade(f"BUY {symbol}", price, usd_spent)
                                                        completed_purchase = True
                                                        logger.success(f"✅ Switch BUY confirmed via {mode}: {symbol} bal={new_bal:.6f}, USD=${new_cash:.2f}")
                                                    else:
                                                        # One last slow recheck
                                                        try:
                                                            re_bal = get_balance()
                                                            rb = float(re_bal.get(symbol, before_target_bal))
                                                            rc = float(re_bal.get("USD", available_cash))
                                                        except Exception:
                                                            rb, rc = before_target_bal, available_cash
                                                        if rb > before_target_bal or (available_cash - rc) >= 0.10:
                                                            holdings = symbol
                                                            last_trade_time_per_symbol[symbol] = datetime.now(timezone.utc)
                                                            last_switch_time = datetime.now(timezone.utc)
                                                            peak_price_after_buy = price
                                                            entry_price_at_buy = price
                                                            partial_profit_taken = False
                                                            try:
                                                                usd_spent_late = available_cash - rc
                                                                if usd_spent_late <= 0:
                                                                    usd_spent_late = None
                                                            except Exception:
                                                                usd_spent_late = None
                                                            record_trade(f"BUY {symbol}", price, usd_spent_late)
                                                            completed_purchase = True
                                                            logger.success(f"✅ Switch BUY confirmed late: {symbol} bal={rb:.6f}, USD=${rc:.2f}")
                                                        else:
                                                            logger.warning(f"❌ Switch BUY not confirmed for {symbol}; balances unchanged.")
                                else:
                                    logger.warning(f"❌ Failed to sell {base_before} during switch; skipping buy")

                # Only buy if no swap/bridge buy happened AND we are flat (no holdings)
                # Prevents buying a new coin without first selling the current holding
                if should_fresh_buy and not completed_purchase:
                    # Safety: double-check no other non-dust crypto exists before buying
                    try:
                        if has_non_dust_crypto(balances, live_prices, dust_usd=1.0):
                            logger.warning("Skipping fresh buy: detected non-dust crypto in balances; enforcing single-position rule.")
                            time.sleep(POLL_INTERVAL_SECONDS)
                            continue
                    except Exception:
                        # On error, be safe and skip buy this cycle
                        logger.warning("Skipping fresh buy due to balance check error; will retry next cycle.")
                        time.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    bals2 = get_balance()
                    usd_balance = float(bals2.get("USD", usd_balance))
                    cash_total = usd_balance
                    if cash_total >= 1:
                        # Fetch candles for momentum/drawdown sizing
                        try:
                            candles = fetch_candles_cached(
                                product_id,
                                (datetime.now(timezone.utc) - timedelta(minutes=TOP_WINDOW_MINUTES)).isoformat(),
                                datetime.now(timezone.utc).isoformat(),
                                GRANULARITY,
                            ) or []
                            candles_sorted = sorted(candles, key=lambda c: c[0])
                            closes = [float(c[4]) for c in candles_sorted]
                            momentum_score = calculate_momentum_score(candles_sorted)
                            drawdown_pct = estimate_recent_drawdown(closes)
                        except Exception:
                            momentum_score = 0.0
                            drawdown_pct = 0.0

                        # Compute sizing multiplier and budget
                        mult = dynamic_position_multiplier(growth, volatility or 0.0, momentum_score, drawdown_pct)
                        # USD-only trading
                        cash_symbol, buy_pid = choose_quote_for_symbol(symbol)
                        if cash_symbol != "USD" or not buy_pid:
                            logger.warning(f"BUY skipped: {symbol}-USD not tradable")
                            continue
                        # Spread guard for buy
                        try:
                            spr = estimate_pair_spread_pct(buy_pid)
                            cap = spread_limit_for_pid(buy_pid)
                            if spr > cap:
                                logger.info(f"⏳ BUY deferred: {buy_pid} spread {spr*100:.2f}% > {cap*100:.2f}%")
                                continue
                        except Exception:
                            pass
                        available_cash = usd_balance
                        total_cash_for_sizing = usd_balance
                        pre_budget = total_cash_for_sizing * MAX_POSITION_PCT
                        sized_budget = pre_budget * (1 - FEE_BUFFER_PERCENT) * mult
                        try:
                            risk_used = max(0.0, float(starting_usd) - float(usd_balance))
                            risk_left = max(0.0, float(MAX_BUY_USD) - risk_used)
                        except Exception:
                            risk_left = float(MAX_BUY_USD)
                        if risk_left < 1.0:
                            logger.info("🛡️ Global risk cap reached; skipping buy")
                            continue
                        quote_budget = min(MAX_BUY_USD, sized_budget, risk_left)
                        quote_budget = floor_8_decimals(max(1.0, quote_budget))

                        if quote_budget <= 0:
                            logger.warning("⚠️ Computed quote_budget <= 0, skipping buy")
                        else:
                            # Enforce product quote_min_size and quote_increment
                            try:
                                prod = client.get_product(buy_pid)
                                qmin = _parse_float_attr(prod, "quote_min_size", 0.0)
                                qinc = _parse_float_attr(prod, "quote_increment", 0.0)
                            except Exception:
                                qmin, qinc = 0.0, 0.0
                            adj_budget = quote_budget
                            if qinc and qinc > 0:
                                adj_budget = math.floor(adj_budget / qinc) * qinc
                            adj_budget = floor_8_decimals(adj_budget)
                            adj_budget = min(adj_budget, risk_left)
                            adj_budget = min(MAX_BUY_USD, adj_budget)
                            if qmin and adj_budget < qmin:
                                logger.warning(f"⚠️ BUY skipped: budget ${adj_budget:.2f} below min quote ${qmin:.2f} for {buy_pid}")
                            else:
                                # Skip dust buys under $1 notional
                                if adj_budget < 1.0:
                                    logger.info("⏳ BUY deferred: computed budget < $1 notional")
                                    continue
                                # USD-only mode
                                logger.info(f"Buying {symbol} sized by signals — budget=USD ${adj_budget:.2f} (mult={mult:.2f})")
                                before_target_bal = get_balance().get(symbol, 0.0)
                                did_maker = False
                                if MAKER_FIRST_ENABLED:
                                    try:
                                        did_maker = try_maker_first_buy(buy_pid, adj_budget)
                                    except Exception:
                                        did_maker = False
                                if not did_maker:
                                    buy_result = safe_place_order(
                                        client.market_order_buy,
                                        client_order_id=f"fresh-buy-{symbol}-{datetime.now().strftime('%H%M%S')}",
                                        product_id=buy_pid,
                                        quote_size=str(adj_budget),
                                    )
                                else:
                                    buy_result = {"status": "FILLED"}
                                if did_maker and DEBUG_MODE:
                                    logger.info(f"🏷️ Maker-first fresh-buy submitted for {buy_pid}")
                                confirmed, new_bal, new_cash, mode = wait_for_buy_confirmation(
                                    symbol, before_target_bal, "USD", available_cash, timeout_s=60, interval_s=3
                                )
                                if confirmed:
                                    holdings = symbol
                                    last_trade_time_per_symbol[symbol] = datetime.now(timezone.utc)
                                    last_switch_time = datetime.now(timezone.utc)
                                    peak_price_after_buy = price
                                    entry_price_at_buy = price
                                    partial_profit_taken = False
                                    try:
                                        usd_spent = available_cash - new_cash
                                        if usd_spent <= 0:
                                            usd_spent = None
                                    except Exception:
                                        usd_spent = None
                                    record_trade(f"BUY {symbol}", price, usd_spent)
                                    completed_purchase = True
                                    logger.success(f"✅ BUY confirmed via {mode}: {symbol} bal={new_bal:.6f}, USD=${new_cash:.2f}")
                                else:
                                    # Late recheck
                                    try:
                                        re_bal = get_balance()
                                        rb = float(re_bal.get(symbol, before_target_bal))
                                        rc = float(re_bal.get("USD", available_cash))
                                    except Exception:
                                        rb, rc = before_target_bal, available_cash
                                    if rb > before_target_bal or (available_cash - rc) >= 0.10:
                                        holdings = symbol
                                        last_trade_time_per_symbol[symbol] = datetime.now(timezone.utc)
                                        last_switch_time = datetime.now(timezone.utc)
                                        peak_price_after_buy = price
                                        entry_price_at_buy = price
                                        partial_profit_taken = False
                                        try:
                                            usd_spent_late = available_cash - rc
                                            if usd_spent_late <= 0:
                                                usd_spent_late = None
                                        except Exception:
                                            usd_spent_late = None
                                        record_trade(f"BUY {symbol}", price, usd_spent_late)
                                        completed_purchase = True
                                        logger.success(f"✅ BUY confirmed late: {symbol} bal={rb:.6f}, USD=${rc:.2f}")
                                    else:
                                        logger.warning(f"❌ BUY not confirmed for {symbol}; balances unchanged.")

                if holdings:
                    # Current holding price (USD)
                    h_price_current = hold_price or 0.0
                    if h_price_current <= 0:
                        try:
                            h_price_current = get_symbol_usd_price(holdings)
                        except Exception:
                            h_price_current = 0.0

                    try:
                        h_pid2 = f"{holdings}-USD"
                        h_candles2 = fetch_candles_cached(
                            h_pid2,
                            (datetime.now(timezone.utc) - timedelta(minutes=TOP_WINDOW_MINUTES)).isoformat(),
                            datetime.now(timezone.utc).isoformat(),
                            GRANULARITY,
                        ) or []
                        h_sorted2 = sorted(h_candles2, key=lambda c: c[0])
                        h_closes2 = [float(c[4]) for c in h_sorted2]
                        h_volumes2 = [float(c[5]) for c in h_sorted2]
                    except Exception:
                        h_closes2, h_volumes2 = [], []

                    # Estimate holding volatility
                    try:
                        h_volatility = (np.std(h_closes2) / h_closes2[0]) if h_closes2 and h_closes2[0] > 0 else 0.0
                    except Exception:
                        h_volatility = 0.0

                    # Update trailing stop base with holding price
                    if peak_price_after_buy is None or (h_price_current and h_price_current > peak_price_after_buy):
                        peak_price_after_buy = h_price_current

                    dyn_stop_pct = min(TRAILING_STOP_BASE + h_volatility * 1.5, 0.04)
                    # Ratchet trailing stop tighter as unrealized gains grow; execute partial profits
                    if ENABLE_PARTIAL_PROFITS and entry_price_at_buy and h_price_current > 0:
                        try:
                            unrealized_gain = (h_price_current - entry_price_at_buy) / entry_price_at_buy
                        except Exception:
                            unrealized_gain = 0.0
                        if unrealized_gain > 0 and RATCHET_THRESHOLDS and RATCHET_TRAIL_PCTS:
                            for thr, tstop in zip(RATCHET_THRESHOLDS, RATCHET_TRAIL_PCTS):
                                if unrealized_gain >= thr:
                                    dyn_stop_pct = min(dyn_stop_pct, tstop)
                        if (not partial_profit_taken) and unrealized_gain >= PARTIAL_PROFIT_1_PCT:
                            try:
                                bal_map_pp = get_balance()
                                cur_amt_pp = bal_map_pp.get(holdings, 0.0)
                                if cur_amt_pp > 0:
                                    try:
                                        prod_pp = client.get_product(f"{holdings}-USD")
                                        base_inc_pp = _parse_float_attr(prod_pp, "base_increment", 1e-8)
                                        base_min_pp = _parse_float_attr(prod_pp, "base_min_size", 0.0)
                                    except Exception:
                                        base_inc_pp, base_min_pp = 1e-8, 0.0
                                    sell_amt_pp = cur_amt_pp * PARTIAL_PROFIT_1_SELL_PCT
                                    sell_amt_pp = floor_to_increment(sell_amt_pp, base_inc_pp)
                                    remaining_pp = cur_amt_pp - sell_amt_pp
                                    if base_min_pp and 0 < remaining_pp < base_min_pp:
                                        if remaining_pp * h_price_current < MIN_SELL_NOTIONAL_USD:
                                            sell_amt_pp = cur_amt_pp
                                    notional_est_pp = sell_amt_pp * h_price_current
                                    if sell_amt_pp > 0 and notional_est_pp >= MIN_SELL_NOTIONAL_USD:
                                        logger.info(f"💰 Partial profit: selling {sell_amt_pp:.6f} {holdings} (${notional_est_pp:.2f}) at gain {unrealized_gain*100:.2f}%")
                                        place_market_sell_sized(f"{holdings}-USD", sell_amt_pp, order_tag="partial-profit")
                                        ok_pp, _ = wait_for_symbol_decrease(holdings, cur_amt_pp, timeout_s=30, interval_s=3)
                                        if ok_pp:
                                            record_trade("SELL-PARTIAL", h_price_current, notional_est_pp)
                                            partial_profit_taken = True
                                            peak_price_after_buy = h_price_current
                                        else:
                                            logger.warning("⚠️ Partial profit sell not confirmed; will retry later")
                            except Exception:
                                if DEBUG_MODE:
                                    logger.debug("Partial profit attempt failed silently")
                    stop_price = (peak_price_after_buy or 0.0) * (1 - dyn_stop_pct)

                    exit_reason = None
                    # Post-entry grace period: suppress exits too soon after last buy/switch to avoid churn
                    try:
                        last_entry_time = None
                        if holdings in last_trade_time_per_symbol:
                            last_entry_time = last_trade_time_per_symbol.get(holdings)
                        if last_entry_time:
                            seconds_since_entry = (datetime.now(timezone.utc) - last_entry_time).total_seconds()
                            if seconds_since_entry < EXIT_GRACE_SECONDS:
                                if DEBUG_MODE:
                                    logger.debug(f"⏳ Exit grace active for {holdings}: {seconds_since_entry:.1f}s < {EXIT_GRACE_SECONDS}s")
                                # Skip exit evaluations this cycle
                                exit_reason = None
                                should_exit = False
                                # Short-circuit to next loop iteration
                                time.sleep(POLL_INTERVAL_SECONDS)
                                continue
                    except Exception:
                        pass
                    # Additional negative trend exits (micro-trend + EMA cross) after grace window
                    try:
                        last_entry_time2 = last_trade_time_per_symbol.get(holdings)
                        seconds_since_entry2 = (datetime.now(timezone.utc) - last_entry_time2).total_seconds() if last_entry_time2 else 999999
                    except Exception:
                        seconds_since_entry2 = 999999
                    if exit_reason is None and seconds_since_entry2 >= NEG_EXIT_MIN_SECONDS_SINCE_ENTRY:
                        # Only evaluate if we have sufficient closes
                        if ENABLE_NEG_CONSEC_EXIT and h_closes2 and len(h_closes2) >= (NEG_CONSEC_CANDLES_EXIT + 1):
                            trigger_neg, avg_drop = detect_consecutive_negative(h_closes2, NEG_CONSEC_CANDLES_EXIT, NEG_MIN_AVG_DROP_PCT)
                            if trigger_neg:
                                exit_reason = f"Consecutive down closes x{NEG_CONSEC_CANDLES_EXIT} avg_drop={avg_drop*100:.2f}%"
                        if exit_reason is None and ENABLE_EMA_TREND_EXIT and h_closes2 and len(h_closes2) >= (EMA_SLOW_PERIOD + 2):
                            try:
                                unrealized = 0.0
                                if entry_price_at_buy and h_price_current:
                                    unrealized = (h_price_current - entry_price_at_buy) / entry_price_at_buy
                                if unrealized >= EMA_EXIT_MIN_GAIN_PCT:
                                    if detect_ema_bearish_cross(h_closes2, EMA_FAST_PERIOD, EMA_SLOW_PERIOD):
                                        exit_reason = f"EMA bearish cross F{EMA_FAST_PERIOD}/S{EMA_SLOW_PERIOD} after gain {unrealized*100:.2f}%"
                            except Exception:
                                pass
                    if h_price_current > 0 and stop_price > 0 and h_price_current <= stop_price:
                        exit_reason = f"Trailing stop hit: {h_price_current:.4f} <= {stop_price:.4f}"
                    if not exit_reason and len(h_closes2) >= 6 and len(h_volumes2) >= 4:
                        should_exit, exit_score, reason_str = calculate_dynamic_exit_signal(
                            h_closes2, h_volumes2, h_price_current, peak_price_after_buy or h_price_current
                        )
                        if should_exit:
                            exit_reason = f"Dynamic exit: {reason_str} (score={exit_score:.2f})"

                    if exit_reason:
                        amt = get_balance().get(holdings, 0.0)
                        if amt > 0:
                            logger.info(f"🚨 Exit {holdings}: {exit_reason}")
                            sell_pid = f"{holdings}-USD"
                            # Always attempt exit even in limit/post-only modes; fallback logic will place limit orders
                            res = place_market_sell_sized(
                                product_id=sell_pid,
                                amount=amt,
                                order_tag="exit",
                            )
                            # Immediate limit fallback if market sell failed to return an order (ad 1)
                            if res is None:
                                try:
                                    prod_fb = client.get_product(sell_pid)
                                    base_inc_fb = _parse_float_attr(prod_fb, "base_increment", 1e-8)
                                    bid_fb, ask_fb = get_orderbook_bid_ask(sell_pid)
                                    # For a SELL we want a price that crosses the spread (<= best bid). Use bid (or slightly below if available)
                                    if bid_fb and bid_fb > 0:
                                        price_fb = bid_fb
                                        # Size to increment
                                        base_size_fb = floor_to_increment(amt, base_inc_fb)
                                        lo_id_fb = f"immediate-fallback-sell-{sell_pid}-{datetime.now().strftime('%H%M%S')}"
                                        logger.warning(f"⚠️ Market sell returned None; attempting immediate crossing limit SELL {sell_pid} size={base_size_fb} price={price_fb}")
                                        try:
                                            client.limit_order_sell(
                                                client_order_id=lo_id_fb,
                                                product_id=sell_pid,
                                                base_size=str(base_size_fb),
                                                price=str(price_fb),
                                                post_only=False,
                                            )
                                        except Exception as le2:
                                            logger.error(f"Immediate limit fallback placement failed: {le2}")
                                        # Short pause to allow fill
                                        time.sleep(4)
                                except Exception:
                                    pass
                            # Extra diagnostics when order response is None
                            if res is None:
                                try:
                                    prod_dbg = client.get_product(sell_pid)
                                    base_inc_dbg = _parse_float_attr(prod_dbg, "base_increment", 1e-8)
                                    base_min_dbg = _parse_float_attr(prod_dbg, "base_min_size", 0.0)
                                    logger.warning(f"🔍 Sell diagnostic {sell_pid}: amt={amt:.10f}, base_min={base_min_dbg}, base_inc={base_inc_dbg}")
                                    if base_min_dbg and amt < base_min_dbg:
                                        # Treat as unrecoverable dust; clear position to avoid repeated failed attempts
                                        sell_failures[holdings] = sell_failures.get(holdings, 0) + 1
                                        logger.warning(f"🧹 Dust position detected (< min size). Marking {holdings} as cleared without execution.")
                                        record_trade("SELL-DUST", None, None)
                                        holdings = None
                                        peak_price_after_buy = None
                                        entry_price_at_buy = None
                                        partial_profit_taken = False
                                        continue
                                except Exception:
                                    pass
                            ok_dec, new_sym_bal = wait_for_symbol_decrease(holdings, amt, timeout_s=45, interval_s=3)
                            if ok_dec:
                                logger.success(f"✅ Sold {amt:.6f} {holdings} at ${h_price_current:.4f}")
                                last_trade_time_per_symbol[holdings] = datetime.now(timezone.utc)
                                record_trade("SELL", h_price_current, (amt or 0.0) * (h_price_current or 0.0))
                                # Reset failure counter (ad 2)
                                sell_failures[holdings] = 0
                                holdings = None
                                peak_price_after_buy = None
                                entry_price_at_buy = None
                                partial_profit_taken = False
                            else:
                                # Slow-path recheck before declaring failure
                                try:
                                    re_bal = get_balance()
                                    new_sym_bal = re_bal.get(holdings, amt)
                                except Exception:
                                    new_sym_bal = amt
                                if new_sym_bal < amt:
                                    logger.success(f"✅ Sold {holdings} (late detect) at ${h_price_current:.4f}")
                                    last_trade_time_per_symbol[holdings] = datetime.now(timezone.utc)
                                    record_trade("SELL", h_price_current, (amt or 0.0) * (h_price_current or 0.0))
                                    sell_failures[holdings] = 0
                                    holdings = None
                                    peak_price_after_buy = None
                                    entry_price_at_buy = None
                                    partial_profit_taken = False
                                else:
                                    # Increment failure counter and apply escalation if persistent
                                    sell_failures[holdings] = sell_failures.get(holdings, 0) + 1
                                    fail_ct = sell_failures[holdings]
                                    logger.error(f"❌ Sell order for {holdings} failed (attempt #{fail_ct}).")
                                    if fail_ct >= MAX_SELL_FAILURES_FORCE_SKIP:
                                        logger.error(f"🛑 Abandoning further sell attempts for {holdings} after {fail_ct} failures (safety ceiling). Marking as force-cleared.")
                                        record_trade("SELL-ABANDON", h_price_current, None)
                                        holdings = None
                                        peak_price_after_buy = None
                                        entry_price_at_buy = None
                                        partial_profit_taken = False
                                        continue
                                    # Escalation: after 3 failures, attempt aggressive crossing limit sell
                                    try:
                                        prod_esc = client.get_product(sell_pid)
                                        base_inc_e = _parse_float_attr(prod_esc, "base_increment", 1e-8)
                                        base_min_e = _parse_float_attr(prod_esc, "base_min_size", 0.0)
                                    except Exception:
                                        base_inc_e, base_min_e = 1e-8, 0.0
                                    if fail_ct >= 3 and amt >= base_min_e:
                                        try:
                                            bid, ask = get_orderbook_bid_ask(sell_pid)
                                            if bid:
                                                # Cross the spread: place limit sell at bid (or tiny epsilon below to force taker fill if allowed)
                                                price = bid if bid > 0 else h_price_current
                                                base_size = floor_to_increment(amt, base_inc_e)
                                                lo_id = f"force-exit-sell-{sell_pid}-{datetime.now().strftime('%H%M%S')}"
                                                logger.warning(f"⚠️ Escalation: attempting crossing limit SELL {sell_pid} size={base_size} price={price}")
                                                try:
                                                    limit_res = client.limit_order_sell(
                                                        client_order_id=lo_id,
                                                        product_id=sell_pid,
                                                        base_size=str(base_size),
                                                        price=str(price),
                                                        post_only=False,
                                                    )
                                                except Exception as le:
                                                    logger.error(f"Limit escalation failed: {le}")
                                                # Short confirmation wait
                                                time.sleep(5)
                                                try:
                                                    post_bal = get_balance().get(holdings, amt)
                                                except Exception:
                                                    post_bal = amt
                                                if post_bal < amt:
                                                    logger.success(f"✅ Escalated SELL succeeded for {holdings}")
                                                    last_trade_time_per_symbol[holdings] = datetime.now(timezone.utc)
                                                    record_trade("SELL", h_price_current, (amt or 0.0) * (h_price_current or 0.0))
                                                    holdings = None
                                                    peak_price_after_buy = None
                                                    entry_price_at_buy = None
                                                    partial_profit_taken = False
                                                    continue
                                        except Exception:
                                            pass
                                    # Chunked selling attempt
                                    if fail_ct >= MAX_SELL_FAILURES_BEFORE_CHUNK and holdings is not None:
                                        try:
                                            if CHUNK_SELL_SPLIT > 1:
                                                chunk_amt = amt / CHUNK_SELL_SPLIT
                                                logger.warning(f"🪓 Attempting chunked sells: {CHUNK_SELL_SPLIT} chunks of ~{chunk_amt:.6f} {holdings}")
                                                for i_chunk in range(CHUNK_SELL_SPLIT):
                                                    cur_bal_pre = get_balance().get(holdings, 0.0)
                                                    if cur_bal_pre <= 0:
                                                        break
                                                    sell_chunk = min(chunk_amt, cur_bal_pre)
                                                    place_market_sell_sized(sell_pid, sell_chunk, order_tag="chunk-exit")
                                                    time.sleep(4)
                                                    cur_bal_post = get_balance().get(holdings, cur_bal_pre)
                                                    if cur_bal_post < cur_bal_pre:
                                                        logger.info(f"✅ Chunk {i_chunk+1}/{CHUNK_SELL_SPLIT} reduced balance {cur_bal_pre:.6f}→{cur_bal_post:.6f}")
                                                        if cur_bal_post <= 0:
                                                            last_trade_time_per_symbol[holdings] = datetime.now(timezone.utc)
                                                            record_trade("SELL", h_price_current, (amt or 0.0) * (h_price_current or 0.0))
                                                            holdings = None
                                                            peak_price_after_buy = None
                                                            entry_price_at_buy = None
                                                            partial_profit_taken = False
                                                            break
                                            # Refresh amt if still holding
                                            if holdings is not None:
                                                amt = get_balance().get(holdings, amt)
                                        except Exception:
                                            pass
                                    # Stable bridge attempt: sell to stable then stable->USD
                                    if fail_ct >= MAX_SELL_FAILURES_BEFORE_BRIDGE and holdings is not None:
                                        try:
                                            cur_bal_bridge = get_balance().get(holdings, amt)
                                            for stable_sym in STABLE_BRIDGE_SYMBOLS:
                                                if stable_sym == holdings:
                                                    continue
                                                bridge_pid = f"{holdings}-{stable_sym}"
                                                if not is_product_tradable(bridge_pid):
                                                    continue
                                                logger.warning(f"🌉 Attempting stable bridge: {holdings} → {stable_sym} (pid {bridge_pid})")
                                                # Sell holdings into stable
                                                place_market_sell_sized(bridge_pid, cur_bal_bridge, order_tag="bridge-sell")
                                                time.sleep(5)
                                                # Now sell stable to USD if possible
                                                stable_bal = get_balance().get(stable_sym, 0.0)
                                                if stable_bal > 0:
                                                    usd_pid = f"{stable_sym}-USD"
                                                    if is_product_tradable(usd_pid):
                                                        place_market_sell_sized(usd_pid, stable_bal, order_tag="bridge-stable-usd")
                                                        time.sleep(5)
                                                # Re-evaluate holdings
                                                post_bal_check = get_balance().get(holdings, 0.0)
                                                if post_bal_check <= 0:
                                                    logger.success(f"✅ Stable bridge cleared {holdings} via {stable_sym}")
                                                    record_trade("SELL", h_price_current, None)
                                                    holdings = None
                                                    peak_price_after_buy = None
                                                    entry_price_at_buy = None
                                                    partial_profit_taken = False
                                                    break
                                        except Exception:
                                            pass
                                    if fail_ct >= 5:
                                        # Final fallback: if position USD value tiny or cannot be sold, mark cleared
                                        try:
                                            mkt_px = get_reliable_usd_price(holdings)
                                            usd_val = amt * mkt_px if mkt_px else 0
                                        except Exception:
                                            usd_val = 0
                                        if usd_val < 1.0 or amt < base_min_e:
                                            logger.warning(f"🧹 Force-clearing stuck/dust position {holdings} after {fail_ct} failed exit attempts (usd≈${usd_val:.2f}).")
                                            record_trade("SELL-FORCE", h_price_current, usd_val or None)
                                            holdings = None
                                            peak_price_after_buy = None
                                            entry_price_at_buy = None
                                            partial_profit_taken = False
                                            continue

            except (requests.ConnectionError, requests.Timeout) as e:
                logger.error(f"💥 Network connection failed: {e}")
                logger.warning("🔄 Retrying in 30 seconds...")
                time.sleep(30)
                continue
            except Exception as e:
                logger.error(f"💥 Unexpected error in iteration: {e}")
                logger.warning("🔄 Continuing after 60 seconds...")
                time.sleep(60)
                continue
        logger.info("⏰ Runtime complete. Initiating final liquidation…")
        try:
            if ENABLE_TRADING:
                robust_final_liquidation()
        finally:
            try:
                if ENABLE_PLOTTING:
                    plot_portfolio()
            except Exception as pe:
                logger.error(f"❌ Error during plotting after scheduled end: {pe}")
    except KeyboardInterrupt:
        logger.warning("👋 Interrupted by user. Liquidating...")
        try:
            if ENABLE_TRADING:
                robust_final_liquidation()
        except Exception as e:
            logger.error(f"❌ Error during final liquidation: {e}")
        finally:
            try:
                if ENABLE_PLOTTING:
                    plot_portfolio()
            except Exception as pe:
                logger.error(f"❌ Error during plotting on interrupt: {pe}")

if __name__ == "__main__":
    main_trading_loop()
