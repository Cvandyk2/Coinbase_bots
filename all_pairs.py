"""Generate a list of ALL tradable Coinbase product IDs (not just USD pairs).

Writes one product_id per line to list_all.txt in the same directory.

Rules for inclusion:
  * trading_disabled == False
  * limit_only == False
  * post_only == False
  * product_id string present

Usage (from repo root):
  python -m coinbase.Valid_Pairs.all_pairs
or
  python coinbase/Valid_Pairs/all_pairs.py

You must have a valid API key JSON at API_KEY_PATH (same as bots).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable
from loguru import logger

try:
	from coinbase.rest import RESTClient
except Exception as e:  # pragma: no cover
	print("coinbase package missing. Install dependencies first.")
	raise

# --- Configuration ---
API_KEY_PATH = "api_key.json"
OUTPUT_FILENAME = "list_all.txt"
MAX_RETRIES = 6
RETRY_SLEEP_BASE = 1.7  # exponential backoff base
RETRY_STATUSES = {429, 500, 502, 503, 504}

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / OUTPUT_FILENAME

logger.remove()
logger.add(sys.stderr, level="INFO")


def init_client() -> RESTClient:
	return RESTClient(key_file=API_KEY_PATH)


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


def fetch_all_products(client: RESTClient):
	last_err = None
	for attempt in range(MAX_RETRIES):
		try:
			resp = client.get_products()
			prods = getattr(resp, 'products', None) or []
			return prods
		except Exception as e:  # noqa: BLE001
			last_err = e
			# Rate-limit / transient handling heuristic
			msg = str(e).lower()
			status = None
			try:
				status = getattr(getattr(e, 'response', None), 'status_code', None)
			except Exception:  # noqa: BLE001
				pass
			if status in RETRY_STATUSES or 'rate' in msg or 'timeout' in msg or 'temporar' in msg:
				sleep_s = min(20, (RETRY_SLEEP_BASE ** attempt) + 0.25)
				logger.warning(f"Retry {attempt+1}/{MAX_RETRIES} after error: {e} (sleep {sleep_s:.1f}s)")
				time.sleep(sleep_s)
				continue
			raise
	raise RuntimeError(f"Failed to fetch products after {MAX_RETRIES} attempts: {last_err}")


def extract_tradable_ids(products: Iterable) -> list[str]:
	out: list[str] = []
	for p in products:
		try:
			pid = getattr(p, 'product_id', None) or (p.get('product_id') if isinstance(p, dict) else None)
			if not pid:
				continue
			if _is_tradable_product(p):
				out.append(pid)
		except Exception:
			continue
	# Deduplicate & sort (case-insensitive but keep original case)
	dedup = list(dict.fromkeys(out))
	dedup.sort(key=lambda s: s.upper())
	return dedup


def write_list(pids: list[str]):
	try:
		with OUT_PATH.open('w', encoding='utf-8') as f:
			for pid in pids:
				f.write(pid + '\n')
		logger.info(f"📝 Wrote {len(pids)} tradable product IDs to {OUT_PATH}")
	except Exception as e:  # noqa: BLE001
		logger.error(f"Failed writing list file: {e}")
		raise


def main():  # pragma: no cover - simple orchestrator
	if not os.path.isfile(API_KEY_PATH):
		logger.error(f"API key file not found: {API_KEY_PATH}")
		sys.exit(1)
	logger.info("🔑 Initializing client…")
	client = init_client()
	logger.info("📦 Fetching products…")
	prods = fetch_all_products(client)
	logger.info(f"Total products returned: {len(prods)}")
	tradable = extract_tradable_ids(prods)
	logger.info(f"Tradable (non-disabled, non-limit/post-only): {len(tradable)}")
	write_list(tradable)
	logger.success("✅ Done.")


if __name__ == "__main__":
	main()

