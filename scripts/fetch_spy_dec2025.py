"""Fetch SPY 1-min bars for December 2025 from IBKR gateway.

Saves to data/candles/spy_1min_dec2025.parquet with columns:
  datetime  — UTC timezone-aware timestamp
  open, high, low, close, volume

Requires the IBKR gateway to be running and logged in (paper port 4002).
Run once; output is gitignored but required by gex_call_wall_hourly.py for
Dec 2025 SPY vol zscore computation.
"""

from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ib_insync import IB, Stock, util

HOST      = "127.0.0.1"
PORT      = 4002
CLIENT_ID = 12
OUT_PATH  = Path("data/candles/spy_1min_dec2025.parquet")


def trading_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ib = IB()
    print(f"Connecting to IBKR gateway {HOST}:{PORT} ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"Connected: {ib.isConnected()}\n")

    spy = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(spy)
    print(f"SPY contract qualified: {spy}\n")

    all_rows = []
    for trade_date in trading_days(date(2025, 12, 1), date(2025, 12, 31)):
        print(f"  Fetching {trade_date} ...", end=" ", flush=True)
        # 16:30 ET = 21:30 UTC (Dec is EST, UTC-5)
        end_dt = f"{trade_date.strftime('%Y%m%d')}-21:30:00"
        try:
            bars = ib.reqHistoricalData(
                spy,
                endDateTime=end_dt,
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if not bars:
            print("no bars returned")
            continue

        for b in bars:
            dt = b.date
            dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            all_rows.append({
                "datetime": dt_utc,
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
            })

        print(f"{len(bars)} bars  ({bars[0].date} – {bars[-1].date})")
        util.sleep(0.4)

    ib.disconnect()

    if not all_rows:
        print("\nNo data fetched — is the gateway running?")
        return

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_PATH)

    print(f"\nSaved {len(df):,} bars → {OUT_PATH}")
    print(f"  Date range: {df['datetime'].min()} – {df['datetime'].max()}")
    print(f"  Trading days: {df['datetime'].dt.date.nunique()}")


if __name__ == "__main__":
    util.startLoop()
    main()
