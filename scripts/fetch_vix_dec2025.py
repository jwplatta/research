"""Fetch VIX 1-min bars for December 2025 from IBKR gateway.

Saves to data/candles/vix_1min_dec2025.parquet with columns:
  datetime  — UTC timezone-aware timestamp
  open, high, low, close, volume

Requires the IBKR gateway to be running and logged in (paper port 4002).
Run once; output is committed to the repo so the main pipeline can read it.
"""

from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ib_insync import IB, Index, util

HOST      = "127.0.0.1"
PORT      = 4002
CLIENT_ID = 10  # distinct from other connections
OUT_PATH  = Path("data/candles/vix_1min_dec2025.parquet")


def trading_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def fetch_day(ib: IB, vix_contract: Index, trade_date: date) -> list:
    """Fetch 1-min VIX bars for a single trading day.

    endDateTime is set to 16:30 ET on that day so IBKR returns the full RTH session.
    """
    # UTC format with dash separator: yyyymmdd-hh:mm:ss
    # 16:30 ET = 21:30 UTC (EST, Dec is UTC-5)
    end_dt = f"{trade_date.strftime('%Y%m%d')}-21:30:00"
    bars = ib.reqHistoricalData(
        vix_contract,
        endDateTime=end_dt,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,  # epoch seconds
    )
    return bars


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ib = IB()
    print(f"Connecting to IBKR gateway {HOST}:{PORT} ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"Connected: {ib.isConnected()}\n")

    vix = Index("VIX", "CBOE")
    ib.qualifyContracts(vix)
    print(f"VIX contract qualified: {vix}\n")

    days = list(trading_days(date(2025, 12, 1), date(2025, 12, 31)))
    all_rows = []

    for trade_date in days:
        print(f"  Fetching {trade_date} ...", end=" ", flush=True)
        try:
            bars = fetch_day(ib, vix, trade_date)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if not bars:
            print("no bars returned")
            continue

        for b in bars:
            # formatDate=2 returns a datetime object (already UTC-aware from ib_insync)
            dt = b.date
            if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                dt_utc = dt.astimezone(timezone.utc)
            else:
                dt_utc = dt.replace(tzinfo=timezone.utc)
            all_rows.append({
                "datetime": dt_utc,
                "open":     b.open,
                "high":     b.high,
                "low":      b.low,
                "close":    b.close,
                "volume":   b.volume,
            })

        print(f"{len(bars)} bars  ({bars[0].date} – {bars[-1].date})")
        util.sleep(0.5)  # be polite to the gateway

    ib.disconnect()

    if not all_rows:
        print("\nNo data fetched — is the gateway running?")
        return

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, OUT_PATH)

    print(f"\nSaved {len(df):,} bars → {OUT_PATH}")
    print(f"  Date range: {df['datetime'].min()} – {df['datetime'].max()}")
    print(f"  Trading days: {df['datetime'].dt.date.nunique()}")


if __name__ == "__main__":
    util.startLoop()
    main()
