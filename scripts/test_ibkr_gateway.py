"""Smoke test for IBKR Gateway connection via ib_insync.

Connects to the local gateway and pulls:
  - Account summary
  - SPX last price (delayed)
  - SPX historical 1-min bars (last 2 trading days)
  - VIX historical 5-min bars (last 2 trading days)

Gateway must be running and logged in before executing.
Paper trading port: 4002  |  Live port: 4001
"""

import math

from ib_insync import IB, Index, util

HOST = "127.0.0.1"
PORT = 4002   # paper trading; change to 4001 for live
CLIENT_ID = 1


def main() -> None:
    ib = IB()
    print(f"Connecting to IBKR gateway at {HOST}:{PORT} ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"Connected: {ib.isConnected()}\n")

    # --- Account summary ---
    print("=== Account Summary ===")
    summary = ib.accountSummary()
    for item in summary:
        if item.tag in {"NetLiquidation", "TotalCashValue", "BuyingPower"}:
            print(f"  {item.tag}: {item.value} {item.currency}")
    print()

    ib.reqMarketDataType(3)  # delayed data for paper account

    # --- SPX snapshot ---
    print("=== SPX Last Price ===")
    spx = Index("SPX", "CBOE")
    ib.qualifyContracts(spx)
    [ticker] = ib.reqTickers(spx)
    util.sleep(2)
    spx_price = next(
        (v for v in (ticker.last, ticker.close) if v and not math.isnan(v)), None
    )
    print(f"  SPX: {spx_price}\n")

    # --- SPX 1-min historical bars ---
    print("=== SPX 1-min bars (last 2 days) ===")
    spx_bars = ib.reqHistoricalData(
        spx,
        endDateTime="",           # empty = now
        durationStr="2 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if spx_bars:
        print(f"  Got {len(spx_bars)} bars  "
              f"first: {spx_bars[0].date}  last: {spx_bars[-1].date}")
        print(f"  Last bar — O:{spx_bars[-1].open} H:{spx_bars[-1].high} "
              f"L:{spx_bars[-1].low} C:{spx_bars[-1].close} V:{spx_bars[-1].volume}")
    else:
        print("  No bars returned")
    print()

    # --- VIX 5-min historical bars ---
    print("=== VIX 5-min bars (last 2 days) ===")
    vix = Index("VIX", "CBOE")
    ib.qualifyContracts(vix)
    vix_bars = ib.reqHistoricalData(
        vix,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if vix_bars:
        print(f"  Got {len(vix_bars)} bars  "
              f"first: {vix_bars[0].date}  last: {vix_bars[-1].date}")
        print(f"  Last bar — C:{vix_bars[-1].close}")
    else:
        print("  No bars returned")

    ib.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    util.startLoop()
    main()
