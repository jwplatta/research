"""Plot SPX 5-min bars with aggregate GEX walls, SPY volume, and ATM IV.

Top panel:    SPX 5-min candlesticks + aggregate (all expiry) call/put GEX walls
              at two times:
                - solid lines:  walls after first hour  (10:30 ET)
                - dashed lines: walls after three hours (12:30 ET)
              ZGL annotated inline or in corner if far from price.
Middle panel: SPY 5-min volume bars.
Bottom panel: ATM IV (avg of nearest 0DTE call + put) over the day.

Outputs one PNG per trading day to ./tmp/gex_walls/

Usage:
    uv run python scripts/chart_gex_walls_daily.py
"""

import sys
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calc.gex import find_raw_wall_strikes, find_zero_gamma_level
from util.minio import get_duckdb_conn, option_parquet_uris
from util.spx_candles import load_spx_5min, load_spy_5min

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

START_DATE  = date(2025, 12, 1)
END_DATE    = date(2026, 9, 1)
WALL_TIME_1 = time(10, 30)   # after first hour
WALL_TIME_2 = time(12, 30)   # after three hours
OUT_DIR     = Path("./tmp/gex_walls/agg_dual")
TZ          = ZoneInfo("America/New_York")

COLUMNS = [
    "sampled_at", "expiration_date", "contract_type",
    "strike", "gamma", "open_interest", "underlying_price", "volatility",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_options(start: date, end: date) -> dict[date, pd.DataFrame]:
    """Load all-expiry SPXW snapshots from MinIO grouped by trade date.

    Also retains a 0DTE-only subset per date for ATM IV calculation.
    Returns dict of {trade_date: full_day_df}.
    """
    uris = option_parquet_uris("SPXW", start, end)
    con = get_duckdb_conn()
    cols = ", ".join(COLUMNS)
    all_frames: list[pd.DataFrame] = []
    for uri in uris:
        try:
            df = con.execute(
                f"SELECT {cols} FROM read_parquet('{uri}') WHERE open_interest > 0"
            ).df()
            all_frames.append(df)
        except Exception:
            pass

    if not all_frames:
        return {}

    df = pd.concat(all_frames, ignore_index=True)
    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert("America/New_York")
    df["trade_date"] = df["sampled_at"].dt.date
    return {d: grp.reset_index(drop=True) for d, grp in df.groupby("trade_date")}


def snap_at(day_opts: pd.DataFrame, target: time) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Snapshot at or just after target time; falls back to earliest available."""
    base = day_opts["sampled_at"].iloc[0].normalize()
    cutoff = base.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    after = day_opts[day_opts["sampled_at"] >= cutoff]
    pool = after if not after.empty else day_opts
    snap_time = pool["sampled_at"].min()
    return pool[pool["sampled_at"] == snap_time], snap_time


# ---------------------------------------------------------------------------
# ATM IV time series (0DTE only)
# ---------------------------------------------------------------------------

def atm_iv_series(day_opts: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Return ATM IV time series from 0DTE chain (avg nearest call + put).

    Returns DataFrame with index = sampled_at, column = atm_iv.
    """
    opts_0dte = day_opts[
        pd.to_datetime(day_opts["expiration_date"]).dt.date == trade_date
    ]
    records = []
    for snap_time, grp in opts_0dte.groupby("sampled_at"):
        spot  = float(grp["underlying_price"].iloc[0])
        calls = grp[grp["contract_type"].str.upper() == "CALL"]
        puts  = grp[grp["contract_type"].str.upper() == "PUT"]
        if calls.empty or puts.empty:
            continue
        atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
        atm_put  = puts.iloc[(puts["strike"]  - spot).abs().argsort().iloc[0]]
        records.append({
            "sampled_at": snap_time,
            "atm_iv": (float(atm_call["volatility"]) + float(atm_put["volatility"])) / 2,
        })

    if not records:
        return pd.DataFrame(columns=["sampled_at", "atm_iv"])
    return pd.DataFrame(records).set_index("sampled_at").sort_index()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def expected_move(spot: float, atm_iv_pct: float, snap_time: pd.Timestamp) -> float:
    """1σ expected move from snap_time to 16:00 ET using ATM IV."""
    close_time = snap_time.normalize().replace(hour=16, minute=0, second=0, microsecond=0)
    mins_remaining = max((close_time - snap_time).total_seconds() / 60, 1)
    T = mins_remaining / (252 * 390)
    return spot * (atm_iv_pct / 100) * math.sqrt(T)


def plot_day(
    trade_date: date,
    spx_candles: pd.DataFrame,
    spy_candles: pd.DataFrame,
    snap1: pd.DataFrame,
    snap1_time: pd.Timestamp,
    snap2: pd.DataFrame | None,
    snap2_time: pd.Timestamp | None,
    iv: pd.DataFrame,
    zgl: float | None,
) -> None:
    if spx_candles.empty:
        return

    has_volume = not spy_candles.empty
    has_iv     = not iv.empty
    n_panels   = 1 + int(has_volume) + int(has_iv)
    ratios     = [3] + ([1] if has_volume else []) + ([1] if has_iv else [])

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 5 + 2 * (n_panels - 1)),
        gridspec_kw={"height_ratios": ratios}, sharex=False,
    )
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.4)

    ax_price = axes[0]
    ax_vol   = axes[1] if has_volume else None
    ax_iv    = axes[2] if (has_volume and has_iv) else (axes[1] if has_iv else None)

    # --- Candlesticks ---
    x_vals = spx_candles.index
    marker_every = max(1, len(x_vals) // 10)
    for ts, row in spx_candles.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax_price.plot([ts, ts], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax_price.plot([ts, ts], [row["open"], row["close"]], color=color, linewidth=3)

    prices = [float(spx_candles["low"].min()), float(spx_candles["high"].max())]
    candle_range = prices[1] - prices[0]
    candle_mid   = (prices[0] + prices[1]) / 2

    def _add_walls(snap, snap_time, linestyle):
        if snap is None or snap.empty:
            return
        spot = float(snap["underlying_price"].iloc[0])
        call_wall, put_wall = find_raw_wall_strikes(snap, spot=spot)
        label_time = snap_time.strftime("%H:%M")
        if call_wall is not None:
            prices.append(call_wall)
            ax_price.plot(x_vals, [call_wall] * len(x_vals),
                          color="#00701a", linewidth=1.5, linestyle=linestyle,
                          marker="^", markevery=marker_every, markersize=5,
                          label=f"Call {label_time} {call_wall:.0f}")
        if put_wall is not None:
            prices.append(put_wall)
            ax_price.plot(x_vals, [put_wall] * len(x_vals),
                          color="#7f0000", linewidth=1.5, linestyle=linestyle,
                          marker="v", markevery=marker_every, markersize=5,
                          label=f"Put  {label_time} {put_wall:.0f}")

    _add_walls(snap1, snap1_time, "-")
    _add_walls(snap2, snap2_time, "--")

    # Expected move range from snap1 ATM IV
    snap1_iv = iv[iv.index == snap1_time]["atm_iv"].values
    if len(snap1_iv) > 0:
        spot = float(snap1["underlying_price"].iloc[0])
        em = expected_move(spot, float(snap1_iv[0]), snap1_time)
        upper = spot + em
        lower = spot - em
        for level, label in [(upper, f"+1σ {upper:.0f}"), (lower, f"−1σ {lower:.0f}")]:
            prices.append(level)
            ax_price.axhline(level, color="#9e9e9e", linewidth=1.0, linestyle="--", alpha=0.7)
        ax_price.axhspan(lower, upper, alpha=0.05, color="grey")
        ax_price.annotate(
            f"±1σ EM: ±{em:.0f}  [{lower:.0f}–{upper:.0f}]",
            xy=(0, upper), xycoords=("axes fraction", "data"),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="#757575",
        )

    # ZGL
    if zgl is not None:
        if abs(zgl - candle_mid) < candle_range * 3:
            prices.append(zgl)
            ax_price.plot(x_vals, [zgl] * len(x_vals),
                          color="#ff9800", linewidth=1.2, linestyle=":",
                          marker="*", markevery=marker_every, markersize=6,
                          label=f"ZGL {zgl:.0f}")
        else:
            ax_price.annotate(f"ZGL {zgl:.0f}", xy=(1, 1), xycoords="axes fraction",
                              xytext=(-8, -8), textcoords="offset points",
                              ha="right", va="top", fontsize=9, color="#ff9800",
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff9800", alpha=0.8))

    y_min, y_max = float(min(prices)), float(max(prices))
    y_pad = (y_max - y_min) * 0.1
    ax_price.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_price.set_ylabel("SPX")
    ax_price.set_title(
        f"SPX 5-min  |  {trade_date}  |  GEX walls (all expiries)"
        f"  [solid={snap1_time.strftime('%H:%M')} ET"
        + (f"  dashed={snap2_time.strftime('%H:%M')} ET" if snap2_time else "") + "]",
        fontsize=11,
    )
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax_price.xaxis.set_major_locator(mdates.HourLocator(tz=TZ))
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(axis="y", alpha=0.3)

    # --- SPY volume panel ---
    if ax_vol is not None and not spy_candles.empty:
        colors = ["#26a69a" if r["close"] >= r["open"] else "#ef5350"
                  for _, r in spy_candles.iterrows()]
        ax_vol.bar(spy_candles.index, spy_candles["volume"],
                   width=pd.Timedelta(minutes=4), color=colors, alpha=0.8)
        ax_vol.set_ylabel("SPY Vol")
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
        ax_vol.xaxis.set_major_locator(mdates.HourLocator(tz=TZ))
        ax_vol.grid(axis="y", alpha=0.3)
        ax_vol.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

    # --- ATM IV panel ---
    if ax_iv is not None and not iv.empty:
        ax_iv.plot(iv.index, iv["atm_iv"],
                   color="#7b1fa2", linewidth=1.5, marker="o", markersize=3)
        ax_iv.set_ylabel("ATM IV (%)")
        ax_iv.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
        ax_iv.xaxis.set_major_locator(mdates.HourLocator(tz=TZ))
        ax_iv.grid(alpha=0.3)

    fig.autofmt_xdate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{trade_date}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges = []
    cur = start.replace(day=1)
    while cur < end:
        nxt = cur.replace(year=cur.year + 1, month=1) if cur.month == 12 else cur.replace(month=cur.month + 1)
        ranges.append((max(cur, start), min(nxt, end)))
        cur = nxt
    return ranges


def _iv_bucket(iv_pct: float) -> str:
    lo = int(iv_pct // 5) * 5
    return f"{lo}-{lo+5}%"


def _print_summary(stats: list[dict]) -> None:
    df = pd.DataFrame(stats)
    if df.empty:
        return

    total = len(df)
    inside = df["closed_inside"].sum()
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total} days")
    print(f"  Closed inside walls: {inside} / {total}  ({100*inside/total:.1f}%)")

    print(f"\n--- By ZGL position relative to put wall ---")
    for zgl_pos, grp in df.dropna(subset=["zgl_vs_put"]).groupby("zgl_vs_put"):
        n = len(grp)
        n_in = grp["closed_inside"].sum()
        print(f"  ZGL {zgl_pos} put wall:  {n_in}/{n} closed inside  ({100*n_in/n:.1f}%)")

    print(f"\n--- By ATM IV bucket (at {WALL_TIME_1}) ---")
    for bucket, grp in df.dropna(subset=["iv_bucket"]).groupby("iv_bucket"):
        n = len(grp)
        n_in = grp["closed_inside"].sum()
        print(f"  IV {bucket:>10}:  {n_in}/{n} closed inside  ({100*n_in/n:.1f}%)")

    print(f"\n--- Cross-tab: IV bucket × ZGL position ---")
    ct = df.dropna(subset=["iv_bucket", "zgl_vs_put"])
    iv_buckets = sorted(ct["iv_bucket"].unique())
    print(f"  {'IV bucket':<12}  {'ZGL above':>20}  {'ZGL below':>20}")
    print("  " + "-" * 56)

    def _fmt(g: pd.DataFrame) -> str:
        if len(g) == 0:
            return f"{'  -':>20}"
        s = f"{int(g['closed_inside'].sum())}/{len(g)} ({100*g['closed_inside'].sum()/len(g):.0f}%)"
        return f"{s:>20}"

    for bucket in iv_buckets:
        row = ct[ct["iv_bucket"] == bucket]
        above = row[row["zgl_vs_put"] == "above"]
        below = row[row["zgl_vs_put"] == "below"]
        print(f"  {bucket:<12}  {_fmt(above)}  {_fmt(below)}")

    out_path = OUT_DIR / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-day stats → {out_path}")


def main() -> None:
    print(f"Loading SPX 5-min candles {START_DATE} → {END_DATE}...")
    spx_all = load_spx_5min(START_DATE, END_DATE)
    print(f"Loading SPY 5-min candles {START_DATE} → {END_DATE}...")
    spy_all = load_spy_5min(START_DATE, END_DATE)
    trading_days = sorted(spx_all.index.normalize().unique().date.tolist())
    print(f"  SPX covers {len(trading_days)} trading days")
    print(f"  SPY covers {spy_all.index.normalize().nunique()} trading days")

    total_skipped = 0
    total_written = 0
    stats: list[dict] = []

    for month_start, month_end in _month_ranges(START_DATE, END_DATE):
        print(f"\nLoading options {month_start} → {month_end}...")
        opts_by_date = load_options(month_start, month_end)
        print(f"  {len(opts_by_date)} days with option data")

        for trade_date in [d for d in trading_days if month_start <= d < month_end]:
            spx_day = spx_all[spx_all.index.date == trade_date]
            if spx_day.empty:
                continue

            if trade_date not in opts_by_date:
                print(f"  {trade_date} SKIP: no option data")
                total_skipped += 1
                continue

            spy_day = spy_all[spy_all.index.date == trade_date]
            day_opts = opts_by_date[trade_date]

            snap1, snap1_time = snap_at(day_opts, WALL_TIME_1)
            snap2, snap2_time = snap_at(day_opts, WALL_TIME_2)
            if snap2_time == snap1_time:
                snap2, snap2_time = None, None

            spot = float(snap1["underlying_price"].iloc[0])
            zgl  = find_zero_gamma_level(snap1, spot=spot)
            iv   = atm_iv_series(day_opts, trade_date)

            # Walls at snap1
            call_wall, put_wall = find_raw_wall_strikes(snap1, spot=spot)

            # Closing price: last SPX bar
            close_price = float(spx_day["close"].iloc[-1])

            # ATM IV at snap1_time
            snap1_iv_vals = iv[iv.index == snap1_time]["atm_iv"].values
            snap1_iv = float(snap1_iv_vals[0]) if len(snap1_iv_vals) > 0 else None

            # Did market close inside the walls?
            closed_inside = (
                call_wall is not None
                and put_wall is not None
                and put_wall <= close_price <= call_wall
            )

            # ZGL vs put wall
            zgl_vs_put = None
            if zgl is not None and put_wall is not None:
                zgl_vs_put = "above" if zgl >= put_wall else "below"

            stats.append({
                "date":          trade_date,
                "close":         close_price,
                "call_wall":     call_wall,
                "put_wall":      put_wall,
                "closed_inside": closed_inside,
                "zgl":           zgl,
                "zgl_vs_put":    zgl_vs_put,
                "atm_iv":        snap1_iv,
                "iv_bucket":     _iv_bucket(snap1_iv) if snap1_iv is not None else None,
            })

            print(f"  {trade_date}  snap1={snap1_time.strftime('%H:%M')}"
                  + (f"  snap2={snap2_time.strftime('%H:%M')}" if snap2_time else "  snap2=n/a")
                  + (f"  zgl={zgl:.0f}" if zgl else "")
                  + f"  close={'IN' if closed_inside else 'OUT'}")

            plot_day(trade_date, spx_day, spy_day, snap1, snap1_time,
                     snap2, snap2_time, iv, zgl)
            total_written += 1

    print(f"\nDone. {total_written} charts written to {OUT_DIR}  ({total_skipped} skipped)")
    _print_summary(stats)


if __name__ == "__main__":
    main()
