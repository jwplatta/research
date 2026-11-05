"""GEX Condor Timing Analysis

Reads wall_stability_30min.csv and answers: when during the day do both
the call wall and the put wall get tested (spot within 10 pts)?  The
"second test bucket" — the later of the two first-test events — marks
the earliest moment an iron condor could safely be initiated knowing
both walls have been confirmed by price.

Input:  out/gex_walls/wall_stability_30min.csv
Output: stdout only
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALL_CSV = Path("out/gex_walls/wall_stability_30min.csv")
APPROACH_PTS = 10.0        # pts from wall = "tested"

# Bucket 0 = 09:30 ET, each bucket is 30 min wide.
# Hours remaining in RTH = time from bucket_start to 16:00.
BUCKET_TIMES = {
    0:  "09:30",
    1:  "10:00",
    2:  "10:30",
    3:  "11:00",
    4:  "11:30",
    5:  "12:00",
    6:  "12:30",
    7:  "13:00",
    8:  "13:30",
    9:  "14:00",
    10: "14:30",
    11: "15:00",
    12: "15:30",
}

# Hours remaining in RTH after bucket start (16:00 is market close)
# e.g. bucket 0 starts 09:30, 6.5 hrs remain; bucket 12 starts 15:30, 0.5 hr remains.
def hours_remaining(bucket: int) -> float:
    hour_offset = bucket * 0.5          # each bucket = 0.5 hr past 09:30
    start_hour = 9.5 + hour_offset      # fractional hour (e.g. 9.5 = 09:30)
    return 16.0 - start_hour            # 16.0 = 16:00


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_stability() -> pd.DataFrame:
    df = pd.read_csv(WALL_CSV, parse_dates=["sample_datetime"])
    # Normalise column names (the CSV uses these exact names per spec)
    df = df.rename(columns=str.strip)
    # Derive a plain date column for grouping
    df["date"] = pd.to_datetime(df["sample_datetime"], utc=True).dt.tz_convert("America/New_York").dt.date
    return df


# ---------------------------------------------------------------------------
# Per-day: find first bucket where spot comes within APPROACH_PTS of a wall
# ---------------------------------------------------------------------------

def find_first_test_bucket(day: pd.DataFrame, wall_col: str) -> int | None:
    """Return the first bucket_num where spot is within APPROACH_PTS of the wall.

    For the call wall: spot >= wall - APPROACH_PTS  (spot approaches from below).
    For the put wall:  spot <= wall + APPROACH_PTS  (spot approaches from above).
    """
    day_sorted = day.sort_values("bucket_num")
    if wall_col == "call_wall":
        close = day_sorted[day_sorted["spot"] >= day_sorted["call_wall"] - APPROACH_PTS]
    else:
        close = day_sorted[day_sorted["spot"] <= day_sorted["put_wall"] + APPROACH_PTS]

    if close.empty:
        return None
    return int(close.iloc[0]["bucket_num"])


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_stability()
    total_days = df["date"].nunique()
    print(f"Loaded {len(df)} rows across {total_days} trading days\n")

    # --- Per-day first-test buckets ---
    records = []
    for day_date, day_df in df.groupby("date"):
        call_bucket = find_first_test_bucket(day_df, "call_wall")
        put_bucket  = find_first_test_bucket(day_df, "put_wall")
        records.append({
            "date":        day_date,
            "call_bucket": call_bucket,
            "put_bucket":  put_bucket,
        })

    per_day = pd.DataFrame(records)

    # --- (1) Days both walls tested ---
    both_tested = per_day.dropna(subset=["call_bucket", "put_bucket"])
    n_both = len(both_tested)
    pct_both = n_both / total_days * 100

    print("=" * 60)
    print("1. BOTH WALLS TESTED")
    print("=" * 60)
    print(f"   Both walls tested: {n_both} of {total_days} days ({pct_both:.1f}%)")
    print()

    if n_both == 0:
        print("No days with both walls tested — nothing further to show.")
        return

    # --- (2) Second-test bucket distribution ---
    # Second test = later of the two first-test buckets
    both_tested = both_tested.copy()
    both_tested["second_bucket"] = both_tested[["call_bucket", "put_bucket"]].max(axis=1).astype(int)

    dist = (
        both_tested.groupby("second_bucket")
        .size()
        .reset_index(name="n_days")
        .sort_values("second_bucket")
    )
    dist["cumul_pct"] = dist["n_days"].cumsum() / n_both * 100
    dist["bucket_time"]     = dist["second_bucket"].map(BUCKET_TIMES)
    dist["hrs_remaining"]   = dist["second_bucket"].apply(hours_remaining)

    print("=" * 60)
    print("2. SECOND-TEST BUCKET DISTRIBUTION  (condor entry timing)")
    print("=" * 60)
    header = f"  {'Bucket':>6}  {'Time':>6}  {'N days':>7}  {'Cumul%':>7}  {'Hrs Left in RTH':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in dist.iterrows():
        print(
            f"  {int(row['second_bucket']):>6}  {row['bucket_time']:>6}  "
            f"{int(row['n_days']):>7}  {row['cumul_pct']:>6.1f}%  "
            f"{row['hrs_remaining']:>16.1f}"
        )
    print()

    # --- (3) Timing milestones ---
    BUCKET_1130 = 4   # 11:30 ET
    BUCKET_1300 = 7   # 13:00 ET

    n_by_1130 = int((both_tested["second_bucket"] <= BUCKET_1130).sum())
    n_by_1300 = int((both_tested["second_bucket"] <= BUCKET_1300).sum())

    print("=" * 60)
    print("3. TIMING MILESTONES")
    print("=" * 60)
    print(f"   Both walls tested by bucket {BUCKET_1130} (11:30 ET): "
          f"{n_by_1130} days ({n_by_1130 / n_both * 100:.1f}% of both-tested days)")
    print(f"   Both walls tested by bucket {BUCKET_1300} (13:00 ET): "
          f"{n_by_1300} days ({n_by_1300 / n_both * 100:.1f}% of both-tested days)")
    print()

    # Bonus: summary of which wall was tested first
    both_tested["first_wall"] = both_tested.apply(
        lambda r: "call" if r["call_bucket"] < r["put_bucket"]
        else ("put" if r["put_bucket"] < r["call_bucket"] else "tie"),
        axis=1,
    )
    first_counts = both_tested["first_wall"].value_counts()
    print("   First wall tested (of both-tested days):")
    for label, cnt in first_counts.items():
        print(f"     {label}: {cnt} ({cnt / n_both * 100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
