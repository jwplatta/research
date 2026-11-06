"""BacktestEngine — loads data, iterates time buckets, calls strategy.step()."""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from backtest.logger import configure as configure_logger
from backtest.portfolio import Portfolio
from backtest.result import BacktestResult
from backtest.strategy import Strategy
from util.minio import get_duckdb_conn, option_parquet_uris

TZ = ZoneInfo("America/New_York")

_BAR_SIZES = {
    "1min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "30min": timedelta(minutes=30),
    "1d": None,  # daily — single step per day, full day's data
}

_RTH_START = time(9, 30)
_RTH_END = time(16, 0)

_LOCAL_LOADERS = {
    "spx_1min": ("util.spx_candles", "load_spx_1min"),
    "spx_5min": ("util.spx_candles", "load_spx_5min"),
    "vix_5min": ("util.spx_candles", "load_vix_5min"),
    "vix9d_1min": ("util.spx_candles", "load_vix9d_1min"),
}


class BacktestEngine:
    """Drives a backtest by iterating trading days and time buckets.

    For each trading day:
    - Loads that day's data for each declared data source.
    - If ``bar_size != '1d'``, splits RTH into fixed-width time buckets and
      calls ``strategy.step()`` once per bucket, passing only data with
      ``sampled_at < bucket_end`` (no lookahead).
    - If ``bar_size == '1d'``, calls ``strategy.step()`` once with the full
      day's data.
    - After all buckets, calls ``strategy.portfolio.record_equity(date)``.

    At the end of the run, calls ``BacktestResult.build()`` to write outputs.

    Args:
        strategy: A fully constructed ``Strategy`` subclass instance.
        start_date: First date to include (inclusive).
        end_date: Last date to exclude (exclusive).
        out_dir: Directory where charts, CSVs, and the log file are saved.
        bar_size: Time bucket granularity — ``'1min'``, ``'5min'``, ``'30min'``,
            ``'60min'``, or ``'1d'`` for one step per day.

    Raises:
        ValueError: If ``bar_size`` is not a supported value.
        ValueError: If ``strategy.data_sources`` contains an unknown source type
            or loader name.
    """

    def __init__(
        self,
        strategy: Strategy,
        start_date: date,
        end_date: date,
        out_dir: str | Path = "./out/backtests",
        bar_size: str = "1d",
    ) -> None:
        if bar_size not in _BAR_SIZES:
            raise ValueError(
                f"bar_size must be one of {list(_BAR_SIZES)}, got '{bar_size}'"
            )
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.out_dir = Path(out_dir)
        self.bar_size = bar_size

        # Inject portfolio now that the strategy is fully constructed
        strategy.portfolio = Portfolio(strategy.start_portfolio_amt)

        self._validate_data_sources()

    def run(self) -> None:
        """Execute the backtest."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.out_dir / f"{self.strategy.name}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log = configure_logger(run_dir, self.strategy.name)
        # Also make the logger available on the strategy for use inside step()
        self.strategy.log = log

        log.info("Running backtest: %s", self.strategy.name)
        log.info("Period: %s → %s", self.start_date, self.end_date)
        log.info("Bar size: %s", self.bar_size)

        local_data = self._load_local_sources(log)
        uri_maps = self._build_minio_uri_maps()
        trading_days = self._build_trading_days(local_data, uri_maps)

        log.info("Trading days: %d", len(trading_days))

        con = get_duckdb_conn() if uri_maps else None
        try:
            for trade_date in trading_days:
                data_slice = self._build_day_slice(trade_date, local_data, uri_maps, con, log)
                if data_slice is None:
                    continue

                if self.bar_size == "1d":
                    day_start = datetime(
                        trade_date.year, trade_date.month, trade_date.day,
                        _RTH_START.hour, _RTH_START.minute, tzinfo=TZ
                    )
                    self.strategy.step(data_slice, timestamp=day_start)
                else:
                    self._run_intraday_buckets(trade_date, data_slice, log)
        finally:
            if con is not None:
                con.close()

        BacktestResult(self.strategy, run_dir, log).build()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_data_sources(self) -> None:
        for name, cfg in self.strategy.data_sources.items():
            src_type = cfg.get("type")
            if src_type not in ("minio", "local"):
                raise ValueError(
                    f"data_source '{name}': unknown type '{src_type}'"
                    " — must be 'minio' or 'local'"
                )
            if src_type == "local":
                loader = cfg.get("loader")
                if loader not in _LOCAL_LOADERS and "path" not in cfg:
                    raise ValueError(
                        f"data_source '{name}': loader '{loader}' not recognised"
                        f" — must be one of {list(_LOCAL_LOADERS)} or provide 'path'"
                    )

    def _load_local_sources(self, log) -> dict[str, pd.DataFrame]:
        """Load all local data sources upfront (full date range)."""
        import importlib

        loaded = {}
        for name, cfg in self.strategy.data_sources.items():
            if cfg["type"] != "local":
                continue

            if "path" in cfg:
                df = pd.read_parquet(cfg["path"])
            else:
                loader_name = cfg["loader"]
                module_path, fn_name = _LOCAL_LOADERS[loader_name]
                mod = importlib.import_module(module_path)
                fn = getattr(mod, fn_name)
                df = fn(self.start_date, self.end_date)

            loaded[name] = df
            log.info("Loaded local source '%s': %d rows", name, len(df))

        return loaded

    def _build_minio_uri_maps(self) -> dict[str, dict[date, str]]:
        """Build {source_name: {date: uri}} for all minio sources."""
        uri_maps = {}
        for name, cfg in self.strategy.data_sources.items():
            if cfg["type"] != "minio":
                continue
            root = cfg.get("root", "SPXW")
            provider = cfg.get("provider", "schwab")
            bucket = cfg.get("bucket", "tractatus-lakehouse")
            uris = option_parquet_uris(root, self.start_date, self.end_date, provider, bucket)
            d = self.start_date
            day_map: dict[date, str] = {}
            for uri in uris:
                day_map[d] = uri
                d += timedelta(days=1)
            uri_maps[name] = day_map

        return uri_maps

    def _build_trading_days(
        self,
        local_data: dict[str, pd.DataFrame],
        uri_maps: dict[str, dict[date, str]],
    ) -> list[date]:
        """Derive a sorted list of candidate trading days from loaded data."""
        days: set[date] = set()

        for df in local_data.values():
            if hasattr(df.index, "date"):
                days.update(df.index.date)
            elif "date" in df.columns:
                days.update(pd.to_datetime(df["date"]).dt.date)

        if not days and not uri_maps:
            d = self.start_date
            while d < self.end_date:
                days.add(d)
                d += timedelta(days=1)
        elif not days and uri_maps:
            first_map = next(iter(uri_maps.values()))
            days = set(first_map.keys())

        return sorted(d for d in days if self.start_date <= d < self.end_date)

    def _build_day_slice(
        self,
        trade_date: date,
        local_data: dict[str, pd.DataFrame],
        uri_maps: dict[str, dict[date, str]],
        con,
        log,
    ) -> dict[str, pd.DataFrame] | None:
        """Load all sources for one trading day, returning None if any minio file is missing.

        Minio sources are loaded once per day using a single DuckDB query.  If the
        data source config includes a ``before_time`` key (a ``datetime.time`` in ET),
        only rows with ``sampled_at < before_time`` are fetched — the filter is pushed
        into the DuckDB WHERE clause so only matching parquet row groups are read from
        MinIO.  Strategies that only need early-morning option chain data (e.g. a
        10:00–10:30 entry window) should set this to avoid loading the full day's file.
        Strategies that need options data throughout the day should omit it.
        """
        data_slice: dict[str, pd.DataFrame] = {}

        for name, day_map in uri_maps.items():
            uri = day_map.get(trade_date)
            if uri is None:
                log.warning("%s SKIP: no minio URI for '%s'", trade_date, name)
                return None

            cfg = self.strategy.data_sources[name]
            before_time: time | None = cfg.get("before_time")
            before_ts: datetime | None = None
            if before_time is not None:
                before_ts = datetime.combine(trade_date, before_time).replace(tzinfo=TZ)
            columns: list[str] | None = cfg.get("columns")

            df = self._load_minio_window(con, uri, before_ts, columns)
            if df is None:
                log.warning("%s SKIP: minio file missing for '%s'", trade_date, name)
                return None
            data_slice[name] = df

        for name, df in local_data.items():
            if hasattr(df.index, "date"):
                day_df = df[df.index.date == trade_date]
            elif "sampled_at" in df.columns:
                day_df = df[pd.to_datetime(df["sampled_at"]).dt.date == trade_date]
            else:
                day_df = df

            if not day_df.empty:
                data_slice[name] = day_df.sort_index()

        return data_slice

    def _load_minio_window(
        self,
        con,
        uri: str,
        before_ts: datetime | None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """Load rows from a MinIO parquet with optional column projection and time filter.

        ``columns`` is pushed down as a SELECT list — parquet's columnar format means
        only those byte ranges are read from MinIO.  ``before_ts`` adds a WHERE clause
        that DuckDB can use to skip row groups outside the time window.
        """
        select = ", ".join(columns) if columns else "*"

        if before_ts is not None:
            from datetime import timezone as _tz
            before_utc = before_ts.astimezone(_tz.utc)
            ts_str = before_utc.strftime("%Y-%m-%d %H:%M:%S")
            query = (
                f"SELECT {select} FROM read_parquet('{uri}', union_by_name=true)"
                f" WHERE sampled_at < TIMESTAMPTZ '{ts_str}'"
            )
        else:
            query = f"SELECT {select} FROM read_parquet('{uri}', union_by_name=true)"

        try:
            df = con.execute(query).df()
        except Exception:
            return None

        if df.empty:
            return None

        if "sampled_at" in df.columns:
            df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(
                "America/New_York"
            )
            df = df.sort_values("sampled_at").reset_index(drop=True)

        return df

    def _run_intraday_buckets(
        self,
        trade_date: date,
        full_day_slice: dict[str, pd.DataFrame],
        log,
    ) -> None:
        """Split RTH into fixed buckets and call strategy.step() for each.

        ``full_day_slice`` is already loaded (once per day).  Each bucket receives
        a pandas slice of that data filtered to ``sampled_at < bucket_end``.
        """
        delta = _BAR_SIZES[self.bar_size]
        assert delta is not None

        rth_start = datetime(
            trade_date.year, trade_date.month, trade_date.day,
            _RTH_START.hour, _RTH_START.minute, tzinfo=TZ
        )
        rth_end = datetime(
            trade_date.year, trade_date.month, trade_date.day,
            _RTH_END.hour, _RTH_END.minute, tzinfo=TZ
        )

        bucket_start = rth_start
        while bucket_start < rth_end:
            bucket_end = bucket_start + delta

            bucket_slice: dict[str, pd.DataFrame] = {}
            missing = []
            for name, df in full_day_slice.items():
                if "sampled_at" in df.columns:
                    sliced = df[df["sampled_at"] < bucket_end]
                elif hasattr(df.index, "tzinfo") and df.index.tzinfo is not None:
                    sliced = df[df.index < bucket_end]
                else:
                    sliced = df

                if sliced.empty:
                    missing.append(name)
                else:
                    bucket_slice[name] = sliced

            if missing:
                log.debug(
                    "%s %s SKIP: no data for %s",
                    trade_date, bucket_start.strftime("%H:%M"), missing,
                )
                bucket_start = bucket_end
                continue

            self.strategy.step(bucket_slice, timestamp=bucket_start)

            bucket_start = bucket_end
