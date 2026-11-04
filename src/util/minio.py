"""MinIO / DuckDB connection helpers for reading from the tractatus-lakehouse cache bucket."""

import os
from datetime import date, timedelta

import duckdb
from dotenv import load_dotenv


def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection configured for MinIO S3 access."""
    load_dotenv()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    endpoint = os.environ["MINIO_ENDPOINT"].replace("http://", "").replace("https://", "")
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]
    con.execute(f"""
        SET s3_endpoint='{endpoint}';
        SET s3_url_style='path';
        SET s3_access_key_id='{access_key}';
        SET s3_secret_access_key='{secret_key}';
        SET s3_use_ssl=false;
    """)
    return con


def option_parquet_uris(
    root: str,
    start: date,
    end: date,
    provider: str = "schwab",
    bucket: str = "tractatus-lakehouse",
) -> list[str]:
    """Generate s3:// URIs for each calendar day in [start, end).

    Cache path pattern:
        s3://{bucket}/options/{provider}/YYYY/MM/DD/{root}_samples_YYYY-MM-DD.parquet
    """
    uris = []
    d = start
    while d < end:
        uris.append(
            f"s3://{bucket}/options/{provider}"
            f"/{d.year}/{d.month:02d}/{d.day:02d}"
            f"/{root}_samples_{d}.parquet"
        )
        d += timedelta(days=1)
    return uris
