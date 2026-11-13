"""Smoke test to verify tractatus tickrake client is installed and can read local data."""

from datetime import date

from tractatus.tickrake import TickrakeClient


def main() -> None:
    client = TickrakeClient()

    print("=== Candles ===")
    providers = client.candles.list_providers()
    print(f"Providers: {providers}")
    for provider in providers:
        freqs = client.candles.list_frequencies(provider)
        print(f"  {provider} frequencies: {freqs}")
        for freq in freqs[:2]:
            symbols = client.candles.list_symbols(provider, frequency=freq)
            print(f"    {freq} symbols ({len(symbols)}): {symbols[:5]}")
            if symbols:
                df = client.candles.read(symbols[0], freq, provider=provider)
                rng = client.candles.date_range(symbols[0], freq, provider=provider)
                print(f"    {symbols[0]} rows={len(df)}, range={rng}")

    print("\n=== Level One ===")
    providers = client.level_one.list_providers()
    print(f"Providers: {providers}")
    for provider in providers:
        dates = client.level_one.list_dates(provider)
        print(f"  {provider} dates: {dates}")
        if dates:
            symbols = client.level_one.list_symbols(provider, sample_date=dates[-1])
            print(f"    {dates[-1]} symbols: {symbols}")
            if symbols:
                df = client.level_one.read(symbols[0], dates[-1], provider=provider)
                print(f"    {symbols[0]} rows={len(df)}, cols={list(df.columns)}")

    print("\n=== Order Book ===")
    providers = client.order_book.list_providers()
    print(f"Providers: {providers}")
    for provider in providers:
        dates = client.order_book.list_dates(provider)
        print(f"  {provider} dates: {dates}")
        if dates:
            symbols = client.order_book.list_symbols(provider, sample_date=dates[-1])
            print(f"    {dates[-1]} symbols: {symbols}")
            if symbols:
                df = client.order_book.read(symbols[0], dates[-1], provider=provider)
                print(f"    {symbols[0]} rows={len(df)}, cols={list(df.columns)}")

    print("\n=== Options ===")
    roots = client.options_filesystem.list_roots("schwab")
    print(f"Roots ({len(roots)}): {roots[:10]}")
    if roots:
        sample_dates = client.options_filesystem.list_sample_dates(roots[0], "schwab")
        print(f"  {roots[0]} sample dates ({len(sample_dates)}): {sample_dates[:5]}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
