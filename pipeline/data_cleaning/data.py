"""Raw AEMO ingestion + chronological splitting.

This module deliberately does NOT perform model feature engineering.

It fetches:
    - DISPATCHPRICE via NEMOSIS
    - DISPATCHREGIONSUM via NEMOSIS
    - DISPATCHINTERCONNECTORRES via NEMOSIS (optional)
    - P5MIN REGIONSOLUTION via NEMSEER (optional)

and returns those source tables separately on a Data object.

Feature joins, lags, rolling statistics, reserve margin, P5MIN alignment,
normalisation, etc. belong in the preprocessing package.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import nemosis
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
NEMOSIS_CACHE_DIR = BASE_DIR / "nemosis_cache"
NEMSEER_RAW_CACHE_DIR = BASE_DIR / "nemseer_cache" / "raw"
NEMSEER_PROCESSED_CACHE_DIR = BASE_DIR / "nemseer_cache" / "processed"

DEFAULT_REGIONS = [
    "NSW1",
    "QLD1",
    "VIC1",
    "SA1",
    "TAS1",
]
DEFAULT_START = datetime(2022, 1, 1)
DEFAULT_END = datetime(2026, 7, 31)

TEST_FRACTION = 0.20

# Keep these on by default because they are part of the intended modelling data.
# They can be disabled while testing the ingestion code on a small sample.
DEFAULT_INCLUDE_INTERCONNECTORS = True
DEFAULT_INCLUDE_P5MIN = True


@dataclass
class Data:
    """Raw source tables for one chronological split.

    ``df`` is retained for backwards compatibility with older model scripts and
    simply aliases ``prices``. New code should use the named source attributes.
    """

    prices: pd.DataFrame
    region_summary: pd.DataFrame
    interconnectors: pd.DataFrame | None
    p5min: pd.DataFrame | None
    regions: list[str]
    start: datetime
    end: datetime

    @property
    def df(self) -> pd.DataFrame:
        """Backwards-compatible alias for the old price-only ``Data.df``."""
        return self.prices


# ---------------------------------------------------------------------------
# NEMOSIS
# ---------------------------------------------------------------------------

def _nemosis_datetime(dt: datetime) -> str:
    return dt.strftime("%Y/%m/%d %H:%M:%S")


def _fetch_region_table(
    table: str,
    regions: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch a regional dispatch table and retain non-intervention rows."""
    NEMOSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = nemosis.dynamic_data_compiler(
        _nemosis_datetime(start),
        _nemosis_datetime(end),
        table,
        str(NEMOSIS_CACHE_DIR),
        filter_cols=["REGIONID", "INTERVENTION"],
        filter_values=(list(regions), [0]),
    )

    return df.reset_index(drop=True)


def _fetch_interconnectors(
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch all NEM interconnector dispatch results.

    DISPATCHINTERCONNECTORRES is not keyed by REGIONID, so it is fetched once
    for the NEM and the preprocessing layer later decides how to use it.
    """
    NEMOSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = nemosis.dynamic_data_compiler(
        _nemosis_datetime(start),
        _nemosis_datetime(end),
        "DISPATCHINTERCONNECTORRES",
        str(NEMOSIS_CACHE_DIR),
        filter_cols=["INTERVENTION"],
        filter_values=([0],),
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# NEMSEER P5MIN
# ---------------------------------------------------------------------------

def _nemseer_datetime(dt: datetime) -> str:
    # NEMSEER uses minute-resolution datetime strings.
    return dt.strftime("%Y/%m/%d %H:%M")


def _month_chunks(start: datetime, end: datetime):
    """Yield [chunk_start, chunk_end) month-sized windows."""
    cursor = start

    while cursor < end:
        if cursor.month == 12:
            next_month = cursor.replace(
                year=cursor.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            next_month = cursor.replace(
                month=cursor.month + 1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        chunk_end = min(next_month, end)
        yield cursor, chunk_end
        cursor = chunk_end


def _fetch_p5min(
    regions: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch raw P5MIN REGIONSOLUTION forecasts with NEMSEER.

    Data is compiled month-by-month to avoid building a four-year P5MIN query in
    memory at once. NEMSEER's cache means reruns reuse already downloaded data.

    The returned table stays raw: RUN_DATETIME and INTERVAL_DATETIME are kept
    separately. The preprocessing layer is responsible for leakage-safe forecast
    alignment.
    """
    try:
        from nemseer import compile_data, generate_runtimes
    except ImportError as exc:
        raise ImportError(
            "P5MIN fetching is enabled but nemseer is not installed. Install the "
            "PR-72 branch with:\n"
            'pip install "nemseer @ '
            'git+https://github.com/timcu/NEMSEER.git@'
            'b66d59651fc550e06d3249f427fc0117984af78c"'
        ) from exc

    NEMSEER_RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NEMSEER_PROCESSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end_exclusive in _month_chunks(start, end):
        # P5MIN forecast intervals are 5-minute aligned. Treat our requested
        # overall end as exclusive, matching the rest of the project.
        forecasted_end = chunk_end_exclusive - timedelta(minutes=5)

        if forecasted_end < chunk_start:
            continue

        forecasted_start_str = _nemseer_datetime(chunk_start)
        forecasted_end_str = _nemseer_datetime(forecasted_end)

        run_start, run_end = generate_runtimes(
            forecasted_start_str,
            forecasted_end_str,
            "P5MIN",
        )

        compiled = compile_data(
            run_start=run_start,
            run_end=run_end,
            forecasted_start=forecasted_start_str,
            forecasted_end=forecasted_end_str,
            forecast_type="P5MIN",
            tables="REGIONSOLUTION",
            raw_cache=str(NEMSEER_RAW_CACHE_DIR),
            processed_cache=str(NEMSEER_PROCESSED_CACHE_DIR),
            data_format="df",
        )

        if not compiled:
            continue

        df = compiled.get("REGIONSOLUTION")
        if df is None or df.empty:
            continue

        if "REGIONID" in df.columns:
            df = df[df["REGIONID"].astype(str).isin(regions)]

        # Keep only columns that are useful to the current preprocessing schema,
        # plus LASTCHANGED when available so forecast availability can be aligned
        # conservatively.
        wanted = [
            "RUN_DATETIME",
            "INTERVAL_DATETIME",
            "REGIONID",
            "RRP",
            "TOTALDEMAND",
            "AVAILABLEGENERATION",
            "LASTCHANGED",
        ]
        keep = [c for c in wanted if c in df.columns]
        frames.append(df[keep].copy())

    if not frames:
        return pd.DataFrame(
            columns=[
                "RUN_DATETIME",
                "INTERVAL_DATETIME",
                "REGIONID",
                "RRP",
                "TOTALDEMAND",
                "AVAILABLEGENERATION",
            ]
        )

    out = pd.concat(frames, ignore_index=True)

    datetime_cols = [
        c
        for c in ("RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED")
        if c in out.columns
    ]
    for col in datetime_cols:
        out[col] = pd.to_datetime(out[col])

    dedupe_keys = [
        c
        for c in ("REGIONID", "RUN_DATETIME", "INTERVAL_DATETIME")
        if c in out.columns
    ]
    if dedupe_keys:
        out = out.drop_duplicates(dedupe_keys, keep="last")

    return out.sort_values(dedupe_keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FULL RAW LOAD
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_raw_cached(
    regions: tuple[str, ...],
    start: datetime,
    end: datetime,
    include_interconnectors: bool,
    include_p5min: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load source tables once per process for a given configuration."""
    prices = _fetch_region_table("DISPATCHPRICE", regions, start, end)
    region_summary = _fetch_region_table("DISPATCHREGIONSUM", regions, start, end)

    interconnectors = (
        _fetch_interconnectors(start, end)
        if include_interconnectors
        else None
    )

    p5min = (
        _fetch_p5min(regions, start, end)
        if include_p5min
        else None
    )

    return prices, region_summary, interconnectors, p5min


# ---------------------------------------------------------------------------
# SPLITTING
# ---------------------------------------------------------------------------

def _cutoff(start: datetime, end: datetime, test_fraction: float) -> datetime:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    return start + (end - start) * (1 - test_fraction)


def _split_by_datetime(
    df: pd.DataFrame | None,
    date_col: str,
    cutoff: datetime,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if df is None:
        return None, None

    if df.empty:
        return df.copy(), df.copy()

    if date_col not in df.columns:
        raise ValueError(f"Expected datetime column {date_col!r} in source table")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    train = work[work[date_col] < cutoff].reset_index(drop=True)
    test = work[work[date_col] >= cutoff].reset_index(drop=True)
    return train, test


def getTrainTestData(
    regions: list[str] = DEFAULT_REGIONS,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    test_fraction: float = TEST_FRACTION,
    include_interconnectors: bool = DEFAULT_INCLUDE_INTERCONNECTORS,
    include_p5min: bool = DEFAULT_INCLUDE_P5MIN,
) -> tuple[Data, Data]:
    """Fetch once, then return chronological raw train/test splits."""
    regions_tuple = tuple(str(r) for r in regions)
    cutoff = _cutoff(start, end, test_fraction)

    prices, region_summary, interconnectors, p5min = _load_raw_cached(
        regions_tuple,
        start,
        end,
        include_interconnectors,
        include_p5min,
    )

    train_prices, test_prices = _split_by_datetime(
        prices,
        "SETTLEMENTDATE",
        cutoff,
    )
    train_region, test_region = _split_by_datetime(
        region_summary,
        "SETTLEMENTDATE",
        cutoff,
    )
    train_ic, test_ic = _split_by_datetime(
        interconnectors,
        "SETTLEMENTDATE",
        cutoff,
    )

    # Forecast data is split by RUN_DATETIME (information time), not by the
    # future interval being forecast. This is essential for leakage-safe tests.
    train_p5, test_p5 = _split_by_datetime(
        p5min,
        "RUN_DATETIME",
        cutoff,
    )

    train = Data(
        prices=train_prices,
        region_summary=train_region,
        interconnectors=train_ic,
        p5min=train_p5,
        regions=list(regions_tuple),
        start=start,
        end=cutoff,
    )

    test = Data(
        prices=test_prices,
        region_summary=test_region,
        interconnectors=test_ic,
        p5min=test_p5,
        regions=list(regions_tuple),
        start=cutoff,
        end=end,
    )

    return train, test


def getTrainingData(
    regions: list[str] = DEFAULT_REGIONS,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    test_fraction: float = TEST_FRACTION,
    include_interconnectors: bool = DEFAULT_INCLUDE_INTERCONNECTORS,
    include_p5min: bool = DEFAULT_INCLUDE_P5MIN,
) -> Data:
    train, _ = getTrainTestData(
        regions=regions,
        start=start,
        end=end,
        test_fraction=test_fraction,
        include_interconnectors=include_interconnectors,
        include_p5min=include_p5min,
    )
    return train


def getTestData(
    regions: list[str] = DEFAULT_REGIONS,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    test_fraction: float = TEST_FRACTION,
    include_interconnectors: bool = DEFAULT_INCLUDE_INTERCONNECTORS,
    include_p5min: bool = DEFAULT_INCLUDE_P5MIN,
) -> Data:
    _, test = getTrainTestData(
        regions=regions,
        start=start,
        end=end,
        test_fraction=test_fraction,
        include_interconnectors=include_interconnectors,
        include_p5min=include_p5min,
    )
    return test
