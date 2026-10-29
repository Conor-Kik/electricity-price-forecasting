from __future__ import annotations

# Canonical names expected by the preprocessing layer. The ingestion layer may
# preserve AEMO's uppercase names; canonicalise_columns() maps common AEMO names.
COLUMN_ALIASES = {
    "SETTLEMENTDATE": "settlement_date",
    "REGIONID": "region",
    "RRP": "rrp",
    "TOTALDEMAND": "totaldemand",
    "AVAILABLEGENERATION": "availablegeneration",
    "DEMANDFORECAST": "demandforecast",
    "DISPATCHABLEGENERATION": "dispatchablegeneration",
    "DISPATCHABLELOAD": "dispatchableload",
    "NETINTERCHANGE": "netinterchange",
    "RUN_DATETIME": "run_datetime",
    "INTERVAL_DATETIME": "forecast_time",
    "LASTCHANGED": "lastchanged",
    "MWFLOW": "mwflow",
    "IMPORTLIMIT": "importlimit",
    "EXPORTLIMIT": "exportlimit",
    "INTERCONNECTORID": "interconnectorid",
}

CORE_ACTUAL_COLUMNS = (
    "settlement_date",
    "region",
    "rrp",
    "totaldemand",
    "availablegeneration",
)

OPTIONAL_ACTUAL_COLUMNS = (
    "demandforecast",
    "dispatchablegeneration",
    "dispatchableload",
    "netinterchange",
)

P5MIN_KEYS = (
    "region",
    "run_datetime",
    "forecast_time",
)

P5MIN_VALUE_COLUMNS = (
    "rrp",
    "totaldemand",
    "availablegeneration",
)

# Features which are model inputs by default after engineering. Extra numeric
# source columns are not automatically included, which prevents feature creep.
BASE_MODEL_FEATURES = (
    "rrp",
    "totaldemand",
    "availablegeneration",
    "demandforecast",
    "dispatchablegeneration",
    "dispatchableload",
    "netinterchange",
    "reserve_margin_mw",
    "reserve_ratio",
)
