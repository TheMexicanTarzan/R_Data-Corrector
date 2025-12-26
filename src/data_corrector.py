import sys
import os
from pathlib import Path
from typing import Callable
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

import polars
from multiprocessing import Pool

from src.input_handlers.csv_reader import read_csv_files_to_polars
from src.modules.errors.sanity_check.sanity_check import (
    fill_negatives_fundamentals,
    fill_negatives_market,
    zero_wipeout,
    mkt_cap_scale_error,
    ohlc_integrity
)
from src.features.lazy_parallelization import parallel_process_tickers


current_dir = Path.cwd()
data_directory = current_dir / ".." / "Input" / "Data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataframe_dict = read_csv_files_to_polars(data_directory, max_files= 500)


fundamental_negatives_columns = [
    "fbs_cash_and_cash_equivalents",
    "fbs_cash_and_shortterm_investments",
    "fbs_net_inventory",
    "fbs_net_property_plant_and_equipment",
    "fbs_goodwill",
    "fbs_net_intangible_assets_excluding_goodwill",
    "fbs_net_intangible_assets_including_goodwill",
    "fbs_assets",
    "fbs_current_assets",
    "fbs_noncurrent_assets",
    "fis_weighted_average_basic_shares_outstanding",
    "fis_weighted_average_diluted_shares_outstanding",
    "fis_revenues"
]

market_negatives_columns = columns = [
    "m_open",
    "m_high",
    "m_low",
    "m_close",
    "m_volume",
    "m_vwap",
    "m_open_split_adjusted",
    "m_high_split_adjusted",
    "m_low_split_adjusted",
    "m_close_split_adjusted",
    "m_volume_split_adjusted",
    "m_vwap_split_adjusted",
    "m_open_dividend_and_split_adjusted",
    "m_high_dividend_and_split_adjusted",
    "m_low_dividend_and_split_adjusted",
    "m_close_dividend_and_split_adjusted",
    "m_volume_dividend_and_split_adjusted",
    "m_vwap_dividend_and_split_adjusted"
]

zero_wipeout_columns = [
    "fis_weighted_average_basic_shares_outstanding",
    "fis_weighted_average_diluted_shares_outstanding"
]

shares_outstanding_10x_columns = [
    "fis_weighted_average_basic_shares_outstanding",
    "c_market_cap"
]

# dataframe_dict_clean_negatives_fundamentals, negative_fundamentals_logs = parallel_process_tickers(
#         data_dict = dataframe_dict,
#         columns = fundamental_negatives_columns,
#         function = fill_negatives_fundamentals
#     )
#
#
# dataframe_dict_clean_zero_wipeout, logs = parallel_process_tickers(
#         data_dict = dataframe_dict,
#         columns = zero_wipeout_columns,
#         function = zero_wipeout
#     )

# dataframe_dict_clean_10x_shares_outstanding, logs = parallel_process_tickers(
#         data_dict = dataframe_dict,
#         columns = shares_outstanding_10x_columns,
#         function = mkt_cap_scale_error
#     )

dataframe_dict_clean_zero_wipeout, logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = market_negatives_columns,
        function = fill_negatives_market,
    )

print(dataframe_dict_clean_zero_wipeout)