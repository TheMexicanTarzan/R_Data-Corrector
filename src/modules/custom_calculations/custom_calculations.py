"""
Easily add your own custom feature calculation function.

To add a custom calculation function, you need to have this file the Config folder under the
project's root directory (and not the templates directory!), and add your functions there.

Each function needs to start with c_ as a prefix, and the rest of the name can be anything as
long as it's a valid Python function name.

Each function declares as arguments the names of each column it needs as input, which
are provided to it as our custom DataColumn objects that act as pyarrow.Array wrappers
but with neat features like:
- operator overloading (so you can directly perform arithmetic operations between columns,
like in pandas)
- automatically casting any operations involving NaN or null elements as null, as we
consider any null a missing value

Each function needs to return an iterable supported by pyarrow.array(), of the same length
(preferably another DataColumn, a pyarrow.Array, a pandas.Series or a 1D numpy.ndarray).
The result will automatically be wrapped in a DataColumn for any successive functions that
use that as input. Yes, you can absolutely chain together functions, and are encouraged to
do so!

Once you've added your function to the file, you need to add its name to the Output_Columns
sheet of the parameters_datacurator.xlsx file. Don't forget that your function name needs to
start with c_ as a prefix!

See more examples of how easy it is to program custom functions by checking out the file
src/kaxanuk/data_curator/features/calculations.py
"""

# Here you'll find helper functions for calculating more complicated features:
from kaxanuk.data_curator.features import helpers

import numpy as np
import pandas as pd


def c_test(m_open, m_close):
    """
    Example features calculation function.

    Receives the market open and market close columns, and returns a column with their difference.

    For this function to generate an output column, you need to:
    1. Place it in the Config/custom_calculations.py file (if it doesn't exist you can copy
    this file there).
    2. Add c_test to the Output_Columns sheet in the Config/parameters_datacurator.xlsx file.

    Parameters
    ----------
    m_open : kaxanuk.data_curator.DataColumn
    m_close : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
    """
    # we're just doing a subtraction here, but you can implement any logic
    # just remember to return the same number of rows in a single column!
    return m_close - m_open


# =============================================================================
# OHLC INTEGRITY CORRECTIONS
# Translated from: sanity_check/market.py - ohlc_integrity()
# =============================================================================

def c_fixed_high(m_open, m_high, m_low, m_close):
    """
    Corrects m_high to ensure it is the maximum of all OHLC values.

    Also handles negative values by forward-filling before integrity check.
    High must be >= max(Open, Close, Low). If violated, High is set to the maximum.

    Parameters
    ----------
    m_open : kaxanuk.data_curator.DataColumn
    m_high : kaxanuk.data_curator.DataColumn
    m_low : kaxanuk.data_curator.DataColumn
    m_close : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected high values where High >= max(Open, Low, Close).
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close, dtype=np.float64))

    ohlc_max = np.fmax(np.fmax(np.fmax(open_arr, high_arr), low_arr), close_arr)

    result = np.where(high_arr < ohlc_max, ohlc_max, high_arr)

    return result


def c_fixed_low(m_open, m_high, m_low, m_close):
    """
    Corrects m_low to ensure it is the minimum of all OHLC values.

    Also handles negative values by forward-filling before integrity check.
    Low must be <= min(Open, Close, High). If violated, Low is set to the minimum.

    Parameters
    ----------
    m_open : kaxanuk.data_curator.DataColumn
    m_high : kaxanuk.data_curator.DataColumn
    m_low : kaxanuk.data_curator.DataColumn
    m_close : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected low values where Low <= min(Open, High, Close).
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close, dtype=np.float64))

    ohlc_min = np.fmin(np.fmin(np.fmin(open_arr, high_arr), low_arr), close_arr)

    result = np.where(low_arr > ohlc_min, ohlc_min, low_arr)

    return result


def c_fixed_vwap(m_open, m_high, m_low, m_close, m_vwap):
    """
    Corrects m_vwap to ensure it falls within the [Low, High] range.

    Also handles negative values by forward-filling before integrity check.
    VWAP must be within [Low, High]. If violated, VWAP is set to the OHLC centroid (O+H+L+C)/4.
    Uses corrected High (max of OHLC) and Low (min of OHLC) for range validation.

    Parameters
    ----------
    m_open : kaxanuk.data_curator.DataColumn
    m_high : kaxanuk.data_curator.DataColumn
    m_low : kaxanuk.data_curator.DataColumn
    m_close : kaxanuk.data_curator.DataColumn
    m_vwap : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected VWAP values within [Low, High] range.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close, dtype=np.float64))
    vwap_arr = _forward_fill_negatives(np.asarray(m_vwap, dtype=np.float64))

    corrected_high = np.fmax(np.fmax(np.fmax(open_arr, high_arr), low_arr), close_arr)
    corrected_low = np.fmin(np.fmin(np.fmin(open_arr, high_arr), low_arr), close_arr)

    ohlc_centroid = (open_arr + corrected_high + corrected_low + close_arr) / 4.0

    is_outside_range = (vwap_arr < corrected_low) | (vwap_arr > corrected_high)
    result = np.where(is_outside_range, ohlc_centroid, vwap_arr)

    return result


def c_fixed_high_split_adjusted(
    m_open_split_adjusted,
    m_high_split_adjusted,
    m_low_split_adjusted,
    m_close_split_adjusted
):
    """
    Corrects m_high_split_adjusted to ensure it is the maximum of all split-adjusted OHLC values.

    Also handles negative values by forward-filling before integrity check.

    Parameters
    ----------
    m_open_split_adjusted : kaxanuk.data_curator.DataColumn
    m_high_split_adjusted : kaxanuk.data_curator.DataColumn
    m_low_split_adjusted : kaxanuk.data_curator.DataColumn
    m_close_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected split-adjusted high values.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open_split_adjusted, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high_split_adjusted, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low_split_adjusted, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close_split_adjusted, dtype=np.float64))

    ohlc_max = np.fmax(np.fmax(np.fmax(open_arr, high_arr), low_arr), close_arr)

    result = np.where(high_arr < ohlc_max, ohlc_max, high_arr)

    return result


def c_fixed_low_split_adjusted(
    m_open_split_adjusted,
    m_high_split_adjusted,
    m_low_split_adjusted,
    m_close_split_adjusted
):
    """
    Corrects m_low_split_adjusted to ensure it is the minimum of all split-adjusted OHLC values.

    Also handles negative values by forward-filling before integrity check.

    Parameters
    ----------
    m_open_split_adjusted : kaxanuk.data_curator.DataColumn
    m_high_split_adjusted : kaxanuk.data_curator.DataColumn
    m_low_split_adjusted : kaxanuk.data_curator.DataColumn
    m_close_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected split-adjusted low values.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open_split_adjusted, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high_split_adjusted, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low_split_adjusted, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close_split_adjusted, dtype=np.float64))

    ohlc_min = np.fmin(np.fmin(np.fmin(open_arr, high_arr), low_arr), close_arr)

    result = np.where(low_arr > ohlc_min, ohlc_min, low_arr)

    return result


def c_fixed_vwap_split_adjusted(
    m_open_split_adjusted,
    m_high_split_adjusted,
    m_low_split_adjusted,
    m_close_split_adjusted,
    m_vwap_split_adjusted
):
    """
    Corrects m_vwap_split_adjusted to ensure it falls within the [Low, High] range.

    Also handles negative values by forward-filling before integrity check.
    Uses corrected High (max of OHLC) and Low (min of OHLC) for range validation.

    Parameters
    ----------
    m_open_split_adjusted : kaxanuk.data_curator.DataColumn
    m_high_split_adjusted : kaxanuk.data_curator.DataColumn
    m_low_split_adjusted : kaxanuk.data_curator.DataColumn
    m_close_split_adjusted : kaxanuk.data_curator.DataColumn
    m_vwap_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected split-adjusted VWAP values.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open_split_adjusted, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high_split_adjusted, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low_split_adjusted, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close_split_adjusted, dtype=np.float64))
    vwap_arr = _forward_fill_negatives(np.asarray(m_vwap_split_adjusted, dtype=np.float64))

    corrected_high = np.fmax(np.fmax(np.fmax(open_arr, high_arr), low_arr), close_arr)
    corrected_low = np.fmin(np.fmin(np.fmin(open_arr, high_arr), low_arr), close_arr)

    ohlc_centroid = (open_arr + corrected_high + corrected_low + close_arr) / 4.0

    is_outside_range = (vwap_arr < corrected_low) | (vwap_arr > corrected_high)
    result = np.where(is_outside_range, ohlc_centroid, vwap_arr)

    return result


def c_fixed_high_dividend_and_split_adjusted(
    m_open_dividend_and_split_adjusted,
    m_high_dividend_and_split_adjusted,
    m_low_dividend_and_split_adjusted,
    m_close_dividend_and_split_adjusted
):
    """
    Corrects m_high_dividend_and_split_adjusted to ensure it is the maximum.

    Also handles negative values by forward-filling before integrity check.

    Parameters
    ----------
    m_open_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_high_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_low_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_close_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected dividend-and-split-adjusted high values.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open_dividend_and_split_adjusted, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high_dividend_and_split_adjusted, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low_dividend_and_split_adjusted, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close_dividend_and_split_adjusted, dtype=np.float64))

    ohlc_max = np.fmax(np.fmax(np.fmax(open_arr, high_arr), low_arr), close_arr)

    result = np.where(high_arr < ohlc_max, ohlc_max, high_arr)

    return result


def c_fixed_low_dividend_and_split_adjusted(
    m_open_dividend_and_split_adjusted,
    m_high_dividend_and_split_adjusted,
    m_low_dividend_and_split_adjusted,
    m_close_dividend_and_split_adjusted
):
    """
    Corrects m_low_dividend_and_split_adjusted to ensure it is the minimum.

    Also handles negative values by forward-filling before integrity check.

    Parameters
    ----------
    m_open_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_high_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_low_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_close_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected dividend-and-split-adjusted low values.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open_dividend_and_split_adjusted, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high_dividend_and_split_adjusted, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low_dividend_and_split_adjusted, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close_dividend_and_split_adjusted, dtype=np.float64))

    ohlc_min = np.fmin(np.fmin(np.fmin(open_arr, high_arr), low_arr), close_arr)

    result = np.where(low_arr > ohlc_min, ohlc_min, low_arr)

    return result


def c_fixed_vwap_dividend_and_split_adjusted(
    m_open_dividend_and_split_adjusted,
    m_high_dividend_and_split_adjusted,
    m_low_dividend_and_split_adjusted,
    m_close_dividend_and_split_adjusted,
    m_vwap_dividend_and_split_adjusted
):
    """
    Corrects m_vwap_dividend_and_split_adjusted to ensure it falls within [Low, High].

    Also handles negative values by forward-filling before integrity check.
    Uses corrected High (max of OHLC) and Low (min of OHLC) for range validation.

    Parameters
    ----------
    m_open_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_high_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_low_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_close_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn
    m_vwap_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected dividend-and-split-adjusted VWAP values.
    """
    open_arr = _forward_fill_negatives(np.asarray(m_open_dividend_and_split_adjusted, dtype=np.float64))
    high_arr = _forward_fill_negatives(np.asarray(m_high_dividend_and_split_adjusted, dtype=np.float64))
    low_arr = _forward_fill_negatives(np.asarray(m_low_dividend_and_split_adjusted, dtype=np.float64))
    close_arr = _forward_fill_negatives(np.asarray(m_close_dividend_and_split_adjusted, dtype=np.float64))
    vwap_arr = _forward_fill_negatives(np.asarray(m_vwap_dividend_and_split_adjusted, dtype=np.float64))

    corrected_high = np.fmax(np.fmax(np.fmax(open_arr, high_arr), low_arr), close_arr)
    corrected_low = np.fmin(np.fmin(np.fmin(open_arr, high_arr), low_arr), close_arr)

    ohlc_centroid = (open_arr + corrected_high + corrected_low + close_arr) / 4.0

    is_outside_range = (vwap_arr < corrected_low) | (vwap_arr > corrected_high)
    result = np.where(is_outside_range, ohlc_centroid, vwap_arr)

    return result


# =============================================================================
# NEGATIVE VALUE CORRECTIONS
# Translated from: sanity_check/market.py - fill_negatives_market()
# and sanity_check/fundamental.py - fill_negatives_fundamentals()
# =============================================================================

def _forward_fill_negatives(arr):
    """
    Helper function to replace negative values with forward-filled last valid value.

    Uses fully vectorized pandas operations for performance.
    """
    series = pd.Series(arr)
    series = series.where(series >= 0, other=np.nan)
    result = series.ffill().to_numpy()
    return result


def c_fixed_open(m_open):
    """
    Corrects negative values in m_open by forward-filling with the last valid value.

    Translated from fill_negatives_market() in sanity_check/market.py.
    Uses forward fill as a simplified vectorized approach (no cubic spline for performance).

    Parameters
    ----------
    m_open : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected open values with negatives replaced.
    """
    arr = np.asarray(m_open, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_close(m_close):
    """
    Corrects negative values in m_close by forward-filling with the last valid value.

    Parameters
    ----------
    m_close : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected close values with negatives replaced.
    """
    arr = np.asarray(m_close, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_volume(m_volume):
    """
    Corrects negative values in m_volume by forward-filling with the last valid value.

    Parameters
    ----------
    m_volume : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected volume values with negatives replaced.
    """
    arr = np.asarray(m_volume, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_open_split_adjusted(m_open_split_adjusted):
    """
    Corrects negative values in m_open_split_adjusted by forward-filling.

    Parameters
    ----------
    m_open_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected split-adjusted open values with negatives replaced.
    """
    arr = np.asarray(m_open_split_adjusted, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_close_split_adjusted(m_close_split_adjusted):
    """
    Corrects negative values in m_close_split_adjusted by forward-filling.

    Parameters
    ----------
    m_close_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected split-adjusted close values with negatives replaced.
    """
    arr = np.asarray(m_close_split_adjusted, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_volume_split_adjusted(m_volume_split_adjusted):
    """
    Corrects negative values in m_volume_split_adjusted by forward-filling.

    Parameters
    ----------
    m_volume_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected split-adjusted volume values with negatives replaced.
    """
    arr = np.asarray(m_volume_split_adjusted, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_open_dividend_and_split_adjusted(m_open_dividend_and_split_adjusted):
    """
    Corrects negative values in m_open_dividend_and_split_adjusted by forward-filling.

    Parameters
    ----------
    m_open_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected dividend-and-split-adjusted open values with negatives replaced.
    """
    arr = np.asarray(m_open_dividend_and_split_adjusted, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_close_dividend_and_split_adjusted(m_close_dividend_and_split_adjusted):
    """
    Corrects negative values in m_close_dividend_and_split_adjusted by forward-filling.

    Parameters
    ----------
    m_close_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected dividend-and-split-adjusted close values with negatives replaced.
    """
    arr = np.asarray(m_close_dividend_and_split_adjusted, dtype=np.float64)
    return _forward_fill_negatives(arr)


def c_fixed_volume_dividend_and_split_adjusted(m_volume_dividend_and_split_adjusted):
    """
    Corrects negative values in m_volume_dividend_and_split_adjusted by forward-filling.

    Parameters
    ----------
    m_volume_dividend_and_split_adjusted : kaxanuk.data_curator.DataColumn

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected dividend-and-split-adjusted volume values with negatives replaced.
    """
    arr = np.asarray(m_volume_dividend_and_split_adjusted, dtype=np.float64)
    return _forward_fill_negatives(arr)


# =============================================================================
# ACCOUNTING IDENTITY CORRECTIONS (Hard Filters)
# Translated from: sanity_check/fundamental.py - validate_financial_equivalencies()
# =============================================================================

def c_fixed_current_assets(fbs_assets, fbs_current_assets, fbs_noncurrent_assets):
    """
    Corrects fbs_current_assets to satisfy: Assets = Current Assets + Noncurrent Assets.

    Uses proportional scaling when components don't sum to total.
    Edge case: If both components sum to 0 but total != 0, current_assets stays at 0.

    Parameters
    ----------
    fbs_assets : kaxanuk.data_curator.DataColumn
        Total assets.
    fbs_current_assets : kaxanuk.data_curator.DataColumn
        Current assets.
    fbs_noncurrent_assets : kaxanuk.data_curator.DataColumn
        Noncurrent assets.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected current assets values.
    """
    total_arr = np.asarray(fbs_assets, dtype=np.float64)
    current_arr = np.asarray(fbs_current_assets, dtype=np.float64)
    noncurrent_arr = np.asarray(fbs_noncurrent_assets, dtype=np.float64)

    tolerance = 0.05
    component_sum = current_arr + noncurrent_arr

    difference = np.abs(total_arr - component_sum)
    threshold = np.abs(total_arr) * tolerance

    is_violation = difference > threshold

    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(component_sum != 0, total_arr / component_sum, 1.0)

    new_current = np.where(
        is_violation,
        np.where(component_sum != 0, current_arr * scaling_factor, 0.0),
        current_arr
    )

    return new_current


def c_fixed_noncurrent_assets(fbs_assets, fbs_current_assets, fbs_noncurrent_assets):
    """
    Corrects fbs_noncurrent_assets to satisfy: Assets = Current Assets + Noncurrent Assets.

    Uses proportional scaling when components don't sum to total.
    Edge case: If both components sum to 0 but total != 0, noncurrent_assets gets the total.

    Parameters
    ----------
    fbs_assets : kaxanuk.data_curator.DataColumn
        Total assets.
    fbs_current_assets : kaxanuk.data_curator.DataColumn
        Current assets.
    fbs_noncurrent_assets : kaxanuk.data_curator.DataColumn
        Noncurrent assets.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected noncurrent assets values.
    """
    total_arr = np.asarray(fbs_assets, dtype=np.float64)
    current_arr = np.asarray(fbs_current_assets, dtype=np.float64)
    noncurrent_arr = np.asarray(fbs_noncurrent_assets, dtype=np.float64)

    tolerance = 0.05
    component_sum = current_arr + noncurrent_arr

    difference = np.abs(total_arr - component_sum)
    threshold = np.abs(total_arr) * tolerance

    is_violation = difference > threshold

    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(component_sum != 0, total_arr / component_sum, 1.0)

    new_noncurrent = np.where(
        is_violation,
        np.where(component_sum != 0, noncurrent_arr * scaling_factor, total_arr),
        noncurrent_arr
    )

    return new_noncurrent


def c_fixed_current_liabilities(fbs_liabilities, fbs_current_liabilities, fbs_noncurrent_liabilities):
    """
    Corrects fbs_current_liabilities to satisfy: Liabilities = Current + Noncurrent.

    Uses proportional scaling when components don't sum to total.

    Parameters
    ----------
    fbs_liabilities : kaxanuk.data_curator.DataColumn
        Total liabilities.
    fbs_current_liabilities : kaxanuk.data_curator.DataColumn
        Current liabilities.
    fbs_noncurrent_liabilities : kaxanuk.data_curator.DataColumn
        Noncurrent liabilities.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected current liabilities values.
    """
    total_arr = np.asarray(fbs_liabilities, dtype=np.float64)
    current_arr = np.asarray(fbs_current_liabilities, dtype=np.float64)
    noncurrent_arr = np.asarray(fbs_noncurrent_liabilities, dtype=np.float64)

    tolerance = 0.05
    component_sum = current_arr + noncurrent_arr

    difference = np.abs(total_arr - component_sum)
    threshold = np.abs(total_arr) * tolerance

    is_violation = difference > threshold

    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(component_sum != 0, total_arr / component_sum, 1.0)

    new_current = np.where(
        is_violation,
        np.where(component_sum != 0, current_arr * scaling_factor, 0.0),
        current_arr
    )

    return new_current


def c_fixed_noncurrent_liabilities(fbs_liabilities, fbs_current_liabilities, fbs_noncurrent_liabilities):
    """
    Corrects fbs_noncurrent_liabilities to satisfy: Liabilities = Current + Noncurrent.

    Uses proportional scaling when components don't sum to total.
    Edge case: If both components sum to 0 but total != 0, noncurrent gets the total.

    Parameters
    ----------
    fbs_liabilities : kaxanuk.data_curator.DataColumn
        Total liabilities.
    fbs_current_liabilities : kaxanuk.data_curator.DataColumn
        Current liabilities.
    fbs_noncurrent_liabilities : kaxanuk.data_curator.DataColumn
        Noncurrent liabilities.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected noncurrent liabilities values.
    """
    total_arr = np.asarray(fbs_liabilities, dtype=np.float64)
    current_arr = np.asarray(fbs_current_liabilities, dtype=np.float64)
    noncurrent_arr = np.asarray(fbs_noncurrent_liabilities, dtype=np.float64)

    tolerance = 0.05
    component_sum = current_arr + noncurrent_arr

    difference = np.abs(total_arr - component_sum)
    threshold = np.abs(total_arr) * tolerance

    is_violation = difference > threshold

    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(component_sum != 0, total_arr / component_sum, 1.0)

    new_noncurrent = np.where(
        is_violation,
        np.where(component_sum != 0, noncurrent_arr * scaling_factor, total_arr),
        noncurrent_arr
    )

    return new_noncurrent


# =============================================================================
# ZERO WIPEOUT CORRECTIONS
# Translated from: sanity_check/fundamental.py - zero_wipeout()
# =============================================================================

def c_fixed_basic_shares(fis_weighted_average_basic_shares_outstanding, m_volume):
    """
    Corrects zero values in basic shares outstanding when volume > 0.

    When shares outstanding is 0 but volume is positive, the 0 is suspicious
    and is replaced with forward-filled last valid value.

    Parameters
    ----------
    fis_weighted_average_basic_shares_outstanding : kaxanuk.data_curator.DataColumn
        Basic shares outstanding values.
    m_volume : kaxanuk.data_curator.DataColumn
        Trading volume.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected basic shares outstanding values.
    """
    shares_arr = np.asarray(fis_weighted_average_basic_shares_outstanding, dtype=np.float64)
    volume_arr = np.asarray(m_volume, dtype=np.float64)

    suspicious_zeros = (shares_arr == 0) & (volume_arr > 0)

    series = pd.Series(shares_arr)
    series = series.where(~suspicious_zeros, other=np.nan)
    result = series.ffill().to_numpy()

    return result


def c_fixed_diluted_shares(fis_weighted_average_diluted_shares_outstanding, m_volume):
    """
    Corrects zero values in diluted shares outstanding when volume > 0.

    When shares outstanding is 0 but volume is positive, the 0 is suspicious
    and is replaced with forward-filled last valid value.

    Parameters
    ----------
    fis_weighted_average_diluted_shares_outstanding : kaxanuk.data_curator.DataColumn
        Diluted shares outstanding values.
    m_volume : kaxanuk.data_curator.DataColumn
        Trading volume.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected diluted shares outstanding values.
    """
    shares_arr = np.asarray(fis_weighted_average_diluted_shares_outstanding, dtype=np.float64)
    volume_arr = np.asarray(m_volume, dtype=np.float64)

    suspicious_zeros = (shares_arr == 0) & (volume_arr > 0)

    series = pd.Series(shares_arr)
    series = series.where(~suspicious_zeros, other=np.nan)
    result = series.ffill().to_numpy()

    return result


# =============================================================================
# SCALE ERROR CORRECTIONS (10x JUMPS)
# Translated from: sanity_check/fundamental.py - mkt_cap_scale_error()
# =============================================================================

def c_fixed_diluted_shares_scale(fis_weighted_average_diluted_shares_outstanding):
    """
    Corrects 10x or greater jumps in diluted shares outstanding.

    When shares outstanding jumps by 10x or more compared to the previous value,
    it's likely a data error and is replaced with forward-filled last valid value.

    Parameters
    ----------
    fis_weighted_average_diluted_shares_outstanding : kaxanuk.data_curator.DataColumn
        Diluted shares outstanding values.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected diluted shares outstanding values without 10x jumps.
    """
    shares_arr = np.asarray(fis_weighted_average_diluted_shares_outstanding, dtype=np.float64)

    prev_shares = np.roll(shares_arr, 1)
    prev_shares[0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        is_10x_jump = (shares_arr >= prev_shares * 10) & np.isfinite(prev_shares) & (prev_shares > 0)

    series = pd.Series(shares_arr)
    series = series.where(~is_10x_jump, other=np.nan)
    result = series.ffill().to_numpy()

    return result


def c_fixed_basic_shares_scale(fis_weighted_average_basic_shares_outstanding):
    """
    Corrects 10x or greater jumps in basic shares outstanding.

    When shares outstanding jumps by 10x or more compared to the previous value,
    it's likely a data error and is replaced with forward-filled last valid value.

    Parameters
    ----------
    fis_weighted_average_basic_shares_outstanding : kaxanuk.data_curator.DataColumn
        Basic shares outstanding values.

    Returns
    -------
    kaxanuk.data_curator.DataColumn
        Corrected basic shares outstanding values without 10x jumps.
    """
    shares_arr = np.asarray(fis_weighted_average_basic_shares_outstanding, dtype=np.float64)

    prev_shares = np.roll(shares_arr, 1)
    prev_shares[0] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        is_10x_jump = (shares_arr >= prev_shares * 10) & np.isfinite(prev_shares) & (prev_shares > 0)

    series = pd.Series(shares_arr)
    series = series.where(~is_10x_jump, other=np.nan)
    result = series.ffill().to_numpy()

    return result


# =============================================================================
# DATA WARNING FLAG
# Translated from: sanity_check/fundamental.py - validate_financial_equivalencies()
# =============================================================================

def c_data_warning(
    fbs_stockholder_equity,
    fbs_common_stock_value,
    fbs_additional_paid_in_capital,
    fbs_retained_earnings,
    fbs_other_stockholder_equity,
    fbs_assets,
    fbs_liabilities,
    fbs_noncontrolling_interest
):
    """
    Generates a boolean data warning flag for soft filter violations.

    Returns True if any of the following soft filter checks fail:
    1. Stockholder Equity != sum of equity components (within 5% tolerance)
    2. Assets != Liabilities + Stockholder Equity + NCI (accounting equation)

    Parameters
    ----------
    fbs_stockholder_equity : kaxanuk.data_curator.DataColumn
    fbs_common_stock_value : kaxanuk.data_curator.DataColumn
    fbs_additional_paid_in_capital : kaxanuk.data_curator.DataColumn
    fbs_retained_earnings : kaxanuk.data_curator.DataColumn
    fbs_other_stockholder_equity : kaxanuk.data_curator.DataColumn
    fbs_assets : kaxanuk.data_curator.DataColumn
    fbs_liabilities : kaxanuk.data_curator.DataColumn
    fbs_noncontrolling_interest : kaxanuk.data_curator.DataColumn

    Returns
    -------
    numpy.ndarray
        Boolean array indicating rows with data quality warnings.
    """
    equity_arr = np.asarray(fbs_stockholder_equity, dtype=np.float64)
    common_stock_arr = np.asarray(fbs_common_stock_value, dtype=np.float64)
    apic_arr = np.asarray(fbs_additional_paid_in_capital, dtype=np.float64)
    retained_arr = np.asarray(fbs_retained_earnings, dtype=np.float64)
    other_equity_arr = np.asarray(fbs_other_stockholder_equity, dtype=np.float64)
    assets_arr = np.asarray(fbs_assets, dtype=np.float64)
    liabilities_arr = np.asarray(fbs_liabilities, dtype=np.float64)
    nci_arr = np.asarray(fbs_noncontrolling_interest, dtype=np.float64)

    nci_arr = np.where(np.isnan(nci_arr), 0.0, nci_arr)

    tolerance = 0.05

    equity_components_sum = common_stock_arr + apic_arr + retained_arr + other_equity_arr
    equity_diff = np.abs(equity_arr - equity_components_sum)
    equity_threshold = np.abs(equity_arr) * tolerance
    equity_mismatch = equity_diff > equity_threshold

    total_claims = liabilities_arr + equity_arr + nci_arr
    bs_diff = np.abs(assets_arr - total_claims)
    bs_threshold = np.abs(assets_arr) * tolerance
    bs_mismatch = bs_diff > bs_threshold

    data_warning = equity_mismatch | bs_mismatch

    return data_warning
