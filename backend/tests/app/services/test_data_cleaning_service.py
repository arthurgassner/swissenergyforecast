import numpy as np
import pandas as pd
import pytest

from app.services import data_cleaning_service


def test__format():
    """Formatting a dateframe sets back its index by 24 and renames its columns."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # When
    formatted_df = data_cleaning_service._format(df=df)

    # Then
    assert len(formatted_df.columns) == 2  # 2 columns
    assert formatted_df.columns[0] == "24h_later_forecast" and formatted_df.columns[1] == "24h_later_load"
    assert (formatted_df.dtypes == "float64").all()  # correct dtype
    assert isinstance(formatted_df.index, pd.DatetimeIndex)
    # correct timezone
    assert formatted_df.index.dtype == "datetime64[ns, Europe/Zurich]"
    # -24h delay post-formatting
    assert (df.index - formatted_df.index == pd.Timedelta(24, "h")).all()


def test__force_1h_frequency():
    """Data with a missing hour ends up with a row of NaN inplace of that missing hour."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, 7890.0],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 21:00", tz="Europe/Zurich"),
                pd.Timestamp("20240101 22:00", tz="Europe/Zurich"),
                pd.Timestamp("20240102 00:00", tz="Europe/Zurich"),
            ]
        ),
    )

    # When
    enforced_frequency_df = data_cleaning_service._force_1h_frequency(df=df)

    # Then

    # data
    assert len(enforced_frequency_df.columns) == 2  # 2 columns
    assert enforced_frequency_df.columns[0] == "Forecasted Load" and enforced_frequency_df.columns[1] == "Actual Load"
    assert enforced_frequency_df.index.freq == "h"  # freq is now hourly
    assert len(enforced_frequency_df) == len(df) + 1  # a new row has been added
    assert enforced_frequency_df.iloc[2].isna().all()  # that row is the 3rd row, and filled with nan

    # index
    assert isinstance(enforced_frequency_df.index, pd.DatetimeIndex)
    assert enforced_frequency_df.index.is_monotonic_increasing
    assert enforced_frequency_df.index.is_unique

    # dtypes
    assert (enforced_frequency_df.dtypes == "float64").all()  # correct dtype
    assert enforced_frequency_df.index.dtype == "datetime64[ns, Europe/Zurich]"  # correct timezone


def test__enforce_data_quality():
    """Check that a df with no data quality issues goes through without changes."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    enforced_data_quality_df = data_cleaning_service._enforce_data_quality(df)

    # then
    assert enforced_data_quality_df.equals(df)


def test__enforce_data_quality__index_type():
    """Check that if not isinstance(df.index, pd.DatetimeIndex), a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DataFrame(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = data_cleaning_service._enforce_data_quality(df)


def test__enforce_data_quality__index_tz():
    """Check that if df.index.dtype != "datetime64[ns, Europe/Zurich]", a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Berlin"),
                pd.Timestamp("20240201 23:45", tz="Europe/Berlin"),
                pd.Timestamp("20240301 23:45", tz="Europe/Berlin"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = data_cleaning_service._enforce_data_quality(df)


def test__enforce_data_quality__two_columns():
    """Check that if len(df.columns) != 2, a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
            "Some Third Column": [0, 0, 0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = data_cleaning_service._enforce_data_quality(df)


def test__enforce_data_quality__column_names():
    """Check that if df.columns != ["Forecasted Load", "Actual Load"], a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Wrong column name": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = data_cleaning_service._enforce_data_quality(df)


def test__enforce_data_quality__dtypes():
    """Check that if df.dtypes.to_list() != ['float64', 'float64'], a ValueError is raised."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": ["a", "b", "c"],
            "Actual Load": ["d", "e", "f"],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when-then
    with pytest.raises(ValueError):
        df = data_cleaning_service._enforce_data_quality(df)


def test__enforce_data_quality__index_is_not_monotonic_increasing():
    """Check that a df with an index that is not monotonic increasing gets sorted."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    index_monotic_increasing_df = data_cleaning_service._enforce_data_quality(df)

    # then
    assert index_monotic_increasing_df.index.is_monotonic_increasing
    # reorder the rows, so that index is monotonic increasing
    expected_df = df.iloc[[0, 2, 1]]
    assert expected_df.equals(index_monotic_increasing_df)


def test__enforce_data_quality__index_is_not_unique():
    """Check that a df with an index that is not unique gets aggregated."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [100.0, np.nan, 200.0, np.nan],
            "Actual Load": [np.nan, 200.0, 300.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    index_unique_increasing_df = data_cleaning_service._enforce_data_quality(df)

    print()
    print(df)
    print(index_unique_increasing_df)

    # then
    assert index_unique_increasing_df.index.is_unique
    assert index_unique_increasing_df.index.is_monotonic_increasing
    assert len(index_unique_increasing_df) == df.index.nunique()
    np.testing.assert_array_equal(index_unique_increasing_df.iloc[0], [100.0, np.nan])  # 1st row is unchanged
    np.testing.assert_array_equal(index_unique_increasing_df.iloc[1], [200.0, 250.0])  # 2nd row is median
    np.testing.assert_array_equal(index_unique_increasing_df.iloc[2], [np.nan, np.nan])  # 3rd row is unchanged
