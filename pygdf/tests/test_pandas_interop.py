import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def test_to_pandas():
    df = DataFrame()
    df['a'] = np.arange(10, dtype=np.int32)
    df['b'] = np.arange(10, 20, dtype=np.float64)

    pdf = df.to_pandas()

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])


def test_from_pandas():
    pdf = pd.DataFrame()
    pdf['a'] = np.arange(10, dtype=np.int32)
    pdf['b'] = np.arange(10, 20, dtype=np.float64)

    df = DataFrame.from_pandas(pdf)

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])


def test_from_pandas_ex1():
    pdf = pd.DataFrame({'a': [0, 1, 2, 3],
                        'b': [0.1, 0.2, None, 0.3]})
    print(pdf)
    df = DataFrame.from_pandas(pdf)
    print(df)

    assert tuple(df.columns) == tuple(pdf.columns)
    assert np.all(df['a'].to_array() == pdf['a'])
    matches = df['b'].to_array() == pdf['b']
    # the 3d element is False due to (nan == nan) == False
    assert np.all(matches == [True, True, False, True])
    assert np.isnan(df['b'].to_array()[2])
    assert np.isnan(pdf['b'][2])
