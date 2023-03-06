import unittest

import pandas as pd

from ananke import utils
from ananke.utils import get_repeated_df


class UtilsTestCase(unittest.TestCase):
    def test_percentile(self):
        percentile_func_no_name = utils.percentile(0.5)

        df = pd.Series(
            [
                0, 5, 10
            ]
        )

        assert df.quantile(0.5) == percentile_func_no_name(df)
        assert percentile_func_no_name.__name__ == 'percentile_50'
        percentile_func_name = utils.percentile(0.5, 'foo')
        assert percentile_func_name.__name__ == 'foo'

    def test_repeated_df(self):
        df = pd.DataFrame({
            'a': [0,1]
        })

        repeated_df = get_repeated_df(df_to_repeat=df, number_of_replications=2)

        self.assertTrue(
            pd.DataFrame({
                'a': [0,0,1,1]
            }).equals(repeated_df)
        )


if __name__ == '__main__':
    unittest.main()
