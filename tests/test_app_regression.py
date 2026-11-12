"""Synthetic regression tests for DataLens AI's deterministic pandas tools."""
import unittest
import pandas as pd

from app import filter_data, group_data


class AnalyticsRegressionTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Customer": ["A", "A", "B", "B", "C", "C", "D"],
            "Segment": ["Gold", "Gold", "Silver", "Silver", "Gold", "Bronze", "Gold"],
            "Region": ["North", "South", "North", "North", "South", "North", "South"],
            "Sales": [100, 250, 300, 50, 400, 125, 200],
        })

    def test_raw_maximum(self):
        self.assertEqual(self.df["Sales"].max(), 400)

    def test_grouped_sum(self):
        result = group_data(self.df, "Segment", "Sales", "sum")
        actual = dict(zip(result["Segment"], result["Sales"]))
        self.assertEqual(actual, {"Bronze": 125, "Gold": 950, "Silver": 350})

    def test_grouped_max(self):
        result = group_data(self.df, "Customer", "Sales", "max")
        actual = dict(zip(result["Customer"], result["Sales"]))
        self.assertEqual(actual, {"A": 250, "B": 300, "C": 400, "D": 200})

    def test_count(self):
        result = group_data(self.df, "Segment", "Sales", "count")
        actual = dict(zip(result["Segment"], result["Count"]))
        self.assertEqual(actual, {"Bronze": 1, "Gold": 4, "Silver": 2})

    def test_top_10(self):
        result = self.df.nlargest(10, "Sales")
        self.assertEqual(result.iloc[0]["Sales"], 400)
        self.assertEqual(len(result), 7)

    def test_multiple_filters(self):
        filtered = filter_data(self.df, "Segment", "==", "Gold")
        filtered = filter_data(filtered, "Region", "==", "South")
        self.assertEqual(filtered["Sales"].tolist(), [250, 400, 200])


if __name__ == "__main__":
    unittest.main()
