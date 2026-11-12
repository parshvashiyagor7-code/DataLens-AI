import unittest
import pandas as pd
from app import rank_numeric_columns

class Phase4SchemaTests(unittest.TestCase):
    def test_resolves_nonstandard_revenue_column(self):
        tables = {"orders": pd.DataFrame({"Net_Turnover": [10, 20], "Units": [1, 2]})}
        column, confidence = rank_numeric_columns("top products by revenue", tables)
        self.assertEqual(column, "Net_Turnover")
        self.assertGreater(confidence, 0)

    def test_resolves_score_from_schema(self):
        tables = {"students": pd.DataFrame({"Exam_Score": [70, 80], "Student_ID": [1, 2]})}
        column, _ = rank_numeric_columns("highest score", tables)
        self.assertEqual(column, "Exam_Score")

    def test_ambiguous_metric_does_not_guess(self):
        tables = {"sales": pd.DataFrame({"Sales_Value": [1, 2], "Sales_Amount": [3, 4]})}
        column, _ = rank_numeric_columns("top items by sales", tables)
        self.assertIsNone(column)

    def test_no_numeric_match_returns_none(self):
        tables = {"people": pd.DataFrame({"Name": ["A", "B"]})}
        column, confidence = rank_numeric_columns("highest revenue", tables)
        self.assertIsNone(column)
        self.assertEqual(confidence, 0.0)

if __name__ == "__main__":
    unittest.main()
