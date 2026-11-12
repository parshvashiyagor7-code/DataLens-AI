import unittest
import pandas as pd

from app import profile_data, resolve_column_name, load_user_dataset


class Phase2SchemaTests(unittest.TestCase):
    def test_profile_includes_sample_values(self):
        df = pd.DataFrame({"Customer Segment": ["Gold", "Silver", "Gold", None]})
        profile = profile_data(df)
        self.assertEqual(profile["sample_values"]["Customer Segment"], ["Gold", "Silver"])

    def test_string_and_category_columns_are_categorical(self):
        df = pd.DataFrame({
            "as_string": pd.Series(["a", "b"], dtype="string"),
            "as_category": pd.Series(["x", "y"], dtype="category"),
        })
        profile = profile_data(df)
        self.assertIn("as_string", profile["categorical_columns"])
        self.assertIn("as_category", profile["categorical_columns"])

    def test_exact_case_insensitive_column_match(self):
        self.assertEqual(resolve_column_name("sales_amount", ["Sales_Amount"]), "Sales_Amount")

    def test_normalized_punctuation_and_spacing_match(self):
        self.assertEqual(resolve_column_name("customer segment", ["Customer_Segment"]), "Customer_Segment")

    def test_conservative_fuzzy_match(self):
        self.assertEqual(resolve_column_name("Sales Amont", ["Sales_Amount", "Order_Date"]), "Sales_Amount")

    def test_ambiguous_fuzzy_match_returns_none(self):
        self.assertIsNone(resolve_column_name("Sales", ["Sales_Amount", "Sales_Value"]))

    def test_no_unsafe_match(self):
        self.assertIsNone(resolve_column_name("completely unrelated", ["Sales_Amount", "Customer_ID"]))


if __name__ == "__main__":
    unittest.main()
