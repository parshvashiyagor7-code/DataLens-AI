import unittest
import pandas as pd

from app import validate_dataset_tables, build_schema_context
from analysis_service import validate_analysis_payload


class Phase5DatasetQualityTests(unittest.TestCase):
    def test_empty_dataset_is_flagged(self):
        issues = validate_dataset_tables({"empty": pd.DataFrame()})
        self.assertTrue(any("no data rows" in issue.lower() for issue in issues))
        self.assertTrue(any("no columns" in issue.lower() for issue in issues))

    def test_duplicate_columns_are_flagged_without_crashing(self):
        df = pd.DataFrame([[1, 2]], columns=["Sales", "Sales"])
        issues = validate_dataset_tables({"orders": df})
        self.assertTrue(any("duplicate column" in issue.lower() for issue in issues))
        schema = build_schema_context({"orders": df})
        self.assertEqual(len(schema["orders"]["columns"]), 2)

    def test_unnamed_and_all_missing_columns_are_flagged(self):
        df = pd.DataFrame({"Unnamed: 0": [1, 2], "Empty": [None, None]})
        issues = validate_dataset_tables({"data": df})
        self.assertTrue(any("unnamed" in issue.lower() for issue in issues))
        self.assertTrue(any("entirely empty" in issue.lower() for issue in issues))

    def test_schema_context_contains_dtype_missing_unique_and_samples(self):
        df = pd.DataFrame({"Region": ["North", "South", "North"], "Sales": [10, None, 30]})
        schema = build_schema_context({"orders": df})["orders"]
        self.assertEqual(schema["rows"], 3)
        region = next(col for col in schema["columns"] if col["name"] == "Region")
        self.assertEqual(region["unique"], 2)
        self.assertEqual(region["samples"], ["North", "South"])


class Phase5PayloadValidationTests(unittest.TestCase):
    def test_appends_unlisted_result_keys(self):
        payload = {"result": [{"Sales": 10, "Region": "North"}], "columns": ["Sales"]}
        normalized = validate_analysis_payload(payload)
        self.assertEqual(normalized["columns"], ["Sales", "Region"])
        self.assertEqual(normalized["row_count"], 1)
        self.assertEqual(normalized["chart_type"], "NONE")

    def test_rejects_non_dictionary_rows(self):
        with self.assertRaises(ValueError):
            validate_analysis_payload({"result": ["bad row"], "columns": []})

    def test_rejects_duplicate_result_columns(self):
        with self.assertRaises(ValueError):
            validate_analysis_payload({"result": [], "columns": ["Sales", "Sales"]})

    def test_rejects_non_list_result(self):
        with self.assertRaises(ValueError):
            validate_analysis_payload({"result": {}, "columns": []})


if __name__ == "__main__":
    unittest.main()
