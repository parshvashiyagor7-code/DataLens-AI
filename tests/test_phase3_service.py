import json
import unittest

from analysis_service import parse_engine_output, validate_analysis_payload


class AnalysisServiceTests(unittest.TestCase):
    def test_parse_payload_with_chart_type(self):
        payload = {"status": "success", "answer": "Done", "result": [{"x": 1}],
                   "columns": ["x"], "chart_type": "BAR"}
        parsed = parse_engine_output("debug\\n__STREAMLIT_RESULT__" + json.dumps(payload))
        self.assertEqual(parsed["chart_type"], "BAR")
        self.assertEqual(parsed["result"], [{"x": 1}])

    def test_defaults_missing_optional_fields(self):
        parsed = parse_engine_output('__STREAMLIT_RESULT__{"status":"success"}')
        self.assertEqual(parsed["chart_type"], "NONE")
        self.assertEqual(parsed["result"], [])
        self.assertEqual(parsed["columns"], [])

    def test_missing_marker_rejected(self):
        with self.assertRaises(ValueError):
            parse_engine_output("no payload")

    def test_invalid_json_rejected(self):
        with self.assertRaises(ValueError):
            parse_engine_output("__STREAMLIT_RESULT__not-json")

    def test_payload_shape_validation(self):
        self.assertEqual(validate_analysis_payload({"result": [], "columns": []})["result"], [])
        with self.assertRaises(ValueError):
            validate_analysis_payload({"result": {}, "columns": []})


if __name__ == "__main__":
    unittest.main()
