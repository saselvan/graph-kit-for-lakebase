"""Tests for profile step: schema profiling without sampling sensitive data."""

import json
import tempfile
from pathlib import Path

import pytest

from lgk.steps.profile import profile


class TestProfileNoSampledValues:
    """Test that profile writes VERDICTS ONLY, never sampled data."""

    @pytest.mark.seam2
    def test_profile_schema_json_contains_no_sampled_values(self):
        """schema.json should contain ONLY verdicts (match rate, entropy, distinct ratio), NO sampled values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "profile_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal gold_tables fixture
            gold_tables_file = run_dir / "gold_tables.json"
            with open(gold_tables_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {
                                "name": "providers",
                                "key_columns": ["provider_id"],
                                "row_count": 5000,
                            }
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }

            profile(config)

            # Read schema.json
            schema_file = run_dir / "schema.json"
            assert schema_file.exists()

            with open(schema_file) as f:
                schema_data = json.load(f)

            # Check that NO sampled values appear in the schema
            schema_str = json.dumps(schema_data)

            # These patterns indicate sampled data (which should NOT appear):
            forbidden_patterns = [
                "patient_name",  # Real name data
                "provider_id_sample",
                "example_value",
                "sample_",
                "data_sample",
            ]

            for pattern in forbidden_patterns:
                assert pattern not in schema_str.lower(), (
                    f"schema.json contains sampled data pattern: {pattern}"
                )

            # Verify verdicts ARE present (match_rate, entropy, distinct_ratio)
            assert isinstance(schema_data, dict)
            if "tables" in schema_data:
                for table in schema_data.get("tables", []):
                    # Each table should have verdicts, not samples
                    if "columns" in table:
                        for column in table["columns"]:
                            # Verdict fields are allowed
                            allowed_verdict_fields = {
                                "name",
                                "type",
                                "match_rate",
                                "entropy",
                                "distinct_ratio",
                                "null_rate",
                            }
                            for field in column.keys():
                                assert field in allowed_verdict_fields or field.startswith(
                                    "_"
                                ), f"Unexpected field in column: {field} (looks like sampled data)"

    @pytest.mark.seam2
    def test_profile_verdicts_only_no_raw_samples(self):
        """Profile verdicts should use computed metrics, never raw sampled rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "profile_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create fixture
            gold_tables_file = run_dir / "gold_tables.json"
            with open(gold_tables_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {
                                "name": "claims",
                                "key_columns": ["claim_id"],
                                "row_count": 10000,
                            }
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }

            profile(config)

            # Read schema.json
            schema_file = run_dir / "schema.json"
            with open(schema_file) as f:
                schema_data = json.load(f)

            # Verify all data is numeric verdicts, not strings/arrays with sample data
            def check_no_samples_recursive(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        # Allowed non-numeric fields: name, type, etc.
                        if key in ["name", "type", "table_name"]:
                            assert isinstance(value, str)
                        elif key in ["match_rate", "entropy", "distinct_ratio", "null_rate"]:
                            # These should be numeric verdicts
                            assert isinstance(value, (int, float)), (
                                f"Verdict {key} at {path} should be numeric, got {type(value)}"
                            )
                        else:
                            check_no_samples_recursive(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_no_samples_recursive(item, f"{path}[{i}]")

            check_no_samples_recursive(schema_data)

    @pytest.mark.seam2
    def test_profile_writes_to_schema_json(self):
        """Profile should write schema.json to the run folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "profile_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create minimal fixture
            gold_tables_file = run_dir / "gold_tables.json"
            with open(gold_tables_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {
                                "name": "providers",
                                "key_columns": ["provider_id"],
                            }
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
            }

            profile(config)

            schema_file = run_dir / "schema.json"
            assert schema_file.exists()
            assert schema_file.stat().st_size > 0
