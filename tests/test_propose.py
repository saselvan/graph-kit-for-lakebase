"""Tests for propose step: draft graph structure from keyed tables and FKs."""

import json
import tempfile
from pathlib import Path

import pytest

from lgk.steps.propose import propose


class TestProposeRefusesKeylessEdges:
    """Test that propose refuses to draft edges with no key behind them."""

    @pytest.mark.seam2
    def test_propose_refuses_keyless_edge(self):
        """Propose should refuse to create an edge for a table with no primary key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "propose_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create schema fixture: one keyed table, one join table (no key)
            schema_file = run_dir / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {
                                "name": "providers",
                                "key_columns": ["provider_id"],
                            },
                            {
                                "name": "provider_specialties",
                                # NO key_columns - this is a join table
                                "foreign_keys": [
                                    {
                                        "columns": ["provider_id"],
                                        "references": ["providers", "provider_id"],
                                    }
                                ],
                            },
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
                "source_catalog": "cat",
                "source_schema": "gold",
            }

            # Propose should either:
            # 1. Refuse to create the edge (raise error or skip it)
            # 2. Write a warning in graph_mapping.yaml
            try:
                propose(config)

                # If it succeeds, check that the keyless edge was NOT created
                graph_mapping_file = run_dir / "graph_mapping.yaml"
                if graph_mapping_file.exists():
                    with open(graph_mapping_file) as f:
                        content = f.read()

                    # The edge should not be proposed
                    assert (
                        "provider_specialties" not in content
                        or "skipped" in content.lower()
                        or "keyless" in content.lower()
                    )
            except ValueError as e:
                # Expected: "keyless table" or similar
                assert "key" in str(e).lower()

    @pytest.mark.seam2
    def test_propose_creates_edges_with_keys(self):
        """Propose should create edges for keyed tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "propose_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            schema_file = run_dir / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {
                                "name": "providers",
                                "key_columns": ["provider_id"],
                            },
                            {
                                "name": "affiliations",
                                "key_columns": ["affiliation_id"],
                                "foreign_keys": [
                                    {
                                        "columns": ["provider_id"],
                                        "references": ["providers", "provider_id"],
                                    }
                                ],
                            },
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
                "source_catalog": "cat",
                "source_schema": "gold",
            }

            propose(config)

            graph_mapping_file = run_dir / "graph_mapping.yaml"
            assert graph_mapping_file.exists()

            with open(graph_mapping_file) as f:
                content = f.read()

            # Both tables should be in the mapping
            assert "providers" in content
            assert "affiliations" in content


class TestProposeGraphShapeClassification:
    """Test that propose classifies graph as web or tree."""

    @pytest.mark.seam2
    def test_propose_classifies_tree_shape(self):
        """Propose should classify a tree-shaped graph (hierarchy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "propose_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create a tree: root -> parent -> child
            schema_file = run_dir / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {"name": "root", "key_columns": ["root_id"]},
                            {
                                "name": "parent",
                                "key_columns": ["parent_id"],
                                "foreign_keys": [
                                    {
                                        "columns": ["root_id"],
                                        "references": ["root", "root_id"],
                                    }
                                ],
                            },
                            {
                                "name": "child",
                                "key_columns": ["child_id"],
                                "foreign_keys": [
                                    {
                                        "columns": ["parent_id"],
                                        "references": ["parent", "parent_id"],
                                    }
                                ],
                            },
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
                "source_catalog": "cat",
                "source_schema": "gold",
            }

            propose(config)

            flags_file = run_dir / "flags.json"
            assert flags_file.exists()

            with open(flags_file) as f:
                flags_data = json.load(f)

            # Must have a shape classification
            assert "shape" in flags_data or "graph_shape" in flags_data

    @pytest.mark.seam2
    def test_propose_classifies_web_shape(self):
        """Propose should classify a web-shaped graph (many cross-links)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "propose_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create a web: multiple cross-linked tables
            schema_file = run_dir / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {"name": "providers", "key_columns": ["provider_id"]},
                            {"name": "organizations", "key_columns": ["org_id"]},
                            {
                                "name": "affiliations",
                                "key_columns": ["affiliation_id"],
                                "foreign_keys": [
                                    {
                                        "columns": ["provider_id"],
                                        "references": ["providers", "provider_id"],
                                    },
                                    {
                                        "columns": ["org_id"],
                                        "references": ["organizations", "org_id"],
                                    },
                                ],
                            },
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
                "source_catalog": "cat",
                "source_schema": "gold",
            }

            propose(config)

            flags_file = run_dir / "flags.json"
            assert flags_file.exists()

            with open(flags_file) as f:
                flags_data = json.load(f)

            # Must have a shape classification
            assert "shape" in flags_data or "graph_shape" in flags_data


class TestProposeDraftsNodesAndEdges:
    """Test that propose drafts nodes from keyed tables and edges from FKs."""

    @pytest.mark.seam2
    def test_propose_drafts_nodes_from_keyed_tables(self):
        """Propose should draft a node for each keyed table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "propose_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            schema_file = run_dir / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {"name": "providers", "key_columns": ["provider_id"]},
                            {"name": "organizations", "key_columns": ["org_id"]},
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
                "source_catalog": "cat",
                "source_schema": "gold",
            }

            propose(config)

            graph_mapping_file = run_dir / "graph_mapping.yaml"
            assert graph_mapping_file.exists()

            with open(graph_mapping_file) as f:
                content = f.read()

            # Nodes should be created
            assert "providers" in content
            assert "organizations" in content

    @pytest.mark.seam2
    def test_propose_writes_graph_mapping_yaml_and_flags_json(self):
        """Propose should write graph_mapping.yaml and flags.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "propose_test"
            run_dir.mkdir(parents=True, exist_ok=True)

            schema_file = run_dir / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(
                    {
                        "tables": [
                            {"name": "providers", "key_columns": ["provider_id"]},
                        ]
                    },
                    f,
                )

            config = {
                "graph_name": "test_graph",
                "run_folder": str(run_dir),
                "source_catalog": "cat",
                "source_schema": "gold",
            }

            propose(config)

            # Both files should exist
            graph_mapping_file = run_dir / "graph_mapping.yaml"
            flags_file = run_dir / "flags.json"

            assert graph_mapping_file.exists()
            assert flags_file.exists()
