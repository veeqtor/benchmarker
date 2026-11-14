"""Production-critical tests for report.py."""

from benchmarker.report import table


class TestTable:
    def test_basic_table(self):
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        result = table(rows, ["a", "b"])
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 rows
        assert "| a | b |" in lines[0]
        assert "| 1 | 2 |" in lines[2]

    def test_escapes_pipes(self):
        rows = [{"col": "a|b"}]
        result = table(rows, ["col"])
        assert "a|b" not in result
        assert "a/b" in result

    def test_empty_rows(self):
        result = table([], ["a", "b"])
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + separator

    def test_missing_values(self):
        rows = [{"a": "1"}]  # missing "b"
        result = table(rows, ["a", "b"])
        assert "| 1 |  |" in result

