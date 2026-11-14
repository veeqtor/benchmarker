"""Production-critical tests for gguf.py."""

import tempfile

from benchmarker.gguf import is_extra_part, is_mmproj, model_files


class TestIsMmproj:
    def test_mmproj_in_name(self):
        assert is_mmproj("/path/mmproj-f16.gguf") is True

    def test_mmproj_lowercase(self):
        assert is_mmproj("/path/MMPROJ.gguf") is True

    def test_not_mmproj(self):
        assert is_mmproj("/path/model.gguf") is False


class TestIsExtraPart:
    def test_extra_shard(self):
        assert is_extra_part("model-00002-of-00003.gguf") is True

    def test_first_shard(self):
        assert is_extra_part("model-00001-of-00003.gguf") is False

    def test_non_sharded(self):
        assert is_extra_part("model.gguf") is False


class TestModelFiles:
    def test_single_file(self):
        assert model_files("/path/model.gguf") == ["/path/model.gguf"]

    def test_sharded_expansion(self):
        import tempfile
        import os
        # Create actual shard files so glob works
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                p = os.path.join(tmpdir, f"model-0000{i}-of-00003.gguf")
                open(p, "w").close()
            result = model_files(os.path.join(tmpdir, "model-00001-of-00003.gguf"))
            assert len(result) == 3


class TestGgufMeta:
    def test_raises_on_non_gguf(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            f.write(b"not a gguf file")
            f.flush()
            from benchmarker.gguf import gguf_meta

            try:
                gguf_meta(f.name)
                assert False, "should raise ValueError"
            except ValueError as e:
                assert "not a GGUF file" in str(e)


class TestModelInfo:
    def test_raises_on_non_gguf(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            f.write(b"not a gguf file")
            f.flush()
            from benchmarker.gguf import model_info

            try:
                model_info(f.name)
                assert False, "should raise ValueError"
            except ValueError:
                pass


class TestFindMmproj:
    def test_no_mmproj(self, tmp_path):
        from benchmarker.gguf import find_mmproj

        # Create a dummy model file
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF" + b"\x00" * 100)
        assert find_mmproj(str(model)) == ""
