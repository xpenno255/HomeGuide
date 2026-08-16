"""Title mixing: the vector nudge that steers queries to the right appliance."""
import numpy as np
import pytest

from app import embeddings


TITLE = "Ninja Air Fryer"


class TestTitleMixing:
    def test_vector_moves_toward_the_title(self, monkeypatch):
        """Body points along x, title along y: the result must lean toward y
        without ever leaving the body's half of the plane."""

        def fake(texts):
            vecs = [[0.0, 1.0] if t == TITLE else [1.0, 0.0] for t in texts]
            return np.array(vecs, dtype=np.float32)

        monkeypatch.setattr(embeddings, "embed_passages", fake)
        out = embeddings.embed_passages_titled(["some chunk"], TITLE)
        assert out is not None
        assert out[0][1] > 0, "no title component mixed in"
        assert out[0][0] > out[0][1], "title overwhelmed the body"

    def test_result_is_normalised(self, monkeypatch):
        def fake(texts):
            return np.array([[0.6, 0.8]] * len(texts), dtype=np.float32)

        monkeypatch.setattr(embeddings, "embed_passages", fake)
        out = embeddings.embed_passages_titled(["a", "b"], "Some Manual")
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_blank_title_leaves_vectors_untouched(self, monkeypatch):
        base = np.array([[1.0, 0.0]], dtype=np.float32)
        monkeypatch.setattr(embeddings, "embed_passages", lambda texts: base.copy())
        assert np.allclose(embeddings.embed_passages_titled(["a"], "   "), base)

    def test_keyword_only_mode_returns_none(self, monkeypatch):
        monkeypatch.setattr(embeddings, "embed_passages", lambda texts: None)
        assert embeddings.embed_passages_titled(["a"], "Manual") is None

    def test_mix_is_a_nudge_not_a_takeover(self):
        """0.3 already loses the cooking charts; keep this well below it."""
        assert 0 < embeddings.TITLE_MIX <= 0.2
