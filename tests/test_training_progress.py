from http_attack_agent import training


def test_progress_batches_reports_total_and_phase(monkeypatch):
    captured = {}

    def fake_tqdm(iterable, **kwargs):
        captured.update(kwargs)
        return iterable

    monkeypatch.setattr(training, "tqdm", fake_tqdm)
    batches = ["first", "second", "third"]

    assert list(training._progress_batches(batches, "canine-c validation")) == batches
    assert captured["total"] == 3
    assert captured["desc"] == "canine-c validation"
    assert captured["unit"] == "batch"
