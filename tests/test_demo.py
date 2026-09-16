from http_attack_agent.demo import run_demo


def test_core_smoke_demo(tmp_path):
    report = run_demo(tmp_path / "smoke", full_dependency_check=False)

    assert report["status"] == "passed"
    assert report["scope"] == "core"
    assert report["dataset"]["rows"] == 96
    assert report["lightweight_model"]["micro_f1"] >= 0.8
    assert "waf_sqli" in report["waf"]["concepts"]
    assert (tmp_path / "smoke" / "smoke_report.json").is_file()
