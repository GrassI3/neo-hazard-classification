"""Headless dashboard test with Streamlit's AppTest: first load and a training run."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_loads_and_trains() -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Train / Re-train Models" in i.value for i in at.info)

    at.sidebar.radio[0].set_value("Mock").run()
    at.sidebar.slider[0].set_value(1500)  # fewer rows keeps the test quick
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert len(at.tabs) == 7
    assert any("Recall" in s.value for s in at.success)


def test_app_trains_on_nasa_data() -> None:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert at.sidebar.radio[0].value == "Real NASA data"
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert len(at.tabs) == 7
