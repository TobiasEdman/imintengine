"""The Stage B scoring template must not ship a deadline sized for one mode."""
from pathlib import Path
import re

import yaml

TEMPLATE = Path(__file__).resolve().parents[1] / "k8s/score-truth-job.template.yaml"


def test_deadline_is_an_explicit_operator_placeholder():
    text = TEMPLATE.read_text()
    header = text.split("apiVersion:", 1)[0]
    assert "__ACTIVE_DEADLINE_SECONDS__" in header
    assert yaml.safe_load(text)["spec"]["activeDeadlineSeconds"] == "__ACTIVE_DEADLINE_SECONDS__"


def test_substituted_template_is_a_job_with_integer_deadline():
    text = TEMPLATE.read_text().replace("__ACTIVE_DEADLINE_SECONDS__", "1800")
    text = re.sub(r"__[A-Z0-9_]+__", "x", text)
    job = yaml.safe_load(text)
    assert job["kind"] == "Job"
    assert job["spec"]["activeDeadlineSeconds"] == 1800
