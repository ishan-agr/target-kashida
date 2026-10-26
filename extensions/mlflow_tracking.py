"""MLflow wrapper for the cheque benchmark: the tracking I wish I had from day one.

STATUS: written for the write-up. The project tracked experiments as timestamped
JSON reports plus a hand-kept "dead path registry" in Markdown. That approach broke
down in four ways, each of which a field below is meant to fix:
  1. Two scorer implementations drifted apart (Arabic-normalised vs frozen pipeline
     matcher)                                          -> log scorer source hash
  2. A scorecard script was deleted, so old numbers could no longer be reproduced
                                                       -> log code + git SHA as artifact
  3. Eval-set membership changed between runs (29 vs 30 crops, 18 vs 83 cheques)
                                                       -> log eval manifest set_hash
  4. Proxy confusion (FP8 used to stand in for AWQ)    -> log exact model id, quant
                                                          format and server flags

Usage (wraps any report the existing harness writes):
  python mlflow_tracking.py --report benchmark/reports/v7a_live_benchmark.json \
      --model cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit --quant awq-w4a16 \
      --eval-manifest eval_frozen/manifest.json --tag dead_path=false
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import mlflow


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "untracked"


def log_benchmark(report: dict, *, model: str, quant: str, eval_manifest: str | None,
                  scorer_fn=None, tags: dict[str, str] | None = None,
                  experiment: str = "cheque-extraction") -> str:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{model.split('/')[-1]}-{quant}") as run:
        mlflow.log_params({"model": model, "quant": quant, "git_sha": _git_sha(),
                           "n_items": report.get("n", len(report.get("rows", [])))})
        if eval_manifest:
            man = json.loads(Path(eval_manifest).read_text())
            mlflow.log_param("eval_set_hash", man.get("set_hash", _sha(eval_manifest)))
        if scorer_fn is not None:
            src = inspect.getsource(scorer_fn).encode()
            mlflow.log_param("scorer_hash", hashlib.sha256(src).hexdigest()[:16])
        for field, acc in report.get("field_accuracy", {}).items():
            mlflow.log_metric(f"acc/{field}", float(acc))
        for k in ("latency_mean_s", "latency_p95_s", "micr_emitted_precision", "micr_coverage"):
            if k in report:
                mlflow.log_metric(k, float(report[k]))
        mlflow.set_tags({"validated": "true", **(tags or {})})
        mlflow.log_dict(report, "report.json")
        return run.info.run_id


def promote_if_better(run_id: str, metric: str = "acc/payee",
                      experiment: str = "cheque-extraction") -> None:
    """Tag `run_id` as champion only if it beats the current champion on `metric`.

    The served weights live on the HF Hub / vLLM host, not in MLflow, so a
    'champion' tag on the benchmark run (which records the exact model id + quant)
    is the promotion record.
    """
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    new = client.get_run(run_id).data.metrics.get(metric, 0.0)
    champs = client.search_runs([exp.experiment_id], "tags.champion = 'true'")
    old = max((r.data.metrics.get(metric, 0.0) for r in champs), default=-1.0)
    if new > old:
        for r in champs:
            client.set_tag(r.info.run_id, "champion", "false")
        client.set_tag(run_id, "champion", "true")
        print(f"promoted {run_id}: {metric} {old:.3f} -> {new:.3f}")
    else:
        print(f"kept champion: {metric} {old:.3f} >= {new:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--eval-manifest")
    ap.add_argument("--tag", action="append", default=[])
    a = ap.parse_args()
    rid = log_benchmark(json.loads(Path(a.report).read_text("utf-8")), model=a.model, quant=a.quant,
                        eval_manifest=a.eval_manifest, tags=dict(t.split("=", 1) for t in a.tag))
    print("run_id", rid)
