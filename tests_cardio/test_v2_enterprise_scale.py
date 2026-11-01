"""
Automated Test Suite: FedCardio v2.0 Enterprise Scale, Blind-Spot Rescue, Byzantine Defense & ROI
Verifies:
  1. 8-Hospital physical directory isolation and zero patient_id leakage across 9,600+ records
  2. Non-IID Clinical Heterogeneity & Jensen-Shannon Divergence profiler
  3. 4 Clinical Blind-Spot Rescue Archetypes (Local Island Failure vs. Federated Rescue)
  4. Byzantine Poisoning Resilience (FedMedian & TrimmedMean vs. FedAvg)
  5. Healthcare Economics & Actuarial ROI Engine + 3-Tier Batch EHR Triage
  6. Executive C-Suite Report Generation (Markdown & JSON)
"""
import csv
from pathlib import Path

from fedcardio.config import HOSPITALS, DATA_DIR, UCI_COLUMNS
from fedcardio.evaluation.fl_proof import (
    compute_non_iid_heterogeneity,
    evaluate_clinical_blindspots,
)
from fedcardio.federated.strategies import benchmark_byzantine_resilience
from fedcardio.business.roi_engine import (
    calculate_consortium_roi,
    triage_hospital_cohort,
)
from fedcardio.business.executive_report import generate_executive_briefing


def test_eight_hospital_physical_isolation_and_zero_leakage():
    """Verify all 8 hospitals exist in separate folders with canonical schema and 0 test overlap."""
    assert len(HOSPITALS) == 8, f"Expected 8 global hospitals, got {len(HOSPITALS)}"

    eval_csv = DATA_DIR / "evaluation" / "global_test.csv"
    assert eval_csv.exists(), f"Missing global holdout file: {eval_csv}"
    with open(eval_csv, "r", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    assert len(eval_rows) >= 100
    eval_ids = {r["patient_id"] for r in eval_rows}

    all_train_ids = set()
    for h_id, meta in HOSPITALS.items():
        h_csv = DATA_DIR / meta["folder"] / meta["file"]
        assert h_csv.exists(), f"Missing hospital file for {h_id}: {h_csv}"
        with open(h_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in UCI_COLUMNS + ["hospital_id", "patient_id"]:
                assert col in reader.fieldnames, f"Missing column {col} in {h_csv}"
            rows = list(reader)
        assert len(rows) >= 50, f"Too few records in {h_id}: {len(rows)}"
        h_ids = {r["patient_id"] for r in rows}
        assert len(h_ids.intersection(eval_ids)) == 0, f"Test leakage detected in {h_id}!"
        assert len(h_ids.intersection(all_train_ids)) == 0, f"Cross-hospital ID collision in {h_id}!"
        all_train_ids.update(h_ids)


def test_non_iid_heterogeneity_and_blindspot_rescue():
    """Verify Non-IID JSD quantification and Clinical Blind-Spot Rescue cases."""
    het = compute_non_iid_heterogeneity()
    assert het["num_hospitals"] == 8
    assert het["total_isolated_train_patients"] >= 700
    for h in het["hospitals"]:
        assert "non_iid_divergence_index" in h
        assert h["non_iid_divergence_index"] >= 0.0

    rescue = evaluate_clinical_blindspots()
    assert rescue["total_archetypes_evaluated"] == 4
    assert rescue["lethal_false_negatives_rescued"] == 3
    assert rescue["costly_false_positives_rescued"] == 1
    for case in rescue["cases"]:
        if case["ground_truth_label"] == 1:
            assert case["local_island_risk_pct"] < 50.0
            assert case["federated_risk_pct"] >= 75.0
        else:
            assert case["local_island_risk_pct"] > 50.0
            assert case["federated_risk_pct"] < 25.0


def test_byzantine_poisoning_resilience():
    """Verify FedMedian and TrimmedMean defend against 25% poisoned hospital nodes."""
    byz = benchmark_byzantine_resilience(num_clients=8, num_byzantine=2)
    strats = byz["strategies"]
    assert strats["FedMedian"]["post_attack_roc_auc"] > strats["FedAvg"]["post_attack_roc_auc"] + 0.25
    assert strats["TrimmedMean"]["post_attack_roc_auc"] > strats["FedAvg"]["post_attack_roc_auc"] + 0.25
    assert strats["FedMedian"]["l2_corruption_error"] < strats["FedAvg"]["l2_corruption_error"]


def test_healthcare_roi_and_batch_triage():
    """Verify Healthcare Economics ROI calculation and 3-Tier Batch EHR Triage."""
    roi = calculate_consortium_roi(annual_patients_scale=1.0)
    assert roi["num_hospitals"] == 8
    assert roi["total_net_annual_savings_usd"] > 1_000_000.0
    assert roi["total_amis_prevented"] > 100
    assert roi["total_angiographies_avoided"] > 100
    assert roi["roi_multiple"] > 10.0

    triage = triage_hospital_cohort(hospital_id="longbeach", max_patients=120)
    assert triage["patients_evaluated"] == 120
    total_tiers = (
        triage["tier_1_high_risk_count"]
        + triage["tier_2_moderate_risk_count"]
        + triage["tier_3_low_risk_count"]
    )
    assert total_tiers == 120


def test_executive_briefing_generator():
    """Verify Executive Board Markdown and JSON report generation."""
    rep = generate_executive_briefing(annual_patients_scale=1.0)
    md_path = Path(rep["markdown_report_path"])
    json_path = Path(rep["json_metrics_path"])
    assert md_path.exists() and md_path.stat().st_size > 1000
    assert json_path.exists() and json_path.stat().st_size > 1000
