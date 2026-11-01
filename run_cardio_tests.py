"""
FedCardio v2.0 Automated Verification & Test Runner
Executes all unit and integration tests across:
  1. 8-Hospital Data Isolation, Canonical Schema & Zero Leakage
  2. Non-IID Heterogeneity & Clinical Blind-Spot Rescue Engine
  3. Byzantine Poisoning Resilience (FedMedian & TrimmedMean)
  4. Healthcare Economics, Actuarial ROI & 3-Tier Batch EHR Triage
  5. Executive Board Report Generator & Full Codebase Bytecode Compilation
  6. PyTorch Federated Pipeline, Per-Sample DP-SGD & Coalitional SHAP (when torch is active)
"""
import sys
import py_compile
import traceback
from pathlib import Path

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_all_modules_bytecode_compile():
    """Verify that every Python module in fedcardio/, web/, tests_cardio/, and CLI compiles cleanly."""
    root = Path(__file__).resolve().parent
    py_files = (
        list((root / "fedcardio").rglob("*.py"))
        + list((root / "web").rglob("*.py"))
        + list((root / "tests_cardio").rglob("*.py"))
        + [root / "cardio_cli.py", root / "run_cardio_tests.py"]
    )
    for f in py_files:
        py_compile.compile(str(f), doraise=True)


def run_all_tests():
    print("\n=====================================================================")
    print("Running FedCardio v2.0 Enterprise Verification & Automated Test Suite")
    print("=====================================================================\n")

    from tests_cardio.test_v2_enterprise_scale import (
        test_eight_hospital_physical_isolation_and_zero_leakage,
        test_non_iid_heterogeneity_and_blindspot_rescue,
        test_byzantine_poisoning_resilience,
        test_healthcare_roi_and_batch_triage,
        test_executive_briefing_generator,
    )

    all_tests = [
        ("test_all_modules_bytecode_compile", test_all_modules_bytecode_compile),
        ("test_eight_hospital_physical_isolation_and_zero_leakage", test_eight_hospital_physical_isolation_and_zero_leakage),
        ("test_non_iid_heterogeneity_and_blindspot_rescue", test_non_iid_heterogeneity_and_blindspot_rescue),
        ("test_byzantine_poisoning_resilience", test_byzantine_poisoning_resilience),
        ("test_healthcare_roi_and_batch_triage", test_healthcare_roi_and_batch_triage),
        ("test_executive_briefing_generator", test_executive_briefing_generator),
    ]

    if HAS_TORCH:
        from tests_cardio.test_data_isolation import (
            test_hospital_files_exist_in_separate_folders,
            test_isolated_hospital_loaders,
            test_global_evaluation_loader,
        )
        from tests_cardio.test_federated_pipeline import (
            test_clinical_net_forward_and_payload,
            test_hospital_client_fit_and_drift,
            test_federated_consortium_2_rounds,
        )
        from tests_cardio.test_privacy_and_explainability import (
            test_dp_engine_clipping_and_noise,
            test_gradient_inversion_defense,
            test_clinical_shap_explainer,
        )
        all_tests.extend([
            ("test_hospital_files_exist_in_separate_folders", test_hospital_files_exist_in_separate_folders),
            ("test_isolated_hospital_loaders", test_isolated_hospital_loaders),
            ("test_global_evaluation_loader", test_global_evaluation_loader),
            ("test_clinical_net_forward_and_payload", test_clinical_net_forward_and_payload),
            ("test_hospital_client_fit_and_drift", test_hospital_client_fit_and_drift),
            ("test_federated_consortium_2_rounds", test_federated_consortium_2_rounds),
            ("test_dp_engine_clipping_and_noise", test_dp_engine_clipping_and_noise),
            ("test_gradient_inversion_defense", test_gradient_inversion_defense),
            ("test_clinical_shap_explainer", test_clinical_shap_explainer),
        ])

    total = 0
    passed = 0
    failures = []

    for name, test_fn in all_tests:
        total += 1
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failures.append((name, str(e)))

    print("\n=====================================================================")
    print(f"Test Results: {passed}/{total} Passed")
    if failures:
        print(f"Failures: {len(failures)}")
        for name, err in failures:
            print(f"  * {name}: {err}")
        sys.exit(1)
    else:
        print("ALL FEDCARDIO v2.0 ENTERPRISE TESTS PASSED SUCCESSFULLY!")
        print("=====================================================================\n")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
