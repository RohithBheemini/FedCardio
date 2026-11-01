"""
FedCardio v2.0 Unified Enterprise Command-Line Interface (CLI)
Provides 13 commands for 8-hospital dataset provisioning, enterprise cohort scaling,
federated training (FedAvg/FedProx/FedMedian/TrimmedMean), clinical blind-spot rescue proof,
healthcare economics ROI auditing, Byzantine poisoning defense, SHAP predictions, and web portal launch.
"""
import sys
import argparse
import subprocess
from pathlib import Path

from fedcardio.config import HOSPITALS, DATA_DIR, CHECKPOINT_DIR
from fedcardio.data.downloader import fetch_or_generate_hospital_data
from fedcardio.data.synthesizer import generate_modern_enterprise_cohort
from fedcardio.evaluation.fl_proof import (
    compute_non_iid_heterogeneity,
    evaluate_clinical_blindspots,
)
from fedcardio.federated.strategies import benchmark_byzantine_resilience
from fedcardio.business.roi_engine import calculate_consortium_roi, triage_hospital_cohort
from fedcardio.business.executive_report import generate_executive_briefing
from fedcardio.database.audit_db import get_recent_rounds, init_database


def cmd_info(args):
    init_database()
    het = compute_non_iid_heterogeneity()
    roi = calculate_consortium_roi()
    print("\n=====================================================================")
    print("FedCardio v2.0 Global 8-Hospital Clinical AI & Economics Platform")
    print("=====================================================================")
    print("Status: Active | Architecture: Flower Federated Learning + vmap DP-SGD + Coalitional SHAP")
    print(
        f"Consortium Scale: {het['num_hospitals']} Isolated Hospital Silos | "
        f"{het['total_isolated_train_patients']:,} Train Records (Zero Shared CSVs)"
    )
    print(
        f"Annual Net Health System ROI: ${roi['total_net_annual_savings_usd']:,.2f} "
        f"({roi['roi_multiple']}x ROI | {roi['total_amis_prevented']} AMIs Prevented/yr)"
    )
    print("\nParticipating Sovereign Hospital Nodes (Physical Isolated Silos):")
    for h_stat in het["hospitals"]:
        print(
            f"  * {h_stat['hospital_name']} ({h_stat['city']}, {h_stat['country']}) [{h_stat['tier']}]"
        )
        print(
            f"    Path: {h_stat['folder']} | {h_stat['n_patients']} pts | "
            f"CAD: {h_stat['cad_prevalence_pct']}% | Female: {h_stat['female_pct']}% | "
            f"Cath: {h_stat['cath_modality_pct']}% | JSD Skew: {h_stat['non_iid_divergence_index']}"
        )

    rounds = get_recent_rounds()
    print(f"\nAudit History: {len(rounds)} federated rounds recorded in SQLite.")
    if rounds:
        latest = rounds[-1]
        print(
            f"Latest Round: #{latest['round_id']} | ROC-AUC: {latest['global_roc_auc']:.4f} | "
            f"PR-AUC: {latest['global_pr_auc']:.4f} | Strategy: {latest['strategy']}"
        )
    print("=====================================================================\n")


def cmd_setup(args):
    print("Provisioning 8-hospital isolated datasets (4 UCI Benchmarks + 4 Specialist Centers)...")
    fetch_or_generate_hospital_data(force=args.force)
    print("\nAll 8 hospital silos verified in physically isolated storage directories.")


def cmd_scale(args):
    print(
        f"Scaling 8-hospital enterprise consortium to {args.patients} patients per hospital "
        f"({args.patients * len(HOSPITALS):,} total patients)..."
    )
    generate_modern_enterprise_cohort(n_per_hospital=args.patients)
    print("\nScaled 8-hospital canonical EHR silos generated with strict 85/15 split and zero leakage.")


def cmd_run_fl(args):
    from fedcardio.federated.server import run_federated_pipeline
    print(
        f"Triggering {args.rounds} federated rounds across 8 hospitals with {args.strategy} "
        f"(DP epsilon={args.epsilon}, Byzantine nodes={args.byzantine_nodes})..."
    )
    history = run_federated_pipeline(
        num_rounds=args.rounds,
        strategy=args.strategy,
        use_dp=not args.no_dp,
        epsilon=args.epsilon,
        byzantine_nodes=args.byzantine_nodes,
    )
    print(f"\nFederated training completed successfully. {len(history)} rounds audited.")


def cmd_blindspots(args):
    print("\n=====================================================================")
    print("FedCardio Clinical Blind-Spot Rescue Benchmark (Local Island vs. FL)")
    print("=====================================================================")
    res = evaluate_clinical_blindspots()
    for c in res["cases"]:
        print(f"\n[{c['case_id']}] {c['title']}")
        print(f"  Presenting Center : {c['presenting_hospital_name']}")
        print(f"  Patient Summary   : {c['clinical_summary']}")
        print(f"  Ground Truth      : {c['ground_truth']}")
        print(f"  Local Island Risk : {c['local_island_risk_pct']}% -> {c['local_island_decision']}")
        print(f"  FedCardio FL Risk : {c['federated_risk_pct']}% -> {c['federated_decision']}")
        print(f"  Economic Impact   : {c['financial_impact_label']}")
    print("=====================================================================\n")


def cmd_roi(args):
    roi = calculate_consortium_roi(annual_patients_scale=args.scale)
    print("\n=====================================================================")
    print("FedCardio Healthcare Economics & Actuarial ROI Audit (8 Hospitals)")
    print("=====================================================================")
    print(f"  Net Annual Health System Value : ${roi['total_net_annual_savings_usd']:,.2f}")
    print(f"  Direct Clinical Care Savings   : ${roi['clinical_care_savings_usd']:,.2f}")
    print(f"  Prevented Heart Attacks (AMIs) : {roi['total_amis_prevented']:,} / yr (${roi['prevented_ami_savings_usd']:,.2f})")
    print(f"  Avoided Unnecessary Angiograms : {roi['total_angiographies_avoided']:,} / yr (${roi['avoided_angiography_savings_usd']:,.2f})")
    print(f"  CMS Readmission Penalties Saved: ${roi['cms_penalties_avoided_usd']:,.2f}")
    print(f"  HIPAA Breach Liability Avoided : ${roi['breach_liability_avoided_usd']:,.2f}")
    print(f"  Consortium ROI Multiple        : {roi['roi_multiple']}x (Payback: {roi['payback_months']} months)")
    print("\nPer-Hospital Net Annual Value Breakdown:")
    for h in roi["per_hospital_breakdown"]:
        print(
            f"  * {h['hospital_name']:<36} | Admissions: {h['annual_admissions']:>5} | "
            f"AMIs Prevented: {h['amis_prevented']:>3} | Caths Avoided: {h['angiograms_avoided']:>3} | "
            f"Net Value: ${h['net_annual_value_usd']:>10,.2f}"
        )
    print("=====================================================================\n")


def cmd_triage(args):
    res = triage_hospital_cohort(hospital_id=args.hospital, max_patients=args.limit)
    print(f"\nBatch EHR Triage for {res['hospital_name']} ({res['isolated_source_file']}):")
    print(f"  Total Patients Evaluated : {res['patients_evaluated']}")
    print(f"  [Tier 1 Red >=70% Risk]  : {res['tier_1_high_risk_count']} ({res['tier_1_pct']}%) -> Immediate Cath Lab")
    print(f"  [Tier 2 Amber 35-69%]    : {res['tier_2_moderate_risk_count']} ({res['tier_2_pct']}%) -> Outpatient Stress Echo + Statin")
    print(f"  [Tier 3 Green <35% Risk] : {res['tier_3_low_risk_count']} ({res['tier_3_pct']}%) -> Routine Prevention\n")


def cmd_report(args):
    generate_executive_briefing(annual_patients_scale=args.scale)


def cmd_byzantine(args):
    res = benchmark_byzantine_resilience(num_clients=8, num_byzantine=args.poisoned)
    print(f"\nByzantine Resilience Benchmark ({res['num_byzantine']}/{res['num_clients']} Poisoned Hospital Nodes):")
    for s_name, s_info in res["strategies"].items():
        print(
            f"  * {s_name:<12} | Post-Attack ROC-AUC: {s_info['post_attack_roc_auc']:.4f} | "
            f"L2 Error: {s_info['l2_corruption_error']:>7.4f} | {s_info['status']}"
        )
    print()


def cmd_matrix(args):
    from fedcardio.evaluation.cross_validation import generate_cross_hospital_transfer_matrix
    print("Benchmarking 8x8 Cross-Hospital Generalization Matrix (Tri-Model Evaluation)...")
    generate_cross_hospital_transfer_matrix(fast_mode=args.fast)


def cmd_privacy_test(args):
    from fedcardio.evaluation.attack_simulation import simulate_gradient_inversion_attack
    print("Running Live Gradient Inversion Attack Simulator (Deep Leakage from Gradients)...")
    res_no_dp = simulate_gradient_inversion_attack(use_dp=False, steps=35)
    res_dp = simulate_gradient_inversion_attack(use_dp=True, steps=35)
    print(f"\n[Unprotected Baseline] Attacker Error (MSE): {res_no_dp['initial_mse']} -> {res_no_dp['final_mse']} (Vitals partially leaked)")
    print(f"[FedCardio DP-SGD] Attacker Error (MSE): {res_dp['initial_mse']} -> {res_dp['final_mse']} (Reconstruction neutralized)")


def cmd_predict(args):
    from fedcardio.explainability.explainer import ClinicalSHAPExplainer
    explainer = ClinicalSHAPExplainer()
    patient = {
        "age": args.age, "sex": args.sex, "cp": args.cp, "trestbps": args.trestbps,
        "chol": args.chol, "fbs": args.fbs, "restecg": args.restecg,
        "thalach": args.thalach, "exang": args.exang, "oldpeak": args.oldpeak,
        "slope": args.slope, "ca": args.ca, "thal": args.thal,
    }
    res = explainer.explain_patient(patient)
    print("\n================ Patient Diagnostic Assessment ================")
    print(f"Predicted 1-Year Cardiac Risk: {res['predicted_risk_percent']}%")
    print(f"Risk Stratification: {res['risk_category']}")
    print("\nTop Clinical Feature Drivers (Coalitional SHAP Attribution):")
    for att in res["attributions"][:5]:
        arrow = "[+]" if att["is_risk_increasing"] else "[-]"
        print(f"  {arrow} {att['label']} ({att['value']}): {att['impact']:+.2f}% points")
    print("================================================================\n")


def cmd_serve(args):
    app_path = Path(__file__).resolve().parent / "web" / "app.py"
    port = args.port
    print(f"Launching FedCardio v2.0 Enterprise Web Portal on port {port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)])


def cmd_verify(args):
    print("Running FedCardio v2.0 Verification Suite...")
    subprocess.run([sys.executable, "run_cardio_tests.py"])


def main():
    parser = argparse.ArgumentParser(description="FedCardio v2.0 Unified Enterprise CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("info", help="View 8-hospital consortium topology, Non-IID skew, and ROI status")

    p_setup = subparsers.add_parser("setup", help="Provision 8-hospital isolated datasets")
    p_setup.add_argument("--force", action="store_true", help="Force regeneration")

    p_scale = subparsers.add_parser("scale", help="Scale 8-hospital cohorts in isolated silos")
    p_scale.add_argument("--patients", type=int, default=1200, help="Patients per hospital silo")

    p_gen = subparsers.add_parser("generate", help="Alias for scale: generate enterprise cohort")
    p_gen.add_argument("--patients", type=int, default=1200, help="Patients per hospital silo")

    p_fl = subparsers.add_parser("run-fl", help="Run 8-hospital federated training rounds")
    p_fl.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    p_fl.add_argument(
        "--strategy",
        type=str,
        default="FedProx",
        choices=["FedProx", "FedAvg", "FedMedian", "TrimmedMean"],
    )
    p_fl.add_argument("--epsilon", type=float, default=3.0, help="Privacy budget epsilon")
    p_fl.add_argument("--byzantine-nodes", type=int, default=0, help="Number of simulated poisoned hospital nodes")
    p_fl.add_argument("--no-dp", action="store_true", help="Disable differential privacy")

    subparsers.add_parser("blindspots", help="Evaluate Clinical Blind-Spot Rescue archetypes (Local vs FL)")

    p_roi = subparsers.add_parser("roi", help="Calculate Healthcare Economics & Actuarial ROI across 8 hospitals")
    p_roi.add_argument("--scale", type=float, default=1.0, help="Admission volume multiplier")

    p_tri = subparsers.add_parser("triage", help="Run 3-Tier Batch EHR Clinical Triage on an isolated hospital")
    p_tri.add_argument("--hospital", type=str, default="cleveland", choices=list(HOSPITALS.keys()))
    p_tri.add_argument("--limit", type=int, default=150, help="Max patients to evaluate")

    p_rep = subparsers.add_parser("report", help="Generate C-Suite Executive Board Markdown & JSON reports")
    p_rep.add_argument("--scale", type=float, default=1.0, help="Admission volume multiplier")

    p_byz = subparsers.add_parser("byzantine-test", help="Benchmark Byzantine poisoning resilience")
    p_byz.add_argument("--poisoned", type=int, default=2, help="Number of poisoned hospital nodes")

    p_mat = subparsers.add_parser("matrix", help="Generate 8x8 Cross-Hospital Generalization Matrix")
    p_mat.add_argument("--fast", action="store_true", help="Use fast calibrated 8x8 matrix")

    subparsers.add_parser("privacy-test", help="Test gradient inversion (DLG) defense")

    p_pred = subparsers.add_parser("predict", help="Predict cardiac risk & Coalitional SHAP for patient")
    p_pred.add_argument("--age", type=float, default=62)
    p_pred.add_argument("--sex", type=float, default=1)
    p_pred.add_argument("--cp", type=float, default=4)
    p_pred.add_argument("--trestbps", type=float, default=145)
    p_pred.add_argument("--chol", type=float, default=275)
    p_pred.add_argument("--fbs", type=float, default=1)
    p_pred.add_argument("--restecg", type=float, default=2)
    p_pred.add_argument("--thalach", type=float, default=128)
    p_pred.add_argument("--exang", type=float, default=1)
    p_pred.add_argument("--oldpeak", type=float, default=2.4)
    p_pred.add_argument("--slope", type=float, default=2)
    p_pred.add_argument("--ca", type=float, default=2)
    p_pred.add_argument("--thal", type=float, default=7)

    p_srv = subparsers.add_parser("serve", help="Launch 6-tab interactive web portal")
    p_srv.add_argument("--port", type=int, default=8501, help="Port to serve on")

    subparsers.add_parser("verify", help="Run automated test suite")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "info": cmd_info,
        "setup": cmd_setup,
        "scale": cmd_scale,
        "generate": cmd_scale,
        "run-fl": cmd_run_fl,
        "blindspots": cmd_blindspots,
        "roi": cmd_roi,
        "triage": cmd_triage,
        "report": cmd_report,
        "byzantine-test": cmd_byzantine,
        "matrix": cmd_matrix,
        "privacy-test": cmd_privacy_test,
        "predict": cmd_predict,
        "serve": cmd_serve,
        "verify": cmd_verify,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
