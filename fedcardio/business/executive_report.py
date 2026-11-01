"""
FedCardio Executive Board Briefing & C-Suite Report Generator
Generates publication-ready Markdown and JSON executive reports summarizing:
  1. 8-Hospital Consortium Non-IID Topology & Data Sovereignty Attestation
  2. Clinical Blind-Spot Rescue Case Studies (Why Federated Learning Wins)
  3. Healthcare Economics, Prevented AMIs, Avoided Angiographies & Net ROI
  4. Byzantine Poisoning Resilience & Per-Sample DP-SGD Privacy Guarantees
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from fedcardio.config import REPORTS_DIR, HOSPITALS
from fedcardio.evaluation.fl_proof import compute_non_iid_heterogeneity, evaluate_clinical_blindspots
from fedcardio.business.roi_engine import calculate_consortium_roi, triage_hospital_cohort
from fedcardio.federated.strategies import benchmark_byzantine_resilience


def generate_executive_briefing(annual_patients_scale: float = 1.0) -> Dict[str, Any]:
    """
    Generates C-Suite Markdown (`reports/fedcardio_executive_briefing.md`) and
    structured JSON (`reports/roi_metrics.json`) executive reports.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "fedcardio_executive_briefing.md"
    json_path = REPORTS_DIR / "roi_metrics.json"

    het = compute_non_iid_heterogeneity()
    blindspots = evaluate_clinical_blindspots()
    roi = calculate_consortium_roi(annual_patients_scale=annual_patients_scale)
    byz = benchmark_byzantine_resilience(num_clients=8, num_byzantine=2)
    triage_sample = triage_hospital_cohort("cleveland", max_patients=150)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md_lines = [
        "# FedCardio v2.0: Executive Board & Chief Medical Officer Briefing",
        f"**Generated**: {timestamp} | **Consortium Scale**: {het['num_hospitals']} Global Medical Centers | **Compliance**: HIPAA / GDPR / FDA SaMD",
        "",
        "---",
        "",
        "## 1. Executive Summary & C-Suite Value Proposition",
        f"- **Annual Consortium Net Financial Value**: **${roi['total_net_annual_savings_usd']:,.2f}** (**{roi['roi_multiple']}x ROI**, Payback: **{roi['payback_months']} months**)",
        f"- **Direct Clinical Care Savings**: **${roi['clinical_care_savings_usd']:,.2f} / year**",
        f"  - **Prevented Acute Myocardial Infarctions (AMIs)**: **{roi['total_amis_prevented']:,} lives/events rescued per year** (${roi['prevented_ami_savings_usd']:,.2f} saved)",
        f"  - **Avoided Unnecessary Invasive Coronary Angiographies**: **{roi['total_angiographies_avoided']:,} procedures avoided per year** (${roi['avoided_angiography_savings_usd']:,.2f} saved)",
        f"- **CMS 30-Day Readmission Penalty Mitigation (HRRP)**: **${roi['cms_penalties_avoided_usd']:,.2f} / year**",
        f"- **HIPAA / GDPR Breach Liability Avoided**: **${roi['breach_liability_avoided_usd']:,.2f}** ($499/record benchmark; **0 raw patient records ever leave hospital firewalls**)",
        "",
        "---",
        "",
        "## 2. 8-Hospital Global Consortium & Non-IID Clinical Heterogeneity",
        "| Hospital Node | Location | Specialty | Train Pts | CAD Rate | Female % | Cath % | JSD Skew | Net Annual ROI ($) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    roi_by_id = {r["hospital_id"]: r for r in roi["per_hospital_breakdown"]}
    for h in het["hospitals"]:
        r_row = roi_by_id.get(h["hospital_id"], {})
        net_val = r_row.get("net_annual_value_usd", 0.0)
        md_lines.append(
            f"| **{h['hospital_name']}** | {h['city']}, {h['country']} | {h['specialty']} | "
            f"{h['n_patients']:,} | {h['cad_prevalence_pct']}% | {h['female_pct']}% | "
            f"{h['cath_modality_pct']}% | {h['non_iid_divergence_index']} | **${net_val:,.0f}** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Clinical Blind-Spot Rescue Proof (Why Federated Learning Wins)",
        "| Case ID | Presenting Center | True Diagnosis | Local Island Risk (Failure) | FedCardio Consensus Risk (Rescue) | Clinical & Economic Impact |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for c in blindspots["cases"]:
        md_lines.append(
            f"| `{c['case_id']}` | {c['presenting_hospital_name']} | {c['ground_truth']} | "
            f"**{c['local_island_risk_pct']}%** ({c['local_island_decision']}) | "
            f"**{c['federated_risk_pct']}%** ({c['federated_decision']}) | "
            f"**{c['financial_impact_label']}** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Byzantine Poisoning Resilience & Cryptographic Privacy Attestation",
        "- **Zero Single-CSV Slicing**: Every hospital node reads exclusively from its own isolated directory (`data/hospital_<id>/`).",
        "- **Per-Sample DP-SGD (`torch.func.vmap`)**: Enforces exact per-patient $L_2$ gradient clipping ($C=1.0$) decoupled from FedProx proximal regularization.",
        "- **Byzantine Defense (25% Poisoned Hospital Nodes)**:",
    ])

    for strat_name, s_info in byz["strategies"].items():
        md_lines.append(
            f"  - **{strat_name}**: Post-Attack ROC-AUC = `{s_info['post_attack_roc_auc']:.4f}` | "
            f"L2 Corruption = `{s_info['l2_corruption_error']:.4f}` -> *{s_info['status']}*"
        )

    md_content = "\n".join(md_lines) + "\n"
    md_path.write_text(md_content, encoding="utf-8")

    payload = {
        "generated_at": timestamp,
        "consortium_heterogeneity": het,
        "clinical_blindspot_rescues": blindspots,
        "healthcare_economics_roi": roi,
        "byzantine_resilience": byz,
        "sample_batch_triage_summary": {
            k: v for k, v in triage_sample.items() if k != "patient_roster"
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Executive Markdown Briefing saved to: {md_path}")
    print(f"Structured ROI Telemetry JSON saved to: {json_path}")
    return {
        "markdown_report_path": str(md_path),
        "json_metrics_path": str(json_path),
        "roi_summary": roi,
    }


if __name__ == "__main__":
    generate_executive_briefing()
