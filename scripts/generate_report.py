#!/usr/bin/env python3
"""
Generate Phase 0 + Phase 1 technical report as PDF.

Includes findings, plots, and next steps for boss review.
"""

import json
import os
import sys
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
PHASE1_DIR = os.path.join(OUTPUT_DIR, "phase1")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=26,
        leading=32,
        spaceAfter=6 * mm,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a4a6a"),
        spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceBefore=10 * mm,
        spaceAfter=4 * mm,
        textColor=colors.HexColor("#1a1a2e"),
        borderWidth=0,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SubHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
        textColor=colors.HexColor("#2d2d4e"),
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        "BulletItem",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        leftIndent=12 * mm,
        bulletIndent=6 * mm,
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "Finding",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        leftIndent=8 * mm,
        borderWidth=1,
        borderColor=colors.HexColor("#2ecc71"),
        borderPadding=6,
        backColor=colors.HexColor("#f0faf0"),
        spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"),
        spaceAfter=6 * mm,
        spaceBefore=2 * mm,
    ))
    return styles


def add_cover(story, styles):
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph(
        "ESN Uncertainty Estimation<br/>for Robotic Arm Manipulation",
        styles["CoverTitle"],
    ))
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(
        width="60%", thickness=1.5,
        color=colors.HexColor("#3498db"),
        spaceAfter=8 * mm,
    ))
    story.append(Paragraph(
        "Phase 0 &amp; Phase 1 Technical Report",
        styles["CoverSubtitle"],
    ))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "Echo State Network prediction error as a real-time uncertainty signal<br/>"
        "for detecting manipulation failures in robotic arms",
        styles["CoverSubtitle"],
    ))
    story.append(Spacer(1, 20 * mm))

    date_str = datetime.now().strftime("%B %d, %Y")
    meta_data = [
        ["Author", "Umar Bin Muzzafar"],
        ["Organization", "OMNIBIO LTD"],
        ["Date", date_str],
        ["Datasets", "Berkeley Autolab UR5, DROID-100 (Franka Panda)"],
        ["Tools", "LeRobot, Rerun.io, Python 3.10"],
    ]
    meta_table = Table(meta_data, colWidths=[40 * mm, 100 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4a4a6a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(PageBreak())


def add_executive_summary(story, styles, phase1_data):
    story.append(Paragraph("Executive Summary", styles["SectionHeader"]))
    story.append(Paragraph(
        "This report presents the development and validation of an Echo State Network (ESN) "
        "based uncertainty estimator for robotic arm manipulation. The system learns what "
        "\"normal\" robot behavior looks like, then flags deviations in real-time — enabling "
        "a robot to detect when it is about to fail <b>before</b> the failure actually happens.",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 4 * mm))

    auc = phase1_data["classification"]["auc"]
    div = phase1_data["temporal"]["divergence_pct"]
    joint = phase1_data["per_joint"]["most_predictive"]
    ratio = phase1_data["per_joint"]["max_ratio"]

    findings = [
        ["Finding", "Result"],
        ["Failure detection", "Failed episodes have 1.35x higher prediction error (sigma)"],
        ["Most predictive joint", f"{joint} (upper arm) — {ratio:.2f}x failure/success ratio"],
        ["Early warning", f"Sigma diverges at {div:.0f}% episode completion — {100-div:.0f}% warning time"],
        ["Classification", f"AUC = {auc:.3f} using sigma alone (moderate classifier)"],
        ["Training speed", "0.95 seconds (closed-form, no gradient descent)"],
    ]
    t = Table(findings, colWidths=[55 * mm, 105 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)


def add_approach(story, styles):
    story.append(Paragraph("Approach: How ESN Uncertainty Works", styles["SectionHeader"]))

    story.append(Paragraph(
        "An Echo State Network (ESN) is a recurrent neural network with <b>fixed random weights</b>. "
        "Only a simple linear readout is trained. This makes it:",
        styles["BodyText2"],
    ))
    bullets = [
        "<b>Fast</b> — trains in under 1 second (closed-form solution, no epochs)",
        "<b>Lightweight</b> — 512 neurons, runs on CPU in real-time",
        "<b>Stable</b> — fixed reservoir means no overfitting, no catastrophic forgetting",
    ]
    for b in bullets:
        story.append(Paragraph(b, styles["BulletItem"], bulletText="•"))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Pipeline:", styles["SubHeader"]))
    story.append(Paragraph(
        "1. <b>Input</b>: current joint state + action command (15 dims for UR5, 14 for Franka)<br/>"
        "2. <b>ESN reservoir</b>: 512 random neurons process the input, maintaining temporal memory<br/>"
        "3. <b>Readout</b>: linear layer predicts the next joint state (ridge regression)<br/>"
        "4. <b>Sigma</b>: prediction error = ||predicted − actual|| = uncertainty<br/>"
        "5. <b>Low sigma</b> = familiar situation. <b>High sigma</b> = something unexpected.",
        styles["BodyText2"],
    ))


def add_phase0(story, styles):
    story.append(Paragraph("Phase 0: Does ESN Uncertainty Work?", styles["SectionHeader"]))

    story.append(Paragraph("Objective", styles["SubHeader"]))
    story.append(Paragraph(
        "Build the ESN pipeline from scratch and test whether prediction error (sigma) "
        "correlates with manipulation failures.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("Datasets", styles["SubHeader"]))
    ds_data = [
        ["Dataset", "Robot", "Episodes", "Success Rate", "State Dims"],
        ["Berkeley UR5", "UR5 6-DOF", "1,000", "100%", "8 (EE pose + gripper)"],
        ["DROID-100", "Franka Panda 7-DOF", "100", "81%", "7 (joint positions)"],
    ]
    t = Table(ds_data, colWidths=[35 * mm, 35 * mm, 25 * mm, 28 * mm, 40 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph(
        "The UR5 dataset was used to build and validate the pipeline (anomaly detection, ESN training, "
        "visualization). However, all 1,000 UR5 episodes are successful — no failures to detect. "
        "We then applied the pipeline to DROID-100, which has 19 failed episodes across 47 diverse "
        "manipulation tasks.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("Anomaly Detection", styles["SubHeader"]))
    story.append(Paragraph(
        "Four anomaly detectors scan each trajectory frame-by-frame:",
        styles["BodyText2"],
    ))
    anom_data = [
        ["Anomaly Type", "What It Detects", "Events Found"],
        ["Velocity spikes", "Sudden jerky end-effector movements", "367"],
        ["Acceleration spikes", "Abrupt speed changes (2nd derivative)", "197"],
        ["Command gaps", "Robot not following commanded actions", "—"],
        ["Gripper oscillation", "Repeated open/close (grasping struggle)", "—"],
    ]
    t = Table(anom_data, colWidths=[38 * mm, 80 * mm, 30 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    story.append(Paragraph("Key Phase 0 Finding", styles["SubHeader"]))
    story.append(Paragraph(
        "<b>Failed episodes have 1.35x higher average sigma than successful episodes.</b> "
        "The ESN, trained only on successful demonstrations, produces measurably higher "
        "prediction error when encountering failure trajectories — even though nobody told "
        "it what \"failure\" looks like.",
        styles["Finding"],
    ))

    sigma_data = [
        ["Metric", "Successful Episodes", "Failed Episodes", "Ratio"],
        ["Sigma mean", "0.219 ± 0.085", "0.296 ± 0.094", "1.35x"],
        ["Training RMSE", "0.260", "—", "—"],
        ["Training time", "0.95 seconds", "—", "—"],
        ["Training samples", "27,733", "—", "—"],
    ]
    t = Table(sigma_data, colWidths=[35 * mm, 42 * mm, 42 * mm, 25 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)


def add_phase1(story, styles, phase1_data):
    story.append(PageBreak())
    story.append(Paragraph("Phase 1: Deep Sigma Analysis", styles["SectionHeader"]))

    story.append(Paragraph(
        "Phase 0 established that sigma correlates with failure. Phase 1 asks three deeper questions: "
        "<b>which joints</b> drive the signal, <b>when</b> does it appear, and <b>how well</b> "
        "can it classify outcomes.",
        styles["BodyText2"],
    ))

    # Analysis 1
    story.append(Paragraph("Analysis 1: Per-Joint Prediction Error", styles["SubHeader"]))
    story.append(Paragraph(
        "Instead of a single sigma number (L2 norm across all 7 joints), we measured prediction "
        "error for each joint independently and compared between success and failure groups.",
        styles["BodyText2"],
    ))

    # Insert plot
    img_path = os.path.join(PHASE1_DIR, "per_joint_comparison.png")
    if os.path.exists(img_path):
        img = Image(img_path, width=155 * mm, height=75 * mm)
        story.append(img)
        story.append(Paragraph(
            "Figure 1: Per-joint prediction error comparison. Numbers above bars show the "
            "failure/success ratio. Joint 2 (upper arm) has the highest ratio at 1.88x.",
            styles["Caption"],
        ))

    story.append(Paragraph(
        f"<b>Finding:</b> {phase1_data['per_joint']['most_predictive']} (upper arm/elbow region) "
        f"is the most predictive joint with a {phase1_data['per_joint']['max_ratio']:.2f}x "
        "failure-to-success ratio. Upper arm joints (0, 1, 2) have higher ratios than wrist "
        "joints (5, 6). This means: <b>when the robot fails, it is the gross arm positioning "
        "that goes wrong, not the fine wrist adjustments.</b>",
        styles["Finding"],
    ))

    # Analysis 2
    story.append(Paragraph("Analysis 2: Temporal Sigma Profile", styles["SubHeader"]))
    story.append(Paragraph(
        "We normalized each episode to 0–100% completion, resampled sigma to 100 bins, "
        "and averaged across success and failure groups. This reveals <b>when</b> the ESN "
        "detects trouble relative to the episode timeline.",
        styles["BodyText2"],
    ))

    img_path = os.path.join(PHASE1_DIR, "temporal_sigma_profile.png")
    if os.path.exists(img_path):
        img = Image(img_path, width=155 * mm, height=75 * mm)
        story.append(img)
        story.append(Paragraph(
            "Figure 2: Average sigma over normalized episode time. Green = successful episodes, "
            "red = failed episodes. Orange dashed line marks the divergence point.",
            styles["Caption"],
        ))

    div = phase1_data["temporal"]["divergence_pct"]
    story.append(Paragraph(
        f"<b>Finding:</b> Failure sigma diverges from success sigma at <b>{div:.0f}%</b> of "
        f"episode completion. This means the ESN can detect trouble when the robot is only "
        f"{div:.0f}% through the task — <b>{100-div:.0f}% of the episode remains as warning "
        f"time</b> for corrective action.",
        styles["Finding"],
    ))

    # Analysis 3
    story.append(Paragraph("Analysis 3: Success/Failure Classification", styles["SubHeader"]))
    story.append(Paragraph(
        "We used sigma_mean as a single-feature binary classifier to test whether "
        "prediction error alone can distinguish successful from failed episodes.",
        styles["BodyText2"],
    ))

    img_path = os.path.join(PHASE1_DIR, "roc_curve.png")
    if os.path.exists(img_path):
        img = Image(img_path, width=95 * mm, height=95 * mm)
        story.append(img)
        story.append(Paragraph(
            "Figure 3: ROC curve for success/failure classification using sigma_mean. "
            "AUC = 0.701 (moderate classifier, clearly above random baseline of 0.5).",
            styles["Caption"],
        ))

    c = phase1_data["classification"]
    class_data = [
        ["Metric", "Value"],
        ["AUC (Area Under Curve)", f"{c['auc']:.3f}"],
        ["Optimal threshold", f"sigma_mean ≥ {c['optimal_threshold']:.4f}"],
        ["Precision", f"{c['precision']:.2f}"],
        ["Recall", f"{c['recall']:.2f} (catches {c['recall']*100:.0f}% of failures)"],
        ["F1 Score", f"{c['f1']:.2f}"],
        ["Accuracy", f"{c['accuracy']:.2f}"],
    ]
    t = Table(class_data, colWidths=[55 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    story.append(Paragraph(
        "<b>Finding:</b> A single feature (sigma_mean) achieves AUC = 0.701 — a moderate "
        "classifier that is clearly better than random. Combining per-joint sigmas, temporal "
        "features, and caution zone counts into a multi-feature classifier should significantly "
        "improve this.",
        styles["Finding"],
    ))


def add_next_steps(story, styles):
    story.append(Paragraph("Next Steps", styles["SectionHeader"]))

    steps = [
        ("<b>Phase 2: Multi-feature classifier</b> — Combine per-joint sigmas, temporal "
         "features, and caution counts. Expected AUC improvement to 0.80+."),
        ("<b>Phase 2: Sigma-conditioned actions</b> — Use sigma to modulate robot "
         "behavior in real-time: high sigma → slow down, very high → try alternative approach."),
        ("<b>Phase 3: Scale to full DROID</b> — Train on 95,000 episodes (currently 81). "
         "More data → stronger ESN → better uncertainty estimates."),
        ("<b>Phase 3: Continual learning</b> — Keep ESN reservoir frozen, update only "
         "the readout online. Robot improves from experience without catastrophic forgetting."),
        ("<b>Integration</b> — Deploy ESN uncertainty as a real-time safety layer on "
         "physical robot arms (UR5, Franka) alongside the VLA model."),
    ]
    for s in steps:
        story.append(Paragraph(s, styles["BulletItem"], bulletText="→"))


def add_tools(story, styles):
    story.append(Paragraph("Tools &amp; Infrastructure", styles["SectionHeader"]))
    tools_data = [
        ["Component", "Technology", "Purpose"],
        ["ESN Core", "NumPy + SciPy", "512-neuron reservoir, ridge regression readout"],
        ["Dataset", "LeRobot (HuggingFace)", "UR5 + DROID-100 manipulation trajectories"],
        ["Visualization", "Rerun.io v0.26", "Interactive trajectory + sigma timeline viewer"],
        ["Plotting", "Matplotlib", "Publication-quality charts"],
        ["Runtime", "Python 3.10, CPU only", "No GPU required"],
    ]
    t = Table(tools_data, colWidths=[30 * mm, 40 * mm, 85 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORT_DIR, "esn_uncertainty_report.pdf")

    # Load Phase 1 results
    phase1_path = os.path.join(PHASE1_DIR, "phase1_results.json")
    with open(phase1_path) as f:
        phase1_data = json.load(f)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = build_styles()
    story = []

    add_cover(story, styles)
    add_executive_summary(story, styles, phase1_data)
    add_approach(story, styles)
    add_phase0(story, styles)
    add_phase1(story, styles, phase1_data)
    add_next_steps(story, styles)
    add_tools(story, styles)

    doc.build(story)
    print(f"Report saved: {pdf_path}")


if __name__ == "__main__":
    main()
