#!/usr/bin/env python3
"""Generate teammate replication guide PDF."""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, Preformatted,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "output")
REPORTS = os.path.join(OUT, "reports")
TASK_DIR = os.path.join(OUT, "task_specific")

# Shorthand colors
DARK = "#1a1a2e"
BLUE = "#3498db"
GREEN = "#27ae60"
RED = "#e74c3c"
ORANGE = "#f39c12"
GRAY = "#7f8c8d"
BG = "#f8f9fa"


def styles():
    s = getSampleStyleSheet()
    def mk(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=s[parent], **kw)
    return {
        "title": mk("T", "Title", fontSize=28, leading=34, spaceAfter=6*mm, textColor=colors.HexColor(DARK)),
        "sub": mk("S", fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#4a4a6a"), spaceAfter=4*mm),
        "h1": mk("H1", "Heading1", fontSize=17, leading=22, spaceBefore=10*mm, spaceAfter=4*mm, textColor=colors.HexColor(DARK)),
        "h2": mk("H2", "Heading2", fontSize=13, leading=17, spaceBefore=6*mm, spaceAfter=3*mm, textColor=colors.HexColor("#2d2d4e")),
        "body": mk("B", fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=3*mm),
        "bullet": mk("BU", fontSize=10.5, leading=15, leftIndent=12*mm, bulletIndent=6*mm, spaceAfter=2*mm),
        "code": mk("C", "Code", fontSize=8.5, leading=12, leftIndent=6*mm, rightIndent=6*mm,
                    backColor=colors.HexColor("#1e1e2e"), textColor=colors.HexColor("#e0e0e0"),
                    borderWidth=0.5, borderColor=colors.HexColor("#444"), borderPadding=8,
                    fontName="Courier", spaceAfter=4*mm, spaceBefore=2*mm),
        "caption": mk("CA", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor(GRAY),
                       spaceAfter=6*mm, spaceBefore=2*mm),
        "tip": mk("TIP", fontSize=10.5, leading=15, leftIndent=8*mm, borderWidth=1,
                   borderColor=colors.HexColor(BLUE), borderPadding=6,
                   backColor=colors.HexColor("#eef6ff"), spaceAfter=4*mm),
        "warn": mk("WARN", fontSize=10.5, leading=15, leftIndent=8*mm, borderWidth=1,
                    borderColor=colors.HexColor(ORANGE), borderPadding=6,
                    backColor=colors.HexColor("#fff8ee"), spaceAfter=4*mm),
        "finding": mk("F", fontSize=10.5, leading=15, leftIndent=8*mm, borderWidth=1,
                       borderColor=colors.HexColor(GREEN), borderPadding=6,
                       backColor=colors.HexColor("#f0faf0"), spaceAfter=4*mm),
    }


def tbl(data, widths, hdr="#1a1a2e"):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(hdr)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(BG)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def code(text, st):
    """Code block as a paragraph."""
    return Paragraph(f"<font face='Courier' size=8.5>{text}</font>", st["code"])


def img(name, w=160, h=70):
    """Load image if exists."""
    for base in [REPORTS, TASK_DIR, os.path.join(TASK_DIR, "video_frames")]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return Image(p, width=w*mm, height=h*mm)
    return None


def main():
    os.makedirs(REPORTS, exist_ok=True)
    pdf_path = os.path.join(REPORTS, "esn_replication_guide.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    st = styles()
    story = []

    # ═══════════════════════════════════════════════════════════════
    # COVER
    # ═══════════════════════════════════════════════════════════════
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("ESN Uncertainty Pipeline", st["title"]))
    story.append(Paragraph("Replication Guide", st["title"]))
    story.append(HRFlowable(width="50%", thickness=1.5, color=colors.HexColor(BLUE), spaceAfter=8*mm))
    story.append(Paragraph(
        "A complete step-by-step guide to replicate the ESN uncertainty analysis<br/>"
        "and adapt it for any robotic arm manipulation task.",
        st["sub"],
    ))
    story.append(Spacer(1, 15*mm))
    meta = [
        ["Author", "Umar Bin Muzzafar"],
        ["Organization", "OMNIBIO LTD"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
        ["Python", "3.10 (required for LeRobot)"],
        ["GPU Required", "No — runs on CPU"],
        ["Training Time", "< 2 seconds"],
    ]
    mt = Table(meta, colWidths=[40*mm, 110*mm])
    mt.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4a4a6a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(mt)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("Contents", st["h1"]))
    toc = [
        "1. What Is an ESN? (Architecture Explained)",
        "2. Project Structure",
        "3. Prerequisites &amp; Setup",
        "4. Step-by-Step: Run the Full Pipeline",
        "5. How to Adapt for a New Dataset / Task",
        "6. Understanding the Output",
        "7. Key Parameters to Tune",
        "8. What We Found (Summary of Results)",
    ]
    for item in toc:
        story.append(Paragraph(item, st["body"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 1. WHAT IS AN ESN?
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("1. What Is an ESN? (Architecture Explained)", st["h1"]))

    story.append(Paragraph(
        "An Echo State Network (ESN) is a type of recurrent neural network. Unlike normal "
        "neural networks, the recurrent connections are <b>random and fixed</b> — they never "
        "get trained. Only a simple linear layer (the readout) is trained. This makes ESNs:",
        st["body"],
    ))
    for b in [
        "<b>Extremely fast</b> — training is a single matrix solve, not thousands of epochs",
        "<b>CPU-only</b> — no GPU needed, runs in under 2 seconds",
        "<b>Stable</b> — no vanishing gradients, no catastrophic forgetting",
        "<b>Good at temporal data</b> — the reservoir maintains memory of past inputs",
    ]:
        story.append(Paragraph(b, st["bullet"], bulletText="•"))

    i = img("esn_architecture.png", 165, 70)
    if i:
        story.append(i)
        story.append(Paragraph("Figure 1: ESN pipeline architecture", st["caption"]))

    story.append(Paragraph("How It Works — Step by Step", st["h2"]))
    steps = [
        ("<b>Input (14 dims)</b>: At each timestep, we concatenate the robot's current "
         "joint state (7 values) + the action command (7 values) = 14 numbers."),
        ("<b>W_in projection (512 x 14)</b>: A random matrix projects the 14-dim input "
         "into a 512-dimensional reservoir space."),
        ("<b>Reservoir update</b>: The 512 neurons update their state using: "
         "<font face='Courier'>x(t) = (1-α)·x(t-1) + α·tanh(W_in·u + W·x(t-1))</font> "
         "where W is a sparse random matrix and α is the leaking rate. "
         "This equation means each neuron mixes its previous state with new input — creating memory."),
        ("<b>W_out prediction (7 x 513)</b>: A trained linear layer maps the 512-dim "
         "reservoir state to a 7-dim predicted next state. The +1 is a bias term."),
        ("<b>Sigma</b>: We compare the prediction to what actually happened: "
         "<font face='Courier'>σ = ||predicted - actual||</font>. "
         "Low σ = familiar. High σ = unexpected = uncertain."),
    ]
    for i_step, s in enumerate(steps, 1):
        story.append(Paragraph(f"{i_step}. {s}", st["body"]))

    story.append(Paragraph("The Spectral Radius", st["h2"]))
    story.append(Paragraph(
        "The spectral radius (SR) of W controls how long the reservoir remembers past inputs. "
        "SR=0.95 means echoes of past inputs persist for many timesteps — good for predicting "
        "robot trajectories. If SR were &gt; 1.0, the network would be unstable (states explode). "
        "If SR &lt; 0.5, memory would be too short.",
        st["body"],
    ))

    story.append(Paragraph("Ridge Regression (Training)", st["h2"]))
    story.append(Paragraph(
        "Training the readout is a single equation, not gradient descent:",
        st["body"],
    ))
    story.append(code(
        "W_out = (S<sup>T</sup>·S + α·I)<sup>-1</sup> · S<sup>T</sup>·Y<br/><br/>"
        "S = matrix of reservoir states (one row per timestep)<br/>"
        "Y = matrix of actual next states (targets)<br/>"
        "α = regularization (prevents overfitting, default 0.01)",
        st,
    ))
    story.append(Paragraph(
        "This solves instantly. No learning rate, no epochs, no batch size. "
        "Just one matrix inversion.",
        st["body"],
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 2. PROJECT STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Project Structure", st["h1"]))

    i = img("project_structure.png", 165, 95)
    if i:
        story.append(i)
        story.append(Paragraph("Figure 2: Full project file tree with descriptions", st["caption"]))

    story.append(tbl([
        ["Directory", "Purpose"],
        ["config.py", "All parameters: ESN size, thresholds, paths. Change these first."],
        ["data/", "Dataset loading. loader.py for UR5, droid_loader.py for DROID-100."],
        ["core/", "The brain: esn.py (reservoir), readout.py (ridge regression), uncertainty.py (sigma)"],
        ["analysis/", "Anomaly detection + Phase 1 analyses + task-specific learning curve"],
        ["viz/", "Rerun.io visualization (interactive trajectory + sigma timeline viewer)"],
        ["scripts/", "Numbered scripts you run in order. Each one does one job."],
        ["pipeline.py", "Wires core/ together: train on episodes, evaluate sigma."],
        ["output/", "All generated results: models, plots, JSON, Rerun recordings."],
    ], [30*mm, 130*mm]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 3. PREREQUISITES
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Prerequisites &amp; Setup", st["h1"]))

    story.append(Paragraph("System Requirements", st["h2"]))
    story.append(tbl([
        ["Requirement", "Details"],
        ["Python", "3.10 (required by LeRobot). Check: python3.10 --version"],
        ["OS", "Linux (tested on Ubuntu 22.04). macOS should work."],
        ["RAM", "4 GB minimum (dataset is ~100MB)"],
        ["GPU", "NOT required. Everything runs on CPU."],
        ["Disk", "~2 GB for dataset cache + outputs"],
    ], [35*mm, 125*mm], "#34495e"))

    story.append(Paragraph("Installation", st["h2"]))
    story.append(code(
        "# 1. Clone the repository<br/>"
        "git clone https://github.com/OMNIBIO-LTD/Active-inference.git<br/>"
        "cd Active-inference<br/>"
        "git checkout Umz_Work<br/><br/>"
        "# 2. Install dependencies<br/>"
        "pip install -r omni3/requirements.txt<br/><br/>"
        "# 3. Verify installation<br/>"
        "python3.10 -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('OK')\"<br/>"
        "python3.10 -c \"import rerun; print('Rerun', rerun.__version__)\"<br/>"
        "python3.10 -c \"import scipy; print('SciPy', scipy.__version__)\"",
        st,
    ))

    story.append(Paragraph(
        "<b>The dataset downloads automatically</b> from HuggingFace on first run. "
        "No manual download needed. It's cached in ~/.cache/huggingface/ after the first run.",
        st["tip"],
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 4. STEP BY STEP
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Step-by-Step: Run the Full Pipeline", st["h1"]))

    story.append(Paragraph(
        "Run these commands from the repository root (where omni3/ directory is). "
        "Each script is standalone and produces output in omni3/output/.",
        st["body"],
    ))

    # Step 0
    story.append(Paragraph("Step 0: Explore the Dataset", st["h2"]))
    story.append(Paragraph(
        "See what's in the UR5 dataset — episodes, tasks, state/action dimensions.",
        st["body"],
    ))
    story.append(code(
        "python3.10 omni3/scripts/00_explore.py<br/><br/>"
        "# For only first 20 episodes (faster):<br/>"
        "python3.10 omni3/scripts/00_explore.py --episodes 20",
        st,
    ))
    story.append(Paragraph(
        "This prints: number of episodes, frame lengths, state/action ranges, task list, "
        "success rate. Takes ~7 seconds for 20 episodes.",
        st["body"],
    ))

    # Step 1
    story.append(Paragraph("Step 1: Detect Anomalies", st["h2"]))
    story.append(code(
        "python3.10 omni3/scripts/01_filter.py --episodes 100",
        st,
    ))
    story.append(Paragraph(
        "Scans 100 episodes for velocity spikes, acceleration spikes, command gaps, "
        "and gripper oscillation. Outputs: omni3/output/filtered_episodes.json",
        st["body"],
    ))

    # Step 2
    story.append(Paragraph("Step 2: Train ESN", st["h2"]))
    story.append(code(
        "python3.10 omni3/scripts/02_train_esn.py --train-episodes 100",
        st,
    ))
    story.append(Paragraph(
        "Trains the ESN readout on 100 episodes using ridge regression. "
        "Takes &lt; 1 second. Saves model to omni3/output/models/esn_readout.npz. "
        "Expected RMSE: ~0.025 for UR5.",
        st["body"],
    ))

    # Step 3
    story.append(Paragraph("Step 3: Validate Sigma", st["h2"]))
    story.append(code(
        "python3.10 omni3/scripts/03_validate.py",
        st,
    ))
    story.append(Paragraph(
        "Evaluates sigma on flagged episodes. Checks whether sigma spikes before anomalies.",
        st["body"],
    ))

    # Step 4
    story.append(Paragraph("Step 4: Visualize in Rerun", st["h2"]))
    story.append(code(
        "# Save to file (can share with others):<br/>"
        "python3.10 omni3/scripts/04_visualize.py --episodes 0 1 2 --save-rrd omni3/output/viz.rrd<br/><br/>"
        "# View the recording:<br/>"
        "rerun omni3/output/viz.rrd",
        st,
    ))
    story.append(Paragraph(
        "Opens an interactive Rerun viewer with: EE trajectory (3D), state time series, "
        "sigma timeline, anomaly markers, and per-dimension prediction error.",
        st["body"],
    ))

    # Step 5
    story.append(Paragraph("Step 5: Success vs Failure (DROID-100)", st["h2"]))
    story.append(code(
        "python3.10 omni3/scripts/06_droid_failures.py --save-rrd omni3/output/droid.rrd",
        st,
    ))
    story.append(Paragraph(
        "Loads DROID-100 (Franka Panda, 81 success + 19 failures). Trains ESN on successes, "
        "measures sigma on failures. Compares sigma between groups.",
        st["body"],
    ))

    # Step 6
    story.append(Paragraph("Step 6: Deep Analysis (Phase 1)", st["h2"]))
    story.append(code(
        "python3.10 omni3/scripts/07_phase1_analysis.py",
        st,
    ))
    story.append(Paragraph(
        "Three analyses: (1) per-joint sigma — which joints predict failure, "
        "(2) temporal profile — when does sigma diverge, "
        "(3) ROC/AUC classification. Plots saved to omni3/output/phase1/.",
        st["body"],
    ))

    # Step 7
    story.append(Paragraph("Step 7: Task-Specific + Learning Curve", st["h2"]))
    story.append(code(
        "python3.10 omni3/scripts/08_task_specific.py",
        st,
    ))
    story.append(Paragraph(
        "<b>The most important script.</b> Focuses on one task (Task 5, 35 success + 19 failure). "
        "Runs the incremental learning curve: trains on 1, 2, ..., 30 episodes and shows "
        "sigma decreasing. Proves the ESN builds a generative model. "
        "Plots saved to omni3/output/task_specific/.",
        st["finding"],
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 5. HOW TO ADAPT FOR A NEW DATASET
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("5. How to Adapt for a New Dataset / Task", st["h1"]))

    story.append(Paragraph(
        "The pipeline works with <b>any LeRobot-format dataset</b>. Here's how to plug in "
        "a new one:",
        st["body"],
    ))

    story.append(Paragraph("Step A: Find a Dataset", st["h2"]))
    story.append(Paragraph(
        "Browse HuggingFace for LeRobot datasets: "
        "https://huggingface.co/datasets?search=lerobot",
        st["body"],
    ))
    story.append(tbl([
        ["Dataset", "Robot", "Episodes", "Has Failures?"],
        ["lerobot/berkeley_autolab_ur5", "UR5", "1,000", "No"],
        ["lerobot/droid_100", "Franka", "100", "Yes (19)"],
        ["lerobot/berkeley_fanuc_manipulation", "Fanuc", "415", "Unknown"],
        ["lerobot/roboturk", "Franka", "1,995", "Unknown"],
        ["lerobot/stanford_kuka_multimodal_dataset", "KUKA", "3,000", "Unknown"],
        ["IPEC-COMMUNITY/kuka_lerobot", "KUKA IIWA", "209,880", "Unknown"],
    ], [60*mm, 25*mm, 25*mm, 35*mm], "#34495e"))

    story.append(Paragraph("Step B: Inspect the Dataset", st["h2"]))
    story.append(code(
        "python3.10 -c \"<br/>"
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset<br/>"
        "ds = LeRobotDataset('YOUR_REPO_ID', episodes=[0], download_videos=False)<br/>"
        "hf = ds.hf_dataset<br/>"
        "sample = hf[0]<br/>"
        "print('Features:', list(ds.features.keys()))<br/>"
        "print('State shape:', sample['observation.state'].shape)<br/>"
        "print('Action shape:', sample['action'].shape)<br/>"
        "print('Episodes:', ds.num_episodes)<br/>"
        "\"",
        st,
    ))

    story.append(Paragraph("Step C: Update config.py", st["h2"]))
    story.append(Paragraph(
        "Edit <font face='Courier'>omni3/config.py</font> with your dataset's parameters:",
        st["body"],
    ))
    story.append(code(
        "DATASET_REPO_ID = 'your/dataset_id'<br/>"
        "DATASET_FPS = 15  # your dataset's FPS<br/><br/>"
        "STATE_DIM = 7   # your observation.state dimension<br/>"
        "ACTION_DIM = 7  # your action dimension<br/>"
        "ESN_INPUT_DIM = STATE_DIM + ACTION_DIM  # auto-computed<br/>"
        "READOUT_OUTPUT_DIM = STATE_DIM  # predict next state<br/><br/>"
        "STATE_NAMES = ['j0', 'j1', ...]  # your joint names<br/>"
        "ACTION_NAMES = ['a0', 'a1', ...]  # your action names",
        st,
    ))

    story.append(Paragraph("Step D: Write a Loader (if needed)", st["h2"]))
    story.append(Paragraph(
        "If your dataset has the same structure as UR5 (observation.state, action, "
        "next.reward, next.done), scripts 00-04 work directly — just change config.py. "
        "If the field names differ, create a new loader in data/ following the pattern "
        "in droid_loader.py. The key is to output a list of dicts with:",
        st["body"],
    ))
    story.append(code(
        "episode = {<br/>"
        "    'index': int,<br/>"
        "    'states': np.ndarray,    # (T, STATE_DIM)<br/>"
        "    'actions': np.ndarray,   # (T, ACTION_DIM)<br/>"
        "    'timestamps': np.ndarray, # (T,)<br/>"
        "    'success': bool,<br/>"
        "    'task': str,<br/>"
        "    'num_frames': int,<br/>"
        "}",
        st,
    ))

    story.append(Paragraph("Step E: Run the Pipeline", st["h2"]))
    story.append(Paragraph(
        "That's it. Run scripts 00-08 in order. The ESN, readout, uncertainty, and analysis "
        "code are dataset-agnostic — they work with any (state, action) pair of any dimension.",
        st["body"],
    ))

    story.append(Paragraph(
        "<b>The ESN_INPUT_DIM and READOUT_OUTPUT_DIM in config.py are the only things you "
        "must change</b> to use a different robot. Everything else adapts automatically.",
        st["tip"],
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 6. UNDERSTANDING OUTPUT
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("6. Understanding the Output", st["h1"]))

    story.append(Paragraph("The Learning Curve (Most Important)", st["h2"]))
    i = img("learning_curve.png", 160, 58)
    if i:
        story.append(i)
        story.append(Paragraph(
            "Figure 3: Validation sigma drops 94% as training data grows (left). "
            "Training RMSE stabilizes after ~15 episodes (right).",
            st["caption"],
        ))
    story.append(Paragraph(
        "<b>Left plot</b>: X-axis = number of training episodes. Y-axis = prediction error "
        "on held-out validation episodes. The curve should go DOWN — this proves the ESN "
        "learns from repeated visits. A flat curve means the ESN isn't learning.<br/><br/>"
        "<b>Right plot</b>: Training RMSE. Goes UP because more data means more variety, "
        "but the model generalizes better (validation sigma goes down).",
        st["body"],
    ))

    story.append(Paragraph("Per-Joint Comparison", st["h2"]))
    i = img("task5_per_joint_comparison.png", 155, 65)
    if i:
        story.append(i)
        story.append(Paragraph(
            "Figure 4: Per-joint prediction error. Numbers above bars = failure/success ratio.",
            st["caption"],
        ))
    story.append(Paragraph(
        "Green bars = average prediction error for successful episodes. "
        "Red bars = average for failed episodes. The ratio (number above) tells you "
        "which joint is most predictive of failure. Higher ratio = that joint deviates "
        "more during failures.",
        st["body"],
    ))

    story.append(Paragraph("ROC Curve", st["h2"]))
    i = img("task5_roc_curve.png", 85, 85)
    if i:
        story.append(i)
        story.append(Paragraph(
            "Figure 5: ROC curve for sigma-based failure classification.",
            st["caption"],
        ))
    story.append(Paragraph(
        "<b>AUC</b> (Area Under Curve): 0.5 = random, 0.7 = moderate, 0.8 = strong, 1.0 = perfect. "
        "Our task-specific AUC is 0.785 using a single feature (sigma_mean). "
        "The curve shows the tradeoff between catching failures (recall) and false alarms (FPR).",
        st["body"],
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 7. PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("7. Key Parameters to Tune", st["h1"]))

    story.append(tbl([
        ["Parameter", "Default", "What It Does", "When to Change"],
        ["RESERVOIR_SIZE", "512", "Number of neurons. More = more capacity.", "Increase for complex tasks (1024, 2048)"],
        ["SPECTRAL_RADIUS", "0.95", "Memory length. Higher = longer memory.", "Lower (0.8) for fast tasks, higher (0.99) for slow"],
        ["LEAKING_RATE", "0.3", "How fast old state decays. 0=keep all, 1=forget all.", "Lower (0.1) for slow robots, higher (0.5) for fast"],
        ["INPUT_SCALING", "0.3", "Scale of input projection.", "Increase if inputs have large range"],
        ["RIDGE_ALPHA", "0.01", "Regularization. Higher = less overfitting.", "Increase if RMSE is good but sigma is noisy"],
        ["ESN_WARMUP_STEPS", "10", "Discard first N frames (transient).", "Increase for slower FPS datasets"],
        ["SIGMA_WINDOW_SIZE", "50", "Running average window for z-score.", "Decrease for shorter episodes"],
    ], [33*mm, 14*mm, 45*mm, 60*mm], "#34495e"))

    story.append(Paragraph(
        "<b>Start with defaults.</b> Only tune if results are poor. The most impactful "
        "parameter is RESERVOIR_SIZE — try 1024 if 512 gives weak sigma separation.",
        st["tip"],
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # 8. SUMMARY
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("8. What We Found (Summary)", st["h1"]))

    story.append(tbl([
        ["Finding", "Result", "Significance"],
        ["Learning curve", "94% sigma decrease over 30 episodes", "ESN builds a generative model"],
        ["Sigma ratio (F/S)", "1.54x (task-specific)", "Failures are distinguishable"],
        ["Classification AUC", "0.785", "Moderate classifier from 1 feature"],
        ["Most predictive joint", "joint_2 (2.64x)", "Upper arm fails, not wrist"],
        ["Early warning", "27% completion", "73% warning time remaining"],
        ["Training time", "< 1 second", "Real-time capable"],
    ], [38*mm, 50*mm, 62*mm], GREEN))

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        "The ESN uncertainty pipeline is validated: it learns task dynamics, detects failures "
        "early, and runs in real-time on CPU. Next steps are BCM weight adaptation and "
        "Curious/Cautious Pi policies for active robot behavior under uncertainty.",
        st["finding"],
    ))

    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(
        "<b>Quick Start (TL;DR)</b>: Install deps, then run "
        "<font face='Courier'>python3.10 omni3/scripts/08_task_specific.py</font> "
        "to see everything.",
        st["warn"],
    ))

    doc.build(story)
    print(f"Guide saved: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
