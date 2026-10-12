"""
create_app_screenshot.py
Generates a realistic high-resolution UI screenshot mockup of the Streamlit web application.
Uses clean typography, styled badges, and aligned layouts with zero font warnings.
Saves to screenshots/app.png.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_screenshot():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    screenshots_dir = os.path.join(base_dir, 'screenshots')
    os.makedirs(screenshots_dir, exist_ok=True)
    out_path = os.path.join(screenshots_dir, 'app.png')

    fig = plt.figure(figsize=(16, 10), facecolor='#f8fafc', dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.axis('off')

    # 1. Sidebar Background
    sidebar = patches.Rectangle((0, 0), 330, 1000, facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1.5)
    ax.add_patch(sidebar)

    # Sidebar Header
    ax.text(30, 945, "PROJECT DETAILS", fontsize=13, fontweight='bold', color='#1e3a8a')
    ax.text(30, 915, "Domain: Energy Prediction", fontsize=11, color='#334155', fontweight='bold')
    ax.text(30, 890, "Target: High Peak Alert vs Normal", fontsize=10, color='#475569')
    ax.text(30, 868, "Outcome: CO5 - ML Evaluation", fontsize=10, color='#475569')

    # Model Badge
    badge = patches.FancyBboxPatch((25, 805), 280, 42, boxstyle="round,pad=0.2", facecolor='#dcfce7', edgecolor='#86efac', linewidth=1.2)
    ax.add_patch(badge)
    ax.text(40, 820, "ACTIVE MODEL: Logistic Regression", fontsize=9.5, fontweight='bold', color='#166534')

    ax.plot([30, 305], [780, 780], color='#e2e8f0', linewidth=1.2)

    # Analytics Section
    ax.text(30, 745, "VISUAL ANALYTICS", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.text(30, 720, "Select Evaluation Plot:", fontsize=10, color='#64748b')
    dropdown = patches.FancyBboxPatch((25, 675), 280, 36, boxstyle="round,pad=0.2", facecolor='#f1f5f9', edgecolor='#cbd5e1', linewidth=1)
    ax.add_patch(dropdown)
    ax.text(38, 687, "Model Comparison Bar Chart   [v]", fontsize=10, color='#1e293b')

    # Thumbnail Plot Box
    thumb_box = patches.FancyBboxPatch((25, 330), 280, 325, boxstyle="round,pad=0.2", facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1)
    ax.add_patch(thumb_box)
    ax.text(35, 625, "Active Plot Preview:", fontsize=10, fontweight='bold', color='#475569')
    
    plot_file = os.path.join(base_dir, 'plots', 'model_comparison.png')
    if os.path.exists(plot_file):
        img = Image.open(plot_file)
        ax.imshow(img, extent=[35, 295, 345, 610], aspect='auto', zorder=2)

    # 2. Main Title Banner
    ax.text(370, 948, "Smart Energy Consumption & Peak Demand Predictor", fontsize=21, fontweight='bold', color='#1e3a8a')
    ax.text(370, 922, "Machine Learning Decision Support System for Real-Time Building Energy Auditing & Demand-Response", fontsize=11, color='#475569')

    # 4 Top KPI Cards
    kpis = [
        ("Dataset Size", "1,500", "Building Observations", 370),
        ("Cross-Validation", "5-Fold", "Stratified CV Splits", 665),
        ("Model Accuracy", "96.67%", "Holdout Test Set", 960),
        ("ROC-AUC Score", "0.990", "Peak Demand Discrimination", 1255)
    ]
    for title, val, sub, x_pos in kpis:
        card = patches.FancyBboxPatch((x_pos, 820), 265, 80, boxstyle="round,pad=0.3", facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1.5)
        ax.add_patch(card)
        ax.text(x_pos + 18, 874, title, fontsize=10.5, color='#64748b', fontweight='bold')
        ax.text(x_pos + 18, 846, val, fontsize=18, color='#1e3a8a', fontweight='bold')
        ax.text(x_pos + 18, 830, sub, fontsize=9.5, color='#94a3b8')

    # Section Heading
    ax.text(370, 775, "Input Building Operational Parameters", fontsize=15, fontweight='bold', color='#0f172a')
    ax.text(370, 755, "Adjust environmental, structural, and load parameters to evaluate whether energy consumption exceeds peak thresholds.", fontsize=10.5, color='#64748b')

    # 3 Input Column Cards
    # Col 1: Structural & Weather
    col1 = patches.FancyBboxPatch((370, 440), 265, 295, boxstyle="round,pad=0.3", facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1.2)
    ax.add_patch(col1)
    ax.text(388, 705, "Structural & Weather", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.text(388, 672, "Square Footage: 2,400 sq ft", fontsize=10, color='#334155')
    ax.plot([388, 615], [658, 658], color='#3b82f6', linewidth=4)
    ax.text(388, 616, "Occupancy: 4 Persons", fontsize=10, color='#334155')
    ax.plot([388, 615], [602, 602], color='#3b82f6', linewidth=4)
    ax.text(388, 560, "Outdoor Temperature: 29.0 C", fontsize=10, color='#334155')
    ax.plot([388, 615], [546, 546], color='#f97316', linewidth=4)
    ax.text(388, 504, "Relative Humidity: 55.0 %", fontsize=10, color='#334155')
    ax.plot([388, 615], [490, 490], color='#3b82f6', linewidth=4)

    # Col 2: Loads & HVAC
    col2 = patches.FancyBboxPatch((665, 440), 265, 295, boxstyle="round,pad=0.3", facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1.2)
    ax.add_patch(col2)
    ax.text(683, 705, "Loads & HVAC", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.text(683, 672, "HVAC Level: High", fontsize=10, fontweight='bold', color='#dc2626')
    ax.text(683, 616, "Appliance Load: 5.20 kW", fontsize=10, color='#334155')
    ax.plot([683, 910], [602, 602], color='#ef4444', linewidth=4)
    ax.text(683, 560, "Lighting Load: 1.50 kW", fontsize=10, color='#334155')
    ax.plot([683, 910], [546, 546], color='#3b82f6', linewidth=4)
    ax.text(683, 504, "Solar Offset: 1.50 kW", fontsize=10, color='#334155')
    ax.plot([683, 910], [490, 490], color='#22c55e', linewidth=4)

    # Col 3: Temporal Factors
    col3 = patches.FancyBboxPatch((960, 440), 265, 295, boxstyle="round,pad=0.3", facecolor='#ffffff', edgecolor='#e2e8f0', linewidth=1.2)
    ax.add_patch(col3)
    ax.text(978, 705, "Temporal Context", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.text(978, 672, "Time of Day: Evening", fontsize=10, fontweight='bold', color='#d97706')
    ax.text(978, 616, "Day Type: Weekday", fontsize=10, color='#334155')

    # Predict Button
    btn = patches.FancyBboxPatch((978, 480), 230, 52, boxstyle="round,pad=0.3", facecolor='#1e3a8a', edgecolor='#172554', linewidth=1.5)
    ax.add_patch(btn)
    ax.text(996, 498, "PREDICT DEMAND STATUS", fontsize=10.5, fontweight='bold', color='#ffffff')

    # Results Section Header
    ax.text(370, 395, "Prediction Results & Operational Audit", fontsize=15, fontweight='bold', color='#0f172a')

    # Result Card 1: Alert Box (Left)
    alert_box = patches.FancyBboxPatch((370, 40), 560, 335, boxstyle="round,pad=0.3", facecolor='#fef2f2', edgecolor='#ef4444', linewidth=2)
    ax.add_patch(alert_box)
    ax.text(395, 335, "HIGH PEAK DEMAND ALERT", fontsize=16, fontweight='bold', color='#991b1b')
    ax.text(395, 305, "Status: Net load predicted to exceed peak capacity (>= 22.0 kW)", fontsize=10.5, color='#b91c1c')
    ax.text(395, 260, "High Demand Probability: 84.7%", fontsize=15, fontweight='bold', color='#dc2626')
    
    # Progress Bar
    ax.plot([395, 890], [228, 228], color='#fee2e2', linewidth=12, solid_capstyle='round')
    ax.plot([395, 815], [228, 228], color='#dc2626', linewidth=12, solid_capstyle='round')

    ax.text(395, 185, "Immediate automated demand-response or load-shedding is", fontsize=10.5, color='#7f1d1d')
    ax.text(395, 162, "recommended to avoid punitive utility peak-tariff penalties.", fontsize=10.5, color='#7f1d1d')

    status_tag = patches.FancyBboxPatch((395, 80), 220, 40, boxstyle="round,pad=0.2", facecolor='#dc2626', edgecolor='#b91c1c', linewidth=1)
    ax.add_patch(status_tag)
    ax.text(415, 94, "ACTION REQUIRED", fontsize=11, fontweight='bold', color='#ffffff')

    # Result Card 2: Recommendations Box (Right)
    rec_box = patches.FancyBboxPatch((950, 40), 610, 335, boxstyle="round,pad=0.3", facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.5)
    ax.add_patch(rec_box)
    ax.text(975, 335, "Actionable Energy Optimization Recommendations", fontsize=13, fontweight='bold', color='#0f172a')
    
    tips = [
        "HVAC Optimization: Thermostat setback (+1.5C to +2.0C) lowers draw by up to 20-25%.",
        "Peak Shifting: Shift heavy appliance cycles (dishwashers/laundry) to off-peak night hours.",
        "Appliance Management: Heavy active load (5.2 kW) detected. Stagger high-draw equipment.",
        "Solar Offset: Maximize daytime self-consumption before sunset to reduce grid dependency.",
        "Lighting Efficiency: Utilize natural daylighting to reduce illumination draw below 1.0 kW."
    ]
    y_text = 290
    for tip in tips:
        ax.text(975, y_text, "- " + tip, fontsize=9.0, color='#334155', wrap=True)
        y_text -= 46

    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[SUCCESS] Clean app screenshot saved at: {out_path}")

if __name__ == '__main__':
    generate_screenshot()
