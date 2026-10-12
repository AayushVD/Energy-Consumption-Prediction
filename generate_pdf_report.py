"""
generate_pdf_report.py
Generates a publication-quality multi-page PDF report (report.pdf) for Assignment 2.
Embeds performance tables, analytical graphs from plots/, and app screenshots.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.units import inch

def build_pdf_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, 'report.pdf')
    plots_dir = os.path.join(base_dir, 'plots')
    screenshots_dir = os.path.join(base_dir, 'screenshots')

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748B'),
        alignment=1, # Centered
        spaceBefore=3,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A'),
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A'),
        alignment=0
    )

    story = []

    # Title & Header Banner
    story.append(Paragraph("Machine Learning Application Report: Energy Consumption Prediction", title_style))
    story.append(Paragraph("<b>Course Outcome:</b> CO5 – Solve real-life problems using appropriate machine learning models and evaluate performance measures<br/><b>Domain:</b> Smart Energy Demand Management &amp; Operational Peak Load Auditing<br/><b>Models Evaluated:</b> Logistic Regression (Champion), Support Vector Machine (RBF), Random Forest Classifier", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=2, spaceAfter=8))

    # 1. Problem Statement
    story.append(Paragraph("1. Problem Statement &amp; Real-World Relevance", h1_style))
    story.append(Paragraph(
        "Modern electrical grids face severe operational bottlenecks due to unanticipated peak power demand spikes. "
        "When commercial facilities and households concurrently surge past standard operational baselines, utilities must activate carbon-intensive "
        "peaker generators, risking localized brownouts and transformer degradation. For consumers and building managers, exceeding contracted peak thresholds "
        "incurs punitive time-of-use (TOU) tariffs. This project establishes an operational machine learning classification system that forecasts whether a facility "
        "will enter a <b>High Peak Energy Consumption Alert (&ge; 22.0 kW Net Load)</b> versus <b>Normal / Efficient Operation (&lt; 22.0 kW)</b>. "
        "Predicting high demand in advance allows automated building management systems (BEMS) to execute proactive load-shedding, precooling, and appliance staggering.",
        body_style
    ))

    # 2. Dataset Description
    story.append(Paragraph("2. Dataset Description &amp; Summary Statistics", h1_style))
    story.append(Paragraph(
        "The investigation utilizes an operational smart-metering dataset of <b>1,500 records</b> capturing building structural metrics, environmental variables, "
        "appliance power draws, and rooftop solar offset. The dataset is derived from real-world telemetry benchmarks (e.g., UCI Appliances Energy and Building Energy Efficiency). "
        "The target variable <code>High_Consumption</code> reflects operational peak conditions (947 Normal samples [63.1%] vs 553 High Peak Alert samples [36.9%]).",
        body_style
    ))

    # Feature table
    feat_data = [
        [Paragraph("<b>Feature Name</b>", table_header_style), Paragraph("<b>Data Type</b>", table_header_style), Paragraph("<b>Unit / Range</b>", table_header_style), Paragraph("<b>Physical Operational Description</b>", table_header_style)],
        [Paragraph("Square_Footage", table_cell_left), Paragraph("Numerical (float)", table_cell_style), Paragraph("800 – 4500 sq ft", table_cell_style), Paragraph("Conditioned floor area dictating baseline heating/cooling load.", table_cell_left)],
        [Paragraph("Occupancy", table_cell_left), Paragraph("Numerical (int)", table_cell_style), Paragraph("1 – 8 persons", table_cell_style), Paragraph("Active occupants contributing internal metabolic heat & device draws.", table_cell_left)],
        [Paragraph("Outdoor_Temperature", table_cell_left), Paragraph("Numerical (float)", table_cell_style), Paragraph("-5.0 – 38.0 °C", table_cell_style), Paragraph("Exterior ambient temperature driving conduction and HVAC work.", table_cell_left)],
        [Paragraph("Humidity", table_cell_left), Paragraph("Numerical (float)", table_cell_style), Paragraph("20.0 – 85.0 %", table_cell_style), Paragraph("Relative humidity affecting latent air-conditioning enthalpy.", table_cell_left)],
        [Paragraph("HVAC_Usage_Level", table_cell_left), Paragraph("Categorical", table_cell_style), Paragraph("Low, Medium, High", table_cell_style), Paragraph("Operational cooling/heating intensity state.", table_cell_left)],
        [Paragraph("Appliance_Load_kW", table_cell_left), Paragraph("Numerical (float)", table_cell_style), Paragraph("0.5 – 11.5 kW", table_cell_style), Paragraph("Active electrical draw of continuous and scheduled appliances.", table_cell_left)],
        [Paragraph("Lighting_Load_kW", table_cell_left), Paragraph("Numerical (float)", table_cell_style), Paragraph("0.2 – 3.2 kW", table_cell_style), Paragraph("Illumination power draw across conditioned zones.", table_cell_left)],
        [Paragraph("Solar_Generation_kW", table_cell_left), Paragraph("Numerical (float)", table_cell_style), Paragraph("0.0 – 7.0 kW", table_cell_style), Paragraph("Photovoltaic generation directly offsetting grid import.", table_cell_left)],
        [Paragraph("Time_of_Day", table_cell_left), Paragraph("Categorical", table_cell_style), Paragraph("Morn, Aft, Eve, Night", table_cell_style), Paragraph("Diurnal tariff period (evening represents maximum peak stress).", table_cell_left)],
        [Paragraph("Day_Type", table_cell_left), Paragraph("Categorical", table_cell_style), Paragraph("Weekday, Weekend", table_cell_style), Paragraph("Occupancy and operational scheduling pattern.", table_cell_left)],
        [Paragraph("High_Consumption", table_cell_left), Paragraph("Binary Target", table_cell_style), Paragraph("0 or 1", table_cell_style), Paragraph("1 if Net Load &ge; 22.0 kW (Peak Demand Alert), else 0.", table_cell_left)]
    ]

    t_feat = Table(feat_data, colWidths=[1.3*inch, 1.0*inch, 1.2*inch, 3.5*inch])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 8))

    # 3. Preprocessing
    story.append(Paragraph("3. Preprocessing &amp; Anti-Leakage Architecture", h1_style))
    story.append(Paragraph(
        "<b>Data Hygiene:</b> Missing values were audited (0 nulls found across all features). Duplicate rows were evaluated (0 duplicates).<br/>"
        "<b>Anti-Leakage Pipeline:</b> Features were standardized via <code>StandardScaler</code> and categorical variables encoded via <code>OneHotEncoder(drop='first')</code>. "
        "Critically, all scalers and encoders were encapsulated inside a <code>sklearn.compose.ColumnTransformer</code> and unified <code>Pipeline</code>. "
        "Scaling statistics (&mu;, &sigma;) were fitted strictly on inner training folds during cross-validation, guaranteeing zero test set contamination.<br/>"
        "<b>Partitioning:</b> An 80/20 stratified split (1,200 training samples, 300 test samples) was executed with <code>stratify=y</code> and fixed <code>random_state=42</code>.",
        body_style
    ))

    # 4. Model Selection & 5-Fold CV
    story.append(Paragraph("4. Model Selection, 5-Fold Cross-Validation &amp; Results", h1_style))
    story.append(Paragraph(
        "Three distinct algorithmic paradigms were benchmarked: <b>Logistic Regression</b> (linear baseline with calibrated probabilities), "
        "<b>Support Vector Machine (RBF Kernel)</b> (margin maximization in reproducing kernel Hilbert space), and <b>Random Forest Classifier</b> (ensemble bagging). "
        "In compliance with academic standards, model selection was governed <b>exclusively by 5-fold cross-validation on the training set</b>.",
        body_style
    ))

    # Metrics Table
    perf_data = [
        [Paragraph("<b>Model Architecture</b>", table_header_style), Paragraph("<b>5-Fold CV F1-Score</b>", table_header_style), Paragraph("<b>Test Accuracy</b>", table_header_style), Paragraph("<b>Test Precision</b>", table_header_style), Paragraph("<b>Test Recall</b>", table_header_style), Paragraph("<b>Test F1-Score</b>", table_header_style), Paragraph("<b>Test ROC-AUC</b>", table_header_style)],
        [Paragraph("<b>Logistic Regression (Champion)</b>", table_cell_left), Paragraph("<b>0.9056 &plusmn; 0.026</b>", table_cell_style), Paragraph("<b>96.67%</b>", table_cell_style), Paragraph("<b>97.20%</b>", table_cell_style), Paragraph("<b>93.69%</b>", table_cell_style), Paragraph("<b>95.41%</b>", table_cell_style), Paragraph("<b>0.9898</b>", table_cell_style)],
        [Paragraph("Support Vector Machine (RBF)", table_cell_left), Paragraph("0.9007 &plusmn; 0.025", table_cell_style), Paragraph("95.33%", table_cell_style), Paragraph("98.02%", table_cell_style), Paragraph("89.19%", table_cell_style), Paragraph("93.40%", table_cell_style), Paragraph("0.9900", table_cell_style)],
        [Paragraph("Random Forest Classifier", table_cell_left), Paragraph("0.8496 &plusmn; 0.027", table_cell_style), Paragraph("90.00%", table_cell_style), Paragraph("91.75%", table_cell_style), Paragraph("80.18%", table_cell_style), Paragraph("85.58%", table_cell_style), Paragraph("0.9736", table_cell_style)]
    ]

    t_perf = Table(perf_data, colWidths=[1.8*inch, 1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FEF3C7'), colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 10))

    # Page Break for Visual Analytics
    story.append(PageBreak())

    # 5. Visualizations & Analytical Interpretations
    story.append(Paragraph("5. Visual Analytics &amp; Empirical Interpretations", h1_style))
    story.append(Paragraph(
        "Six analytical plots were generated and saved at 300 DPI to assess feature relationships, prediction distributions, and comparative decision dynamics.",
        body_style
    ))

    # Grid of images: Figure 1 & Figure 2
    img_row1 = []
    p1 = os.path.join(plots_dir, 'class_distribution.png')
    p2 = os.path.join(plots_dir, 'correlation_heatmap.png')
    if os.path.exists(p1) and os.path.exists(p2):
        img_row1 = [
            [Image(p1, width=3.3*inch, height=2.4*inch), Image(p2, width=3.5*inch, height=2.4*inch)],
            [Paragraph("<b>Figure 1:</b> Target Class Distribution (63.1% Normal vs 36.9% Peak)", caption_style),
             Paragraph("<b>Figure 2:</b> Feature Correlation Matrix Heatmap", caption_style)]
        ]
        t_img1 = Table(img_row1, colWidths=[3.5*inch, 3.5*inch])
        t_img1.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_img1)

    story.append(Spacer(1, 6))

    # Grid of images: Figure 3 & Figure 4
    p3 = os.path.join(plots_dir, 'confusion_matrix.png')
    p4 = os.path.join(plots_dir, 'roc_curves.png')
    if os.path.exists(p3) and os.path.exists(p4):
        img_row2 = [
            [Image(p3, width=3.2*inch, height=2.4*inch), Image(p4, width=3.4*inch, height=2.4*inch)],
            [Paragraph("<b>Figure 3:</b> Confusion Matrix (Champion Logistic Regression)", caption_style),
             Paragraph("<b>Figure 4:</b> Multi-Model ROC Curves Comparison", caption_style)]
        ]
        t_img2 = Table(img_row2, colWidths=[3.5*inch, 3.5*inch])
        t_img2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_img2)

    story.append(Spacer(1, 6))

    # Grid of images: Figure 5 & Figure 6
    p5 = os.path.join(plots_dir, 'model_comparison.png')
    p6 = os.path.join(plots_dir, 'feature_importance.png')
    if os.path.exists(p5) and os.path.exists(p6):
        img_row3 = [
            [Image(p5, width=3.4*inch, height=2.3*inch), Image(p6, width=3.4*inch, height=2.3*inch)],
            [Paragraph("<b>Figure 5:</b> Performance Metric Comparison Across Algorithms", caption_style),
             Paragraph("<b>Figure 6:</b> Random Forest Feature Importance Ranking", caption_style)]
        ]
        t_img3 = Table(img_row3, colWidths=[3.5*inch, 3.5*inch])
        t_img3.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_img3)

    # Page Break for In-Depth Discussion & Deployment
    story.append(PageBreak())

    # 6. In-Depth Results Analysis & Critical Discussion
    story.append(Paragraph("6. In-Depth Results Analysis &amp; Critical Discussion", h1_style))
    story.append(Paragraph(
        "<b>Why Did Logistic Regression Win?</b><br/>"
        "Net electrical power consumption is fundamentally governed by additive conservation of energy laws: "
        "<code>Net Load = Base Load + &Sigma; Subsystem Loads - Solar Offset</code>. "
        "While non-linearities exist around thermal comfort setpoints, the primary discriminator between normal and peak states is additive accumulation. "
        "Logistic Regression establishes a regularized planar boundary that matches this additive physics. On 1,200 training instances, high-capacity models "
        "like Random Forest suffered from orthogonal partition boundaries that hurt generalization on continuous edge cases.<br/><br/>"
        "<b>Cost-of-Error Analysis: Which Metric Matters Most?</b><br/>"
        "The operational costs of False Positives (FP) and False Negatives (FN) are highly asymmetric in power systems:<br/>"
        "&bull; <b>False Negative (Missed Peak Demand):</b> The model predicts Normal (0), but actual draw surges into peak territory (&ge; 22.0 kW). "
        "The facility fails to shed load, triggering severe utility penalty surcharges and grid transformer stress. <i>Cost: Catastrophic</i>.<br/>"
        "&bull; <b>False Positive (False Alarm):</b> The model predicts High Demand (1), but actual draw remains normal (0). "
        "The facility executes minor precautionary actions (e.g., setpoint setback by 1.5°C or shifting laundry). <i>Cost: Negligible</i>.<br/>"
        "Therefore, <b>Recall (Sensitivity) is the primary metric of interest</b>. Logistic Regression achieved the highest Recall (<b>93.69%</b>, "
        "missing only 7 peak events out of 111), substantially outperforming Random Forest (Recall = 80.18%, missing 22 peak events).<br/><br/>"
        "<b>Limitations:</b> Observations were treated as conditionally independent (i.i.d.), whereas real buildings exhibit thermal inertia and time-series autocorrelation. "
        "Future work should integrate autoregressive lag terms and dynamic tariff thresholds.",
        body_style
    ))

    # 7. CO5 Mapping
    story.append(Paragraph("7. Mapping to Course Outcome (CO5)", h1_style))
    story.append(Paragraph(
        "<b>Course Outcome:</b> <i>'CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures'</i><br/>"
        "<b>1. Real-Life Problem Formulation:</b> Addressed electrical peak demand surges in smart building operations, framing a high-stakes decision problem.<br/>"
        "<b>2. Appropriate Algorithm Selection:</b> Formulated hypotheses across linear, kernelized margin, and ensemble tree models, explaining their suitability.<br/>"
        "<b>3. Leakage-Free Preprocessing:</b> Implemented <code>StandardScaler</code> and encoders inside <code>Pipeline</code> to ensure zero leakage.<br/>"
        "<b>4. Comprehensive Evaluation:</b> Assessed models across Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices using 5-Fold Stratified Cross-Validation.<br/>"
        "<b>5. Real-World Deployment:</b> Deployed the champion pipeline into an interactive web interface and native Android Studio mobile application.",
        body_style
    ))

    # 8. Deployment Overview
    story.append(Paragraph("8. Deployment: Streamlit Web Dashboard &amp; Native Android App", h1_style))
    story.append(Paragraph(
        "<b>Interactive Web Interface (Streamlit - <code>app.py</code>):</b> Provides interactive sliders, dropdowns, real-time prediction cards, "
        "and prescriptive energy-saving recommendations.<br/>"
        "<b>Native Android Application (<code>android_app/</code>):</b> Formulated for Android Studio with an on-device Kotlin ML engine "
        "(<code>EnergyPredictor.kt</code>) executing the exact mathematical transformation with &lt; 1 ms latency and 100% offline capability.",
        body_style
    ))

    # Embed App Screenshot
    app_img_path = os.path.join(screenshots_dir, 'app.png')
    if os.path.exists(app_img_path):
        story.append(Spacer(1, 4))
        story.append(Image(app_img_path, width=7.0*inch, height=4.2*inch))
        story.append(Paragraph("<b>Figure 7:</b> Interactive Streamlit Web Application Interface (app.py) with On-Demand Inference &amp; Recommendations", caption_style))

    # Build document
    doc.build(story)
    print(f"[SUCCESS] High-quality PDF report generated at: {pdf_path}")
    return pdf_path

if __name__ == '__main__':
    build_pdf_report()
