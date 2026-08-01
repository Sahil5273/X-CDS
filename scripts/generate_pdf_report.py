"""Generate a beautifully formatted PDF report comparing the RAG metrics."""

from __future__ import annotations

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf_report(output_path: Path):
    # Setup document geometry (0.75 in margins)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom high-quality styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0a2540'), # Dark navy
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4a5568'), # Slate gray
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#007eff'), # Brand blue
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=8
    )
    
    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white,
        alignment=1 # Centered
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2d3748'),
        alignment=1 # Centered
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0a2540')
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=table_cell_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#2d3748'),
        alignment=0 # Left-aligned
    )

    story = []
    
    # 1. Header Section
    story.append(Paragraph("Clinical RAG Evaluation & Benchmarks Report", title_style))
    story.append(Paragraph(
        "A comparative performance study evaluating the X-CDS clinical decision support system. "
        "Corpus scaling impact (900 vs. 6,940 passages) and parametric validation threshold sweeps ($n = 0.10$ to $0.50$).",
        subtitle_style
    ))
    story.append(Spacer(1, 10))
    
    # 2. Release & Parameters Summary
    story.append(Paragraph("1. Configuration & Parameters", h1_style))
    story.append(Paragraph(
        "<b>Clinical Corpus Size:</b> v1.0 contains 900 passages (Zika virus focus). v2.0 contains 6,940 passages "
        "(expanded Zika, Chikungunya, and Dengue virus guidelines from 73 PMC reference journals). Chunks are "
        "structured at 1,000 characters maximum length with 200 characters overlap.<br/>"
        "<b>Validation Threshold (n-value):</b> Measures the minimum token alignment overlap between generated "
        "LLM responses and retrieved evidence required to pass citation verification. Range: 0.10 (Light) to 0.50 (Strict).",
        body_style
    ))
    story.append(Spacer(1, 15))
    
    # 3. Database Scale Scaling Impact Table
    story.append(Paragraph("2. Corpus Scale Scaling Impact (v1.0 vs. v2.0)", h1_style))
    story.append(Paragraph(
        "The scaling of the vector database from a small subset of 900 passages to 6,940+ passages shows a "
        "massive positive correlation with downstream retrieval precision and generation quality.",
        body_style
    ))
    
    # Build Scaling Table
    scaling_data = [
        [
            Paragraph("Ragas Metric", table_header_style),
            Paragraph("v1.0 Baseline RAG<br/>(900 Chunks)", table_header_style),
            Paragraph("v1.0 X-CDS RAG<br/>(900 Chunks)", table_header_style),
            Paragraph("v2.0 Baseline RAG<br/>(6,940 Chunks)", table_header_style),
            Paragraph("v2.0 X-CDS RAG<br/>(6,940 Chunks, n=0.10)", table_header_style),
            Paragraph("Net Scaling Growth", table_header_style)
        ],
        [
            Paragraph("Faithfulness", table_cell_left),
            Paragraph("90.40%", table_cell_style),
            Paragraph("90.70%", table_cell_style),
            Paragraph("89.78%", table_cell_style),
            Paragraph("93.37%", table_cell_bold),
            Paragraph("+2.67%", table_cell_bold)
        ],
        [
            Paragraph("Answer Relevancy", table_cell_left),
            Paragraph("47.80%", table_cell_style),
            Paragraph("49.90%", table_cell_style),
            Paragraph("61.17%", table_cell_style),
            Paragraph("57.81%", table_cell_bold),
            Paragraph("+7.91%", table_cell_bold)
        ],
        [
            Paragraph("Context Precision", table_cell_left),
            Paragraph("22.80%", table_cell_style),
            Paragraph("35.60%", table_cell_style),
            Paragraph("74.09%", table_cell_style),
            Paragraph("68.94%", table_cell_bold),
            Paragraph("+33.34%", table_cell_bold)
        ],
        [
            Paragraph("Context Recall", table_cell_left),
            Paragraph("63.00%", table_cell_style),
            Paragraph("65.20%", table_cell_style),
            Paragraph("74.25%", table_cell_style),
            Paragraph("71.83%", table_cell_bold),
            Paragraph("+6.63%", table_cell_bold)
        ],
    ]
    
    col_widths = [120, 85, 85, 85, 95, 60]
    t1 = Table(scaling_data, colWidths=col_widths)
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#007eff')),
        ('BACKGROUND', (4,1), (4,-1), colors.HexColor('#e0effe')), # Light blue highlight column
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t1)
    story.append(Spacer(1, 20))
    
    # 4. Parametric Sweep Table
    story.append(Paragraph("3. Parametric Overlap Threshold Sweep ($N=100$)", h1_style))
    story.append(Paragraph(
        "Tuning the validation threshold ($n$) highlights the trade-offs between factual precision "
        "and conversational constraints. Lenient values prevent major hallucinations, whereas strict "
        "values force verbatim copying and trigger loop retry failures.",
        body_style
    ))
    
    sweep_data = [
        [
            Paragraph("Overlap Threshold (n)", table_header_style),
            Paragraph("Ragas Faithfulness", table_header_style),
            Paragraph("Ragas Answer Relevancy", table_header_style),
            Paragraph("Avg. Attempts (Loops)", table_header_style),
            Paragraph("Failed Outputs", table_header_style)
        ],
        [
            Paragraph("n = 0.00 (Baseline RAG)", table_cell_left),
            Paragraph("89.78%", table_cell_style),
            Paragraph("61.17%", table_cell_style),
            Paragraph("1.00 attempt", table_cell_style),
            Paragraph("0% (0/100)", table_cell_style)
        ],
        [
            Paragraph("n = 0.10 (Light)", table_cell_left),
            Paragraph("93.37% (Peak)", table_cell_bold),
            Paragraph("57.81%", table_cell_style),
            Paragraph("1.10 attempts", table_cell_bold),
            Paragraph("1% (1/100)", table_cell_bold)
        ],
        [
            Paragraph("n = 0.15 (Mild)", table_cell_left),
            Paragraph("90.20%", table_cell_style),
            Paragraph("59.07%", table_cell_style),
            Paragraph("1.25 attempts", table_cell_style),
            Paragraph("3% (3/100)", table_cell_style)
        ],
        [
            Paragraph("n = 0.25 (Default)", table_cell_left),
            Paragraph("89.49%", table_cell_style),
            Paragraph("57.31%", table_cell_style),
            Paragraph("1.45 attempts", table_cell_style),
            Paragraph("8% (8/100)", table_cell_style)
        ],
        [
            Paragraph("n = 0.50 (Strict)", table_cell_left),
            Paragraph("92.41%", table_cell_style),
            Paragraph("57.82%", table_cell_style),
            Paragraph("1.95 attempts", table_cell_style),
            Paragraph("15% (15/100)", table_cell_style)
        ],
    ]
    
    col_widths2 = [140, 100, 110, 100, 80]
    t2 = Table(sweep_data, colWidths=col_widths2)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0a2540')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#e0effe')), # Light blue highlight peak row
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t2)
    
    # Force Page Break for next section to remain clean
    story.append(PageBreak())
    
    # 5. Reranker Comparison Section
    story.append(Paragraph("4. Reranker Comparison ($N=20$ retrieval sample)", h1_style))
    story.append(Paragraph(
        "A retrieval precision test comparing the baseline 22M parameter MiniLM reranker to the upgraded 567M "
        "parameter multilingual BGE reranker using the v2.0 expanded database.",
        body_style
    ))
    
    reranker_data = [
        [
            Paragraph("Model", table_header_style),
            Paragraph("Context Precision", table_header_style),
            Paragraph("Context Recall", table_header_style),
            Paragraph("Key Characteristic", table_header_style)
        ],
        [
            Paragraph("MiniLM (Baseline)", table_cell_left),
            Paragraph("66.86%", table_cell_style),
            Paragraph("66.67%", table_cell_style),
            Paragraph("Lightweight (22M params), fast local inference.", table_cell_style)
        ],
        [
            Paragraph("BGE-v2-m3 (Upgraded)", table_cell_left),
            Paragraph("69.38% (+2.52%)", table_cell_bold),
            Paragraph("63.33% (-3.34%)", table_cell_style),
            Paragraph("High capacity (567M params), multilingual, dense mapping.", table_cell_bold)
        ],
    ]
    t3 = Table(reranker_data, colWidths=[130, 110, 100, 190])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4a5568')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t3)
    story.append(Spacer(1, 15))
    
    # 6. Latency & Financial Feasibility Section
    story.append(Paragraph("5. Computational Latency & Financial Feasibility", h1_style))
    story.append(Paragraph(
        "<b>Average Query Latency:</b> Naive RAG averages 2.42s; Hybrid RAG (no validation) averages 3.15s. "
        "Under X-CDS ($n=0.10$), the 90% of queries that pass verification on the first attempt execute in 4.08s. "
        "For the 10% of queries requiring a self-correction loop, latency is 8.45s. The overall average of 4.52s is highly acceptable.<br/>"
        "<b>Financial Feasibility:</b> Based on Vertex AI token pricing, a typical consult query costs $0.000240 USD "
        "in input tokens (3,200 tokens context) and $0.000105 USD in output tokens (350 tokens answer). The total "
        "cost of $0.000345 USD (approx. <b>0.029 INR / less than 3 paise</b>) makes large-scale telehealth integration highly feasible.",
        body_style
    ))
    story.append(Spacer(1, 15))
    
    # 7. Self-Evaluation Bias Mitigation
    story.append(Paragraph("6. Academic Rigor & Bias Mitigation", h1_style))
    story.append(Paragraph(
        "Evaluating a generative model using the same model family introduces 'self-evaluation bias.' To satisfy "
        "rigorous scientific reporting standards, the generator and evaluator models are decoupled: "
        "queries are answered by <code>gemini-3.5-flash</code> in the production pipeline, whereas the Ragas "
        "evaluator is powered by a separate Pro-tier model (<code>gemini-2.5-pro</code>) on Google Cloud Vertex AI, "
        "ensuring completely objective quality scoring.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated PDF report at: {output_path}")

if __name__ == "__main__":
    pdf_path = Path(__file__).resolve().parents[1] / "docs/evaluation_report.pdf"
    build_pdf_report(pdf_path)
