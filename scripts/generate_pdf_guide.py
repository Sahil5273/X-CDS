import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def make_callout(text, style, title=""):
    """Helper to generate a blockquote callout box with a blue left border."""
    content = []
    if title:
        content.append(Paragraph(f"<b>{title}</b>", style))
        content.append(Spacer(1, 4))
    content.append(Paragraph(text, style))
    
    t = Table([[content]], colWidths=[430])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('LINELEFT', (0, 0), (-1, -1), 3.5, colors.HexColor('#2B6CB0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    return t

def build_pdf(filename="docs/xcds_documentation_guide.pdf"):
    # Target folder setup
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Page layout definitions (margins: 50pt = ~0.7 inch)
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles definitions
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1A365D'),
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4A5568'),
        alignment=TA_CENTER,
        spaceAfter=25
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER,
        spaceAfter=40
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'PreformattedCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1A202C'),
        backColor=colors.HexColor('#EDF2F7'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )

    story = []
    
    # ----------------------------------------------------
    # TITLE SECTION
    # ----------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("X-CDS: Explainable Clinical Decision Support", title_style))
    story.append(Paragraph("A to Z Comprehensive System Documentation & Technical Reference", subtitle_style))
    story.append(Paragraph("Submitted by: Sahil Kumar (Reg: 23BAI10224)<br/>VIT Bhopal University | School of Computer Science & Engineering (SCSE)<br/>Advisor: Dr. Abdul Rahman", meta_style))
    story.append(Spacer(1, 20))
    
    divider = Table([[""]], colWidths=[490])
    divider.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 1.5, colors.HexColor('#2B6CB0')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 15))
    
    # Intro Callout
    intro_txt = (
        "This guide outlines the complete codebase architecture, ingestion loops, re-ranking logic, "
        "and guardrail systems behind the Explainable Clinical Decision Support (X-CDS) framework. "
        "X-CDS mitigates the hallucination risks of Large Language Models in healthcare via hybrid "
        "retrieval and deterministic token-level validation loops."
    )
    story.append(make_callout(intro_txt, body_style, "Document Objective"))
    story.append(Spacer(1, 15))
    
    # ----------------------------------------------------
    # SECTION 1: DATA FLOW & ARCHITECTURE
    # ----------------------------------------------------
    story.append(Paragraph("1. Data Flow & System Architecture", h1_style))
    story.append(Paragraph(
        "The X-CDS system is organized as a pipeline that ensures clinical suggestions are directly "
        "anchored in validated literature. Below is the step-by-step breakdown of a query's lifecycle:",
        body_style
    ))
    
    steps = [
        "<b>1. Clinical Query Submission:</b> The user (clinician) enters a question in the React split-screen dashboard.",
        "<b>2. Dual-Channel Retrieval:</b> The pipeline concurrently runs a dense semantic query (using BAAI/bge-small vector search in ChromaDB) and a sparse keyword query (using BM25 on the ingested corpus).",
        "<b>3. Reciprocal Rank Fusion (RRF):</b> Merges the results based on their reciprocal ranking order. This balances precise keyword identification (drug names) and vector semantics.",
        "<b>4. Cross-Encoder Re-ranking:</b> Top-N candidates are parsed by a Transformer Cross-Encoder model to determine a joint query-passage relevance score. The top-K documents are selected.",
        "<b>5. LangGraph Generation Node:</b> Generates a response with inline citations using Gemini (mapped via the new <i>langchain-google-genai</i> package and Vertex AI GCP credentials).",
        "<b>6. Token Overlap Citation Validator:</b> Sentinel logic splits the claims and ensures each citation matches the source document with at least 25% alphanumeric overlap (excluding stopwords).",
        "<b>7. Stateful Self-Correction Loop:</b> If validation fails, error details are mapped to state variables, routing the query back to the generation node for up to 3 correction attempts.",
        "<b>8. Independent Ragas Evaluation:</b> Measures pipeline performance using Faithfulness, Answer Relevancy, Context Precision, and Context Recall metrics."
    ]
    for step in steps:
        story.append(Paragraph(f"&bull; {step}", bullet_style))
        
    story.append(Spacer(1, 15))
    
    # ----------------------------------------------------
    # SECTION 2: FILE-BY-FILE BREAKDOWN
    # ----------------------------------------------------
    story.append(Paragraph("2. Core Codebase File-by-File Breakdown", h1_style))
    story.append(Paragraph(
        "Below is a structural description of the core files inside the <code>backend/app/</code> directory, "
        "explaining their exact code design and how they interface:",
        body_style
    ))
    
    # File 1
    story.append(Paragraph("A. app/config/settings.py", h2_style))
    story.append(Paragraph(
        "Loads and validates environment configurations using Pydantic Settings. Crucially, it manages relative path resolution "
        "for the GCP credentials file (<code>GOOGLE_APPLICATION_CREDENTIALS</code>) relative to the project root, "
        "and sets up the process environment variables for Vertex AI API client discovery.",
        body_style
    ))
    
    # File 2
    story.append(Paragraph("B. app/llm/generation.py", h2_style))
    story.append(Paragraph(
        "Implements <code>RobustChatVertexAI</code> using the new <code>ChatGoogleGenerativeAI</code> model class. "
        "It includes automated retry logic for transient Google Cloud API failures (429 rate limits, 503 unavailable) "
        "and falls back dynamically to <code>gemini-3.1-flash-lite</code> if the primary model is unavailable. "
        "It also wraps generation schema validation in a try-except block to return validation failure states to the Graph loop.",
        body_style
    ))
    
    # File 3
    story.append(Paragraph("C. app/llm/graph.py", h2_style))
    story.append(Paragraph(
        "Constructs the stateful LangGraph graph. It compiles nodes for generation and validation, defining conditional edges "
        "(<code>should_retry_generation</code>) that route execution back to the generation node if citations fail alignment checks.",
        body_style
    ))
    
    # File 4
    story.append(Paragraph("D. app/guardrail/validator.py", h2_style))
    story.append(Paragraph(
        "Implements the alphanumeric token overlap scoring. It parses bracketed inline citations (e.g. <code>[1]</code>), "
        "matches them against context document indices, splits paragraphs into sentences, filters stopwords, "
        "and flags claims that do not meet the <code>CITATION_MIN_TOKEN_OVERLAP = 0.25</code> threshold.",
        body_style
    ))
    
    # File 5
    story.append(Paragraph("E. app/guardrail/loop.py", h2_style))
    story.append(Paragraph(
        "Coordinates state transitions between the validator outputs and graph routing state, mapping validator logs to "
        "<code>correction_feedback</code> messages that prompt the LLM to rewrite and cite correctly on successive loop attempts.",
        body_style
    ))
    
    # File 6
    story.append(Paragraph("F. app/eval/ragas_eval.py", h2_style))
    story.append(Paragraph(
        "Executes offline evaluation using the Ragas framework. It initializes Ragas wrappers using Vertex AI classes, "
        "computes metrics (Faithfulness, Relevancy, Context Precision, and Context Recall), aggregates results from NumPy formats, "
        "and outputs a JSON report (<code>ragas_report.json</code>).",
        body_style
    ))
    
    story.append(Spacer(1, 15))
    
    # ----------------------------------------------------
    # SECTION 3: DESIGN CHOICES & ALTERNATIVES
    # ----------------------------------------------------
    story.append(Paragraph("3. Design Choices, Alternatives & Clinical Justifications", h1_style))
    story.append(Paragraph(
        "Deploying generative models in medicine requires prioritising factual safety and explainability over output speed or flexibility. "
        "The following analysis outlines our design choices and why they are appropriate for clinical settings:",
        body_style
    ))
    
    # Alternative Table
    table_data = [
        [
            Paragraph("<b>Architecture / Node</b>", bullet_style), 
            Paragraph("<b>Selected Choice</b>", bullet_style), 
            Paragraph("<b>Discarded Alternative</b>", bullet_style), 
            Paragraph("<b>Clinical Justification</b>", bullet_style)
        ],
        [
            Paragraph("State Management", body_style),
            Paragraph("<b>LangGraph Cyclic Graph</b>", body_style),
            Paragraph("LCEL Linear Chains", body_style),
            Paragraph("LCEL cannot loop. LangGraph allows cyclic self-correction loops when citations fail validation.", body_style)
        ],
        [
            Paragraph("Information Retrieval", body_style),
            Paragraph("<b>Hybrid Search + RRF</b>", body_style),
            Paragraph("Pure Vector Search", body_style),
            Paragraph("Pure vector search misses critical exact terms (e.g. drug formulas). Keyword search ensures precision.", body_style)
        ],
        [
            Paragraph("Citation Alignment", body_style),
            Paragraph("<b>Deterministic Token Overlap</b>", body_style),
            Paragraph("LLM-as-a-Judge (Real-time)", body_style),
            Paragraph("LLM checkers are slow, expensive, and can hallucinate. Algorithmic overlap checks are fast, local, and 100% deterministic.", body_style)
        ]
    ]
    
    col_widths = [110, 110, 110, 160]
    design_table = Table(table_data, colWidths=col_widths)
    design_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
    ]))
    
    # Update text colors for header row in table data
    for i in range(4):
        table_data[0][i].style.textColor = colors.white
        
    story.append(design_table)
    story.append(Spacer(1, 15))
    
    # Conclusion Callout
    conclusion_txt = (
        "By enforcing strict algorithmic bounds on LLM generations, X-CDS guarantees that patient "
        "treatment suggestions are traceable directly to medical literature. This framework establishes "
        "the standard for deploying generative models safely in clinical settings."
    )
    story.append(make_callout(conclusion_txt, body_style, "Clinical Significance"))
    
    # Build Document
    doc.build(story)

if __name__ == "__main__":
    output_path = "docs/xcds_documentation_guide.pdf"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    build_pdf(output_path)
    print(f"Successfully generated PDF documentation guide at: {output_path}")
