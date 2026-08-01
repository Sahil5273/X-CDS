import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Define color scheme (blue/indigo primary, slate borders)
    box_color = "#EBF5FF"
    border_color = "#1E40AF"
    accent_color = "#D1FAE5"
    accent_border = "#065F46"

    # Helper function to draw rectangles with text
    def draw_box(ax, x, y, w, h, text, is_accent=False):
        face = accent_color if is_accent else box_color
        edge = accent_border if is_accent else border_color
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=face, edgecolor=edge, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#1F2937", wrap=True
        )

    # Draw boxes
    draw_box(ax, 0.5, 6.5, 2.0, 0.8, "Clinical Query")
    draw_box(ax, 3.5, 6.5, 2.5, 0.8, "Hybrid Search\n(Dense + BM25)")
    draw_box(ax, 7.0, 6.5, 2.5, 0.8, "Reciprocal Rank\nFusion (RRF)")
    
    draw_box(ax, 7.0, 4.5, 2.5, 0.8, "Cross-Encoder\nRe-ranking")
    draw_box(ax, 3.5, 4.5, 2.5, 0.8, "LangGraph\nOrchestrator")
    draw_box(ax, 0.5, 4.5, 2.0, 0.8, "Gemini Generation\nNode")
    
    # Diamond for validator
    validator_poly = patches.Polygon(
        [[2.5, 2.5], [4.0, 3.2], [5.5, 2.5], [4.0, 1.8]],
        closed=True, facecolor="#FEF3C7", edgecolor="#92400E", linewidth=2
    )
    ax.add_patch(validator_poly)
    ax.text(4.0, 2.5, "Citation Overlap\nValidator\n(T_min = 0.10)", ha="center", va="center", fontsize=8, fontweight="bold", color="#1F2937")

    draw_box(ax, 7.0, 2.1, 2.5, 0.8, "Output to\nClinical Portal", is_accent=True)
    draw_box(ax, 0.5, 2.1, 2.0, 0.8, "State Correction\nFeedback")

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate(
            text, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#4B5563", edgecolor="#4B5563", shrink=0.08, width=1.5, headwidth=6, headlength=6),
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4B5563"
        )

    # Draw flow arrows
    draw_arrow(2.5, 6.9, 3.5, 6.9)
    draw_arrow(6.0, 6.9, 7.0, 6.9)
    draw_arrow(8.25, 6.5, 8.25, 5.3)
    draw_arrow(7.0, 4.9, 6.0, 4.9)
    draw_arrow(3.5, 4.9, 2.5, 4.9)
    draw_arrow(1.5, 4.5, 1.5, 2.9)
    draw_arrow(1.5, 2.5, 2.5, 2.5)
    
    # Output path
    draw_arrow(5.5, 2.5, 7.0, 2.5, "Pass (>= 0.10)")
    # Retry path
    draw_arrow(4.0, 1.8, 1.5, 2.1, "Fail (< 0.10)")

    plt.tight_layout()
    plt.savefig("d:/X-CDS/docs/v2.0_expanded_release/figures/system_architecture.png", bbox_inches="tight", dpi=300)
    plt.close()

def generate_guardrail_flow():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_color = "#F3F4F6"
    border_color = "#4B5563"
    accent_color = "#ECFDF5"
    accent_border = "#047857"

    def draw_box(ax, x, y, w, h, text, is_accent=False):
        face = accent_color if is_accent else box_color
        edge = accent_border if is_accent else border_color
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=face, edgecolor=edge, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(
            x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=8, color="#1F2937"
        )

    # Draw boxes
    draw_box(ax, 0.5, 4.8, 1.8, 0.6, "Initial Generation\n(Citations [n])")
    draw_box(ax, 2.8, 4.8, 2.0, 0.6, "Token Overlap Ratio\nComputation")
    
    # Decision Diamond
    poly = patches.Polygon(
        [[6.0, 5.1], [6.8, 5.6], [7.6, 5.1], [6.8, 4.6]],
        closed=True, facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.5
    )
    ax.add_patch(poly)
    ax.text(6.8, 5.1, "Aligned\n(>=0.10)?", ha="center", va="center", fontsize=8)

    draw_box(ax, 6.0, 3.2, 1.6, 0.6, "Accept Response", is_accent=True)
    draw_box(ax, 2.8, 3.2, 2.0, 0.6, "Extract Mismatched\nClaims & Chunks")
    draw_box(ax, 2.8, 1.8, 2.0, 0.6, "Compile Correction\nState Feedback")
    draw_box(ax, 0.5, 1.8, 1.8, 0.6, "Re-Generate\nResponse Node")

    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate(
            text, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#4B5563", edgecolor="#4B5563", shrink=0.05, width=1.0, headwidth=4, headlength=4),
            ha="center", va="bottom", fontsize=7, color="#4B5563"
        )

    draw_arrow(2.3, 5.1, 2.8, 5.1)
    draw_arrow(4.8, 5.1, 6.0, 5.1)
    draw_arrow(6.8, 4.6, 6.8, 3.8, "Yes")
    draw_arrow(6.0, 5.1, 4.8, 3.5, "No")
    draw_arrow(3.8, 3.2, 3.8, 2.4)
    draw_arrow(2.8, 2.1, 2.3, 2.1)
    draw_arrow(1.4, 2.4, 1.4, 4.8, "Loop Retry")

    plt.tight_layout()
    plt.savefig("d:/X-CDS/docs/v2.0_expanded_release/figures/guardrail_retry_flow.png", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_system_architecture()
    generate_guardrail_flow()
    print("Exported diagram PNGs successfully.")
