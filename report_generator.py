
from pathlib import Path
from datetime import datetime

from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
)


def add_report_image(image_path, max_width=165 * mm, max_height=85 * mm):
    """Create a proportionally scaled image for the PDF."""

    image_path = Path(image_path)

    with PILImage.open(image_path) as source:
        width, height = source.size

    scale = min(
        max_width / width,
        max_height / height
    )

    return Image(
        str(image_path),
        width=width * scale,
        height=height * scale
    )


def generate_report(report_data, original_path, overlay_path, output_path):
    """Generate a PDF solar panel inspection report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="SolarInspect AI - Inspection Report",
        author="SolarInspect AI"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#172B4D"),
        alignment=TA_CENTER,
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#66758A"),
        alignment=TA_CENTER,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#238849"),
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#344054")
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=body_style,
        fontSize=8,
        leading=11
    )

    story = []

    # ------------------------------------------------
    # REPORT HEADER
    # ------------------------------------------------

    story.append(Paragraph("SolarInspect AI", title_style))

    story.append(
        Paragraph(
            "AI-Based Solar Panel Inspection Report",
            subtitle_style
        )
    )

    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#DCE5EC")
    ))

    story.append(Spacer(1, 10))

    report_time = datetime.now().astimezone().strftime(
        "%d %B %Y, %I:%M %p %Z"
    )

    metadata = [
        ["Report ID", report_data["report_id"]],
        ["Inspection date", report_time],
        ["Inspection type", "AI-assisted image inspection"],
        ["Model", "Cleaned Solar Panel Classification Model"]
    ]

    metadata_table = Table(
        metadata,
        colWidths=[42 * mm, 125 * mm]
    )

    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1),
         colors.HexColor("#EEF3F7")),
        ("TEXTCOLOR", (0, 0), (-1, -1),
         colors.HexColor("#344054")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4,
         colors.HexColor("#DCE5EC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))

    story.append(metadata_table)

    # ------------------------------------------------
    # PREDICTION SUMMARY
    # ------------------------------------------------

    story.append(
        Paragraph("1. AI Inspection Result", heading_style)
    )

    prediction = report_data["predicted_class"].replace(
        "_", " "
    ).title()

    summary = [
        ["Predicted condition", prediction],
        [
            "Model confidence",
            f'{report_data["confidence"]:.2f}%'
        ],
        [
            "Assessment",
            report_data["recommendation"]["status"]
        ]
    ]

    summary_table = Table(
        summary,
        colWidths=[55 * mm, 112 * mm]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1),
         colors.HexColor("#EEF7F0")),
        ("TEXTCOLOR", (0, 0), (-1, -1),
         colors.HexColor("#344054")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4,
         colors.HexColor("#DCE5EC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))

    story.append(summary_table)

    # ------------------------------------------------
    # CLASS PROBABILITIES
    # ------------------------------------------------

    story.append(
        Paragraph("2. Class Probability Distribution", heading_style)
    )

    probability_rows = [
        ["Class", "Model score"]
    ]

    for class_name, probability in report_data["probabilities"].items():
        probability_rows.append([
            class_name.replace("_", " ").title(),
            f"{probability:.2f}%"
        ])

    probability_table = Table(
        probability_rows,
        colWidths=[110 * mm, 57 * mm]
    )

    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#E8F5E9")),
        ("TEXTCOLOR", (0, 0), (-1, -1),
         colors.HexColor("#344054")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4,
         colors.HexColor("#DCE5EC")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))

    story.append(probability_table)

    # ------------------------------------------------
    # ORIGINAL IMAGE
    # ------------------------------------------------

    story.append(
        Paragraph("3. Uploaded Solar Panel Image", heading_style)
    )

    story.append(add_report_image(original_path))

    # ------------------------------------------------
    # GRAD-CAM IMAGE
    # ------------------------------------------------

    story.append(
        Paragraph("4. Grad-CAM Visualization", heading_style)
    )

    story.append(add_report_image(overlay_path))

    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "Grad-CAM provides a visual indication of image regions "
        "influencing the model's prediction. It does not confirm "
        "the physical location or extent of a defect.",
        small_style
    ))

    # ------------------------------------------------
    # IMAGE QUALITY
    # ------------------------------------------------

    story.append(
        Paragraph("5. Image Quality Assessment", heading_style)
    )

    quality = report_data["quality"]

    quality_rows = [
        ["Parameter", "Measured value"],
        [
            "Resolution",
            f'{quality["width"]} × {quality["height"]}'
        ],
        ["Brightness", str(quality["brightness"])],
        ["Contrast", str(quality["contrast"])],
        ["Sharpness", str(quality["sharpness"])]
    ]

    quality_table = Table(
        quality_rows,
        colWidths=[110 * mm, 57 * mm]
    )

    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#EEF3F7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4,
         colors.HexColor("#DCE5EC")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))

    story.append(quality_table)

    warnings = quality["warnings"]

    if warnings:
        warning_text = "<b>Warnings:</b><br/>" + "<br/>".join(
            f"- {warning}" for warning in warnings
        )
    else:
        warning_text = (
            "<b>Image quality:</b> No image-quality warnings "
            "were detected by the current assessment rules."
        )

    story.append(Spacer(1, 8))
    story.append(Paragraph(warning_text, body_style))

    # ------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # ------------------------------------------------

    story.append(
        Paragraph("6. Maintenance Recommendation", heading_style)
    )

    story.append(Paragraph(
        report_data["recommendation"]["status"],
        body_style
    ))

    story.append(Spacer(1, 5))

    story.append(Paragraph(
        report_data["recommendation"]["recommendation"],
        body_style
    ))

    # ------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------

    story.append(
        Paragraph("7. Important Disclaimer", heading_style)
    )

    story.append(Paragraph(
        "This report presents an AI-assisted visual assessment "
        "based on an uploaded image. Model confidence scores are "
        "not calibrated probabilities of actual panel condition. "
        "The system is not a certified diagnostic instrument. "
        "Image-quality checks and Grad-CAM visualizations do not "
        "establish physical panel health. Human physical inspection "
        "is recommended before maintenance decisions.",
        small_style
    ))

    story.append(Spacer(1, 12))

    story.append(HRFlowable(
        width="100%",
        thickness=0.7,
        color=colors.HexColor("#DCE5EC")
    ))

    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "Generated by SolarInspect AI | "
        "AI-Based Solar Panel Inspection Project",
        subtitle_style
    ))

    document.build(story)

    return output_path
