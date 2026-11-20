"""Generate a printable review bound to a single procurement proposal."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ascis.event_evidence import HISTORY


EMERALD = colors.HexColor("#0F5C45")
DEEP = colors.HexColor("#07392E")
TANGERINE = colors.HexColor("#F28C28")
INK = colors.HexColor("#182E26")
MUTED = colors.HexColor("#53665E")
PALE = colors.HexColor("#F0F6F1")
LINE = colors.HexColor("#DBE6DF")


def procurement_review(*, event, hotel, proposal, forecast, assessment, status) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=42, rightMargin=42, topMargin=39, bottomMargin=44,
        title="ASCIS Procurement Review", author="ASCIS",
    )
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=15, textColor=INK)
    muted = ParagraphStyle("Muted", parent=body, textColor=MUTED, fontSize=8.5, leading=13)
    kicker = ParagraphStyle("Kicker", parent=muted, fontName="Helvetica-Bold", textColor=EMERALD)
    title = ParagraphStyle("Title", parent=body, fontName="Times-Roman", fontSize=32, leading=34, textColor=DEEP)
    heading = ParagraphStyle("Heading", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=17, textColor=DEEP)
    value = ParagraphStyle("Value", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=19, textColor=DEEP)

    def para(text, style=body):
        return Paragraph(escape(str(text).replace("—", "-").replace("–", "-")), style)

    def source(label, url):
        return Paragraph(f'<link href="{escape(url)}" color="#0F5C45">{escape(label)}</link>', muted)

    def inr(number):
        return f"INR {number:,.0f}"

    def metric_table(entries):
        table = Table([[[
            para(label, kicker),
            para(metric, value),
        ] for label, metric in entries]], colWidths=[doc.width / len(entries)] * len(entries))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("LINEBEFORE", (1, 0), (-1, -1), .5, LINE),
        ]))
        return table

    action = {
        "HOLD_300_PLUS_300": "Open a phased allocation",
        "EXERCISE_300": "Exercise the optional allocation",
        "BUY_300": "Commit 300 firm room-nights",
        "BUY_500": "Commit 500 firm room-nights",
        "RELEASE_OPTION": "Release the optional allocation",
    }.get(proposal["action"], proposal["action"].replace("_", " ").capitalize())
    contents = [
        para("ASCIS  /  PROCUREMENT OPERATIONS", kicker), Spacer(1, 8),
        para("Procurement review", title), para("Simulation only  |  " + status.capitalize(), muted),
        Spacer(1, 19), para(hotel, heading),
        para(f"{event['destination']}  |  {event['title']}  |  {event['date']}"),
        Spacer(1, 15),
        metric_table([
            ("FIRM ROOM-NIGHTS", f"{proposal['firm_units']:,}"),
            ("OPTIONAL ROOM-NIGHTS", f"{proposal['option_units']:,}"),
            ("SIMULATED COMMITMENT", inr(proposal["cost_inr"])),
        ]),
        Spacer(1, 16), para("Decision for approval", heading),
        para(f"{action}. Maximum modeled cash exposure for this proposal is {inr(proposal['maximum_cash_inr'])}. "
             f"Decision window: T{proposal['day']} in the demo. Supplier pool: {proposal['pool'] or 'none'}."),
        Spacer(1, 13), para("Evidence and model assessment", heading),
        para(f"Forecast room-nights: P10 {forecast['p10']:,}, median {forecast['p50']:,}, P90 {forecast['p90']:,}. "
             "The model is fitted on synthetic censored bookings. Organizer attendance supplies market context."),
        para(f"Expected incremental contribution: {inr(assessment['expected_profit'])}. "
             f"Modeled 90% downside CVaR: {inr(assessment['downside_cvar90'])}."),
        Spacer(1, 7),
    ]
    if event["url"]:
        contents.append(source("Organizer announcement", event["url"]))
    else:
        contents.append(para("Synthetic event and dates supplied by Scenario Lab.", muted))
    for row in HISTORY.get(event["id"], ()):
        contents.append(source(f"{row['edition']} | {row['count']:,} {row['metric'].lower()}", row["url"]))
    contents.extend([
        Spacer(1, 13), para("Human approval", heading),
        para("Review the allocation, terms, and modeled exposure. Use the approval inbox to approve or decline "
             "this exact proposal. Approval writes fictional stock to the demo ledger. Payment and supplier reservation are outside this simulation."),
        Spacer(1, 8), para(f"Company policy {proposal['company_policy']} | Report V{proposal['report_version']}", muted),
        para(f"Proposal ID {proposal['id']}", muted),
    ])

    def finish_page(canvas, document):
        width, _ = A4
        canvas.setStrokeColor(TANGERINE)
        canvas.setLineWidth(1.25)
        canvas.line(42, 36, width - 42, 36)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 23, "ASCIS  /  DEMO PROCUREMENT")
        canvas.drawRightString(width - 42, 23, str(document.page))

    doc.build(contents, onFirstPage=finish_page, onLaterPages=finish_page)
    return buffer.getvalue()
