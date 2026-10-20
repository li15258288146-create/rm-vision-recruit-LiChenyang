"""生成 A4 横向、10x7 方格、20 mm 格长的矢量打印 PDF。"""
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def main():
    out = Path(__file__).resolve().parent / "board"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checkerboard_A4_20mm_9x6.pdf"
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    c.setTitle("Camera calibration - 9x6 inner corners - 20 mm squares")
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(148.5 * mm, 194 * mm, "CAMERA CALIBRATION BOARD")
    c.setFont("Helvetica", 10)
    c.drawCentredString(148.5 * mm, 186 * mm, "10 x 7 squares  |  9 x 6 inner corners  |  Square: 20.00 mm")
    # 白色外边界至少 20 mm；棋盘区 200 x 140 mm。
    left, bottom, square = 48.5 * mm, 40 * mm, 20 * mm
    c.setFillColorRGB(0, 0, 0)
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                c.rect(left + col * square, bottom + row * square,
                       square, square, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(148.5 * mm, 29 * mm, "PRINT AT 100% / ACTUAL SIZE - DO NOT FIT TO PAGE")
    c.setFont("Helvetica", 9)
    c.drawCentredString(148.5 * mm, 23 * mm, "Mount flat. Measure printed squares in BOTH directions before calibration.")
    c.setLineWidth(.5)
    c.line(98.5 * mm, 13 * mm, 198.5 * mm, 13 * mm)
    for x in [98.5, 198.5]:
        c.line(x * mm, 11 * mm, x * mm, 15 * mm)
    c.setFont("Helvetica", 8)
    c.drawCentredString(148.5 * mm, 7 * mm, "Reference length: 100 mm")
    c.showPage()
    c.save()
    print(path)


if __name__ == "__main__":
    main()
