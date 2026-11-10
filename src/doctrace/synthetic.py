"""Procedural documents for UI demos and explicitly synthetic negative checks."""
import random
from PIL import Image, ImageDraw, ImageFont


def make_document(seed=0, size=(512, 512)):
    rng = random.Random(seed)
    image = Image.new("RGB", size, (rng.randint(239, 255),) * 3)
    draw = ImageDraw.Draw(image)
    font = lambda n: ImageFont.load_default(size=n)
    ink = (rng.randint(15, 70),) * 3
    margin = 28
    draw.text((margin, 24), "NORTHLINE / LAB RECORD", fill=ink, font=font(20))
    draw.text((margin, 56), f"Document {seed + 1:04d}    |    Internal sample", fill=ink, font=font(12))
    draw.line((margin, 83, size[0]-margin, 83), fill=ink, width=2)
    draw.text((margin, 104), rng.choice(["MATERIAL SUMMARY", "SERVICE STATEMENT", "DELIVERY RECORD"]), fill=ink, font=font(16))
    y = 145
    for i in range(rng.randint(5, 9)):
        draw.text((margin, y), f"{i+1:02d}   {rng.choice(['Calibration', 'Inspection', 'Assembly', 'Transport'])}", fill=ink, font=font(rng.randint(12, 16)))
        draw.text((size[0]-135, y), f"{rng.randint(10, 990)}.00", fill=ink, font=font(14))
        if seed % 3 == 0:
            draw.line((margin, y+23, size[0]-margin, y+23), fill=(180,)*3)
        y += 31
    draw.line((margin, 442, size[0]-margin, 442), fill=ink)
    draw.text((margin, 460), "Synthetic sample - no real personal information", fill=ink, font=font(11))
    return image
