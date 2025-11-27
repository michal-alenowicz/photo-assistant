# vision.py
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential

VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
VISION_API_KEY = os.getenv("AZURE_VISION_KEY")

client = ImageAnalysisClient(
    endpoint=VISION_ENDPOINT,
    credential=AzureKeyCredential(VISION_API_KEY)
)

def analyze_image(image_bytes: bytes) -> dict:
   

    features = [
        "Caption",
        "DenseCaptions",
        "Read",
        "Tags"
    ]

    result = client.analyze(
        image_data=image_bytes,
        visual_features=features,
        gender_neutral_caption=False
    )

    summary = {}

     # ---------------------------------------------------------
    # Caption
    # ---------------------------------------------------------
    if hasattr(result, "caption") and result.caption:
        summary["main_caption"] = [{
                "text": result.caption.text,
                "confidence": result.caption.confidence,
            }]
    

    # ---------------------------------------------------------
    # Dense Captions
    # ---------------------------------------------------------
    if hasattr(result, "dense_captions") and result.dense_captions:
        summary["dense_captions"] = []
        for cap in result.dense_captions.list:
            summary["dense_captions"].append({
                "text": cap.text,
                "confidence": cap.confidence,
                "bbox": {
                    "x": cap.bounding_box.x,
                    "y": cap.bounding_box.y,
                    "w": cap.bounding_box.width,
                    "h": cap.bounding_box.height,
                },
            })

   # -----------------------------------------------------
    # Tags (may contain landmarks)
    # -----------------------------------------------------
    if hasattr(result, "tags") and result.tags:
        summary["tags"] = [
            {"tag": tag.name, "confidence": tag.confidence}
            for tag in result.tags.list
        ]

        # possible landmark detection
        summary["landmarks"] = [
            {"name": tag.name, "confidence": tag.confidence}
            for tag in result.tags.list
            if "landmark" in tag.name.lower()
        ]

    # -----------------------------------------------------
    # OCR (Read)
    # -----------------------------------------------------
    if hasattr(result, "read") and result.read and result.read.blocks:
        all_text = []
        for block in result.read.blocks:
            for line in block.lines:
                all_text.append(line.text)
        summary["ocr_text"] = "\n".join(all_text)

 

    return summary