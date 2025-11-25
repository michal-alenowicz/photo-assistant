# vision.py
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os

endpoint = os.getenv("AZURE_VISION_ENDPOINT")
key = os.getenv("AZURE_VISION_KEY")
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

def analyze_image_bytes(image_bytes):
    """Return dict: captions, tags (with confidence), ocr_text (if any)."""
    from io import BytesIO
    stream = BytesIO(image_bytes)
    # describe
    desc = client.describe_image_in_stream(stream, max_candidates=3)
    captions = [{"text": c.text, "confidence": c.confidence} for c in (desc.captions or [])]
    # Reset stream pointer
    stream.seek(0)
    # tags
    tags_res = client.tag_image_in_stream(stream)
    tags = [{"name": t.name, "confidence": t.confidence} for t in (tags_res.tags or [])]
    # OCR (read)
    stream.seek(0)
    ocr = client.read_in_stream(stream, raw=True)
    # azure read is async — get result
    operation_location_remote = ocr.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    import time
    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(0.5)
    ocr_text = ""
    if result.status == 'succeeded':
        for page in result.analyze_result.read_results:
            for line in page.lines:
                ocr_text += line.text + "\n"
    return {"captions": captions, "tags": tags, "ocr_text": ocr_text}
