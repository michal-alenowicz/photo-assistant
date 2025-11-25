import os
import json
import openai
from openai import AzureOpenAI

AZURE_OPENAI_API_VERSION='2025-01-01-preview'
AZURE_OPENAI_EMBEDDINGS_API_VERSION='2023-05-15'

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=AZURE_OPENAI_API_VERSION
)


CHAT_DEPLOYMENT = "gpt-5-chat"             
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"  



def generate_caption_and_tags(vision_summary_json):

    system_prompt = (
        "Jesteś asystentem generującym krótki opis zdjęcia i listę tagów w języku polskim.\n"
        "Weź pod uwagę wykryte etykiety, opis i tekst (OCR) dostarczony poniżej.\n"
        "Opis zdjęcia powinien być utrzymany w dziennikarskim stylu i nadawać się do publikacji (1-2 zdania). Nie może być suchym opisem widoku\n"
        "Nie opisuj oczywistych elementów widocznych na zdjęciu (kolorów i kształtów); unikaj sformułowań typu 'scena oddaje...', 'na zdjęciu widać...'. Odwołaj się do kontekstu i ogólnej wiedzy o widocznym zjawisku, możesz przywołać ogólne prawdy i znane fakty związane z tematem.\n"
        "Zwróć w formacie JSON z kluczami: 'caption' (1-2 zdania), "
        "'tags' (lista obiektów {tag, confidence}).\n"
        "Użyj naturalnego języka polskiego."
    )

    user_prompt = f"Vision JSON: {vision_summary_json}"

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,          # IMPORTANT for Azure OpenAI
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.2,
    )

    text = response.choices[0].message.content

    # Expected JSON output
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}



def get_embedding(text: str):
    emb = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return emb.data[0].embedding
