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



def generate_caption_and_tags(vision_summary_json, user_context=""):

    system_prompt = (
        "Jesteś asystentem generującym krótki opis zdjęcia i listę tagów (5-8 tagów) w języku polskim.\n"
        "Weź pod uwagę wykryte etykiety, opis i tekst (OCR) dostarczony poniżej.\n"
        "Możliwe, że dziennikarz dostarczy krótki opis postaci, miejsc, wydarzeń i kontekstu (opcjonalnie). W takim wypadku należy KONIECZNIE uwzględnić dodatkowy kontekst od użytkownika.\n"
        "Opis zdjęcia powinien być utrzymany w dziennikarskim stylu i nadawać się do publikacji (1-2 zdania). Opis będzie opublikowany razem ze zdjęciem, nie może być suchym opisem widoku. Unikaj opisywania tego, co i tak już widać na zdjęciu.\n"
        "Nie opisuj oczywistych elementów widocznych na zdjęciu (kolorów, kształtów, wzajemnego położenia elementów), nie opisuj atmosfery zdjęcia; unikaj sformułowań typu 'scena oddaje...', 'na zdjęciu widać...', 'zdjęcie przedstawia...'.\n"
        "Nie pisz wprost, że 'obiekt X symbolizuje pojęcie Y', 'obiekt X przypomina o pojęciu Y', ale odnieś się OD RAZU do pojęcia Y, nie opisując obiektu X.\n"
        "Nie opisuj wyglądu (włosów, twarzy, ubioru) ludzi. Nie opisuj gestów i czynności wykonywanych przez ludzi; nie pisz wprost, co te gesty wyrażają. Odwołuj się od razu w sposób ogólny do wyrażanych przez nie abstrakcyjnych pojęć oraz powiązanej problematyki.\n"
        "Odwołaj się do kontekstu i ogólnej wiedzy o widocznym zjawisku, możesz przywołać ogólne prawdy i znane fakty związane z tematem, powiązaną problematykę, problemy społeczne, również historyczne fakty z życia widocznych osób, narodów czy grup społecznych.\n"
        "Użyj naturalnego języka polskiego.\n"
        "Zwróć odpowiedź TYLKO w formacie JSON: z kluczami: 'caption' (1-2 zdania), \n"
        "'tags' (lista obiektów 'tag').\n"
        
    )

    user_prompt = f"Vision JSON: {vision_summary_json}"

    # ========== ADD USER CONTEXT IF PROVIDED ==========
    if user_context and user_context.strip():
        user_prompt += f"\n\n{'='*60}\n"
        user_prompt += f"DODATKOWY KONTEKST OD UŻYTKOWNIKA:\n"
        user_prompt += f"{user_context}\n"
        user_prompt += f"{'='*60}\n"
        user_prompt += """
WAŻNE: Ten kontekst został dostarczony przez dziennikarza i jest wiarygodny. 
Wykorzystaj te informacje, aby wzbogacić opis zdjęcia. Jeśli kontekst zawiera 
nazwiska osób, daty, nazwy miejsc lub wydarzeń, uwzględnij je w opisie i tagach.
"""

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
