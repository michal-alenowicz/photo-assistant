import os
import json
import openai
from openai import AzureOpenAI
import config



client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_version=config.AZURE_OPENAI_API_VERSION
)

 



def generate_caption_and_tags(vision_summary_json, user_context=""):

    system_prompt = (
        "Jesteś asystentem generującym krótki opis zdjęcia i listę tagów (5-8 tagów) w języku polskim.\n"
        "Weź pod uwagę wykryte etykiety, opis i tekst (OCR) dostarczony poniżej.\n"
        "Możliwe, że dziennikarz dostarczy krótki opis postaci, miejsc, wydarzeń i kontekstu (opcjonalnie). W takim wypadku należy KONIECZNIE uwzględnić dodatkowy kontekst od użytkownika.\n"
        "Zwróć odpowiedź TYLKO w formacie JSON: z kluczami: {\"caption\": \"(string 1-2 zdania)\",\"tags\": [lista obiektów 'string']}.\n"
        "ZASADY KRYTYCZNE: 1) Odpowiadaj TYLKO poprawnym obiektem JSON. 2) NIE dodawaj żadnych wyjaśnień, tekstu, bloków kodu ani formatowania Markdown. 3) NIE otaczaj JSON-a blokami '''json ani '''. 4) NIE dodawaj komentarzy ani przecinków na końcu listy/obiektu. 5) Wynik MUSI być ściśle poprawnym JSON-em."
    )

    user_prompt = (
        "Wygeneruj opis krótki zdjęcia oraz listę tagów (5-8 tagów). Opis zdjęcia powinien być utrzymany w dziennikarskim stylu i nadawać się do publikacji (1-2 zdania). Opis będzie opublikowany razem ze zdjęciem, nie może być suchym opisem widoku.\n" 
        "Jeśli użytkownik nie podał żadnego kontekstu i jeśli analiza OCR nie pozwala na identyfikację obiektów, to w ramach opisu ('caption') Masz ZAKAZ opisywania tego (osób, czynności, obiektów), co i tak już widać na zdjęciu.\n"
        "Nie opisuj oczywistych elementów widocznych na zdjęciu (kolorów, kształtów, wzajemnego położenia elementów), nie opisuj atmosfery zdjęcia; unikaj sformułowań w rodzaju: 'scena oddaje...', 'na zdjęciu widać...', 'zdjęcie przedstawia...', 'zdjęcie zwraca uwagę na', 'ujęcie oddaje', 'fotografia przywołuje' i wszystkich równoważnych. Opis będzie opublikowany pod zdjęciem (obrazem), a więc unikaj słów typu 'zdjęcie', 'obraz', 'rysuek', 'scena' .\n"
        "W przypadku zdjęć symbolicznych/ilustracyjnych nie pisz jednoznacznie, że 'obiekt X symbolizuje pojęcie Y', 'osoba X robi czynność Z, przywołując pojęcie Y' czy 'X to symbol Y'. Odwołaj się od razu do abstrakcyjnego pojęcia Y, omijając opis obiektu X oraz opis treści zdjęcia. \n"
        "Nie opisuj wyglądu (włosów, twarzy, ubioru) ludzi. Nie opisuj gestów i czynności wykonywanych przez ludzi; nie pisz wprost, co te gesty wyrażają. Odwołuj się od razu w sposób ogólny do wyrażanych przez nie abstrakcyjnych pojęć, emocji oraz powiązanej problematyki.\n"
        "Odwołaj się do kontekstu i ogólnej wiedzy o widocznym zjawisku, możesz przywołać ogólne prawdy i znane fakty związane z tematem, powiązaną problematykę, problemy społeczne, również historyczne fakty z życia widocznych osób, narodów czy grup społecznych.\n"
        "Użyj naturalnego języka polskiego. Staraj się, by opis był utrzymany w tonie profesjonalnego dziennikarstwa, unikaj romantyzmu i sensacyjności. Nie używaj sformułowań wyrażających spekulację (typu 'być może', 'zapewne') - opis musi nadawać się do publikacji.\n"
        "Zwróć odpowiedź TYLKO w formacie JSON: z kluczami: {\"caption\": \"(string 1-2 zdania)\",\"tags\": [lista obiektów 'string']}.\n"
        "ZASADY KRYTYCZNE: 1) Odpowiadaj TYLKO poprawnym obiektem JSON. 2) NIE dodawaj żadnych wyjaśnień, tekstu, bloków kodu ani formatowania Markdown. 3) NIE otaczaj JSON-a blokami '''json ani '''. 4) NIE dodawaj komentarzy ani przecinków na końcu listy/obiektu. 5) Wynik MUSI być ściśle poprawnym JSON-em.\n"
    )
    
    
    user_prompt += f"\n\n{'='*60}\n"
    user_prompt += f"Vision JSON: {vision_summary_json}."

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
        model = config.CHAT_DEPLOYMENT,          # IMPORTANT for Azure OpenAI
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=350,
        temperature=0.2,
    )

    text = response.choices[0].message.content

    # Expected JSON output
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}



# def get_embedding(text: str):
#     emb = client.embeddings.create(
#         model = config.EMBEDDING_DEPLOYMENT,
#         input=text
#     )
#     return emb.data[0].embedding
