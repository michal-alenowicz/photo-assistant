from google.cloud import vision
from google.oauth2 import service_account
import openai
from typing import Dict, Optional
import json
import os
import config


class ImageAnalyzer:
    """
    Image analyzer using Google Cloud Vision + OpenAI GPT
    """
    
    def __init__(
        self,
        google_credentials_path: str,
        google_project_id: str,
        openai_api_key: str,
        openai_model: str = "gpt-4.1"
    ):
        """
        Initialize analyzer with Google and OpenAI credentials
        
        """
        # Initialize Google Cloud Vision client
        if os.path.exists(google_credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                google_credentials_path
            )
            self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # For Streamlit Cloud - credentials from secrets
            import streamlit as st
            if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
                creds_dict = dict(st.secrets['google_credentials'])
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                # Default credentials (if GOOGLE_APPLICATION_CREDENTIALS env var is set)
                self.vision_client = vision.ImageAnnotatorClient()
        
        self.project_id = google_project_id
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model

        # Detect if using GPT-5 family for parameter compatibility
        self.is_gpt5 = openai_model.startswith("gpt-5")
    
    def analyze_with_computer_vision(self, image_bytes: bytes) -> Dict:
        """
        Analyze image using Google Cloud Vision API
        Returns structured summary
        """
        image = vision.Image(content=image_bytes)
        
        # Request multiple features in one API call
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=20),
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=10),
            vision.Feature(type_=vision.Feature.Type.FACE_DETECTION, max_results=10),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION, max_results=5),
        ]
        
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = self.vision_client.annotate_image(request=request)
        
        # Parse response into structured format
        summary = self._parse_vision_results(response)
        
        return summary
    
    def _parse_vision_results(self, response) -> Dict:
        """
        Parse Google Vision API response into structured format
        """
        summary = {}
        
        # Labels
        if response.label_annotations:
            summary["tags"] = [
                {
                    "tag": label.description,
                    "confidence": label.score
                }
                for label in response.label_annotations
            ]
        
        # Objects with localization
        if response.localized_object_annotations:
            summary["objects"] = [
                {
                    "name": obj.name,
                    "confidence": obj.score
                }
                for obj in response.localized_object_annotations
            ]
        
        # Face detection (emotions)
        if response.face_annotations:
            summary["faces"] = []
            for face in response.face_annotations:
                emotions = {
                    "joy": face.joy_likelihood.name,
                    "sorrow": face.sorrow_likelihood.name,
                    "anger": face.anger_likelihood.name,
                    "surprise": face.surprise_likelihood.name,
                }
                summary["faces"].append({
                    "emotions": emotions,
                    "confidence": face.detection_confidence
                })
        
        # Text detection (OCR)
        if response.text_annotations:
            # First annotation contains full text
            if len(response.text_annotations) > 0:
                summary["ocr_text"] = response.text_annotations[0].description
        
                
        # Landmarks
        if response.landmark_annotations:
            summary["landmarks"] = [
                {
                    "name": landmark.description,
                    "confidence": landmark.score
                }
                for landmark in response.landmark_annotations
            ]
        
        # Generate main caption from labels (Google doesn't have built-in captions)
        if summary.get("tags"):
            top_labels = [tag["tag"] for tag in summary["tags"][:3]]
            summary["main_caption"] = [{
                "text": f"Image showing {', '.join(top_labels)}",
                "confidence": summary["tags"][0]["confidence"]
            }]
        
        
        return summary
    
    def _build_prompt(self, vision_summary: Dict, user_context: str = "", safety_context: Dict = None) -> str:
        """
        Build structured prompt for GPT from vision analysis results
        """
        prompt = (
            "Wygeneruj opis krótki zdjęcia oraz listę tagów (5-8 tagów). Opis zdjęcia powinien być utrzymany w dziennikarskim stylu i nadawać się do publikacji (1-2 zdania). Opis będzie opublikowany razem ze zdjęciem, nie może być suchym opisem widoku.\n" 
            "Nie opisuj oczywistych elementów widocznych na zdjęciu (kolorów, kształtów, wzajemnego położenia elementów), nie opisuj atmosfery zdjęcia; unikaj sformułowań w rodzaju: 'scena oddaje...', 'na zdjęciu widać...', 'zdjęcie przedstawia...', 'zdjęcie zwraca uwagę na', 'ujęcie oddaje', 'fotografia przywołuje', 'widok przypomina' i wszystkich równoważnych. Opis będzie opublikowany pod zdjęciem (obrazem), a więc unikaj słów typu 'zdjęcie', 'obraz', 'rysuek', 'scena'.\n"
            "W przypadku zdjęć symbolicznych/ilustracyjnych nie pisz jednoznacznie, że 'obiekt X symbolizuje pojęcie Y', 'osoba X wykonuje czynność Z, przywołując pojęcie Y' czy 'X to symbol Y'. Odwołaj się od razu do abstrakcyjnego pojęcia Y, omijając opis obiektu X oraz opis treści zdjęcia.\n"
            "Nie opisuj wyglądu (włosów, twarzy, ubioru) ludzi. Nie opisuj gestów i czynności wykonywanych przez ludzi; nie pisz wprost, co te gesty wyrażają. Odwołuj się od razu w sposób ogólny do wyrażanych przez nie abstrakcyjnych pojęć, emocji oraz powiązanej problematyki.\n"
            "Odwołaj się do kontekstu i ogólnej wiedzy o widocznym zjawisku, możesz przywołać ogólne prawdy i znane fakty związane z tematem, powiązaną problematykę, problemy społeczne, również historyczne fakty z życia widocznych osób, narodów czy grup społecznych.\n"
            "Użyj naturalnego języka polskiego. Staraj się, by opis był utrzymany w tonie profesjonalnego dziennikarstwa, unikaj romantyzmu i sensacyjności. Nie używaj sformułowań wyrażających spekulację (typu 'być może', 'zapewne') - opis musi nadawać się do publikacji.\n"
            "Zwróć odpowiedź TYLKO w formacie JSON: z kluczami: {\"caption\": \"(string 1-2 zdania)\",\"tags\": [lista obiektów 'string']}.\n"
            "ZASADY KRYTYCZNE: 1) Odpowiadaj TYLKO poprawnym obiektem JSON. 2) NIE dodawaj żadnych wyjaśnień, tekstu, bloków kodu ani formatowania Markdown. 3) NIE otaczaj JSON-a blokami '''json ani '''. 4) NIE dodawaj komentarzy ani przecinków na końcu listy/obiektu. 5) Wynik MUSI być ściśle poprawnym JSON-em.\n"
            "Poniżej analiza zdjęcia (wyniki z Google Cloud Vision):\n\n"
        )
        
        #cleaned_summary = vision_summary
        
        prompt += f"\n\n{'='*60}\n"
        prompt += f"Vision JSON: {vision_summary}."
        
        # Safety context
        if safety_context and not safety_context.get('is_safe'):
            prompt += f"\n\n{'='*60}\n"
            prompt += "INFORMACJA O MODERACJI TREŚCI:\n"
            prompt += "System wykrył potencjalnie wrażliwe treści w tej kategorii:\n"
            
            # Category names in Polish
            category_names_pl = {
                'adult': 'Treści seksualne/dla dorosłych',
                'violence': 'Przemoc/Treści drastyczne',
                'racy': 'Treści prowokacyjne',
                'spoof': 'Zmanipulowane treści (deepfake)',
                'medical': 'Treści medyczne/chirurgiczne'
            }
            
            # Get numeric scores for all categories (even those below threshold)
            numeric_scores = safety_context.get('numeric_scores', {})
            
            # Show flagged categories first (those that exceeded threshold)
            for flag in safety_context.get('flags', []):
                cat_name = flag['category']
                cat_pl = category_names_pl.get(cat_name, cat_name)
                likelihood = safety_context['details'][cat_name]['likelihood_label']
                severity = flag.get('severity', flag.get('likelihood_value', 0))
                
                prompt += f"⚠️ **{cat_pl}**: {likelihood} (poziom {severity}/5)\n"
            
            prompt += "\n"
            prompt += "INSTRUKCJE dla treści wrażliwych:\n"
            prompt += "- Możesz nazwać rzeczy po imieniu, jeśli to stosowne w kontekście dziennikarskim\n"
            prompt += "- Unikaj przesadnego dystansowania się od tematu\n"
            prompt += "- W przypadku obrazów o silnym charakterze seksualnym (5/5) założ, że masz do czynienia z treściami pornograficznymi / uprzedmiotowieniem człowieka, jeżeli nie jest to ewidentnie dzieło sztuki.\n"
            prompt += "- Kontekst społeczny/kulturowy jest ważniejszy niż naiwny opis\n"
            prompt += "- Informacja o charakterze 'kontrowersyjnych' treści ma ci posłużyć do lepszego zrozumienia kontekstu - nie generuj żadnych ostrzeżeń ani przestróg dla użytkownika.\n"
            prompt += f"{'='*60}\n\n"
        
        # User context
        if user_context and user_context.strip():
            prompt += f"{'='*60}\n"
            prompt += f"DODATKOWY KONTEKST OD UŻYTKOWNIKA:\n"
            prompt += f"{user_context}\n"
            prompt += f"{'='*60}\n"
            prompt += """
WAŻNE: Wykorzystaj te informacje, aby wzbogacić opis zdjęcia. Jeśli kontekst zawiera 
nazwiska osób, daty, nazwy miejsc lub wydarzeń, uwzględnij je w opisie i tagach. W przypadku konfliktu z treścią odczytaną z OCR, potraktuj priorytetowo OCR.
"""
        
        return prompt
    

    
    def generate_caption_and_tags(
        self, 
        vision_summary: Dict, 
        user_context: str = "",
        safety_context: Dict = None
    ) -> Dict:
        """
        Generate Polish caption and tags using OpenAI GPT
        
        Args:
            vision_summary: Output from analyze_with_computer_vision()
            user_context: Optional user-provided context
            safety_context: Optional content safety analysis results
            
        Returns:
            Dict with 'caption' and 'tags' keys
        """
        system_prompt = (
            "Jesteś asystentem generującym krótki opis zdjęcia i listę tagów (5-8 tagów) w języku polskim.\n"
            "Weź pod uwagę wykryte etykiety, opis i tekst (OCR) dostarczony poniżej.\n"
            "Możliwe, że dziennikarz dostarczy krótki opis postaci, miejsc, wydarzeń i kontekstu (opcjonalnie). W takim wypadku należy KONIECZNIE uwzględnić dodatkowy kontekst od użytkownika.\n"
            "Jeśli otrzymasz informację o wykryciu wrażliwych treści, możesz wykorzystać ją jako informację o ogólnym charakterze zdjęcia.\n"
            "Zwróć odpowiedź TYLKO w formacie JSON: z kluczami: {\"caption\": \"(string 1-2 zdania)\",\"tags\": [lista obiektów 'string']}.\n"
            "ZASADY KRYTYCZNE: 1) Odpowiadaj TYLKO poprawnym obiektem JSON. 2) NIE dodawaj żadnych wyjaśnień, tekstu, bloków kodu ani formatowania Markdown. 3) NIE otaczaj JSON-a blokami '''json ani '''. 4) NIE dodawaj komentarzy ani przecinków na końcu listy/obiektu. 5) Wynik MUSI być ściśle poprawnym JSON-em."
        )
        
        user_prompt = self._build_prompt(vision_summary, user_context, safety_context)
        
        try:
            # GPT-5 compatible parameters
            if self.is_gpt5:
                # GPT-5 uses max_completion_tokens and reasoning parameters
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    reasoning_effort="none",
                    verbosity='low'
                )
            else:
                # GPT-4 and earlier use max_tokens and temperature
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=350,
                    temperature=0.2,
                )
            
            text = response.choices[0].message.content
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
        
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}
    
    def analyze_image(self, image_bytes: bytes, user_context: str = "", safety_context: Dict = None) -> Dict:
        """
        Complete image analysis pipeline
        
        Args:
            image_bytes: Image binary data
            user_context: Optional user-provided context
            safety_context: Optional content safety analysis results
            
        Returns:
            Dict with 'caption', 'tags', and 'vision_summary' keys
        """
        # Step 1: Analyze with Google Cloud Vision
        vision_summary = self.analyze_with_computer_vision(image_bytes)
        
        # Step 2: Generate Polish caption and tags with OpenAI
        result = self.generate_caption_and_tags(vision_summary, user_context, safety_context)
        
        # Step 3: Include vision summary for debugging
        result['vision_summary'] = vision_summary
        
        return result