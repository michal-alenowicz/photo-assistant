from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from typing import Dict, Optional
import json
import config


class ImageAnalyzer:
    """
    Image analyzer using Azure Computer Vision + Azure OpenAI
    Combines vision analysis and caption/tag generation in one class
    """
    
    def __init__(
        self,
        azure_vision_endpoint: str,
        azure_vision_key: str,
        azure_openai_endpoint: str,
        azure_openai_key: str,
        chat_deployment: str,
        api_version: str = "2025-01-01-preview"
    ):
        """
        Initialize analyzer with Azure credentials
        
        Args:
            azure_vision_endpoint: Azure Computer Vision endpoint
            azure_vision_key: Azure Computer Vision API key
            azure_openai_endpoint: Azure OpenAI endpoint
            azure_openai_key: Azure OpenAI API key
            chat_deployment: Name of GPT deployment (e.g., "gpt-5-chat")
            api_version: Azure OpenAI API version
        """
        # Initialize Computer Vision client
        self.vision_client = ImageAnalysisClient(
            endpoint=azure_vision_endpoint,
            credential=AzureKeyCredential(azure_vision_key)
        )
        
        # Initialize OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=api_version
        )
        
        self.chat_deployment = chat_deployment
    
    def analyze_with_computer_vision(self, image_bytes: bytes) -> Dict:
        """
        Analyze image using Azure Computer Vision API
        Returns structured summary with captions, tags, OCR
        """
        features = [
            "Caption",
            "DenseCaptions",
            "Read",
            "Tags"
        ]
        
        result = self.vision_client.analyze(
            image_data=image_bytes,
            visual_features=features,
            gender_neutral_caption=False
        )
        
        summary = {}
        
        # Main Caption
        if hasattr(result, "caption") and result.caption:
            summary["main_caption"] = [{
                "text": result.caption.text,
                "confidence": result.caption.confidence,
            }]
        
        # Dense Captions
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
        
        # Tags (including possible landmarks)
        if hasattr(result, "tags") and result.tags:
            summary["tags"] = [
                {"tag": tag.name, "confidence": tag.confidence}
                for tag in result.tags.list
            ]
            
            # Landmark detection
            summary["landmarks"] = [
                {"name": tag.name, "confidence": tag.confidence}
                for tag in result.tags.list
                if "landmark" in tag.name.lower()
            ]
        
        # OCR (Read)
        if hasattr(result, "read") and result.read and result.read.blocks:
            all_text = []
            for block in result.read.blocks:
                for line in block.lines:
                    all_text.append(line.text)
            summary["ocr_text"] = "\n".join(all_text)
        
        return summary
    
    def _build_prompt(self, vision_summary: Dict, user_context: str = "", safety_context: Dict = None) -> str:
        """
        Build structured prompt for GPT from vision analysis results
        Adds user context when present
        """


        prompt = (
        "Wygeneruj opis krótki zdjęcia oraz listę tagów (5-8 tagów). Opis zdjęcia powinien być utrzymany w dziennikarskim stylu i nadawać się do publikacji (1-2 zdania). Opis będzie opublikowany razem ze zdjęciem, nie może być suchym opisem widoku.\n" 
        "Nie opisuj oczywistych elementów widocznych na zdjęciu (kolorów, kształtów, wzajemnego położenia elementów), nie opisuj atmosfery zdjęcia; unikaj sformułowań w rodzaju: 'scena oddaje...', 'na zdjęciu widać...', 'zdjęcie przedstawia...', 'zdjęcie zwraca uwagę na', 'ujęcie oddaje', 'fotografia przywołuje', 'widok przypomina' i wszystkich równoważnych. Opis będzie opublikowany pod zdjęciem (obrazem), a więc unikaj słów typu 'zdjęcie', 'obraz', 'rysuek', 'scena' .\n"
        "W przypadku zdjęć symbolicznych/ilustracyjnych nie pisz jednoznacznie, że 'obiekt X symbolizuje pojęcie Y', 'osoba X wykonuje czynność Z, przywołując pojęcie Y' czy 'X to symbol Y'. Odwołaj się od razu do abstrakcyjnego pojęcia Y, omijając opis obiektu X oraz opis treści zdjęcia. \n"
        "Nie opisuj wyglądu (włosów, twarzy, ubioru) ludzi. Nie opisuj gestów i czynności wykonywanych przez ludzi; nie pisz wprost, co te gesty wyrażają. Odwołuj się od razu w sposób ogólny do wyrażanych przez nie abstrakcyjnych pojęć, emocji oraz powiązanej problematyki.\n"
        "Odwołaj się do kontekstu i ogólnej wiedzy o widocznym zjawisku, możesz przywołać ogólne prawdy i znane fakty związane z tematem, powiązaną problematykę, problemy społeczne, również historyczne fakty z życia widocznych osób, narodów czy grup społecznych.\n"
        "Użyj naturalnego języka polskiego. Staraj się, by opis był utrzymany w tonie profesjonalnego dziennikarstwa, unikaj romantyzmu i sensacyjności. Nie używaj sformułowań wyrażających spekulację (typu 'być może', 'zapewne') - opis musi nadawać się do publikacji.\n"
        "Zwróć odpowiedź TYLKO w formacie JSON: z kluczami: {\"caption\": \"(string 1-2 zdania)\",\"tags\": [lista obiektów 'string']}.\n"
        "ZASADY KRYTYCZNE: 1) Odpowiadaj TYLKO poprawnym obiektem JSON. 2) NIE dodawaj żadnych wyjaśnień, tekstu, bloków kodu ani formatowania Markdown. 3) NIE otaczaj JSON-a blokami '''json ani '''. 4) NIE dodawaj komentarzy ani przecinków na końcu listy/obiektu. 5) Wynik MUSI być ściśle poprawnym JSON-em.\n"
        "Poniżej analiza zdjęcia (wyniki z Azure Image Analysis):\n\n"
        )
        
        cleaned_summary = self._strip_bboxes(vision_summary)
    
        prompt += f"\n\n{'='*60}\n"
        prompt += f"Vision JSON: {cleaned_summary}."
        
        
        #moderation/safety results
        if safety_context and not safety_context.get('is_safe'):
            prompt += f"\n\n{'='*60}\n"
            prompt += "INFORMACJA O MODERACJI TREŚCI:\n"
            prompt += "System wykrył potencjalnie wrażliwe treści w tej kategorii:\n"
            
            for flag in safety_context.get('flags', []):
                category_pl = {
                    'Sexual': 'treści seksualne',
                    'Violence': 'przemoc',
                    'Hate': 'mowa nienawiści',
                    'SelfHarm': 'samookaleczenie'
                }.get(flag['category'], flag['category'])
                
                severity = flag['severity']
                prompt += f"- {category_pl.capitalize()} (poziom {severity}/6)\n"
            
            prompt += "\n"
            prompt += "INSTRUKCJE dla treści wrażliwych:\n"
            prompt += "- Możesz nazwać rzeczy po imieniu, jeśli to stosowne w kontekście dziennikarskim\n"
            prompt += "- Unikaj przesadnego dystansowania się od tematu\n"
            prompt += "- W przypadku obrazów o silnym charakterze seksualnym (ocena na poziomie 6/6) załóż, że masz do czynienia z treściami pornograficznymi / uprzedmiotowieniem człowieka. Napisz to w opisie zdjęcia.\n"
            prompt += "- Kontekst społeczny/kulturowy jest ważniejszy niż naiwny opis\n"
            prompt += "- Informacja o charakterze 'kontrowersyjnych' treści ma ci posłużyć do lepszego zrozumienia kontekstu i odniesienia się do nich - nie generuj żadnych ostrzeżeń ani przestróg dla użytkownika.\n"
            prompt += f"{'='*60}\n\n"
        
        
        
        # User context (if provided)
        if user_context and user_context.strip():
            prompt += f"{'='*60}\n"
            prompt += f"DODATKOWY KONTEKST OD UŻYTKOWNIKA:\n"
            prompt += f"{user_context}\n"
            prompt += f"{'='*60}\n"
            prompt += """
WAŻNE: Ten kontekst został dostarczony przez dziennikarza i jest wiarygodny. 
Wykorzystaj te informacje, aby wzbogacić opis zdjęcia. Jeśli kontekst zawiera 
nazwiska osób, daty, nazwy miejsc lub wydarzeń, uwzględnij je w opisie i tagach.

"""
        
        return prompt
    

    def _strip_bboxes(self, vision_summary: Dict) -> Dict:
        """
        Remove bounding box coordinates from vision summary
        Keeps only text descriptions that are useful
        """
        cleaned = {}
        
        # Main caption (no bbox)
        if "main_caption" in vision_summary:
            cleaned["main_caption"] = [
                {"text": cap["text"], "confidence": cap["confidence"]}
                for cap in vision_summary["main_caption"]
            ]
        
        # Dense captions (remove bbox)
        if "dense_captions" in vision_summary:
            cleaned["dense_captions"] = [
                {"text": cap["text"], "confidence": cap["confidence"]}
                for cap in vision_summary["dense_captions"]
            ]
        
        # Tags (no bbox)
        if "tags" in vision_summary:
            cleaned["tags"] = vision_summary["tags"]
        
        # Landmarks (no bbox)
        if "landmarks" in vision_summary:
            cleaned["landmarks"] = vision_summary["landmarks"]
        
        # OCR text (no bbox)
        if "ocr_text" in vision_summary:
            cleaned["ocr_text"] = vision_summary["ocr_text"]
        
        return cleaned


    def generate_caption_and_tags(
        self, 
        vision_summary: Dict, 
        user_context: str = "",
        safety_context: Dict = None
    ) -> Dict:
        """
        Generate Polish caption and tags using Azure OpenAI GPT
        
        Args:
            vision_summary: Output from analyze_with_computer_vision()
            user_context: Optional user-provided context
            
        Returns:
            Dict with 'caption' and 'tags' keys, or 'raw' if JSON parsing fails
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
            response = self.openai_client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,
                temperature=0.2,
            )
            
            text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                return {"raw": text}
        
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}
    
    def analyze_image(self, image_bytes: bytes, user_context: str = "", safety_context: Dict = None) -> Dict:
        """
        Complete image analysis pipeline
        
        Args:
            image_bytes: Image binary data
            user_context: Optional user-provided context
            
        Returns:
            Dict with 'caption', 'tags', and 'vision_summary' keys
        """
        # Step 1: Analyze with Computer Vision
        vision_summary = self.analyze_with_computer_vision(image_bytes)
        
        # Step 2: Generate Polish caption and tags
        result = self.generate_caption_and_tags(vision_summary, user_context, safety_context)
        
        # Step 3: Include vision summary for debugging
        result['vision_summary'] = vision_summary
        
        return result