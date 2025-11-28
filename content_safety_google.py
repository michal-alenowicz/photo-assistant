# content_safety_google.py - ENHANCED with all 5 categories + numeric conversion

from google.cloud import vision
from google.oauth2 import service_account
from typing import Dict
import os


class ContentSafetyChecker:
    """
    Google Cloud Vision SafeSearch for image moderation
    Categories: adult, medical, violence
    Likelihood levels: UNKNOWN, VERY_UNLIKELY, UNLIKELY, POSSIBLE, LIKELY, VERY_LIKELY
    """
    
    # Likelihood to numeric mapping (0-5 scale, similar to Azure's 0-6)
    LIKELIHOOD_MAP = {
        'UNKNOWN': 0,
        'VERY_UNLIKELY': 1,
        'UNLIKELY': 2,
        'POSSIBLE': 3,
        'LIKELY': 4,
        'VERY_LIKELY': 5
    }
    
    # Thresholds for journalism (permissive - news may contain disturbing content)
    THRESHOLDS = {
        'adult': 3,      # Alert on POSSIBLE or higher (level 3+)
        'medical': 4,    # Alert on LIKELY or higher (graphic medical)
        'violence': 4   # Alert on LIKELY or higher (allow news violence)
    }
    
    def __init__(self, google_credentials_path: str):
        """Initialize SafeSearch checker"""
        if os.path.exists(google_credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                google_credentials_path
            )
            self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
                    creds_dict = dict(st.secrets['google_credentials'])
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                else:
                    self.vision_client = vision.ImageAnnotatorClient()
            except:
                self.vision_client = vision.ImageAnnotatorClient()
    
    def analyze_image(self, image_bytes: bytes) -> Dict:
        """
        Analyze image for harmful content using Google SafeSearch
        
        Returns:
            Dict with:
            - is_safe (bool): Overall safety status
            - flags (list): Categories that exceeded thresholds
            - details (dict): All 5 categories with scores
            - numeric_scores (dict): Numeric version (0-5 scale)
        """
        try:
            image = vision.Image(content=image_bytes)
            response = self.vision_client.safe_search_detection(image=image)
            safe_search = response.safe_search_annotation
            
            result = self._parse_results(safe_search)
            return result
        
        except Exception as e:
            print(f"⚠️ SafeSearch API error: {e}")
            return {
                'is_safe': True,
                'flags': [],
                'details': {},
                'numeric_scores': {},
                'error': str(e)
            }
    
    def _parse_results(self, safe_search) -> Dict:
        """
        Parse Google SafeSearch results with ALL 5 categories
        """
        results = {
            'is_safe': True,
            'flags': [],
            'details': {},
            'numeric_scores': {},
            'raw': {
                'adult': safe_search.adult.name,
                'medical': safe_search.medical.name,
                'violence': safe_search.violence.name
            }
        }
        
        # All 5 categories
        categories = {
            'adult': safe_search.adult.name,
            'medical': safe_search.medical.name,
            'violence': safe_search.violence.name
        }
        
        for category, likelihood_name in categories.items():
            likelihood_value = self.LIKELIHOOD_MAP.get(likelihood_name, 0)
            
            # Store details with both string and numeric
            results['details'][category] = {
                'likelihood': likelihood_name,
                'likelihood_value': likelihood_value,
                'likelihood_label': self._get_likelihood_label(likelihood_value),
                'severity': likelihood_value  # Alias for Azure compatibility
            }
            
            # Store numeric scores separately for easy access
            results['numeric_scores'][category] = likelihood_value
            
            # Check threshold
            threshold = self.THRESHOLDS.get(category, 3)
            if likelihood_value >= threshold:
                results['is_safe'] = False
                results['flags'].append({
                    'category': category,
                    'likelihood': likelihood_name,
                    'likelihood_value': likelihood_value,
                    'severity': likelihood_value,  
                    'threshold': threshold
                })
        
        return results
    
    def _get_likelihood_label(self, likelihood_value: int) -> str:
        """Convert numeric likelihood to Polish label"""
        if likelihood_value == 0:
            return "Nieznane"
        elif likelihood_value == 1:
            return "Bardzo mało prawdopodobne"
        elif likelihood_value == 2:
            return "Mało prawdopodobne"
        elif likelihood_value == 3:
            return "Możliwe"
        elif likelihood_value == 4:
            return "Prawdopodobne"
        else:  # 5
            return "Bardzo prawdopodobne"
    
    def get_alert_message(self, results: Dict) -> str:
        """Generate Polish alert message for UI"""
        if results['is_safe']:
            return "✅ Treść bezpieczna - brak ostrzeżeń"
        
        category_names = {
            'adult': 'Treści seksualne/dla dorosłych',
            'medical': 'Treści medyczne/chirurgiczne',
            'violence': 'Przemoc/Treści drastyczne'
        }
        
        alert = "⚠️ OSTRZEŻENIE - wykryto potencjalnie niewłaściwą treść:\n\n"
        
        for flag in results['flags']:
            cat_pl = category_names.get(flag['category'], flag['category'])
            likelihood = results['details'][flag['category']]['likelihood_label']
            numeric = flag['likelihood_value']
            alert += f"• **{cat_pl}**: {likelihood} (poziom {numeric}/5)\n"
        
        return alert
    
    def get_all_details(self, results: Dict) -> str:
        """Get detailed breakdown of all 5 categories"""
        category_names = {
            'adult': 'Treści dla dorosłych',
            'medical': 'Treści medyczne',
            'violence': 'Przemoc'
        }
        
        details = "**Szczegółowa analiza moderacji (5 kategorii):**\n\n"
        
        for cat_name, cat_data in results['details'].items():
            cat_pl = category_names.get(cat_name, cat_name)
            likelihood_value = cat_data['likelihood_value']
            label = cat_data['likelihood_label']
            
            # Emoji based on severity
            if likelihood_value <= 1:
                emoji = "✅"
            elif likelihood_value == 2:
                emoji = "🟡"
            elif likelihood_value == 3:
                emoji = "🟠"
            else:  # 4-5
                emoji = "🔴"
            
            details += f"{emoji} **{cat_pl}**: {label} ({likelihood_value}/5)\n\n"
        
        return details
    
    def get_numeric_summary(self, results: Dict) -> str:
        """
        Get numeric summary (similar to Azure format)
        Useful for logging/analytics
        """
        scores = results.get('numeric_scores', {})
        
        summary = "Numeric Scores (0-5 scale):\n"
        for category, score in scores.items():
            summary += f"  {category}: {score}/5\n"
        
        return summary