import requests
import base64
from typing import Dict, List
import config


class ContentSafetyChecker:
    """
    Azure AI Content Safety for image moderation
    Categories: Hate, Violence, Sexual, SelfHarm
    Severity levels: 0 (safe) to 6 (severe)
    """
    
    def __init__(self, endpoint: str, key: str):
        """
        Initialize Content Safety checker
        
        Args:
            endpoint: Azure Content Safety endpoint
            key: Azure Content Safety API key
        """
        self.endpoint = endpoint.rstrip('/')
        self.key = key
        self.api_version = "2024-02-15-preview"
    
    def analyze_image(self, image_bytes: bytes) -> Dict:
        """
        Analyze image for harmful content
        
        Args:
            image_bytes: Image binary data
            
        Returns:
            Dict with safety analysis results
        """
        url = f"{self.endpoint}/contentsafety/image:analyze"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json"
        }
        
        params = {
            "api-version": self.api_version
        }
        
        # Convert image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        body = {
            "image": {
                "content": image_base64
            }
        }
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                json=body,
                timeout=30
            )
            response.raise_for_status()
            
            return self._parse_results(response.json())
        
        except requests.exceptions.RequestException as e:
            # If Content Safety fails, don't block - just return safe result
            print(f"⚠️ Content Safety API error: {e}")
            return {
                'is_safe': True,
                'flags': [],
                'details': {},
                'error': str(e)
            }
    
    def _parse_results(self, raw_results: Dict) -> Dict:
        """
        Parse and interpret Content Safety results
        Severity: 0 (Safe) to 6 (Severe)
        """
        categories = raw_results.get('categoriesAnalysis', [])
        
        results = {
            'is_safe': True,
            'flags': [],
            'details': {},
            'raw': raw_results  # Keep for debugging
        }
        
        # Thresholds for journalism (more permissive than default)
        # News images may contain violence/disturbing content
        THRESHOLDS = {
            'Hate': 4,      # Alert only on severe hate speech
            'SelfHarm': 4,  # Alert on severe self-harm
            'Sexual': 2,    # Alert on moderate sexual content
            'Violence': 4   # Allow news violence
        }
        
        for category in categories:
            cat_name = category.get('category')
            severity = category.get('severity', 0)
            
            # Store all details
            results['details'][cat_name] = {
                'severity': severity,
                'severity_label': self._get_severity_label(severity)
            }
            
            # Check if exceeds threshold
            threshold = THRESHOLDS.get(cat_name, 2)
            if severity >= threshold:
                results['is_safe'] = False
                results['flags'].append({
                    'category': cat_name,
                    'severity': severity,
                    'threshold': threshold,
                    'severity_label': self._get_severity_label(severity)
                })
        
        return results
    
    def _get_severity_label(self, severity: int) -> str:
        """Convert numeric severity to Polish label"""
        if severity == 0:
            return "Bezpieczne"
        elif severity <= 2:
            return "Niskie"
        elif severity <= 4:
            return "Średnie"
        else:
            return "Wysokie"
    
    def get_alert_message(self, results: Dict) -> str:
        """
        Generate Polish alert message for UI
        
        Args:
            results: Output from analyze_image()
            
        Returns:
            Polish alert message
        """
        if results['is_safe']:
            return "✅ Treść bezpieczna - brak ostrzeżeń"
        
        # Map categories to Polish
        category_names = {
            'Hate': 'Mowa nienawiści',
            'Violence': 'Przemoc/Treści drastyczne',
            'Sexual': 'Treści seksualne',
            'SelfHarm': 'Samookaleczenie'
        }
        
        alert = "⚠️ OSTRZEŻENIE - wykryto potencjalnie niewłaściwą treść:\n\n"
        
        for flag in results['flags']:
            cat_pl = category_names.get(flag['category'], flag['category'])
            alert += f"• **{cat_pl}**: {flag['severity_label']} (poziom {flag['severity']}/6)\n"
        
        return alert
    
    def get_all_details(self, results: Dict) -> str:
        """
        Get detailed breakdown of all categories (for expander)
        """
        category_names = {
            'Hate': 'Mowa nienawiści',
            'Violence': 'Przemoc',
            'Sexual': 'Treści seksualne',
            'SelfHarm': 'Samookaleczenie'
        }
        
        details = "**Szczegółowa analiza moderacji:**\n\n"
        
        for cat_name, cat_data in results['details'].items():
            cat_pl = category_names.get(cat_name, cat_name)
            severity = cat_data['severity']
            label = cat_data['severity_label']
            
            # Add emoji based on severity
            if severity == 0:
                emoji = "✅"
            elif severity <= 2:
                emoji = "🟡"
            elif severity <= 4:
                emoji = "🟠"
            else:
                emoji = "🔴"
            
            details += f"{emoji} **{cat_pl}**: {label} ({severity}/6)\n\n"
        
        return details