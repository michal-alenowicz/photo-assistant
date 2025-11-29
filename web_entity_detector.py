
from google.cloud import vision
from google.oauth2 import service_account
from typing import Dict, List, Optional
import os


class WebEntityDetector:
    """
    Detect web entities, best guess labels, and matching pages
    Uses Google Cloud Vision Web Detection API
    """
    
    def __init__(self, google_credentials_path: str):
        """Initialize Web Entity Detector with Google credentials"""
        
        # Initialize Google Cloud Vision client (same as ImageAnalyzer)
        if os.path.exists(google_credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                google_credentials_path
            )
            self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            print("✅ Web Entity Detector initialized from file")
        else:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
                    creds_dict = dict(st.secrets['google_credentials'])
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                    print("✅ Web Entity Detector initialized from Streamlit secrets")
                else:
                    raise Exception("Google credentials not found")
            except Exception as e:
                print(f"❌ Error loading credentials: {e}")
                raise
    
    def detect_web_context(self, image_bytes: bytes) -> Dict:
        """
        Detect web entities and context from image
        
        Args:
            image_bytes: Image binary data
            
        Returns:
            Dict with:
            - best_guess_label: Most likely description of the image
            - web_entities: List of detected entities (people, places, things)
            - matching_pages: Pages where this image appears
            - suggested_context: Auto-generated context string for user
        """
        try:
            image = vision.Image(content=image_bytes)
            
            # Call Web Detection API
            response = self.vision_client.web_detection(image=image)
            annotations = response.web_detection
            
            result = self._parse_web_detection(annotations)
            
            # Generate suggested context for user
            result['suggested_context'] = self._generate_context_suggestion(result)
            
            return result
        
        except Exception as e:
            print(f"⚠️ Web Detection error: {e}")
            return {
                'best_guess_label': None,
                'web_entities': [],
                'matching_pages': [],
                'suggested_context': '',
                'error': str(e)
            }
    
    def _parse_web_detection(self, annotations) -> Dict:
        """Parse Web Detection API response"""
        result = {
            'best_guess_label': None,
            'web_entities': [],
            'matching_pages': [],
            'visually_similar_images': []
        }
        
        # Best guess label (most likely description)
        if annotations.best_guess_labels:
            # Usually only one, but take the first
            result['best_guess_label'] = annotations.best_guess_labels[0].label
        
        # Web entities (people, places, things detected)
        if annotations.web_entities:
            for entity in annotations.web_entities:
                # Only include entities with description and reasonable score
                if entity.description and entity.score > 0.5:
                    result['web_entities'].append({
                        'description': entity.description,
                        'score': entity.score,
                        'entity_id': entity.entity_id if hasattr(entity, 'entity_id') else None
                    })
            
            # Sort by score (highest first)
            result['web_entities'].sort(key=lambda x: x['score'], reverse=True)
        
        # Pages with matching images
        if annotations.pages_with_matching_images:
            for page in annotations.pages_with_matching_images[:5]:  # Limit to top 5
                page_info = {
                    'url': page.url,
                    'page_title': page.page_title if hasattr(page, 'page_title') else None,
                    'full_matches_count': len(page.full_matching_images) if page.full_matching_images else 0,
                    'partial_matches_count': len(page.partial_matching_images) if page.partial_matching_images else 0
                }
                result['matching_pages'].append(page_info)
        
        # Visually similar images (optional - can be useful for verification)
        if annotations.visually_similar_images:
            for img in annotations.visually_similar_images[:3]:  # Limit to top 3
                result['visually_similar_images'].append({
                    'url': img.url
                })
        
        return result
    
    def _generate_context_suggestion(self, detection_result: Dict) -> str:
        """
        Generate suggested context string from web detection results
        This can be used to auto-populate the context field
        """
        context_parts = []
        
        # Add best guess label if available
        if detection_result['best_guess_label']:
            context_parts.append(detection_result['best_guess_label'])
        
        # Add top web entities (max 5, score > 0.7 for higher confidence)
        high_confidence_entities = [
            entity['description'] 
            for entity in detection_result['web_entities'][:3]
            if entity['score'] > 0.99
        ]
        
        if high_confidence_entities:
            context_parts.extend(high_confidence_entities)
        
        # Join with commas
        return ', '.join(context_parts) if context_parts else ''
    
    def get_entity_summary(self, detection_result: Dict) -> str:
        """
        Get human-readable summary of detected entities
        Useful for displaying to user
        """
        summary = []
        
        if detection_result['best_guess_label']:
            summary.append(f"🎯 **Najlepsze dopasowanie**: {detection_result['best_guess_label']}")
        
        if detection_result['web_entities']:
            summary.append(f"\n📋 **Wykryte elementy** ({len(detection_result['web_entities'])}):")
            for i, entity in enumerate(detection_result['web_entities'][:8], 1):
                confidence = int(entity['score'] * 100)
                summary.append(f"  {i}. {entity['description']} ({confidence}%)")
        
        if detection_result['matching_pages']:
            summary.append(f"\n🔗 **Strony z podobnymi obrazami**: {len(detection_result['matching_pages'])}")
            for page in detection_result['matching_pages'][:3]:
                title = page['page_title'] or 'bez tytułu'
                summary.append(f"  • {title}")
        
        return '\n'.join(summary) if summary else 'Brak wykrytych elementów'