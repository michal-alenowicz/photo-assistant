# storage_manager.py - Azure Blob Storage for Google app version

from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime
import json
import io


class AzureStorageManager:
    """Manage user uploads in Azure Blob Storage (for Google Cloud Vision + OpenAI app)"""
    
    def __init__(self, connection_string: str, container_name: str = "user-uploads"):
        """Initialize Azure Blob Storage client"""
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.container_client = self.blob_service.get_container_client(container_name)
        
        # Create container if doesn't exist
        try:
            self.container_client.create_container()
            print(f"✅ Created container: {container_name}")
        except Exception as e:
            # Container already exists
            print(f"✅ Using existing container: {container_name}")
    
    def save_analysis(
        self, 
        image_bytes: bytes, 
        image_name: str, 
        result_json: dict, 
        user_context: str = "",
        web_context: dict = None,
        safety_results: dict = None
    ) -> str:
        """
        Save image and complete analysis results to Azure Blob Storage
        
        Args:
            image_bytes: Original image binary data
            image_name: Original filename
            result_json: Analysis results (caption, tags, vision_summary)
            user_context: User-provided or auto-detected context
            web_context: Web entity detection results (optional)
            safety_results: Content safety check results (optional)
            
        Returns:
            analysis_id: Unique identifier for this analysis
        """
        # Generate unique ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        analysis_id = f"analysis_{timestamp}"
        
        # ===== SAVE IMAGE =====
        image_blob_name = f"images/{analysis_id}_{image_name}"
        image_blob_client = self.container_client.get_blob_client(image_blob_name)
        
        # Determine content type from filename
        content_type = self._get_content_type(image_name)
        
        image_blob_client.upload_blob(
            image_bytes,
            content_settings=ContentSettings(content_type=content_type),
            overwrite=True
        )
        
        # ===== PREPARE COMPREHENSIVE JSON =====
        json_data = {
            # Basic metadata
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "original_filename": image_name,
            "file_size_bytes": len(image_bytes),
            "image_blob_name": image_blob_name,
            
            # Context information (NEW - tracking all context)
            "context": {
                "user_provided": user_context,  # What user typed or accepted
                "was_auto_detected": bool(web_context and web_context.get('suggested_context')),
                "web_detection_used": web_context is not None
            },
            
            # Web entity detection results (NEW)
            "web_detection": web_context if web_context else None,
            
            # Content safety results (NEW - Google SafeSearch)
            "safety_check": {
                "is_safe": safety_results.get('is_safe', True) if safety_results else True,
                "flags": safety_results.get('flags', []) if safety_results else [],
                "numeric_scores": safety_results.get('numeric_scores', {}) if safety_results else {}
            } if safety_results else None,
            
            # Analysis results (caption, tags, vision data)
            "results": {
                "caption": result_json.get("caption"),
                "tags": result_json.get("tags", []),
                "vision_summary": result_json.get("vision_summary", {})
            },
            
            # Technology stack info
            "technology": {
                "vision_api": "Google Cloud Vision",
                "llm_model": "OpenAI GPT-5",
                "safety_api": "Google SafeSearch",
                "web_detection": "Google Cloud Vision Web Detection"
            }
        }
        
        # ===== SAVE JSON =====
        json_blob_name = f"results/{analysis_id}.json"
        json_blob_client = self.container_client.get_blob_client(json_blob_name)
        
        json_bytes = json.dumps(json_data, ensure_ascii=False, indent=2).encode('utf-8')
        json_blob_client.upload_blob(
            json_bytes,
            content_settings=ContentSettings(content_type="application/json"),
            overwrite=True
        )
        
        print(f"✅ Saved analysis {analysis_id} to Azure Blob Storage")
        
        return analysis_id
    
    def _get_content_type(self, filename: str) -> str:
        """Determine MIME type from filename"""
        extension = filename.lower().split('.')[-1]
        
        content_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp',
            'ico': 'image/x-icon',
            'tiff': 'image/tiff',
            'mpo': 'image/mpo'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    def get_storage_stats(self) -> dict:
        """Get storage usage statistics"""
        try:
            blob_list = self.container_client.list_blobs()
            
            total_size = 0
            image_count = 0
            json_count = 0
            
            for blob in blob_list:
                total_size += blob.size
                if blob.name.startswith("images/"):
                    image_count += 1
                elif blob.name.startswith("results/"):
                    json_count += 1
            
            return {
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "image_count": image_count,
                "json_count": json_count,
                "total_analyses": json_count
            }
        
        except Exception as e:
            print(f"⚠️ Error getting storage stats: {e}")
            return {
                "total_size_mb": 0,
                "image_count": 0,
                "json_count": 0,
                "total_analyses": 0
            }