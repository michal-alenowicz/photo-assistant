# storage_manager.py (NEW FILE)

from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime
import json
import io

class AzureStorageManager:
    """Manage user uploads in Azure Blob Storage"""
    
    def __init__(self, connection_string: str, container_name: str = "user-uploads"):
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.container_client = self.blob_service.get_container_client(container_name)
        
        # Create container if doesn't exist
        try:
            self.container_client.create_container()
        except:
            pass  # Container already exists
    
    def save_analysis(self, image_bytes: bytes, image_name: str, result_json: dict, user_context: str = "") -> str:
        """
        Save image and results to Azure Blob Storage
        
        Returns: analysis_id
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        analysis_id = f"analysis_{timestamp}"
        
        # Save image
        image_blob_name = f"images/{analysis_id}_{image_name}"
        image_blob_client = self.container_client.get_blob_client(image_blob_name)
        
        image_blob_client.upload_blob(
            image_bytes,
            content_settings=ContentSettings(content_type="image/jpeg"),
            overwrite=True
        )
        
        # Prepare and save JSON
        json_data = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "original_filename": image_name,
            "image_blob_name": image_blob_name,
            "user_context": user_context,
            "results": result_json
        }
        
        json_blob_name = f"results/{analysis_id}.json"
        json_blob_client = self.container_client.get_blob_client(json_blob_name)
        
        json_bytes = json.dumps(json_data, ensure_ascii=False, indent=2).encode('utf-8')
        json_blob_client.upload_blob(
            json_bytes,
            content_settings=ContentSettings(content_type="application/json"),
            overwrite=True
        )
        
        return analysis_id
    
    def get_storage_stats(self) -> dict:
        """Get storage usage statistics"""
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
            "total_size_mb": total_size / (1024 * 1024),
            "image_count": image_count,
            "json_count": json_count
        }