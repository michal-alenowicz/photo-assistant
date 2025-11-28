# review_uploads.py - Streamlit app to review user uploads

import streamlit as st
from azure.storage.blob import BlobServiceClient
from PIL import Image
import json
import config
import io
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get from your .env or Streamlit secrets
AZURE_STORAGE_CONNECTION_STRING = config.AZURE_STORAGE_CONNECTION_STRING
CONTAINER_NAME = config.AZURE_STORAGE_CONTAINER

# ============================================================================
# BLOB STORAGE HELPER
# ============================================================================

@st.cache_resource
def get_blob_service():
    """Initialize Azure Blob Storage client"""
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def list_all_analyses():
    """Get list of all analysis IDs from JSON files"""
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    
    analyses = []
    
    # List all JSON files in results folder
    blob_list = container_client.list_blobs(name_starts_with="results/")
    
    for blob in blob_list:
        if blob.name.endswith('.json'):
            # Extract analysis ID from filename
            # e.g., "results/analysis_20241128_120530_123456.json"
            analysis_id = blob.name.replace("results/", "").replace(".json", "")
            
            analyses.append({
                'id': analysis_id,
                'timestamp': blob.last_modified,
                'json_blob': blob.name
            })
    
    # Sort by timestamp (newest first)
    analyses.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return analyses

def load_analysis(analysis_id):
    """Load image and JSON for a specific analysis"""
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    
    result = {
        'id': analysis_id,
        'image': None,
        'json_data': None,
        'error': None
    }
    
    try:
        # Load JSON
        json_blob_name = f"results/{analysis_id}.json"
        json_blob = container_client.get_blob_client(json_blob_name)
        json_content = json_blob.download_blob().readall()
        result['json_data'] = json.loads(json_content.decode('utf-8'))
        
        # Get image filename from JSON
        image_blob_name = result['json_data'].get('image_blob_name')
        
        if not image_blob_name:
            # Fallback: try to find image with same analysis ID
            image_blobs = container_client.list_blobs(name_starts_with=f"images/{analysis_id}")
            for blob in image_blobs:
                image_blob_name = blob.name
                break
        
        if image_blob_name:
            # Load image
            image_blob = container_client.get_blob_client(image_blob_name)
            image_bytes = image_blob.download_blob().readall()
            result['image'] = Image.open(io.BytesIO(image_bytes))
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def get_storage_stats():
    """Get storage statistics"""
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    
    stats = {
        'total_size_mb': 0,
        'image_count': 0,
        'json_count': 0,
        'total_analyses': 0
    }
    
    blob_list = container_client.list_blobs()
    
    for blob in blob_list:
        stats['total_size_mb'] += blob.size / (1024 * 1024)
        
        if blob.name.startswith("images/"):
            stats['image_count'] += 1
        elif blob.name.startswith("results/"):
            stats['json_count'] += 1
    
    stats['total_analyses'] = stats['json_count']
    
    return stats

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Upload Reviewer",
    layout="wide",
    page_icon="📊"
)

st.title("📊 User Uploads Reviewer")
st.markdown("Review images and analysis results from Azure Blob Storage")

# ============================================================================
# SIDEBAR - STATS & CONTROLS
# ============================================================================

with st.sidebar:
    st.header("📈 Storage Stats")
    
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.cache_resource.clear()
    
    try:
        stats = get_storage_stats()
        
        st.metric("Total Analyses", stats['total_analyses'])
        st.metric("Images", stats['image_count'])
        st.metric("Storage Used", f"{stats['total_size_mb']:.2f} MB")
        
        st.divider()
        
        # Download all option
        st.header("⬇️ Download Options")
        
        if st.button("📦 Download All JSONs", use_container_width=True):
            st.info("Feature coming soon!")
        
        if st.button("🖼️ Download All Images", use_container_width=True):
            st.info("Feature coming soon!")
    
    except Exception as e:
        st.error(f"Error loading stats: {str(e)}")

# ============================================================================
# MAIN AREA - ANALYSIS LIST & VIEWER
# ============================================================================

st.header("🔍 Browse Analyses")

try:
    # Load all analyses
    analyses = list_all_analyses()
    
    if not analyses:
        st.warning("No analyses found in storage.")
        st.stop()
    
    st.success(f"Found {len(analyses)} analyses")
    
    # ========== VIEW MODE SELECTOR ==========
    view_mode = st.radio(
        "View Mode:",
        ["📋 Single Analysis", "📜 Scroll All"],
        horizontal=True
    )
    
    st.divider()
    
    # ========== SINGLE ANALYSIS MODE ==========
    if view_mode == "📋 Single Analysis":
        
        # Select analysis from dropdown
        analysis_options = [
            f"{a['id']} - {a['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            for a in analyses
        ]
        
        selected_idx = st.selectbox(
            "Select Analysis:",
            range(len(analysis_options)),
            format_func=lambda i: analysis_options[i]
        )
        
        selected_analysis = analyses[selected_idx]
        
        # Load and display
        with st.spinner("Loading analysis..."):
            data = load_analysis(selected_analysis['id'])
        
        if data['error']:
            st.error(f"Error: {data['error']}")
        else:
            # Two columns: image and JSON
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🖼️ Image")
                if data['image']:
                    st.image(data['image'], use_container_width=True)
                else:
                    st.warning("Image not found")
            
            with col2:
                st.subheader("📄 Analysis Results")
                
                if data['json_data']:
                    # Pretty display
                    st.markdown(f"**Analysis ID:** `{data['id']}`")
                    st.markdown(f"**Timestamp:** {data['json_data'].get('timestamp', 'N/A')}")
                    st.markdown(f"**Original Filename:** {data['json_data'].get('original_filename', 'N/A')}")
                    
                    # User context
                    if data['json_data'].get('user_context'):
                        st.markdown(f"**User Context:** _{data['json_data']['user_context']}_")
                    
                    st.divider()
                    
                    # Results
                    results = data['json_data'].get('results', {})
                    
                    st.markdown("**📝 Caption:**")
                    st.info(results.get('caption', 'N/A'))
                    
                    st.markdown("**🏷️ Tags:**")
                    tags = results.get('tags', [])
                    if tags:
                        tags_html = " ".join([
                            f'<span style="background-color: #e3f2fd; padding: 5px 10px; '
                            f'margin: 2px; border-radius: 15px; display: inline-block;">{t}</span>'
                            for t in tags
                        ])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    
                    # Safety check
                    safety = results.get('safety_check', {})
                    if safety and not safety.get('is_safe'):
                        st.warning("⚠️ Content Safety Flags Detected")
                        for flag in safety.get('flags', []):
                            st.markdown(f"- {flag.get('category')}: Level {flag.get('severity')}")
                    
                    # Raw JSON
                    with st.expander("📋 Raw JSON"):
                        st.json(data['json_data'])
    
    # ========== SCROLL ALL MODE ==========
    elif view_mode == "📜 Scroll All":
        
        st.info(f"Displaying all {len(analyses)} analyses. Scroll to browse.")
        
        # Load and display all analyses
        for idx, analysis in enumerate(analyses, 1):
            
            with st.container():
                st.markdown(f"### 📸 Analysis #{idx} - `{analysis['id']}`")
                st.caption(f"Uploaded: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Load data
                data = load_analysis(analysis['id'])
                
                if data['error']:
                    st.error(f"Error: {data['error']}")
                    st.divider()
                    continue
                
                # Two columns layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if data['image']:
                        st.image(data['image'], use_container_width=True)
                        
                        # Image metadata
                        if data['json_data']:
                            st.caption(f"📁 {data['json_data'].get('original_filename', 'N/A')}")
                            file_size = data['json_data'].get('file_size_bytes', 0)
                            if file_size:
                                st.caption(f"💾 {file_size / 1024:.1f} KB")
                    else:
                        st.warning("Image not found")
                
                with col2:
                    if data['json_data']:
                        results = data['json_data'].get('results', {})
                        
                        # User context
                        user_context = data['json_data'].get('user_context', '')
                        if user_context:
                            st.markdown(f"**Context:** _{user_context}_")
                        
                        # Caption
                        st.markdown("**Caption:**")
                        caption = results.get('caption', 'N/A')
                        st.info(caption)
                        
                        # Tags
                        st.markdown("**Tags:**")
                        tags = results.get('tags', [])
                        if tags:
                            st.write(", ".join(tags))
                        
                        # Safety flags
                        safety = results.get('safety_check', {})
                        if safety and not safety.get('is_safe'):
                            st.warning("⚠️ Safety flags present")
                
                st.divider()

except Exception as e:
    st.error(f"Error loading analyses: {str(e)}")
    st.code(str(e))

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💾 Connected to Azure Blob Storage")