# review_uploads.py - Streamlit app to review user uploads (Google app version)

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

AZURE_STORAGE_CONNECTION_STRING = config.AZURE_STORAGE_CONNECTION_STRING
CONTAINER_NAME = 'user-uploads-google'

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
    page_title="Upload Reviewer - Google App",
    layout="wide",
    page_icon="📊"
)

st.title("📊 User Uploads Reviewer (Google Cloud Vision + OpenAI)")
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
        
        # Technology info
        st.header("🔧 Technology Stack")
        st.caption("Vision API: Google Cloud Vision")
        st.caption("LLM: OpenAI GPT-5")
        st.caption("Safety: Google SafeSearch")
        st.caption("Storage: Azure Blob")
        
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
                    
                    # Image metadata
                    if data['json_data']:
                        st.caption(f"📁 {data['json_data'].get('original_filename', 'N/A')}")
                        file_size = data['json_data'].get('file_size_bytes', 0)
                        if file_size:
                            st.caption(f"💾 {file_size / 1024:.1f} KB")
                else:
                    st.warning("Image not found")
            
            with col2:
                st.subheader("📄 Analysis Results")
                
                if data['json_data']:
                    # Basic metadata
                    st.markdown(f"**Analysis ID:** `{data['id']}`")
                    st.markdown(f"**Timestamp:** {data['json_data'].get('timestamp', 'N/A')}")
                    
                    st.divider()
                    
                    # ===== CONTEXT INFORMATION (NEW) =====
                    context_data = data['json_data'].get('context', {})
                    user_context = context_data.get('user_provided', '')
                    was_auto_detected = context_data.get('was_auto_detected', False)
                    web_detection_used = context_data.get('web_detection_used', False)
                    
                    if user_context:
                        st.markdown("### 🎯 Context Used")
                        st.info(user_context)
                        
                        # Context source indicators
                        if was_auto_detected:
                            st.caption("✨ Auto-detected by Google Web Detection")
                        else:
                            st.caption("✍️ Manually provided by user")
                    
                    # ===== WEB DETECTION RESULTS (NEW) =====
                    if web_detection_used:
                        web_data = data['json_data'].get('web_detection', {})
                        
                        with st.expander("🌐 Web Entity Detection Details"):
                            # Best guess label
                            best_guess = web_data.get('best_guess_label')
                            if best_guess:
                                st.markdown(f"**🎯 Best Guess:** {best_guess}")
                            
                            # Web entities
                            web_entities = web_data.get('web_entities', [])
                            if web_entities:
                                st.markdown("**📋 Detected Entities:**")
                                for i, entity in enumerate(web_entities[:5], 1):
                                    score = int(entity['score'] * 100)
                                    st.write(f"{i}. {entity['description']} ({score}%)")
                            
                            # Matching pages
                            matching_pages = web_data.get('matching_pages', [])
                            if matching_pages:
                                st.markdown(f"**🔗 Found on {len(matching_pages)} pages**")
                    
                    st.divider()
                    
                    # ===== ANALYSIS RESULTS =====
                    results = data['json_data'].get('results', {})
                    
                    st.markdown("### 📝 Caption")
                    caption = results.get('caption', 'N/A')
                    st.success(caption)
                    
                    st.markdown("### 🏷️ Tags")
                    tags = results.get('tags', [])
                    if tags:
                        tags_html = " ".join([
                            f'<span style="background-color: #e3f2fd; padding: 5px 10px; '
                            f'margin: 2px; border-radius: 15px; display: inline-block;">{t}</span>'
                            for t in tags
                        ])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.caption("No tags generated")
                    
                    st.divider()
                    
                    # ===== SAFETY CHECK (NEW - Google SafeSearch) =====
                    safety_check = data['json_data'].get('safety_check', {})
                    
                    if safety_check:
                        is_safe = safety_check.get('is_safe', True)
                        flags = safety_check.get('flags', [])
                        numeric_scores = safety_check.get('numeric_scores', {})
                        
                        st.markdown("### 🛡️ Content Safety (Google SafeSearch)")
                        
                        if is_safe:
                            st.success("✅ No safety concerns")
                        else:
                            st.warning("⚠️ Safety flags detected")
                            for flag in flags:
                                category = flag.get('category', 'unknown')
                                severity = flag.get('severity', 0)
                                st.markdown(f"- **{category.title()}**: Level {severity}/5")
                        
                        # Show all scores
                        with st.expander("📊 All Safety Scores (0-5)"):
                            for category, score in numeric_scores.items():
                                # Emoji based on score
                                if score <= 1:
                                    emoji = "✅"
                                elif score == 2:
                                    emoji = "🟡"
                                elif score == 3:
                                    emoji = "🟠"
                                else:
                                    emoji = "🔴"
                                
                                st.markdown(f"{emoji} **{category.title()}**: {score}/5")
                    
                    st.divider()
                    
                    # ===== TECHNOLOGY INFO =====
                    tech = data['json_data'].get('technology', {})
                    if tech:
                        with st.expander("🔧 Technology Stack Used"):
                            st.markdown(f"**Vision API:** {tech.get('vision_api', 'N/A')}")
                            st.markdown(f"**LLM Model:** {tech.get('llm_model', 'N/A')}")
                            st.markdown(f"**Safety API:** {tech.get('safety_api', 'N/A')}")
                            st.markdown(f"**Web Detection:** {tech.get('web_detection', 'N/A')}")
                    
                    # ===== RAW JSON =====
                    with st.expander("📋 Complete Raw JSON"):
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
                        # Context
                        context_data = data['json_data'].get('context', {})
                        user_context = context_data.get('user_provided', '')
                        was_auto_detected = context_data.get('was_auto_detected', False)
                        
                        if user_context:
                            context_label = "🌐 Auto-detected" if was_auto_detected else "✍️ Manual"
                            st.markdown(f"**Context ({context_label}):**")
                            st.caption(user_context)
                        
                        # Results
                        results = data['json_data'].get('results', {})
                        
                        # Caption
                        st.markdown("**📝 Caption:**")
                        caption = results.get('caption', 'N/A')
                        st.info(caption)
                        
                        # Tags
                        st.markdown("**🏷️ Tags:**")
                        tags = results.get('tags', [])
                        if tags:
                            st.write(", ".join(tags))
                        else:
                            st.caption("No tags")
                        
                        # Safety flags (if any)
                        safety = data['json_data'].get('safety_check', {})
                        if safety and not safety.get('is_safe'):
                            flags = safety.get('flags', [])
                            st.warning(f"⚠️ Safety: {len(flags)} flag(s)")
                            for flag in flags:
                                st.caption(f"- {flag.get('category')}: {flag.get('severity')}/5")
                
                st.divider()

except Exception as e:
    st.error(f"Error loading analyses: {str(e)}")
    st.code(str(e))

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("💾 Connected to Azure Blob Storage | 🤖 Google Cloud Vision + OpenAI GPT-5")