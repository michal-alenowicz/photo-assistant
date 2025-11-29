import streamlit as st
from image_analyzer import ImageAnalyzer
from faq_system import FAQSystem
from content_safety_google import ContentSafetyChecker
from web_entity_detector import WebEntityDetector
from storage_manager import AzureStorageManager
import json
import io
from PIL import Image
import config


ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "GIF", "BMP", "WEBP", "ICO", "TIFF", "MPO"}
MAX_FILE_MB = 20
MIN_DIM = 50
MAX_DIM = 16000


st.set_page_config(page_title="Asystent opisywania zdjęć", layout="centered")

col1, col2 = st.columns([6, 1])
with col1:
    st.title("Donal POC - Asystent opisywania i tagowania zdjęć")
    st.caption("wersja Google Cloud + OpenAI API")

with col2:
    st.write("")
    banner = Image.open("donal.png")
    st.image(banner, width='stretch')
    st.caption('(przeciągnij mnie na pole upload)')


@st.cache_resource
def init_image_analyzer():
    """Initialize Image Analyzer with Google + OpenAI credentials"""
    return ImageAnalyzer(
        google_credentials_path=config.GOOGLE_APPLICATION_CREDENTIALS,
        google_project_id=config.GOOGLE_PROJECT_ID,
        openai_api_key=config.OPENAI_API_KEY,
        openai_model=config.OPENAI_MODEL
    )


@st.cache_resource
def init_faq_system():
    """Initialize FAQ System with OpenAI"""
    return FAQSystem(
        openai_api_key=config.OPENAI_API_KEY,
        chat_model=config.OPENAI_MODEL,
        embedding_model=config.OPENAI_EMBEDDING_MODEL,
        faq_file_path="faq_data.json"
    )


@st.cache_resource
def init_content_safety():
    """Initialize Content Safety with Google SafeSearch"""
    return ContentSafetyChecker(
        google_credentials_path=config.GOOGLE_APPLICATION_CREDENTIALS
    )

@st.cache_resource
def init_web_detector():
    """Initialize Web Entity Detector""" 
    return WebEntityDetector(
        google_credentials_path=config.GOOGLE_APPLICATION_CREDENTIALS
    )

@st.cache_resource
def init_storage_manager():
    """Initialize Azure Storage Manager"""  # ← NEW
    if config.AZURE_STORAGE_CONNECTION_STRING:
        return AzureStorageManager(
            connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
            container_name=config.AZURE_STORAGE_CONTAINER
        )
    return None

analyzer = init_image_analyzer()
faq_system = init_faq_system()
content_safety = init_content_safety()
web_detector = init_web_detector()
storage_manager = init_storage_manager()


# Create tabs
tab1, tab2 = st.tabs(["🖼️ Analiza Zdjęć", "❓ FAQ"])

if "prev_filename" not in st.session_state:
        st.session_state.prev_filename = None
if "context_input" not in st.session_state:
        st.session_state.context_input = ""
if "web_context_data" not in st.session_state: 
    st.session_state.web_context_data = None

# ==================== TAB 1: IMAGE ANALYSIS ====================

with tab1:
    st.empty()
    st.header("Prześlij zdjęcie do analizy")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Wybierz plik graficzny (uwaga, zastosowane narzędzia do analizy obrazów narzucają limit: 20 MB)",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "ico", "tiff", "mpo"]
    )
    
    # Reset flag when user uploads their own
    if uploaded_file is not None and not isinstance(uploaded_file, io.BufferedReader):
        st.session_state.sample_preloaded = False
    
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.prev_filename:
            # New image detected - wipe context
            st.session_state.context_input = ""
            st.session_state.web_context_data = None
            st.session_state.prev_filename = uploaded_file.name
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Przesłane zdjęcie")
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')
            
            # ---- File validations ----
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_MB:
                st.error(f"❌ Plik zbyt duży ({file_size_mb:.1f} MB). Dopuszczalne max.: {MAX_FILE_MB} MB.")
                st.stop()
            
            try:
                img = Image.open(uploaded_file)
                fmt = img.format.upper()
                
                if fmt not in ALLOWED_FORMATS:
                    st.error(f"❌ Niedozwolony format: {fmt}")
                    st.stop()
            
            except Exception as e:
                st.error("❌ Błąd odczytu obrazu: " + str(e))
                st.stop()
            
            width, height = img.size
            
            if width < MIN_DIM or height < MIN_DIM:
                st.error(f"❌ Obraz zbyt mały: {width}×{height}px. Minimum to {MIN_DIM}×{MIN_DIM}.")
                st.stop()
            
            if width > MAX_DIM or height > MAX_DIM:
                st.error(f"❌ Obraz zbyt duży: {width}×{height}px. Maksimum to {MAX_DIM}×{MAX_DIM}.")
                st.stop()
            
            # SUCCESS - Image info
            st.caption(f"Nazwa pliku: {uploaded_file.name}")
            st.caption(f"Rozmiar: {uploaded_file.size / 1024:.1f} KB")
            st.caption(f"Wymiary: {image.size[0]} x {image.size[1]} px")
            
            if image.size[0] < 150 or image.size[1] < 150:
                st.warning(
                    f"⚠️ Uwaga: Obraz jest bardzo mały!\n\n"
                    f"Zalecane minimalne wymiary to **150 × 150 px**. "
                    "Jeśli użyjesz tego obrazu, wyniki analizy mogą być mniej wiarygodne."
                )
        
        with col2:
            st.subheader("Analiza")
            
            # ========== SIMPLIFIED CONTEXT SECTION (COMBINED) ==========
            st.markdown("### ⚙️ Kontekst zdjęcia")
            
            st.markdown("""
            **Automatyczne wykrywanie:** web search rozpozna osoby, miejsca i wydarzenia  
            **Lub edytuj ręcznie:** Dodaj nazwiska, lokalizacje, okoliczności
            """)
            
            # Detect context button at the top
            if st.button("🌐 Wykryj kontekst [zalecane!]", use_container_width=True, type="secondary"):
                with st.spinner("Wyszukiwanie w sieci..."):
                    try:
                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format or 'JPEG')
                        img_byte_arr.seek(0)
                        image_data = img_byte_arr.read()
                        
                        # Detect web entities
                        web_result = web_detector.detect_web_context(image_data)
                        
                        if web_result.get('error'):
                            st.error(f"❌ Błąd: {web_result['error']}")
                            st.session_state.web_context_data = None  # ← NEW
                        elif web_result['suggested_context']:
                            st.session_state.context_input = web_result['suggested_context']
                            st.session_state.web_context_data = web_result  # ← NEW - store full result
                            st.success(f"✅ Wykryto: {web_result['suggested_context']}")
                            if web_result['best_guess_label']:
                                st.info(f"🎯 {web_result['best_guess_label']}")
                            st.rerun()
                        else:
                            st.warning("⚠️ Nie wykryto kontekstu. Wpisz ręcznie.")
                            st.session_state.web_context_data = None  # ← NEW
                    except Exception as e:
                        st.error(f"❌ Błąd: {str(e)}")
                        st.session_state.web_context_data = None  # ← NEW
            
            # Context text area (always visible, editable)
            user_context = st.text_area(
                "Kontekst (edytowalny) - w dowolnym języku:",
                placeholder="Kliknij 'Wykryj kontekst' lub wpisz ręcznie",
                height=100,
                help="Automatycznie wykryty kontekst lub wpisany ręcznie. Możesz edytować.",
                key="context_input"
            )
            
            if user_context:
                char_count = len(user_context)
                if char_count > 200:
                    st.warning(f"⚠️ Kontekst zbyt długi ({char_count}/200 znaków). Skróć dla lepszych wyników.")
                else:
                    st.caption(f"✓ {char_count}/200 znaków")
            
            st.markdown("---")  # Visual separator
            
            # ========== ANALYZE BUTTON ==========
            
            # Analyze button
            if st.button("🔍 Analizuj", type="primary"):
                with st.spinner("Analizuję zdjęcie..."):
                    try:
                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format or 'JPEG')
                        img_byte_arr.seek(0)
                        image_data = img_byte_arr.read()
                        
                        # ===== STEP 1: CONTENT SAFETY CHECK (NEW) =====
                        safety_results = content_safety.analyze_image(image_data)
                        
                                                
                        # Show alert if content flagged (but don't block)
                        if not safety_results['is_safe']:
                            alert_msg = content_safety.get_alert_message(safety_results)
                            st.warning(alert_msg)
                            
                            # Show detailed breakdown in expander
                            with st.expander("🔍 Szczegóły moderacji treści"):
                                st.markdown(content_safety.get_all_details(safety_results))
                            
                            # Tell user they can continue
                            st.info(
                                "ℹ️ **Możesz kontynuować pomimo ostrzeżenia.** Materiały dziennikarskie mogą zawierać szokujące treści. "
                                "System nie blokuje analizy - decyzja należy do Ciebie."
                            )
                            
                            # User can still proceed - just click analyze again
                            # Or add explicit checkbox:
                            # if not st.checkbox("Rozumiem, chcę kontynuować"):
                            #     st.stop()
                        else:
                            # All clear
                            st.success("✅ Weryfikacja treści: Brak ostrzeżeń")
                        

                        # ===== STEP 2: REGULAR ANALYSIS (continues regardless) =====
                        stripped_context = user_context.strip() if user_context else ""
                        
                        result = analyzer.analyze_image(image_data, user_context=stripped_context, safety_context=safety_results)
                        
                        st.success("✅ Analiza zakończona!")

                        # ===== SAVE TO AZURE STORAGE (NEW) =====
                        if storage_manager:
                            try:
                                analysis_id = storage_manager.save_analysis(
                                    image_bytes=image_data,
                                    image_name=uploaded_file.name,
                                    result_json=result,
                                    user_context=stripped_context,
                                    web_context=st.session_state.web_context_data,  # ← NEW - include web detection
                                    safety_results=safety_results
                                )
                                st.caption(f"💾 Zapisano: {analysis_id}")
                            except Exception as e:
                                st.warning(f"⚠️ Nie udało się zapisać do storage: {str(e)}")




                        
                        if stripped_context:
                            st.info(f"ℹ️ Użyto kontekstu: *{stripped_context[:100]}...*")
                        
                        # Caption
                        st.markdown("### 📝 Opis zdjęcia")
                        st.write(result.get("caption") or result.get("raw"))
                        
                        # Tags
                        st.markdown("### 🏷️ Tagi")
                        tags = result.get("tags", [])
                        if tags:
                            tags_html = " ".join([
                                f'<span style="background-color: #e3f2fd; padding: 5px 10px; '
                                f'margin: 2px; border-radius: 15px; display: inline-block;">{t}</span>'
                                for t in tags
                            ])
                            st.markdown(tags_html, unsafe_allow_html=True)
                        
                        # Results JSON
                        with st.expander("📄 Opis i tagi w JSON"):
                            st.json({
                                "caption": result.get("caption"),
                                "tags": result.get("tags")
                            })
                        
                        # Vision analysis debug
                        with st.expander("🔍 Pośrednia analiza CV (debug)"):
                            st.json(result.get("vision_summary", {}))
                    
                    except Exception as e:
                        st.error(f"❌ Błąd: {str(e)}")


# ==================== TAB 2: FAQs ====================

with tab2:
    st.header("Pytania i odpowiedzi")
    
    # Display FAQ stats
    faq_count = faq_system.get_faq_count()
    st.caption(f"📚 Baza zawiera {faq_count} pytań i odpowiedzi")
    
 
    st.subheader("Zadaj pytanie")
    
    # Wrap in form to prevent rerun on focus loss
    with st.form(key="faq_form", clear_on_submit=False):
        user_question = st.text_input(
            "Twoje pytanie:",
            placeholder="Np. Jak mogę przesłać zdjęcie do analizy?",
            key="faq_question_input"
        )
        
        # Submit button inside form
        submit_button = st.form_submit_button(
            "🔎 Znajdź odpowiedź",
            type="primary",
            use_container_width=True
        )
    
    # Search button
    if submit_button:
        if user_question.strip():
            with st.spinner("Szukam odpowiedzi..."):
                try:
                    result = faq_system.answer_question(user_question)
                    
                    # Display answer with appropriate styling based on confidence
                    st.markdown("### 💬 Odpowiedź:")
                    
                    if result['confidence'] == 'high':
                        st.success(result['answer'])
                    elif result['confidence'] == 'medium':
                        st.info(result['answer'])
                    else:
                        st.warning(result['answer'])
                    
                    # Show similarity score
                    if result.get('top_similarity', 0) > 0:
                        similarity_percent = result['top_similarity'] * 100
                        st.caption(f"🎯 Trafność dopasowania: {similarity_percent:.0f}%")
                    
                    # Show matched FAQs
                    if result['matched_faqs']:
                        with st.expander("📚 Powiązane pytania z FAQ"):
                            for i, matched in enumerate(result['matched_faqs'], 1):
                                st.markdown(
                                    f"{i}. **{matched['question']}** "
                                    f"(podobieństwo: {matched['similarity_percent']})"
                                )
                
                except Exception as e:
                    st.error(f"❌ Błąd: {str(e)}")
        else:
            st.warning("⚠️ Proszę wpisać pytanie")
    

