import streamlit as st
from vision4 import analyze_image
from llm4 import generate_caption_and_tags
from faq_system import FAQSystem
import json
import io
from PIL import Image
import config


ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "GIF", "BMP", "WEBP", "ICO", "TIFF", "MPO"}
MAX_FILE_MB = 20
MIN_DIM = 50
MAX_DIM = 16000


st.set_page_config(page_title="Asystent opisywania zdjęć", layout="centered")
st.title("Asystent opisywania i tagowania zdjęć (Polski)")



@st.cache_resource
def init_faq_system():
    """Initialize FAQ System with Azure OpenAI"""
    return FAQSystem(
        azure_openai_key=config.AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=config.AZURE_OPENAI_ENDPOINT,
        chat_deployment=config.CHAT_DEPLOYMENT,
        embedding_deployment=config.EMBEDDING_DEPLOYMENT,
        faq_file_path="faq_data.json",
        api_version=config.AZURE_OPENAI_API_VERSION,
        embedding_api_version=config.AZURE_OPENAI_EMBEDDINGS_API_VERSION
    )


faq_system = init_faq_system()


# Create tabs
tab1, tab2 = st.tabs(["🖼️ Analiza Zdjęć", "❓ FAQ"])

if "prev_filename" not in st.session_state:
        st.session_state.prev_filename = None
if "context_input" not in st.session_state:
        st.session_state.context_input = ""

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
            st.session_state.prev_filename = uploaded_file.name


    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Przesłane zdjęcie")
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')

            # ---- 1) File size check ----
            file_size_mb = uploaded_file.size / (1024 * 1024)

            if file_size_mb > MAX_FILE_MB:
                st.error(f"❌ Plik zbyt duży ({file_size_mb:.1f} MB). Dopuszczalne max.: {MAX_FILE_MB} MB.")
                st.stop()

            # ---- 2) File format check ----
            try:
                img = Image.open(uploaded_file)
                fmt = img.format.upper()

                if fmt not in ALLOWED_FORMATS:
                    st.error(f"❌ Niedozwolony format: {fmt}")
                    st.stop()

            except Exception as e:
                st.error("❌ Błąd odczytu obrazu: " + str(e))
                st.stop()

            # ---- 3) Resolution check ----
            width, height = img.size

            if width < MIN_DIM or height < MIN_DIM:
                st.error(f"❌ Obraz zbyt mały: {width}×{height}px. Minimum to {MIN_DIM}×{MIN_DIM}.")
                st.stop()

            if width > MAX_DIM or height > MAX_DIM:
                st.error(f"❌ Obraz zbyt duży: {width}×{height}px. Maksimum to {MAX_DIM}×{MAX_DIM}.")
                st.stop()
            

            # SUCKCES - Image info
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
            
            # ========== OPTIONAL CONTEXT FIELD (COLLAPSIBLE) ==========
            with st.expander("⚙️ Dodatkowy kontekst (opcjonalnie)", expanded=False):
                st.markdown("""
                Podaj hasłowo dodatkowe informacje, które pomogą w lepszym opisie zdjęcia:
                - **Osoby**: Nazwiska znanych osób, polityków
                - **Miejsca**: Nazwy lokalizacji, miast
                - **Wydarzenia**: Nazwa wydarzenia, okoliczności
                """)
                
                user_context = st.text_area(
                    "Kontekst:",
                    value=st.session_state.context_input,
                    placeholder="Np. 'Donald Trump, Waszyngton, konferencja prasowa'",
                    height=100,
                    help="Ten kontekst zostanie przekazany do AI, aby wygenerować bardziej precyzyjny opis",
                    key="context_input"
                )
                
                # Show character count
                if user_context:
                    st.caption(f"Znaki: {len(user_context)}/200")
                    if len(user_context) > 200:
                        st.warning("⚠️ Kontekst jest zbyt długi. Zalecamy maksymalnie 200 znaków.")
            

            # Analyze button
            if st.button("🔍 Analizuj", type="primary"):
                with st.spinner("Analizuję zdjęcie..."):
                    try:
                        # Convert image to bytes
                        # img_bytes = uploaded_file.read()
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format or 'JPEG')
                        img_byte_arr.seek(0)
                        image_data = img_byte_arr.read()
                        

                        # Get user context (empty string if None or whitespace)
                        stripped_context = user_context.strip() if 'user_context' in locals() and user_context else ""

                        # Analyze
                        vision = analyze_image(image_data)
                        res = generate_caption_and_tags(vision, user_context=stripped_context)
                        
                        # Display results
                        st.success("✅ Analiza zakończona!")
                        
                        # Show if context was used
                        if stripped_context:
                            st.info(f"ℹ️ Użyto dodatkowego kontekstu: *{stripped_context[:100]}{'...' if len(stripped_context) > 100 else ''}*")
                        

                        # Caption
                        st.markdown("### 📝 Opis zdjęcia")
                        st.write(res.get("caption") or res.get("raw"))
                        
                        # Tags
                        st.markdown("### 🏷️ Tagi")
                        tags = res.get("tags", [])
                        if tags:
                            tags_html = " ".join([
                                f'<span style="background-color: #e3f2fd; padding: 5px 10px; '
                                f'margin: 2px; border-radius: 15px; display: inline-block;">{t}</span>'
                                for t in tags
                            ])
                            st.markdown(tags_html, unsafe_allow_html=True)
                        
                        # Show results as json
                        with st.expander("🔍 Opis i tagi w json"):
                            st.json(res)

                        # Show detailed insights in expander
                        with st.expander("🔍 Pośrednia analiza CV (debug)"):
                            st.json(vision)
                    
                    except Exception as e:
                        st.error(f"❌ Błąd podczas analizy: {str(e)}")


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
    

