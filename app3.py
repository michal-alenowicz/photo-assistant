import streamlit as st
from vision3 import analyze_image
from llm3 import generate_caption_and_tags
#from faq_store import FAQStore
import json
import io
from PIL import Image



ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "GIF", "BMP", "WEBP", "ICO", "TIFF", "MPO"}
MAX_FILE_MB = 20
MIN_DIM = 50
MAX_DIM = 16000


st.set_page_config(page_title="Asystent opisywania zdjęć", layout="centered")
st.title("Asystent opisywania i tagowania zdjęć (Polski)")

#faq = FAQStore("faq.json")

# Create tabs
tab1, tab2 = st.tabs(["🖼️ Analiza Zdjęć", "❓ FAQ"])

if "prev_filename" not in st.session_state:
        st.session_state.prev_filename = None
if "context_input" not in st.session_state:
        st.session_state.context_input = ""

# ==================== TAB 1: IMAGE ANALYSIS ====================

with tab1:

    #   # Show requirements in an info box
    # with st.expander("📋 Wymagania dotyczące zdjęć", expanded=False):
    #     st.markdown("""
    #     **Obsługiwane formaty:** JPEG, PNG, GIF, BMP, WEBP, ICO, TIFF, MPO
        
    #     **Wymagania techniczne:**
    #     - Maksymalny rozmiar pliku: **20 MB** (zalecane: do 10 MB)
    #     - Minimalne wymiary: **50 x 50 pikseli** (zalecane min. 150 x 150)
    #     - Maksymalne wymiary: **16,000 x 16,000 pikseli**
        
    #     """)



    st.header("Prześlij zdjęcie do analizy")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Wybierz plik graficzny (uwaga, narzędzia do analizy obrazów narzucają limit: 20 MB)",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "ico", "tiff", "mpo"]
    )
    
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




    
    
    
    
# # FAQ question box
# st.subheader("Zapytaj FAQ lub ogólne pytanie")
# q = st.text_input("Twoje pytanie (PL)")
# if st.button("Zadaj pytanie"):
#     if not q.strip():
#         st.warning("Wpisz pytanie.")
#     else:
#         hits = faq.query(q, topk=1)
#         if hits and hits[0]["score"] >= 0.75:
#             st.success("Znaleziono podobne pytanie w FAQ:")
#             st.write(hits[0]["faq"]["a"])
#             st.caption(f"Similarity: {hits[0]['score']:.2f}")
#         else:
#             # fallback to LLM: answer using vision + faq context
#             prompt = {
#                 "vision": vision,
#                 "user_question": q,
#                 "faq": faq.faqs
#             }
#             # call LLM to answer
#             llm_ans = generate_caption_and_tags(prompt)  # reuse or write a separate answer function
#             st.write(llm_ans.get("raw") or llm_ans)
