import streamlit as st
from vision2 import analyze_image
from llm2 import generate_caption_and_tags
#from faq_store import FAQStore
import json

st.set_page_config(page_title="Asystent opisywania zdjęć", layout="centered")
st.title("Asystent opisywania i tagowania zdjęć (Polski)")

#faq = FAQStore("faq.json")

uploaded = st.file_uploader("Prześlij zdjęcie", type=["jpg","jpeg","png"])

if uploaded:
    img_bytes = uploaded.read()
    with st.spinner("Analizuję zdjęcie..."):
        vision = analyze_image(img_bytes)
        st.subheader("Surowe wyniki analizy obrazu")
        st.json(vision)
        # ask LLM to produce Polish caption & tags
        res = generate_caption_and_tags(vision)
    st.subheader("Wynik LLM (polski)")
    st.write(res.get("caption") or res.get("raw"))
    st.write("Tagi:")
    tags = res.get("tags", [])
    if tags:
        for t in tags:
            st.write(f"- {t.get('tag')} (confidence: {t.get('confidence')})")
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
