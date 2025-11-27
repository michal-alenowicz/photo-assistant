# debug_faq_matching.py
from faq_system import FAQSystem
import config

faq = FAQSystem(
    azure_openai_key=config.AZURE_OPENAI_API_KEY,
    azure_openai_endpoint=config.AZURE_OPENAI_ENDPOINT,
    chat_deployment=config.CHAT_DEPLOYMENT,
    embedding_deployment=config.EMBEDDING_DEPLOYMENT,
    faq_file_path="faq_data.json",
    api_version=config.AZURE_OPENAI_API_VERSION,
    embedding_api_version=config.AZURE_OPENAI_EMBEDDINGS_API_VERSION
)

# Test various questions
test_questions = [
    "Jak przesłać zdjęcie?",
    "jak wrzucić zdjęcie",
    "jak dodać obraz",
    "Jakie formaty są obsługiwane?",
    "czy mogę użyć PNG",
    "Ile kosztuje analiza?",
    "jaki jest koszt",
    "Czy system jest zgodny z RODO?",
    "czy to bezpieczne",
]

print("=" * 80)
print("FAQ MATCHING DEBUG")
print("=" * 80)

for question in test_questions:
    print(f"\n📝 Question: '{question}'")
    
    # Get top 3 matches with scores
    matches = faq.find_similar_faqs(question, top_k=3)
    
    if matches:
        for i, (faq_item, score) in enumerate(matches, 1):
            print(f"  {i}. Score: {score:.3f} ({score*100:.1f}%) - {faq_item['question'][:60]}...")
    else:
        print("  ❌ No matches found")
    
    print(f"  → Would match at threshold 0.7? {'✅ YES' if matches and matches[0][1] >= 0.7 else '❌ NO'}")
    print(f"  → Would match at threshold 0.5? {'✅ YES' if matches and matches[0][1] >= 0.5 else '❌ NO'}")