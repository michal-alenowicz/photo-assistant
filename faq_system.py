# (AZURE OPENAI VERSION)
from openai import AzureOpenAI
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
import hashlib
import os
from datetime import datetime

class FAQSystem:
    def __init__(
        self, 
        azure_openai_key: str,
        azure_openai_endpoint: str,
        chat_deployment: str,
        embedding_deployment: str,
        faq_file_path: str,
        api_version: str = "2025-01-01-preview",
        embedding_api_version: str = "2023-05-15"
    ):
        """
        Initialize FAQ System with Azure OpenAI

        """
        # Initialize Azure OpenAI client for chat
        self.chat_client = AzureOpenAI(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=api_version
        )
        
        # Initialize Azure OpenAI client for embeddings
        self.embedding_client = AzureOpenAI(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=embedding_api_version
        )
        
        self.chat_deployment = chat_deployment
        self.embedding_deployment = embedding_deployment
        self.faq_file_path = faq_file_path
        self.faq_data = []
        self.embeddings = []
        self.embeddings_cache_file = "faq_embeddings_cache.pkl"
        
        # Load FAQ data
        self.load_faq(faq_file_path)
        
        # Calculate current file hash
        current_hash = self._get_file_hash(faq_file_path)
        
        # Try to load cached embeddings
        if self._load_cached_embeddings(current_hash):
            print(f"✅ Loaded {len(self.embeddings)} embeddings from cache")
        else:
            print(f"⏳ Generating embeddings for {len(self.faq_data)} FAQs...")
            self.generate_all_embeddings()
            self._save_embeddings_cache(current_hash)
            print(f"💾 Saved embeddings to cache")
    
    def load_faq(self, file_path: str):
        """Load FAQ data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.faq_data = data['faqs']
            print(f"📚 Loaded {len(self.faq_data)} FAQs from {file_path}")
        except FileNotFoundError:
            raise Exception(f"FAQ file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in FAQ file: {e}")
        except KeyError:
            raise Exception("FAQ file must contain 'faqs' key")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of FAQ file"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_cached_embeddings(self, current_hash: str) -> bool:
        """
        Load embeddings from cache if valid
        """
        if not os.path.exists(self.embeddings_cache_file):
            return False
        
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            # Validate cache
            if (cache['file_hash'] == current_hash and 
                cache['faq_count'] == len(self.faq_data) and
                cache['model'] == self.embedding_deployment):
                
                self.embeddings = cache['embeddings']
                return True
            else:
                print("⚠️ Cache invalid (FAQ file changed or different model), regenerating...")
                return False
        
        except Exception as e:
            print(f"⚠️ Cache load error: {e}, regenerating...")
            return False
    
    def _save_embeddings_cache(self, file_hash: str):
        """Save embeddings with metadata"""
        cache = {
            'embeddings': self.embeddings,
            'file_hash': file_hash,
            'faq_count': len(self.faq_data),
            'model': self.embedding_deployment,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache, f)
            print(f"💾 Saved embeddings cache to {self.embeddings_cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Azure OpenAI
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []
    
    def generate_all_embeddings(self):
        """
        Generate embeddings for all FAQ questions
        """
        print("🔄 Generating embeddings for FAQ database...")
        
        for i, faq in enumerate(self.faq_data, 1):
            question_text = faq['question']        
            embedding = self.get_embedding(question_text)
            
            if embedding:
                self.embeddings.append(embedding)
                print(f"  ✓ Generated embedding {i}/{len(self.faq_data)}")
            else:
                self.embeddings.append([])
                print(f"  ✗ Failed to generate embedding {i}/{len(self.faq_data)}")
        
        print(f"✅ Generated {len(self.embeddings)} embeddings")
    

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_faqs(self, question: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Find most similar FAQs to user question
        Returns list of (faq_dict, similarity_score) tuples
        """
        # Generate embedding for user question
        question_embedding = self.get_embedding(question)
        
        if not question_embedding:
            print("❌ Failed to generate embedding for user question")
            return []
        
        # Calculate similarities
        similarities = []
        for i, faq_embedding in enumerate(self.embeddings):
            if faq_embedding:  # Skip empty embeddings
                similarity = self.cosine_similarity(question_embedding, faq_embedding)
                similarities.append((self.faq_data[i], similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return similarities[:top_k]
    
    def answer_question(self, question: str, similarity_threshold: float = 0.3) -> Dict:
        """
        Answer user question using FAQ system + LLM
        """
        # Find similar FAQs
        similar_faqs = self.find_similar_faqs(question, top_k=3)
        
        # Check if we have good matches
        if not similar_faqs or similar_faqs[0][1] < similarity_threshold:
            return {
                "answer": "Przepraszam, nie znalazłem odpowiedzi na to pytanie w bazie FAQ. Czy możesz je sformułować inaczej lub zadać bardziej szczegółowe pytanie?",
                "matched_faqs": [],
                "confidence": "low",
                "top_similarity": 0.0
            }
        
        # Build context from matched FAQs
        context = self._build_context(similar_faqs)
        
        # Generate answer using Azure OpenAI GPT-4
        answer = self._generate_answer_with_llm(question, context)
        
        return {
            "answer": answer,
            "matched_faqs": [
                {
                    "question": faq['question'],
                    "similarity": f"{score:.2f}",
                    "similarity_percent": f"{score * 100:.0f}%"
                }
                for faq, score in similar_faqs
            ],
            "confidence": "high" if similar_faqs[0][1] > 0.85 else "medium",
            "top_similarity": similar_faqs[0][1]
        }
    
    def _build_context(self, similar_faqs: List[Tuple[Dict, float]]) -> str:
        """Build context string from matched FAQs"""
        context = "Powiązane pytania i odpowiedzi z bazy FAQ:\n\n"
        
        for i, (faq, score) in enumerate(similar_faqs, 1):
            context += f"{i}. PYTANIE: {faq['question']}\n"
            context += f"   ODPOWIEDŹ: {faq['answer']}\n"
            context += f"   (podobieństwo: {score:.2f})\n\n"
        
        return context
    
    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """Use Azure OpenAI GPT-4 to generate natural answer based on FAQ context"""
        
        prompt = f"""{context}

PYTANIE UŻYTKOWNIKA:
{question}

ZADANIE:
Na podstawie powyższych informacji z FAQ, odpowiedz na pytanie użytkownika w naturalny, 
przyjazny sposób. Jeśli pytanie jest bezpośrednio związane z którymś z FAQ, użyj tej 
informacji. Jeśli nie ma dokładnego dopasowania, ale możesz pomóc na podstawie 
dostępnego kontekstu, zrób to. Odpowiedź powinna być zwięzła (2-4 zdania) i pomocna.

Jeśli żadna z informacji nie jest związana z pytaniem, powiedz o tym szczerze.
"""
        
        try:
            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment,  # Your deployment name (e.g., "gpt-4", "gpt-4-turbo")
                messages=[
                    {
                        "role": "system",
                        "content": """Jesteś pomocnym asystentem FAQ dla systemu analizy 
                        zdjęć prasowych. Odpowiadasz na pytania użytkowników w sposób przyjazny, 
                        profesjonalny i zwięzły. Używasz informacji z bazy FAQ, ale 
                        formułujesz odpowiedzi w naturalny sposób."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Przepraszam, wystąpił błąd podczas generowania odpowiedzi: {str(e)}"
    
    def get_all_faqs(self) -> List[Dict]:
        """Get all FAQ entries"""
        return self.faq_data
    
    def get_faq_by_id(self, faq_id: int) -> Optional[Dict]:
        """Get specific FAQ by ID"""
        for faq in self.faq_data:
            if faq.get('id') == faq_id:
                return faq
        return None
    
    def get_faq_count(self) -> int:
        """Get total number of FAQs"""
        return len(self.faq_data)
    
    # def search_faqs_by_keyword(self, keyword: str) -> List[Dict]:
    #     """
    #     Simple keyword search in questions and answers
    #     """
    #     keyword_lower = keyword.lower()
    #     results = []
        
    #     for faq in self.faq_data:
    #         if (keyword_lower in faq['question'].lower() or 
    #             keyword_lower in faq['answer'].lower()):
    #             results.append(faq)
        
    #     return results