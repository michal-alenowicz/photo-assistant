import openai
import numpy as np
from typing import List, Dict, Tuple
import json
import pickle
import hashlib
import os
from datetime import datetime


class FAQSystem:
    def __init__(
        self, 
        openai_api_key: str,
        chat_model: str,
        embedding_model: str,
        faq_file_path: str
    ):
        """
        Initialize FAQ System with OpenAI API
        
        """
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.faq_file_path = faq_file_path
        self.faq_data = []
        self.embeddings = []
        self.embeddings_cache_file = "faq_embeddings_cache.pkl"
        
        # Detect if using GPT-5 family
        self.is_gpt5 = chat_model.startswith("gpt-5")


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
        """Load embeddings from cache if valid"""
        if not os.path.exists(self.embeddings_cache_file):
            return False
        
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            # Validate cache
            if (cache['file_hash'] == current_hash and 
                cache['faq_count'] == len(self.faq_data) and
                cache['model'] == self.embedding_model):
                
                self.embeddings = cache['embeddings']
                return True
            else:
                print("⚠️ Cache invalid, regenerating...")
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
            'model': self.embedding_model,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache, f)
            print(f"Saved embeddings cache")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    

    def generate_all_embeddings(self):
        """Generate embeddings for all FAQ questions"""
        print("🔄 Generating embeddings...")
        
        for i, faq in enumerate(self.faq_data, 1):
            question_text = faq['question']        
            embedding = self.get_embedding(question_text)
            
            if embedding:
                self.embeddings.append(embedding)
                print(f"Generated {i}/{len(self.faq_data)}")
            else:
                self.embeddings.append([])
                print(f"Failed {i}/{len(self.faq_data)}")
        
        print(f"✅ Generated {len(self.embeddings)} embeddings")
    

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
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
        """Find most similar FAQs"""
        question_embedding = self.get_embedding(question)
        
        if not question_embedding:
            return []
        
        similarities = []
        for i, faq_embedding in enumerate(self.embeddings):
            if faq_embedding:
                similarity = self.cosine_similarity(question_embedding, faq_embedding)
                similarities.append((self.faq_data[i], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    

    def answer_question(self, question: str, similarity_threshold: float = 0.35) -> Dict:
        """Answer user question using FAQ + OpenAI"""
        similar_faqs = self.find_similar_faqs(question, top_k=3)
        
        if not similar_faqs or similar_faqs[0][1] < similarity_threshold:
            return {
                "answer": "Przepraszam, nie znalazłem odpowiedzi na to pytanie w bazie FAQ.",
                "matched_faqs": [],
                "confidence": "low",
                "top_similarity": 0.0
            }
        
        context = self._build_context(similar_faqs)
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
        """Build context from matched FAQs"""
        context = "Powiązane pytania z FAQ:\n\n"
        
        for i, (faq, score) in enumerate(similar_faqs, 1):
            context += f"{i}. PYTANIE: {faq['question']}\n"
            context += f"   ODPOWIEDŹ: {faq['answer']}\n"
            context += f"   (podobieństwo: {score:.2f})\n\n"
        
        return context
    
    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """Generate answer using OpenAI"""
        prompt = f"""{context}

PYTANIE UŻYTKOWNIKA:
{question}

Odpowiedz na pytanie w naturalny sposób (2-4 zdania) na podstawie informacji z FAQ.
"""
        
        try:
            # GPT-5 compatible parameters
            if self.is_gpt5:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Jesteś pomocnym asystentem FAQ. Odpowiadasz zwięźle i profesjonalnie."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    reasoning_effort="none",
                    verbosity='low' 
                )
            else:
                # GPT-4 and earlier
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Jesteś pomocnym asystentem FAQ. Odpowiadasz zwięźle i profesjonalnie."
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
            return f"Przepraszam, wystąpił błąd: {str(e)}"
    
    def get_faq_count(self) -> int:
        """Get total FAQ count"""
        return len(self.faq_data)