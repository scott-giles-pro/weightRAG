#!/usr/bin/env python3
"""
ValueRAG: A locally-deployed RAG system for ideological bias mitigation in LLMs
Implements query-time document weighting and value-conditioned prompting
for Christian and American constitutional value alignment
"""

import requests
import json
import sys
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Document weights mapping (user-configurable)
DOCUMENT_WEIGHTS = {
    "bible": 3.0, "constitution": 3.0, "declaration": 3.0, "federalist": 2.0,
    "city_of_god": 3.0, "augustine": 3.0, "summa": 2.0, "aquinas": 2.0,
    "institutes": 2.0, "calvin": 2.0, "imitation": 1.0, "kempis": 1.0,
    "pilgrim": 1.0, "bunyan": 1.0, "treatise": 3.0, "locke": 3.0,
    "common_sense": 1.0, "paine": 1.0, "spirit": 3.0, "montesquieu": 3.0,
    "reflections": 1.0, "burke": 1.0, "farmer": 1.0, "dickinson": 1.0,
    "mayflower": 1.0, "democracy": 2.0, "tocqueville": 2.0, "organon": 3.0,
    "aristotle": 2.0, "meditations": 1.0, "aurelius": 1.0, "ethics": 1.0,
    "duties": 1.0, "cicero": 1.0, "seneca": 2.0, "magna_carta": 2.0,
    "politics": 2.0, "plutarch": 1.0, "machiavelli": 1.0, "wealth": 2.0,
    "smith": 2.0, "vindiciae": 1.0, "mornay": 1.0, "leviathan": 1.0, "hobbes": 1.0
}

# Value-conditioned system prompt
VALUE_CONDITIONED_PROMPT = """You are an AI assistant aligned with biblical Christian principles and American constitutional values. 
Your responses should reflect:

1. Biblical Wisdom: Ground moral reasoning in Scripture, emphasizing truth, justice, mercy, and righteousness as defined in Christian theology.

2. Constitutional Principles: Uphold natural rights, rule of law, separation of powers, federalism, and individual liberty as articulated in the U.S. Constitution and founding documents.

3. Classical Reasoning: Apply logical analysis informed by Aristotelian logic and Western philosophical tradition.

4. Discernment: Exercise careful judgment in evaluating claims, recognizing potential for disinformation and ideological manipulation (1 Thessalonians 5:21: "Test everything; hold fast what is good").

When relevant passages from foundational texts are provided, cite them appropriately and use them to support your reasoning. Your responses should be well-reasoned, truthful, and maintain scholarly rigor while remaining accessible."""

# Plain baseline prompt (no value conditioning)
BASELINE_PROMPT = """You are a helpful AI assistant. Provide clear, accurate, and well-reasoned responses to user queries."""


class DocumentRAG:
    """Retrieval-Augmented Generation system with weighted document retrieval."""
    
    def __init__(self, docs_folder: str = "ragReferenceDocs", enable_weighting: bool = True):
        self.docs_folder = Path(docs_folder)
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        self.loaded = False
        self.enable_weighting = enable_weighting
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}", file=sys.stderr)
            return ""
    
    def get_document_weight(self, filename: str) -> float:
        """Get the weight for a document based on its filename."""
        if not self.enable_weighting:
            return 1.0
            
        filename_lower = filename.lower().replace('.pdf', '').replace('_', ' ')
        
        # Direct match
        for key, weight in DOCUMENT_WEIGHTS.items():
            if key.replace('_', ' ') in filename_lower:
                return weight
        
        return 1.0  # default weight
    
    def load_documents(self) -> bool:
        """Load and process all PDF documents from the reference folder."""
        if not self.docs_folder.exists():
            print(f"Documents folder '{self.docs_folder}' not found!", file=sys.stderr)
            return False
        
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in '{self.docs_folder}'", file=sys.stderr)
            return False
        
        print(f"Loading {len(pdf_files)} documents from {self.docs_folder}...")
        
        for pdf_path in pdf_files:
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                # Split into chunks for better retrieval
                chunks = self.split_into_chunks(text, chunk_size=1000, overlap=200)
                weight = self.get_document_weight(pdf_path.stem)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append({
                        'filename': pdf_path.name,
                        'chunk_id': i,
                        'content': chunk,
                        'weight': weight
                    })
                
                print(f"  Loaded: {pdf_path.name} (weight: {weight}, chunks: {len(chunks)})")
        
        if self.documents:
            # Create TF-IDF vectors for all document chunks
            print(f"\nVectorizing {len(self.documents)} document chunks...")
            texts = [doc['content'] for doc in self.documents]
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            self.doc_vectors = self.vectorizer.fit_transform(texts)
            self.loaded = True
            print("RAG system ready!\n")
            return True
        
        return False
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks by words."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def search_documents(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
        """Search for relevant document chunks based on the query."""
        if not self.loaded:
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Apply weights if enabled
            if self.enable_weighting:
                weighted_scores = []
                for i, sim in enumerate(similarities):
                    weight = self.documents[i]['weight']
                    # Weight multiplier: higher weight = higher priority
                    weighted_score = sim * (1.0 + (weight - 1.0) * 0.3)
                    weighted_scores.append((weighted_score, i))
            else:
                weighted_scores = [(sim, i) for i, sim in enumerate(similarities)]
            
            # Sort by score and get top results
            top_indices = sorted(weighted_scores, reverse=True)[:top_k]
            
            results = []
            for score, idx in top_indices:
                if similarities[idx] > min_similarity:
                    results.append({
                        'content': self.documents[idx]['content'],
                        'filename': self.documents[idx]['filename'],
                        'chunk_id': self.documents[idx]['chunk_id'],
                        'weight': self.documents[idx]['weight'],
                        'similarity': float(similarities[idx]),
                        'weighted_score': float(score)
                    })
            
            return results
        except Exception as e:
            print(f"Error searching documents: {e}", file=sys.stderr)
            return []


class ValueRAG:
    """Main ValueRAG system coordinating LLM and RAG components."""
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 use_value_conditioning: bool = True,
                 use_weighted_retrieval: bool = True,
                 docs_folder: str = "ragReferenceDocs"):
        
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        
        # Configuration flags
        self.use_value_conditioning = use_value_conditioning
        self.use_weighted_retrieval = use_weighted_retrieval
        
        # Initialize RAG
        self.rag = DocumentRAG(docs_folder=docs_folder, enable_weighting=use_weighted_retrieval)
        self.rag_enabled = self.rag.load_documents()
        
        # Session tracking
        self.session_history = []
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                # Check for exact match or with :latest tag
                if any(self.model_name in name for name in model_names):
                    return True
                else:
                    print(f"Model '{self.model_name}' not found.", file=sys.stderr)
                    print(f"Available models: {model_names}", file=sys.stderr)
                    return False
            return False
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Make sure it's running with: ollama serve", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error checking Ollama status: {e}", file=sys.stderr)
            return False
    
    def create_enhanced_prompt(self, user_query: str, top_k: int = 3) -> tuple[str, List[Dict]]:
        """Create an enhanced prompt with relevant document passages."""
        if not self.rag_enabled:
            return user_query, []
        
        # Search for relevant passages
        relevant_docs = self.rag.search_documents(user_query, top_k=top_k)
        
        if not relevant_docs:
            return user_query, []
        
        # Build enhanced prompt with context
        enhanced_prompt = f"Query: {user_query}\n\n"
        enhanced_prompt += "Relevant passages from foundational texts:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            source_info = f"[Source: {doc['filename']}, Weight: {doc['weight']}, Relevance: {doc['similarity']:.3f}]"
            # Truncate content for prompt
            content_preview = doc['content'][:600] + "..." if len(doc['content']) > 600 else doc['content']
            enhanced_prompt += f"Passage {i} {source_info}:\n{content_preview}\n\n"
        
        enhanced_prompt += "\nPlease respond to the query drawing upon these passages and other relevant principles from the foundational texts."
        
        return enhanced_prompt, relevant_docs
    
    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """Process a query and return response with metadata."""
        
        # Get system prompt based on configuration
        system_prompt = VALUE_CONDITIONED_PROMPT if self.use_value_conditioning else BASELINE_PROMPT
        
        # Enhance the prompt with RAG if available
        enhanced_query, retrieved_docs = self.create_enhanced_prompt(user_query, top_k=top_k)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_query}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            assistant_response = result['message']['content']
            
            # Build response object
            response_obj = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "response": assistant_response,
                "configuration": {
                    "model": self.model_name,
                    "value_conditioning": self.use_value_conditioning,
                    "weighted_retrieval": self.use_weighted_retrieval,
                    "rag_enabled": self.rag_enabled
                },
                "retrieved_documents": [
                    {
                        "filename": doc['filename'],
                        "weight": doc['weight'],
                        "similarity": doc['similarity'],
                        "weighted_score": doc['weighted_score']
                    } for doc in retrieved_docs
                ] if retrieved_docs else [],
                "error": None
            }
            
            # Store in session history
            self.session_history.append(response_obj)
            
            return response_obj
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "response": None,
                "error": f"Error communicating with Ollama: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "response": None,
                "error": f"Unexpected error: {e}"
            }
    
    def export_session(self, filename: str = None) -> str:
        """Export session history to JSON file."""
        if filename is None:
            filename = f"valuerag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.session_history, f, indent=2, ensure_ascii=False)
        
        return filename


def interactive_mode():
    """Run ValueRAG in interactive mode."""
    print("=" * 70)
    print("ValueRAG: Bias Mitigation System for LLMs")
    print("=" * 70)
    print()
    
    # Configuration
    print("Configuration:")
    print("  [1] Full ValueRAG (weighted retrieval + value conditioning)")
    print("  [2] Baseline RAG (equal weights + neutral prompting)")
    print("  [3] Custom configuration")
    print()
    
    config_choice = input("Select configuration [1/2/3]: ").strip()
    
    if config_choice == "1":
        use_value = True
        use_weights = True
        config_name = "Full ValueRAG"
    elif config_choice == "2":
        use_value = False
        use_weights = False
        config_name = "Baseline RAG"
    elif config_choice == "3":
        use_value = input("Enable value-conditioned prompting? [y/n]: ").lower().startswith('y')
        use_weights = input("Enable weighted document retrieval? [y/n]: ").lower().startswith('y')
        config_name = "Custom"
    else:
        use_value = True
        use_weights = True
        config_name = "Full ValueRAG (default)"
    
    print(f"\nInitializing {config_name}...\n")
    
    # Initialize system
    system = ValueRAG(
        use_value_conditioning=use_value,
        use_weighted_retrieval=use_weights
    )
    
    if not system.check_ollama_status():
        print("\nError: Cannot connect to Ollama or model not found.")
        return
    
    print(f"\n{'=' * 70}")
    print(f"System ready! Configuration: {config_name}")
    print(f"Value Conditioning: {'✓' if use_value else '✗'}")
    print(f"Weighted Retrieval: {'✓' if use_weights else '✗'}")
    print(f"RAG Database: {'✓' if system.rag_enabled else '✗'}")
    print(f"{'=' * 70}\n")
    print("Commands: 'quit' or 'exit' to end, 'export' to save session\n")
    
    # Main loop
    while True:
        try:
            user_input = input("Query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting ValueRAG...")
                break
            
            if user_input.lower() == 'export':
                filename = system.export_session()
                print(f"\nSession exported to: {filename}\n")
                continue
            
            print("\nProcessing query...\n")
            result = system.query(user_input)
            
            if result['success']:
                print("-" * 70)
                print("RESPONSE:")
                print("-" * 70)
                print(result['response'])
                print()
                
                if result['retrieved_documents']:
                    print("-" * 70)
                    print("RETRIEVED DOCUMENTS:")
                    for i, doc in enumerate(result['retrieved_documents'], 1):
                        print(f"  {i}. {doc['filename']} (weight: {doc['weight']}, "
                              f"similarity: {doc['similarity']:.3f}, "
                              f"weighted: {doc['weighted_score']:.3f})")
                    print()
            else:
                print(f"\nError: {result['error']}\n")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")
    
    # Export on exit
    if system.session_history:
        save = input("\nSave session history? [y/n]: ").lower().startswith('y')
        if save:
            filename = system.export_session()
            print(f"Session saved to: {filename}")


def batch_evaluation_mode(queries_file: str):
    """Run batch evaluation with test queries."""
    print("Running batch evaluation mode...")
    print(f"Loading queries from: {queries_file}\n")
    
    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    # Run both configurations
    configs = [
        ("baseline", False, False),
        ("valuerag", True, True)
    ]
    
    results = {}
    
    for config_name, use_value, use_weights in configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {config_name.upper()}")
        print(f"{'=' * 70}\n")
        
        system = ValueRAG(
            use_value_conditioning=use_value,
            use_weighted_retrieval=use_weights
        )
        
        if not system.check_ollama_status():
            print("Error: Cannot connect to Ollama")
            continue
        
        config_results = []
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")
            result = system.query(query)
            config_results.append(result)
            print("  ✓ Complete\n")
        
        results[config_name] = config_results
        
        # Export results
        filename = f"evaluation_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_results, f, indent=2, ensure_ascii=False)
        print(f"Results exported to: {filename}")
    
    print("\n" + "=" * 70)
    print("Batch evaluation complete!")
    print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("ValueRAG: Bias Mitigation System for LLMs")
            print("\nUsage:")
            print("  python valuerag.py              # Interactive mode")
            print("  python valuerag.py queries.txt  # Batch evaluation mode")
            print("\nEnvironment variables:")
            print("  OLLAMA_MODEL - Model name (default: llama3.1:8b)")
            print("  OLLAMA_URL   - Ollama API URL (default: http://localhost:11434)")
            return
        else:
            # Batch mode with queries file
            batch_evaluation_mode(sys.argv[1])
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
