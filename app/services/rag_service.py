import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
import cohere
from app.models.document import Document
from app.models.chat import Conversation, ChatMessage
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
import json
import openai

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.co = cohere.Client(settings.COHERE_API_KEY)
        # Configure OpenAI for chat completion
        openai.api_key = settings.OPENAI_API_KEY
        
    async def retrieve_relevant_documents(
        self, 
        query: str, 
        user_id: int, 
        db: Session, 
        top_k: int = 5,
        similarity_threshold: float = 0.05,
        use_hybrid_search: bool = True,
        selected_document_ids: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a given query using semantic search
        """
        try:
            # Get query embedding
            query_embedding = self.document_processor.create_embeddings(query)
            
            # Get user documents (filtered by selection if provided)
            query_filter = db.query(Document).filter(Document.user_id == user_id)
            
            if selected_document_ids:
                query_filter = query_filter.filter(Document.id.in_(selected_document_ids))
                logger.info(f"Filtering to selected documents: {selected_document_ids}")
            
            documents = query_filter.all()
            
            logger.info(f"Found {len(documents)} documents for user {user_id} (selected: {bool(selected_document_ids)})")
            
            if not documents:
                logger.info("No documents found for user")
                return []
            
            relevant_docs = []
            
            # Enhanced vector similarity search
            for doc in documents:
                if doc.embedding:
                    # Calculate semantic similarity
                    semantic_similarity = cosine_similarity(
                        [query_embedding],
                        [doc.embedding]
                    )[0][0]
                    
                    # Calculate keyword similarity for hybrid approach
                    keyword_similarity = 0
                    if use_hybrid_search:
                        keyword_similarity = self._calculate_keyword_similarity(query, doc)
                    
                    # Combine semantic and keyword scores
                    final_score = (semantic_similarity * 0.7) + (keyword_similarity * 0.3)
                    
                    logger.info(f"Document '{doc.title}': semantic={semantic_similarity:.3f}, keyword={keyword_similarity:.3f}, final={final_score:.3f} (threshold={similarity_threshold})")
                    
                    if final_score > similarity_threshold:
                        # Get relevant excerpt from document
                        excerpt = self._extract_relevant_excerpt(doc.content, query, max_length=300)
                        
                        relevant_docs.append({
                            "document_id": doc.id,
                            "title": doc.title,
                            "excerpt": excerpt,
                            "similarity_score": float(final_score),
                            "semantic_score": float(semantic_similarity),
                            "keyword_score": float(keyword_similarity),
                            "metadata": doc.metadata_col,
                            "file_type": doc.file_type
                        })
                        logger.info(f"✅ Document '{doc.title}' added to relevant docs (score: {final_score:.3f})")
                        logger.info(f"📄 Excerpt for query '{query}': {excerpt[:200]}...")
                    else:
                        logger.info(f"❌ Document '{doc.title}' filtered out (score: {final_score:.3f} < threshold: {similarity_threshold})")
                else:
                    logger.warning(f"⚠️ Document '{doc.title}' has no embedding - skipping semantic search")
                    # For documents without embeddings, use only keyword similarity
                    if use_hybrid_search:
                        keyword_similarity = self._calculate_keyword_similarity(query, doc)
                        final_score = keyword_similarity  # 100% keyword-based
                        
                        logger.info(f"Document '{doc.title}' (no embedding): keyword={keyword_similarity:.3f}, final={final_score:.3f} (threshold={similarity_threshold})")
                        
                        if final_score > similarity_threshold:
                            excerpt = self._extract_relevant_excerpt(doc.content, query, max_length=300)
                            
                            relevant_docs.append({
                                "document_id": doc.id,
                                "title": doc.title,
                                "excerpt": excerpt,
                                "similarity_score": float(final_score),
                                "semantic_score": 0.0,
                                "keyword_score": float(keyword_similarity),
                                "metadata": doc.metadata_col,
                                "file_type": doc.file_type
                            })
                            logger.info(f"✅ Document '{doc.title}' added to relevant docs (keyword-only score: {final_score:.3f})")
                            logger.info(f"📄 Excerpt for query '{query}': {excerpt[:200]}...")
                        else:
                            logger.info(f"❌ Document '{doc.title}' filtered out (keyword-only score: {final_score:.3f} < threshold: {similarity_threshold})")
                    else:
                        logger.info(f"❌ Document '{doc.title}' skipped (no embedding and hybrid search disabled)")
            
            # Sort by combined score and return top k
            relevant_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            final_docs = relevant_docs[:top_k]
            logger.info(f"Returning {len(final_docs)} relevant documents out of {len(documents)} total documents")
            return final_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {str(e)}")
            return []
    
    def _calculate_keyword_similarity(self, query: str, document: Document) -> float:
        """
        Calculate keyword-based similarity between query and document with enhanced phrase matching
        """
        try:
            query_lower = query.lower()
            title_lower = document.title.lower() if document.title else ""
            content_lower = document.content.lower() if document.content else ""
            doc_text = f"{title_lower} {content_lower}"
            
            # Check for exact query phrase match first
            if query_lower in doc_text:
                logger.info(f"Found exact phrase match for '{query}' in document")
                return 0.9  # High score for exact phrase match
            
            # Split query into meaningful terms
            query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 2]
            
            if not query_terms:
                return 0.0
            
            # Calculate different types of matches
            exact_matches = 0
            partial_matches = 0
            phrase_matches = 0
            
            # 1. Exact word matches
            doc_words = set(word.strip('.,!?":;()[]') for word in doc_text.split())
            for term in query_terms:
                if term in doc_words:
                    exact_matches += 1
                else:
                    # Check for partial matches (term contained in larger words)
                    if any(term in doc_word for doc_word in doc_words):
                        partial_matches += 1
            
            # 2. Multi-word phrase matches
            if len(query_terms) > 1:
                for i in range(len(query_terms)):
                    for j in range(i + 2, min(len(query_terms) + 1, i + 4)):  # Check 2-3 word phrases
                        phrase = " ".join(query_terms[i:j])
                        if phrase in doc_text:
                            phrase_matches += 1
                            logger.info(f"Found phrase match: '{phrase}' in document")
            
            # 3. Calculate scores
            exact_score = exact_matches / len(query_terms) if query_terms else 0
            partial_score = partial_matches / len(query_terms) if query_terms else 0
            phrase_score = phrase_matches / max(len(query_terms) - 1, 1) if len(query_terms) > 1 else 0
            
            # 4. Title bonus (if query terms appear in title, it's more relevant)
            title_bonus = 0
            if title_lower:
                title_words = set(title_lower.split())
                title_matches = sum(1 for term in query_terms if term in title_words)
                title_bonus = (title_matches / len(query_terms)) * 0.3
            
            # 5. Check metadata keywords
            metadata_bonus = 0
            if document.metadata_col and 'keywords' in document.metadata_col:
                doc_keywords = set(k.lower() for k in document.metadata_col['keywords'])
                metadata_matches = sum(1 for term in query_terms if term in doc_keywords)
                metadata_bonus = (metadata_matches / len(query_terms)) * 0.2
            
            # 6. Combine scores with weights
            final_score = (exact_score * 0.4) + (phrase_score * 0.3) + (partial_score * 0.1) + title_bonus + metadata_bonus
            
            # Cap at 1.0
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating keyword similarity: {str(e)}")
            return 0.0
    
    def _extract_relevant_excerpt(self, content: str, query: str, max_length: int = 800) -> str:
        """
        Extract the most relevant excerpt from document content based on query
        """
        try:
            # Enhanced approach: find paragraphs and sentences containing query terms
            query_terms = [term.lower().strip() for term in query.lower().split() if len(term.strip()) > 2]
            
            # First try to find exact phrase matches
            query_lower = query.lower()
            content_lower = content.lower()
            
            # Check for exact query match
            if query_lower in content_lower:
                start_pos = content_lower.find(query_lower)
                start_excerpt = max(0, start_pos - 300)
                end_excerpt = min(len(content), start_pos + len(query) + 500)
                excerpt = content[start_excerpt:end_excerpt]
                
                if start_excerpt > 0:
                    excerpt = "..." + excerpt
                if end_excerpt < len(content):
                    excerpt = excerpt + "..."
                    
                logger.info(f"Found exact phrase match for '{query}' at position {start_pos}")
                return excerpt
            
            # Check for related terms that might be in the document
            # For "muffin" query, also look for "Cell food muffins"
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            # Look for compound phrases that contain the query term
            for term in query_terms:
                # Find sentences that contain this term
                sentences = content.split('.')
                for i, sentence in enumerate(sentences):
                    if term in sentence.lower():
                        # Found a sentence with the term, extract surrounding context
                        start_idx = max(0, i - 1)
                        end_idx = min(len(sentences), i + 3)
                        context_sentences = sentences[start_idx:end_idx]
                        excerpt = '. '.join(context_sentences).strip()
                        
                        if len(excerpt) > max_length:
                            excerpt = excerpt[:max_length] + "..."
                        
                        logger.info(f"Found term '{term}' in sentence context")
                        return excerpt
            
            # Split into paragraphs first (better context)
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            if not paragraphs:
                sentences = [s.strip() for s in content.split('. ') if s.strip()]
            else:
                sentences = paragraphs
            
            # Score sentences/paragraphs based on query term presence
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                # Count exact term matches
                exact_matches = sum(1 for term in query_terms if term in sentence_lower)
                # Count partial matches (for compound words)
                partial_matches = sum(1 for term in query_terms 
                                    if any(term in word for word in sentence_lower.split()))
                
                total_score = (exact_matches * 2) + partial_matches
                
                if total_score > 0:
                    scored_sentences.append((sentence, total_score, i))
            
            if not scored_sentences:
                # Return first substantial part of content if no matches
                return content[:max_length] + "..." if len(content) > max_length else content
            
            # Get the best matching sentences with more context
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Take top scoring sentence and add surrounding context
            best_sentence = scored_sentences[0]
            start_idx = max(0, best_sentence[2] - 1)
            end_idx = min(len(sentences), best_sentence[2] + 3)  # More context
            
            # Include multiple high-scoring sentences if they're nearby
            for other_sentence in scored_sentences[1:3]:  # Check next 2 best
                if abs(other_sentence[2] - best_sentence[2]) <= 2:  # If nearby
                    start_idx = min(start_idx, other_sentence[2] - 1)
                    end_idx = max(end_idx, other_sentence[2] + 2)
            
            excerpt = ' '.join(sentences[start_idx:end_idx])
            
            if len(excerpt) > max_length:
                # Try to cut at sentence boundary
                excerpt_words = excerpt.split()
                truncated = []
                char_count = 0
                for word in excerpt_words:
                    if char_count + len(word) + 1 > max_length - 3:
                        break
                    truncated.append(word)
                    char_count += len(word) + 1
                excerpt = ' '.join(truncated) + "..."
                
            return excerpt
            
        except Exception as e:
            logger.error(f"Error extracting excerpt: {str(e)}")
            return content[:max_length] + "..." if len(content) > max_length else content
    
    async def generate_rag_response(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG (Retrieval-Augmented Generation)
        """
        try:
            # Prepare context from retrieved documents
            context_text = ""
            if context_documents:
                context_text = "\n\n".join([
                    f"Document: {doc['title']}\nContent: {doc['excerpt']}"
                    for doc in context_documents
                ])
            
            # Prepare conversation history
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in conversation_history[-5:]  # Last 5 messages for context
                ])
            
            # Create the prompt
            system_prompt = """You are an intelligent assistant that helps users understand and analyze their documents. 
            
            IMPORTANT INSTRUCTIONS:
            1. Use ONLY the provided document context to answer questions
            2. Quote directly from the documents when possible
            3. If information exists in the context, provide a comprehensive answer
            4. Always cite which documents you're referencing
            5. If the answer cannot be found in the provided context, clearly state this
            6. Pay close attention to specific phrases, names, and terms mentioned in the query"""
            
            user_prompt = f"""
            Context from documents:
            {context_text}
            
            Conversation history:
            {history_text}
            
            User question: {query}
            
            Please provide a comprehensive answer based on the document context provided."""
            
            # Try OpenAI first, fallback to Cohere
            try:
                response = await self._generate_openai_response(system_prompt, user_prompt)
            except Exception as openai_error:
                logger.warning(f"OpenAI API error, falling back to Cohere: {openai_error}")
                response = await self._generate_cohere_response(user_prompt)
            
            return {
                "response": response,
                "context_documents": context_documents,
                "sources_used": len(context_documents)
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "context_documents": [],
                "sources_used": 0
            }
    
    async def _generate_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI GPT"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise e
    
    async def _generate_cohere_response(self, prompt: str) -> str:
        """Generate response using Cohere as fallback"""
        try:
            # Updated to use Cohere Chat API with current model
            response = self.co.chat(
                model='command-r-08-2024',
                message=prompt,
                max_tokens=1000,
                temperature=0.7,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise e
    
    async def create_conversation(self, title: str, user_id: int, db: Session) -> Conversation:
        """Create a new conversation"""
        try:
            conversation = Conversation(
                title=title,
                user_id=user_id
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            return conversation
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise e
    
    async def add_message_to_conversation(
        self, 
        conversation_id: int, 
        role: str, 
        content: str, 
        context_documents: List[Dict] = None,
        db: Session = None
    ) -> ChatMessage:
        """Add a message to a conversation"""
        try:
            message = ChatMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                context_documents=context_documents,
                message_metadata={"timestamp": "now"}
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            return message
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise e
    
    async def get_conversation_history(self, conversation_id: int, db: Session) -> List[Dict[str, str]]:
        """Get conversation message history"""
        try:
            messages = db.query(ChatMessage).filter(
                ChatMessage.conversation_id == conversation_id
            ).order_by(ChatMessage.created_at).all()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
