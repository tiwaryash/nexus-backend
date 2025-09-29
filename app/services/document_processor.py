from typing import BinaryIO, Dict
import PyPDF2
from docx import Document as DocxDocument
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import cohere
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data at startup
import nltk

# Initialize Cohere client
co = cohere.Client(api_key=settings.COHERE_API_KEY)

logger = logging.getLogger(__name__)

def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('words')

download_nltk_data()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english'
        )
        
        # Import templates
        from app.templates.academic_template import ACADEMIC_TEMPLATE
        from app.templates.business_template import BUSINESS_TEMPLATE
        # from app.templates.technical_template import TECHNICAL_TEMPLATE
        # from app.templates.letter_template import LETTER_TEMPLATE
        
        # Initialize templates dictionary
        self.templates = {
            'academic': ACADEMIC_TEMPLATE,
            'business': BUSINESS_TEMPLATE,
            # 'technical': TECHNICAL_TEMPLATE,
            # 'letter': LETTER_TEMPLATE
        }

    def process_pdf(self, file: BinaryIO) -> str:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def process_docx(self, file: BinaryIO) -> str:
        doc = DocxDocument(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def process_markdown(self, content: str) -> str:
        return markdown.markdown(content)

    def create_embeddings(self, text: str) -> list[float]:
        """Create document embeddings using TF-IDF as fallback"""
        try:
            # Fit and transform the text
            vector = self.vectorizer.fit_transform([text])
            # Convert sparse matrix to dense and normalize
            dense_vector = vector.todense()
            normalized = dense_vector / np.linalg.norm(dense_vector)
            # Convert to list and ensure consistent size
            embedding = normalized.tolist()[0]
            # Pad or truncate to exactly 512 dimensions
            if len(embedding) < 512:
                embedding.extend([0] * (512 - len(embedding)))
            return embedding[:512]
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return [0] * 512  # Return zero vector as fallback

    async def analyze_document(self, text: str) -> Dict:
        try:
            chunks = self.text_splitter.split_text(text)
            first_chunk = chunks[0] if chunks else text[:4000]
            
            try:
                # Updated to use Cohere Chat API with current model
                response = co.chat(
                    model='command-r-08-2024',
                    message=f"""Analyze this text and provide the following in JSON format:
                    1) summary: A concise 2-3 sentence summary
                    2) keywords: Array of 5-10 most important keywords
                    3) entities: Array of named entities (people, organizations, locations)
                    4) category: The document category/type

                    Text: {first_chunk}

                    Output in JSON format only.""",
                    max_tokens=500,
                    temperature=0.3,
                )
                
                ai_analysis = json.loads(response.text)
                embedding = self.create_embeddings(text)
                
                logger.info(f"AI Analysis successful: {ai_analysis}")

                return {
                    "summary": ai_analysis.get("summary", ""),
                    "keywords": ai_analysis.get("keywords", []),
                    "entities": ai_analysis.get("entities", []),
                    "category": ai_analysis.get("category", "Unknown"),
                    "embedding": embedding
                }

            except Exception as e:
                logger.error(f"Cohere API error: {str(e)}")
                return self.basic_analysis(text)
            
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return self.basic_analysis(text)

    def extract_keywords(self, text: str) -> list[str]:
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # Keep only nouns and adjectives
            important_words = [word for word, tag in tagged 
                             if tag.startswith(('NN', 'JJ')) 
                             and word not in self.stop_words
                             and len(word) > 2]
            
            # Get frequency distribution
            freq_dist = Counter(important_words)
            
            # Return top 10 keywords
            return [word for word, _ in freq_dist.most_common(10)]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def basic_analysis(self, text: str) -> Dict:
        """Fallback analysis when AI is unavailable"""
        keywords = self.extract_keywords(text)
        category = self.classify_document_by_keywords(keywords, text)
        
        logger.info(f"Basic analysis - Keywords: {keywords[:5]}, Category: {category}")
        logger.info(f"Full text preview: {text[:100]}...")
        
        return {
            "summary": text[:200] + "...",  # Basic summary
            "keywords": keywords,
            "entities": self.extract_entities(text),
            "category": category,
            "embedding": self.create_embeddings(text)
        }

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities using NLTK"""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        
        named_entities = []
        for chunk in entities:
            if hasattr(chunk, 'label'):
                named_entities.append(' '.join(c[0] for c in chunk))
        
        return list(set(named_entities))[:10]  # Return up to 10 unique entities

    def classify_document_by_keywords(self, keywords: list[str], text: str) -> str:
        """Classify document based on keywords and content patterns"""
        # Convert keywords and text to lowercase for matching
        keywords_lower = [k.lower() for k in keywords]
        text_lower = text.lower()
        
        # Define category patterns
        category_patterns = {
            "Academic/Research": [
                "research", "study", "analysis", "methodology", "hypothesis", "conclusion",
                "abstract", "literature", "theory", "experiment", "data", "results",
                "findings", "publication", "journal", "academic", "university", "scholar",
                "behavior", "outcome", "distinction", "concept", "principle", "ethics",
                "philosophy", "psychology", "sociology", "investigation", "observation"
            ],
            "Business/Finance": [
                "business", "finance", "financial", "revenue", "profit", "market", "sales",
                "investment", "budget", "strategy", "management", "company", "corporate",
                "quarterly", "annual", "report", "earnings", "roi", "kpi", "metrics"
            ],
            "Technical/Programming": [
                "code", "programming", "software", "development", "algorithm", "function",
                "api", "database", "server", "framework", "library", "documentation",
                "technical", "system", "architecture", "deployment", "debugging"
            ],
            "Legal/Contracts": [
                "legal", "contract", "agreement", "clause", "terms", "conditions",
                "law", "regulation", "compliance", "policy", "procedure", "rights",
                "obligations", "liability", "jurisdiction", "attorney", "court"
            ],
            "Healthcare/Medical": [
                "health", "medical", "patient", "treatment", "diagnosis", "symptoms",
                "medicine", "healthcare", "clinical", "therapy", "hospital", "doctor",
                "nurse", "pharmaceutical", "disease", "wellness", "surgery"
            ],
            "Education/Training": [
                "education", "training", "learning", "course", "curriculum", "lesson",
                "student", "teacher", "instructor", "classroom", "assessment", "grade",
                "knowledge", "skill", "tutorial", "workshop", "seminar"
            ],
            "Marketing/Communications": [
                "marketing", "advertising", "promotion", "campaign", "brand", "customer",
                "communication", "social", "media", "content", "engagement", "outreach",
                "publicity", "pr", "messaging", "audience", "demographic"
            ],
            "Personal/Miscellaneous": [
                "personal", "journal", "diary", "note", "memo", "reminder", "todo",
                "list", "plan", "schedule", "appointment", "meeting", "event",
                "right", "wrong", "moral", "ethical", "values", "beliefs", "opinion",
                "thoughts", "reflection", "perspective", "viewpoint"
            ]
        }
        
        # Score each category
        category_scores = {}
        for category, patterns in category_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in keywords_lower:
                for pattern in patterns:
                    if pattern in keyword or keyword in pattern:
                        score += 2  # Higher weight for keyword matches
            
            # Check text content
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            
            category_scores[category] = score
        
        # Return the category with the highest score, or "General" if no clear match
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return "General"

    async def apply_template(self, content: str, template_id: str) -> str:
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        try:
            # Split content into smaller chunks
            chunks = self.text_splitter.split_text(content)
            first_chunk = chunks[0] if chunks else content[:2000]

            prompt = f"""Analyze and reorganize this document into a structured format.

            Required sections: {', '.join(template['structure'])}

            Instructions:
            1. Extract relevant content for each section
            2. If a section's content isn't found, write a placeholder
            3. Format MUST follow this structure exactly:

            <BEGIN>
            [SECTION]title[/SECTION]
            [CONTENT]Document title here[/CONTENT]

            [SECTION]next_section[/SECTION]
            [CONTENT]Content for this section[/CONTENT]
            <END>

            Here's the document:
            {first_chunk}
            """

            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1,
                stop_sequences=["<END>"]
            )

            # Parse sections using regex
            import re
            sections = {}
            pattern = r'\[SECTION\](.*?)\[/SECTION\]\s*\[CONTENT\](.*?)\[/CONTENT\]'
            matches = re.findall(pattern, response.generations[0].text, re.DOTALL)
            
            for section_name, content in matches:
                sections[section_name.strip().lower()] = content.strip()

            # Process remaining chunks for missing sections
            remaining_text = ' '.join(chunks[1:]) if len(chunks) > 1 else ''
            
            # Ensure all template sections exist with meaningful content
            for section in template['structure']:
                if section not in sections or not sections[section].strip():
                    if remaining_text:
                        # Extract relevant content for this section from remaining text
                        section_prompt = f"Extract content relevant to the '{section}' section from this text: {remaining_text[:1000]}"
                        section_response = co.generate(
                            model='command',
                            prompt=section_prompt,
                            max_tokens=500,
                            temperature=0.1
                        )
                        sections[section] = section_response.generations[0].text.strip()
                        remaining_text = remaining_text[1000:]
                    else:
                        sections[section] = f"[{section} section to be added]"

            # Format according to template
            formatted_content = template['format'].format(**sections)
            
            # Clean up formatting
            formatted_content = '\n'.join(line.strip() for line in formatted_content.splitlines() if line.strip())
            
            return formatted_content

        except Exception as e:
            logger.error(f"Error in template application: {str(e)}")
            # Don't return original content on error, return basic formatted version
            return self._format_basic_template(content, template)

    def _format_basic_template(self, content: str, template: dict) -> str:
        """Fallback formatting when AI processing fails"""
        sections = {}
        content_parts = content.split('\n\n')
        
        # Basic content distribution
        sections['title'] = content_parts[0] if content_parts else "Untitled Document"
        remaining_parts = content_parts[1:] if len(content_parts) > 1 else []
        
        for i, section in enumerate(template['structure'][1:], 1):
            if i < len(remaining_parts):
                sections[section] = remaining_parts[i]
            else:
                sections[section] = f"[{section} section]"
        
        return template['format'].format(**sections) 