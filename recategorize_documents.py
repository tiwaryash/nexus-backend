#!/usr/bin/env python3
"""
Script to recategorize existing documents with the new categorization system
"""

import sys
import os
import asyncio
from sqlalchemy.orm import Session

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import SessionLocal
from app.models.document import Document
from app.services.document_processor import DocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def recategorize_documents():
    """Recategorize all existing documents"""
    db = SessionLocal()
    processor = DocumentProcessor()
    
    try:
        # Get all documents
        documents = db.query(Document).all()
        logger.info(f"Found {len(documents)} documents to recategorize")
        
        updated_count = 0
        
        for doc in documents:
            try:
                # Skip if document already has a good category (not generic ones)
                current_category = doc.metadata_col.get("category", "") if doc.metadata_col else ""
                if current_category and current_category not in ["Document", "Uncategorized", "Unknown", ""]:
                    logger.info(f"Skipping '{doc.title}' - already has category: {current_category}")
                    continue
                
                logger.info(f"Recategorizing document: {doc.title}")
                
                # Get existing metadata or create new
                metadata = doc.metadata_col or {}
                
                # Extract keywords if not present
                keywords = metadata.get("keywords", [])
                if not keywords:
                    keywords = processor.extract_keywords(doc.content)
                    metadata["keywords"] = keywords
                
                # Classify the document
                new_category = processor.classify_document_by_keywords(keywords, doc.content)
                metadata["category"] = new_category
                
                # Update the document
                doc.metadata_col = metadata
                db.commit()
                
                updated_count += 1
                logger.info(f"✅ Updated '{doc.title}' -> Category: {new_category}")
                
            except Exception as e:
                logger.error(f"❌ Error processing document '{doc.title}': {str(e)}")
                db.rollback()
                continue
        
        logger.info(f"✅ Recategorization complete! Updated {updated_count} documents")
        
    except Exception as e:
        logger.error(f"Error in recategorization: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(recategorize_documents())
