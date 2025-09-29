from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.document_processor import DocumentProcessor
from app.models.document import Document
from app.core.config import settings
from app.api.auth import get_current_user  # Add this import
from app.models.user import User
from typing import Annotated
import os
import uuid
from openai import OpenAIError
import logging
from starlette.responses import FileResponse
import json
from sqlalchemy import or_, cast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import String
from io import BytesIO  # Add this import
from storage.storage import S3StorageProvider  # Add this import if missing
from app.models.share_access import ShareAccess, AccessLevel
from app.services.email import EmailService
from sqlalchemy.exc import OperationalError
import time

logger = logging.getLogger(__name__)
router = APIRouter()
document_processor = DocumentProcessor()

@router.post("/upload")
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Validate file extension
        ext = file.filename.split(".")[-1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="File type not supported")

        # Read file content
        file_content = await file.read()
        
        # Process content
        content = ""
        if ext == "pdf":
            content = document_processor.process_pdf(BytesIO(file_content))
        elif ext == "docx":
            content = document_processor.process_docx(BytesIO(file_content))
        elif ext == "md":
            content = document_processor.process_markdown(file_content.decode())

        # Analyze document content
        analysis = await document_processor.analyze_document(content)

             # Upload to S3
        storage = S3StorageProvider()
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = await storage.upload_file(file_content, filename)

        # Create document record
        document = Document(
            title=file.filename,
            content=content,
            file_path=file_path,
            file_type=ext,
            embedding=analysis["embedding"],
            metadata_col={
                "summary": analysis["summary"],
                "keywords": analysis["keywords"],
                "entities": analysis.get("entities", []),
                "category": analysis.get("category", "General")
            },
            user_id=current_user.id
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@router.get("")
async def get_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    documents = db.query(Document).filter(
        Document.user_id == current_user.id
    ).order_by(Document.created_at.desc()).all()
    
    return documents 

@router.get("/search")
async def search_documents(
    q: str,
    filters: str | None = None,
    search_type: str = "hybrid",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Parse filters if provided
    filter_dict = json.loads(filters) if filters else {}
    
    # Preprocess search query
    query_terms = q.lower().split()
    
    # Base query
    query = db.query(Document).filter(Document.user_id == current_user.id)
    
    # Apply metadata filters
    if filter_dict:
        for key, value in filter_dict.items():
            if value:
                query = query.filter(Document.metadata_col[key].contains(value))

    results = []
    scored_results = []
    
    if search_type in ["semantic", "hybrid"]:
        # Get all documents for scoring
        all_docs = query.all()
        query_embedding = document_processor.create_embeddings(q)
        
        # Score documents using embeddings
        for doc in all_docs:
            if doc.embedding:
                similarity = cosine_similarity(
                    [query_embedding],
                    [doc.embedding]
                )[0][0]
                scored_results.append((doc, float(similarity)))
    
    if search_type in ["keyword", "hybrid"]:
        # Get keyword matches
        keyword_matches = query.filter(
            or_(
                Document.title.ilike(f"%{q}%"),
                Document.content.ilike(f"%{q}%"),
                Document.metadata_col['keywords'].cast(String).ilike(f"%{q}%")
            )
        ).all()
        
        # Score keyword matches
        for doc in keyword_matches:
            score = 0
            if q.lower() in doc.title.lower():
                score += 0.8
            if q.lower() in doc.content.lower():
                score += 0.5
            if doc.metadata_col.get('keywords') and \
               any(q.lower() in k.lower() for k in doc.metadata_col['keywords']):
                score += 0.3
                
            scored_results.append((doc, score))
    
    # Remove duplicates and sort by score
    seen_ids = set()
    unique_results = []
    for doc, score in sorted(scored_results, key=lambda x: x[1], reverse=True):
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            unique_results.append({
                "id": doc.id,
                "title": doc.title,
                "excerpt": doc.content[:200] + "...",
                "metadata_col": doc.metadata_col,
                "similarity_score": score,
                "file_type": doc.file_type,
                "created_at": doc.created_at
            })
    
    return unique_results[:10]  # Return top 10 results

@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    storage = S3StorageProvider()
    try:
        file_url = await storage.get_file_url(document.file_path)
        return {
            "metadata": document,
            "url": file_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error accessing file")

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from S3
    storage = S3StorageProvider()
    try:
        success = await storage.delete_file(document.file_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete file from storage")
    except Exception as e:
        logger.error(f"Error deleting file from S3: {e}")
        raise HTTPException(status_code=500, detail="Error deleting file from storage")
        
    # Delete from database
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}

@router.get("/graph")
async def get_document_graph(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        documents = db.query(Document).filter(Document.user_id == current_user.id).all()
        
        nodes = []
        links = []
        seen_keywords = set()
        
        # Create graph data
        for doc in documents:
            doc_id = str(doc.id)
            nodes.append({
                "id": doc_id,
                "name": doc.title,
                "category": "document",
                "val": 1,
                "color": "#ff4444"
            })
            
            # Add keyword nodes and links
            if doc.metadata_col and "keywords" in doc.metadata_col:
                for keyword in doc.metadata_col["keywords"]:
                    keyword_id = f"keyword-{keyword}"
                    if keyword_id not in seen_keywords:
                        nodes.append({
                            "id": keyword_id,
                            "name": keyword,
                            "category": "keyword",
                            "val": 0.5,
                            "color": "#4444ff"
                        })
                        seen_keywords.add(keyword_id)
                    
                    links.append({
                        "source": doc_id,
                        "target": keyword_id,
                        "strength": 1
                    })
        
        return {
            "nodes": nodes,
            "links": links
        }
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating knowledge graph"
        )

@router.get("/{document_id}/metadata")
async def get_document_metadata(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "metadata_col": document.metadata_col
        }

    except Exception as e:
        logger.error(f"Error retrieving document metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document metadata: {str(e)}"
        )

@router.post("/share")
def share_document(
    data: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        document = db.query(Document).filter(
            Document.id == data["documentId"],
            Document.user_id == current_user.id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if already shared with this email
        existing_share = db.query(ShareAccess).filter(
            ShareAccess.document_id == document.id,
            ShareAccess.shared_with_email == data["email"]
        ).first()

        if existing_share:
            raise HTTPException(
                status_code=400,
                detail="Document already shared with this email"
            )

        share_access = ShareAccess(
            document_id=document.id,
            shared_by_id=current_user.id,
            shared_with_email=data["email"],
            access_level=AccessLevel.VIEW
        )
        
        db.add(share_access)
        db.commit()
        db.refresh(share_access)

        # Send email notification
        email_service = EmailService()
        email_sent = email_service.send_share_notification(
            recipient_email=data["email"],
            document_title=document.title,
            shared_by_name=current_user.name
        )
        
        return {
            "message": "Document shared successfully",
            "email_sent": email_sent
        }

    except Exception as e:
        logger.error(f"Error sharing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error sharing document"
        )

@router.get("/shared-with-me")
async def get_shared_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    shared_docs = db.query(Document).join(ShareAccess).filter(
        ShareAccess.shared_with_email == current_user.email
    ).all()
    
    return shared_docs

@router.post("/{document_id}/apply-template")
async def apply_template(
    document_id: int,
    template_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Get document with retry logic
        retries = 3
        retry_delay = 1
        document = None
        
        for attempt in range(retries):
            try:
                document = db.query(Document).filter(
                    Document.id == document_id,
                    Document.user_id == current_user.id
                ).first()
                break
            except OperationalError as e:
                if attempt == retries - 1:
                    raise HTTPException(
                        status_code=503,
                        detail="Database connection error, please try again later"
                    )
                time.sleep(retry_delay)
                db = next(get_db())
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        template_id = template_data.get("templateId")  # Get templateId from request data
        if not template_id:
            raise HTTPException(status_code=400, detail="Template ID is required")

        # Apply template to document content
        new_content = await document_processor.apply_template(
            document.content,
            template_id
        )
        
        # Update document with retry logic
        for attempt in range(retries):
            try:
                document.content = new_content
                document.metadata_col["formatted"] = True  # Add flag to indicate formatted content
                db.commit()
                break
            except OperationalError as e:
                if attempt == retries - 1:
                    raise HTTPException(
                        status_code=503,
                        detail="Failed to save changes, please try again later"
                    )
                time.sleep(retry_delay)
                db = next(get_db())
        
        return {
            "message": "Template applied successfully",
            "content": new_content  # Return the formatted content
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error applying template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error applying template"
        )