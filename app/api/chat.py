from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.api.auth import get_current_user
from app.models.user import User
from app.models.chat import Conversation, ChatMessage
from app.services.rag_service import RAGService
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()
rag_service = RAGService()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    selected_document_ids: Optional[List[int]] = None

class ConversationCreate(BaseModel):
    title: str

class ChatResponse(BaseModel):
    message: str
    context_documents: List[Dict[str, Any]]
    conversation_id: int
    sources_used: int

@router.post("/conversations", response_model=dict)
async def create_conversation(
    conversation_data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new conversation"""
    try:
        conversation = await rag_service.create_conversation(
            title=conversation_data.title,
            user_id=current_user.id,
            db=db
        )
        return {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating conversation")

@router.get("/conversations")
async def get_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all conversations for the current user"""
    try:
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id
        ).order_by(Conversation.updated_at.desc()).all()
        
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                "message_count": len(conv.messages)
            }
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving conversations")

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all messages in a conversation"""
    try:
        # Verify conversation ownership
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        ).order_by(ChatMessage.created_at).all()
        
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "context_documents": msg.context_documents,
                "created_at": msg.created_at.isoformat() if msg.created_at else None
            }
            for msg in messages
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving messages")

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Chat with documents using RAG
    """
    try:
        conversation_id = chat_request.conversation_id
        
        # Create new conversation if not provided
        if not conversation_id:
            # Generate title from first message
            title = chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message
            conversation = await rag_service.create_conversation(
                title=title,
                user_id=current_user.id,
                db=db
            )
            conversation_id = conversation.id
        else:
            # Verify conversation ownership
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            ).first()
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Save user message
        await rag_service.add_message_to_conversation(
            conversation_id=conversation_id,
            role="user",
            content=chat_request.message,
            db=db
        )
        
        # Retrieve relevant documents
        relevant_docs = await rag_service.retrieve_relevant_documents(
            query=chat_request.message,
            user_id=current_user.id,
            db=db,
            top_k=5,
            selected_document_ids=chat_request.selected_document_ids
        )
        
        # Get conversation history for context
        conversation_history = await rag_service.get_conversation_history(
            conversation_id=conversation_id,
            db=db
        )
        
        # Generate RAG response
        rag_response = await rag_service.generate_rag_response(
            query=chat_request.message,
            context_documents=relevant_docs,
            conversation_history=conversation_history[:-1]  # Exclude the message we just added
        )
        
        # Save assistant response
        await rag_service.add_message_to_conversation(
            conversation_id=conversation_id,
            role="assistant",
            content=rag_response["response"],
            context_documents=relevant_docs,
            db=db
        )
        
        return ChatResponse(
            message=rag_response["response"],
            context_documents=relevant_docs,
            conversation_id=conversation_id,
            sources_used=len(relevant_docs)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        db.delete(conversation)
        db.commit()
        
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")

# WebSocket endpoint for real-time chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}  # conversation_id -> websocket

    async def connect(self, websocket: WebSocket, conversation_id: int):
        await websocket.accept()
        self.active_connections[conversation_id] = websocket

    def disconnect(self, conversation_id: int):
        if conversation_id in self.active_connections:
            del self.active_connections[conversation_id]

    async def send_message(self, message: dict, conversation_id: int):
        websocket = self.active_connections.get(conversation_id)
        if websocket:
            await websocket.send_text(json.dumps(message))

    async def send_streaming_chunk(self, chunk: str, conversation_id: int, is_final: bool = False):
        """Send streaming response chunk"""
        message = {
            "type": "stream",
            "content": chunk,
            "is_final": is_final,
            "timestamp": "now"
        }
        await self.send_message(message, conversation_id)

manager = ConnectionManager()

@router.websocket("/ws/{conversation_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    conversation_id: int
):
    """WebSocket endpoint for real-time chat with streaming responses"""
    await manager.connect(websocket, conversation_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get('message', '')
            user_token = message_data.get('token', '')
            
            if not user_message.strip():
                continue
            
            # Verify user authentication (simplified)
            # In production, you'd verify the JWT token properly
            if not user_token:
                await manager.send_message({
                    "type": "error",
                    "content": "Authentication required",
                    "timestamp": "now"
                }, conversation_id)
                continue
            
            try:
                # Send typing indicator
                await manager.send_message({
                    "type": "typing",
                    "content": "AI is thinking...",
                    "timestamp": "now"
                }, conversation_id)
                
                # Process the chat message with streaming
                await process_streaming_chat(user_message, conversation_id, user_token)
                
            except Exception as e:
                logger.error(f"Error in WebSocket chat: {str(e)}")
                await manager.send_message({
                    "type": "error",
                    "content": "Sorry, something went wrong. Please try again.",
                    "timestamp": "now"
                }, conversation_id)
            
    except WebSocketDisconnect:
        manager.disconnect(conversation_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(conversation_id)

async def process_streaming_chat(message: str, conversation_id: int, user_token: str):
    """Process chat message with streaming response"""
    try:
        # This is a simplified streaming simulation
        # In a real implementation, you'd integrate with streaming LLM APIs
        
        # Simulate AI processing and streaming response
        response_parts = [
            "Based on your documents, ",
            "I found some relevant information. ",
            "Let me analyze the content... ",
            "Here's what I discovered: ",
            "The documents suggest that this topic is quite complex and involves multiple factors. ",
            "Would you like me to elaborate on any specific aspect?"
        ]
        
        full_response = ""
        for i, part in enumerate(response_parts):
            # Send chunk
            await manager.send_streaming_chunk(part, conversation_id, is_final=False)
            full_response += part
            
            # Simulate processing delay
            await asyncio.sleep(0.5)
        
        # Send final message
        await manager.send_message({
            "type": "message_complete",
            "content": full_response.strip(),
            "sources": [],  # Would include actual document sources
            "timestamp": "now"
        }, conversation_id)
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {str(e)}")
        await manager.send_message({
            "type": "error",
            "content": "Error processing your message",
            "timestamp": "now"
        }, conversation_id)
