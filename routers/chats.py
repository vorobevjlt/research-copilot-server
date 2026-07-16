from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from storage import supabase
from auth import get_current_user
from langchain_core.messages import HumanMessage, SystemMessage
from tasks import llm

router = APIRouter(
    tags=["chats"]
)  

class ChatCreate(BaseModel):
    title: str
    project_id: str


class SendMessageRequest(BaseModel):
    content: str

@router.post("/api/chats")
async def create_chat(
    chat: ChatCreate, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").insert({
            "title": chat.title, 
            "project_id": chat.project_id, 
            "clerk_id": clerk_id
        }).execute()

        return {
            "message": "Chat created successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to create chat: {str(e)}")


@router.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str, 
    clerk_id: str = Depends(get_current_user)
):

    try:
        deleted_result = supabase.table("chats").delete().eq("id", chat_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data: 
            raise HTTPException(status_code=404, detail="Chat not found or access denied")

        return {
            "message": "Chat Deleted Successfully", 
            "data": deleted_result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to delete chat: {str(e)}")

@router.get("/api/projects/{project_id}/chats")
async def get_project_chats(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at").execute()

        return {
            "success": True,
            "message": "Chats retrieved",
            "data": result.data or []
    }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while retrieving chats: {str(e)}"
        )


@router.post("/api/projects/{project_id}/chats/{chat_id}/messages")
async def create_chat_message(
    project_id: str,
    chat_id: str,
    request: SendMessageRequest,
    clerk_id: str = Depends(get_current_user)
):
    try:
        message = request.content

        message_result = supabase.table("messages").insert({
            "content": message,
            "role": "user",
            "chat_id": chat_id,
            "clerk_id": clerk_id
        }).execute()
        user_message = message_result.data[0]

        messages = [
            SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
            HumanMessage(content=message),
        ]

        response = llm.invoke(messages)
        ai_response = response.content

        ai_message_result = supabase.table('messages').insert({
            "chat_id": chat_id,
            "content": ai_response,
            "role": "assistant",
            "clerk_id": clerk_id,
            "citations": []
        }).execute()

        ai_message = ai_message_result.data[0]
        
        return {
            "message": "Message created successfully",
            "data": {
                "userMessage": user_message,
                "aiMessage": ai_message
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create message: {str(e)}")

@router.get("/api/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Get the chat and verify it belongs to the user AND has a project_id
        result = supabase.table('chats').select('*').eq('id', chat_id).eq('clerk_id', clerk_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
        
        chat = result.data[0]
        
        # Get messages for this chat
        messages_result = supabase.table('messages').select('*').eq('chat_id', chat_id).order('created_at', desc=False).execute()
        
        # Add messages to chat object
        chat['messages'] = messages_result.data or []
        
        return {
            "message": "Chat retrieved successfully",
            "data": chat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat: {str(e)}")
