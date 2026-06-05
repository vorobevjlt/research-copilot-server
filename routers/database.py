from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import supabase
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

supabase_url = os.getenv("SUPABASE_API_URL")
supabase_role_secret = os.getenv("SUPABASE_ROLE_SECRET")

if not supabase_url or not supabase_role_secret:
    raise ValueError("SUPABASE_API_URL and SUPABASE_ROLE_SECRET must be set")

supabase: Client = create_client(supabase_url, supabase_role_secret)