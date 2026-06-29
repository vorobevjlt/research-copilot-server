from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import boto3

load_dotenv()

supabase_url = os.getenv("SUPABASE_API_URL")
supabase_role_secret = os.getenv("SUPABASE_ROLE_SECRET")

if not supabase_url or not supabase_role_secret:
    raise ValueError("SUPABASE_API_URL and SUPABASE_ROLE_SECRET must be set")

supabase: Client = create_client(supabase_url, supabase_role_secret)

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = os.getenv('S3_BUCKET_NAME')