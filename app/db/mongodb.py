import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in .env")

if not MONGODB_DATABASE:
    raise ValueError("MONGODB_DATABASE is not set in .env")

if not MONGODB_COLLECTION:
    raise ValueError("MONGODB_COLLECTION is not set in .env")

client = MongoClient(MONGODB_URI)

db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]