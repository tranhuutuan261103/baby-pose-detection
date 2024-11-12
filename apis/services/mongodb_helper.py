
import os
from pymongo import MongoClient

DATABASE_URL = os.getenv('DATABASE_URL')

# Connect to MongoDB
client = MongoClient(DATABASE_URL)
db = client['baby-pose-detection']