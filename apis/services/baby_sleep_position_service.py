from services.mongodb_helper import db

class BabySleepPositionService:
    def __init__(self):
        self.collection_name = "baby_sleep_positions"
        self.collection = db[self.collection_name]
        
        # Create time series collection if it does not exist
        # self._create_time_series_collection()
    
    def _create_time_series_collection(self):
        try:
            db.create_collection(
                self.collection_name,
                timeseries={
                    "timeField": "timestamp",
                    "metaField": "userId",
                    "granularity": "seconds"
                }
            )
            print("Time series collection created successfully.")
        except Exception as e:
            print("Collection creation failed (might already exist):", e)
    
    def insert_sleep_position(self, data):
        try:
            self.collection.insert_one(data)
            print("Documents inserted successfully.")
        except Exception as e:
            print("Document insertion failed:", e)
    
    def get_all_sleep_positions(self, userId: str):
        # Retrieve all documents in the collection
        try:
            documents = list(self.collection.find(
                {"userId": userId},
            ))
            # Convert ObjectId to string for each document
            for doc in documents:
                doc['_id'] = str(doc['_id'])
            return documents
        except Exception as e:
            print("Failed to retrieve documents:", e)
            return []