from typing import List, Dict, Optional
import json
from datetime import datetime
import redis
import os
from dotenv import load_dotenv

class BlueprintMemory:
    def __init__(self):
        load_dotenv()
        # Initialize Redis client with SSL
        self.redis = redis.Redis(
            host='enabling-rabbit-59482.upstash.io',
            port=6379,
            password=os.getenv('UPSTASH_REDIS_PASSWORD'),
            ssl=True,
            decode_responses=True  # Automatically decode responses to strings
        )
        
    def save_conversation(self, blueprint_id: str, messages: List[Dict]):
        """Save a conversation about a specific blueprint."""
        conversation_key = f"blueprint:{blueprint_id}:conversation"
        timestamp = datetime.now().isoformat()
        
        conversation_data = {
            'timestamp': timestamp,
            'messages': messages
        }
        
        # Store conversation
        self.redis.lpush(conversation_key, json.dumps(conversation_data))
        # Keep only last 50 conversations
        self.redis.ltrim(conversation_key, 0, 49)
        
    def get_conversations(self, blueprint_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversations about a blueprint."""
        conversation_key = f"blueprint:{blueprint_id}:conversation"
        conversations = self.redis.lrange(conversation_key, 0, limit - 1)
        return [json.loads(conv) for conv in conversations]
    
    def save_blueprint_analysis(self, blueprint_id: str, analysis_data: Dict):
        """Save analysis results for a blueprint."""
        analysis_key = f"blueprint:{blueprint_id}:analysis"
        timestamp = datetime.now().isoformat()
        
        analysis_data['timestamp'] = timestamp
        self.redis.set(analysis_key, json.dumps(analysis_data))
    
    def get_blueprint_analysis(self, blueprint_id: str) -> Optional[Dict]:
        """Get stored analysis for a blueprint."""
        analysis_key = f"blueprint:{blueprint_id}:analysis"
        analysis = self.redis.get(analysis_key)
        return json.loads(analysis) if analysis else None
    
    def save_measurement_history(self, blueprint_id: str, room_id: str, measurement: Dict):
        """Track measurement history for a specific room."""
        measurement_key = f"blueprint:{blueprint_id}:room:{room_id}:measurements"
        timestamp = datetime.now().isoformat()
        
        measurement_data = {
            'timestamp': timestamp,
            **measurement
        }
        
        self.redis.lpush(measurement_key, json.dumps(measurement_data))
        # Keep last 100 measurements
        self.redis.ltrim(measurement_key, 0, 99)
    
    def get_measurement_history(self, blueprint_id: str, room_id: str) -> List[Dict]:
        """Get measurement history for a room."""
        measurement_key = f"blueprint:{blueprint_id}:room:{room_id}:measurements"
        measurements = self.redis.lrange(measurement_key, 0, -1)
        return [json.loads(m) for m in measurements]
    
    def save_user_preferences(self, user_id: str, preferences: Dict):
        """Save user preferences for blueprint analysis."""
        pref_key = f"user:{user_id}:preferences"
        self.redis.set(pref_key, json.dumps(preferences))
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """Get user preferences."""
        pref_key = f"user:{user_id}:preferences"
        prefs = self.redis.get(pref_key)
        return json.loads(prefs) if prefs else None 