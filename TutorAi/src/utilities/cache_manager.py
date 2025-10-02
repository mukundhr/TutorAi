"""
Cache manager for storing and retrieving generated questions
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('..')
import config

class CacheManager:
    def __init__(self, cache_dir: Path = config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, text: str, params: dict) -> str:
        """Generate a unique cache key based on text and parameters"""
        content = f"{text}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, params: dict) -> dict:
        """Retrieve cached questions if available"""
        cache_key = self._generate_cache_key(text, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    return cached_data
            except Exception as e:
                print(f"Error reading cache: {e}")
                return None
        
        return None
    
    def set(self, text: str, params: dict, questions: list):
        """Store generated questions in cache"""
        cache_key = self._generate_cache_key(text, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'questions': questions
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Error writing cache: {e}")
    
    def clear(self):
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Error deleting cache file: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'num_cached_items': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }
