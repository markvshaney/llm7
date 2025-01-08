import requests
from typing import List, Dict
import yaml
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class Scraper:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['scraping']
        
        # Setup session with retries
        self.session = requests.Session()
        retries = Retry(
            total=self.config['max_retries'],
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.last_request_time = 0
    
    def _respect_rate_limit(self):
        """Implement rate limiting."""
        if self.config.get('rate_limit'):
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < 1.0 / self.config['rate_limit']:
                time.sleep(1.0 / self.config['rate_limit'] - time_since_last)
            self.last_request_time = time.time()

    def scrape_url(self, url: str) -> Dict:
        """Scrape content from a single URL with improved error handling."""
        self._respect_rate_limit()
        try:
            response = self.session.get(
                url, 
                timeout=self.config['timeout'],
                headers=self.config.get('headers', {})
            )
            response.raise_for_status()
            return {
                'url': url,
                'content': response.text,
                'status': 'success',
                'timestamp': time.time()
            }
        except requests.exceptions.RequestException as e:
            return {
                'url': url,
                'content': None,
                'status': f'error: {str(e)}',
                'timestamp': time.time()
            }

    def batch_scrape(self, urls: List[str]) -> List[Dict]:
        """Scrape content from multiple URLs in batches."""
        results = []
        for i in range(0, len(urls), self.config['batch_size']):
            batch = urls[i:i + self.config['batch_size']]
            batch_results = [self.scrape_url(url) for url in batch]
            results.extend(batch_results)
        return results
