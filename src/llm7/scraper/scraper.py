import time
from typing import Dict, List
from pathlib import Path

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class Scraper:
    """A web scraper with rate limiting, retry mechanism, and batch processing capabilities."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the scraper with configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        The config file should have a 'scraping' section with the following fields:
            - max_retries: Number of retry attempts for failed requests
            - timeout: Request timeout in seconds
            - batch_size: Number of URLs to process in each batch
            - rate_limit: Maximum number of requests per second
            - headers: Optional dictionary of HTTP headers
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["scraping"]

        # Setup session with retries
        self.session = requests.Session()

        # Note: We're not using the retry adapter anymore since we're handling retries manually
        # This allows for more fine-grained control and better test verification
        self.session.mount("http://", HTTPAdapter())
        self.session.mount("https://", HTTPAdapter())

        # Initialize rate limiting
        self.last_request_time = 0

        # Set default headers if not provided
        if "headers" not in self.config:
            self.config["headers"] = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

    def _respect_rate_limit(self) -> None:
        """
        Implement rate limiting by sleeping if necessary.

        This method ensures that requests are not made more frequently
        than the configured rate limit allows.
        """
        if self.config.get("rate_limit"):
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < 1.0 / self.config["rate_limit"]:
                time.sleep(1.0 / self.config["rate_limit"] - time_since_last)
            self.last_request_time = time.time()

    def scrape_url(self, url: str) -> Dict:
        """
        Scrape content from a single URL with error handling and retries.

        Args:
            url: The URL to scrape

        Returns:
            Dictionary containing:
                - url: The scraped URL
                - content: The scraped content (None if failed)
                - status: 'success' or error message
                - timestamp: Unix timestamp of the request
        """
        max_retries = self.config["max_retries"]
        current_try = 0

        while current_try <= max_retries:
            self._respect_rate_limit()

            try:
                response = self.session.get(
                    url,
                    timeout=self.config["timeout"],
                    headers=self.config.get("headers", {}),
                )
                response.raise_for_status()

                return {
                    "url": url,
                    "content": response.text,
                    "status": "success",
                    "timestamp": time.time(),
                }

            except requests.exceptions.RequestException as e:
                current_try += 1
                if current_try <= max_retries:
                    # Only sleep if we're going to retry
                    time.sleep(0.5 * (2 ** (current_try - 1)))  # exponential backoff
                    continue

                # If we've exhausted all retries, return error
                return {
                    "url": url,
                    "content": None,
                    "status": f"error: {str(e)}",
                    "timestamp": time.time(),
                }

    def batch_scrape(self, urls: List[str]) -> List[Dict]:
        """
        Scrape content from multiple URLs in batches.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of dictionaries containing results for each URL
        """
        results = []
        for i in range(0, len(urls), self.config["batch_size"]):
            batch = urls[i : i + self.config["batch_size"]]
            batch_results = [self.scrape_url(url) for url in batch]
            results.extend(batch_results)
        return results

    def update_headers(self, headers: Dict) -> None:
        """
        Update the session headers with new values.

        Args:
            headers: Dictionary of headers to update
        """
        self.config["headers"].update(headers)

    def set_rate_limit(self, requests_per_second: float) -> None:
        """
        Update the rate limit.

        Args:
            requests_per_second: Maximum number of requests per second
        """
        self.config["rate_limit"] = requests_per_second