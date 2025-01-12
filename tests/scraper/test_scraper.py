import pytest
from unittest.mock import Mock, patch
import requests
from config.settings import Settings  # Import from root config
from llm7.scraper.scraper import Scraper

# Test configuration constant
TEST_CONFIG = {
    "scraping.batch_size": 100,
    "scraping.max_retries": 3,
    "scraping.timeout": 30,
    "scraping.headers": {
        "User-Agent": "LLM7 Scraper/0.1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "en-US,en;q=0.5"
    },
    "scraping.rate_limit": 1,
    "scraping.async_enabled": True
}

@pytest.fixture
def mock_settings():
    """Create a mock settings instance."""
    settings = Mock(spec=Settings)
    settings.get.side_effect = lambda key, default=None: TEST_CONFIG.get(key, default)
    return settings

@pytest.fixture
def scraper(mock_settings):
    """Create a Scraper instance for testing."""
    with patch('config.settings.settings', mock_settings):  # Updated patch path
        return Scraper()

@pytest.fixture
def mock_response():
    """Create a base mock response."""
    response = Mock()
    response.text = "test content"
    response.raise_for_status.return_value = None
    return response

def test_successful_scrape(scraper, mock_response, mocker):
    """Test successful URL scraping."""
    mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    result = scraper.scrape_url("https://test.com")

    assert result["status"] == "success"
    assert result["content"] == "test content"
    scraper.session.get.assert_called_once_with(
        "https://test.com",
        headers=TEST_CONFIG["scraping.headers"],
        timeout=TEST_CONFIG["scraping.timeout"]
    )

def test_failed_scrape(scraper, mocker):
    """Test failed URL scraping."""
    error_response = Mock()
    error_response.raise_for_status.side_effect = requests.exceptions.RequestException("Test error")
    mocker.patch.object(scraper.session, 'get', return_value=error_response)

    result = scraper.scrape_url("https://test.com")

    assert result["status"].startswith("error")
    assert "Test error" in result.get("error", "")

def test_retry_mechanism(scraper, mock_response, mocker):
    """Test retry mechanism with temporary failures."""
    error_response = Mock()
    error_response.status_code = 503
    error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "503 Server Error",
        response=error_response
    )

    mock_get = mocker.patch.object(
        scraper.session,
        'get',
        side_effect=[error_response, mock_response]
    )

    result = scraper.scrape_url("https://test.com")

    assert result["status"] == "success"
    assert result["content"] == "test content"
    assert mock_get.call_count == 2  # Verify retry happened

def test_max_retries_exceeded(scraper, mocker):
    """Test behavior when max retries are exceeded."""
    error_response = Mock()
    error_response.status_code = 503
    error_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "503 Server Error",
        response=error_response
    )

    mocker.patch.object(
        scraper.session,
        'get',
        return_value=error_response
    )

    result = scraper.scrape_url("https://test.com")

    assert result["status"].startswith("error")
    assert "Max retries exceeded" in result.get("error", "")

@pytest.mark.timeout(5)  # Ensure test doesn't hang
def test_rate_limiting(scraper, mock_response, mocker):
    """Test rate limiting functionality."""
    mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    import time
    start_time = time.time()

    # Make 3 requests with rate limit of 1 per second
    results = [
        scraper.scrape_url("https://test.com")
        for _ in range(3)
    ]

    duration = time.time() - start_time

    assert duration >= 2.0  # Should take at least 2 seconds due to rate limiting
    assert all(r["status"] == "success" for r in results)
    assert len(results) == 3