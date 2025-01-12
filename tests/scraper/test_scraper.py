import pytest
import requests
from unittest.mock import Mock
import time
import yaml
from pathlib import Path

from llm7.scraper.scraper import Scraper

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_content = {
        "scraping": {
            "max_retries": 3,
            "timeout": 10,
            "batch_size": 2,
            "rate_limit": 2,
            "headers": {
                "User-Agent": "Test Agent"
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    return str(config_path)

@pytest.fixture
def scraper(config_file):
    """Create a Scraper instance with the test config."""
    return Scraper(config_file)

def test_initialization(config_file):
    """Test scraper initialization with config file."""
    scraper = Scraper(config_file)
    assert scraper.config["max_retries"] == 3
    assert scraper.config["timeout"] == 10
    assert scraper.config["batch_size"] == 2
    assert scraper.config["headers"]["User-Agent"] == "Test Agent"

def test_initialization_missing_config():
    """Test scraper initialization with missing config file."""
    with pytest.raises(FileNotFoundError):
        Scraper("nonexistent_config.yaml")

def test_single_url_scraping(scraper, mocker):
    """Test successful scraping of a single URL."""
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.raise_for_status.return_value = None

    mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    result = scraper.scrape_url("https://test.com")

    assert result["status"] == "success"
    assert result["content"] == "Test content"
    assert result["url"] == "https://test.com"
    assert isinstance(result["timestamp"], float)

def test_batch_scraping(scraper, mocker):
    """Test batch scraping functionality."""
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.raise_for_status.return_value = None

    mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    urls = ["https://test1.com", "https://test2.com", "https://test3.com"]
    results = scraper.batch_scrape(urls)

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result["status"] == "success"
        assert result["content"] == "Test content"
        assert result["url"] == urls[i]
        assert isinstance(result["timestamp"], float)

def test_rate_limiting(scraper, mocker):
    """Test rate limiting functionality."""
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.raise_for_status.return_value = None

    mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    start_time = time.time()
    scraper.scrape_url("https://test1.com")
    scraper.scrape_url("https://test2.com")
    end_time = time.time()

    # With rate_limit of 2, two requests should take at least 0.5 seconds
    assert end_time - start_time >= 0.5

def test_error_handling(scraper, mocker):
    """Test error handling with mocked request failure."""
    mock_get = mocker.patch.object(
        scraper.session,
        'get',
        side_effect=requests.exceptions.RequestException("Test error")
    )

    result = scraper.scrape_url("https://test.com")

    assert result["status"].startswith("error:")
    assert result["content"] is None
    assert result["url"] == "https://test.com"
    assert isinstance(result["timestamp"], float)

def test_retry_mechanism(scraper, mocker):
    """Test retry mechanism with temporary failures."""
    mock_success = Mock()
    mock_success.text = "success content"
    mock_success.raise_for_status.return_value = None

    mock_get = mocker.patch.object(
        scraper.session,
        'get',
        side_effect=[
            requests.exceptions.HTTPError("503 Server Error", response=Mock(status_code=503)),
            mock_success
        ]
    )

    result = scraper.scrape_url("https://test.com")

    assert result["status"] == "success"
    assert result["content"] == "success content"
    assert result["url"] == "https://test.com"
    assert isinstance(result["timestamp"], float)

def test_custom_headers(scraper, mocker):
    """Test custom headers are properly set."""
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.raise_for_status.return_value = None

    mock_get = mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    scraper.scrape_url("https://test.com")

    actual_headers = mock_get.call_args[1]['headers']
    assert actual_headers['User-Agent'] == "Test Agent"

def test_update_headers(scraper):
    """Test updating headers at runtime."""
    new_headers = {"Accept": "application/json"}
    scraper.update_headers(new_headers)
    assert scraper.config["headers"]["Accept"] == "application/json"
    # Ensure original headers are preserved
    assert scraper.config["headers"]["User-Agent"] == "Test Agent"

def test_set_rate_limit(scraper):
    """Test updating rate limit at runtime."""
    new_rate_limit = 5.0
    scraper.set_rate_limit(new_rate_limit)
    assert scraper.config["rate_limit"] == new_rate_limit

def test_batch_size_respect(scraper, mocker):
    """Test that batch processing respects the batch size."""
    mock_response = Mock()
    mock_response.text = "Test content"
    mock_response.raise_for_status.return_value = None

    mock_get = mocker.patch.object(scraper.session, 'get', return_value=mock_response)

    urls = ["url1", "url2", "url3", "url4", "url5"]
    results = scraper.batch_scrape(urls)

    # With batch_size=2, should process in 3 batches
    assert mock_get.call_count == 5
    assert len(results) == 5

def test_timeout_configuration(scraper, mocker):
    """Test that timeout configuration is respected."""
    mock_get = mocker.patch.object(
        scraper.session,
        'get',
        side_effect=requests.exceptions.Timeout("Timeout")
    )

    result = scraper.scrape_url("https://test.com")

    assert result["status"].startswith("error:")
    assert "Timeout" in result["status"]
    assert mock_get.call_args[1]["timeout"] == scraper.config["timeout"]