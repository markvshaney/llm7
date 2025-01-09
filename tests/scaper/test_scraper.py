from scraper import Scraper


def test_scraper():
    scraper = Scraper()
    test_urls = ["https://example.com", "https://httpbin.org/get"]

    # Test single URL scraping
    result = scraper.scrape_url(test_urls[0])
    print("Single URL scrape result:", result["status"])

    # Test batch scraping
    results = scraper.batch_scrape(test_urls)
    print(f"Batch scrape results: {len(results)} URLs processed")


if __name__ == "__main__":
    test_scraper()
