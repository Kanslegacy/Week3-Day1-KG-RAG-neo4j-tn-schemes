import requests
import time

URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def fetch_raw_html():
    try:
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()  # raises error if status code is not 200 (success)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None

from bs4 import BeautifulSoup
import json

BASE_URL = "https://www.tn.gov.in/"

def parse_scheme_list(html):
    soup = BeautifulSoup(html, "html.parser")

    # Find the <ul> that contains all the scheme <li> items
    content_list = soup.find("ul", id="content")

    schemes = []
    if content_list:
        # Find every <a> tag inside this <ul> — each one is one scheme link
        links = content_list.find_all("a")
        for link in links:
            name = link.get_text(strip=True)          # the visible scheme name
            relative_url = link.get("href")            # e.g. scheme_details.php?id=XXXX
            full_url = BASE_URL + relative_url         # make it a complete, usable URL

            schemes.append({
                "name": name,
                "url": full_url
            })
    return schemes

if __name__ == "__main__":
    html = fetch_raw_html()
    if html:
        with open("scraper/raw_page.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("✅ Page saved successfully to scraper/raw_page.html")

        # Parse the schemes from this HTML
        schemes = parse_scheme_list(html)
        print(f"✅ Found {len(schemes)} schemes")

        # Save the structured list to a JSON file for later use
        with open("scraper/scheme_list.json", "w", encoding="utf-8") as f:
            json.dump(schemes, f, indent=2, ensure_ascii=False)
        print("✅ Saved scheme list to scraper/scheme_list.json")
    else:
        print("❌ Could not fetch the page.")