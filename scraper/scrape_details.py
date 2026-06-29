import requests
import time
import json
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def fetch_html(url):
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Error fetching {url}: {e}")
        return None

def parse_scheme_detail(html, name, url):
    """Extracts label-value pairs from a scheme detail page."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="d-flex justify-content-center")

    details = {"name": name, "url": url}
    if not container:
        return details

    rows = container.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) == 2:                      # only real data rows have 2 cells
            label_tag = cells[0].find("b")
            if label_tag:
                label = label_tag.get_text(strip=True).rstrip(":")
                value = cells[1].get_text(separator=" ", strip=True)
                if value:                          # skip empty fields
                    details[label] = value
    return details

def main():
    # Load the 54 schemes we already scraped in Part B
    with open("scraper/scheme_list.json", "r", encoding="utf-8") as f:
        scheme_list = json.load(f)

    all_scheme_data = []

    for i, scheme in enumerate(scheme_list, start=1):
        print(f"[{i}/{len(scheme_list)}] Fetching: {scheme['name']}")
        html = fetch_html(scheme["url"])

        if html:
            details = parse_scheme_detail(html, scheme["name"], scheme["url"])
            all_scheme_data.append(details)

        time.sleep(1.5)   # be polite — small delay between each request

    # Save everything to one file
    with open("scraper/schemes_data.json", "w", encoding="utf-8") as f:
        json.dump(all_scheme_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved {len(all_scheme_data)} scheme records to scraper/schemes_data.json")

if __name__ == "__main__":
    main()