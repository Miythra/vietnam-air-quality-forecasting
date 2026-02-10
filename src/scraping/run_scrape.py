import os
import psycopg
import requests
import sys
import time
import pytz
from datetime import datetime

# Token API trouvé ensemble
API_URL = "https://apiserver.aqi.in/aqi/v2/getLocationDetailsBySlug"
AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySUQiOjEsImlhdCI6MTc3MDcwNzMxNSwiZXhwIjoxNzcxMzEyMTE1fQ.kQoOvSuZwuTRk7QqDExOoetSkDcvKvr2APdWR8QzEfQ"

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'authorization': AUTH_TOKEN
}

LOCATIONS_SLUGS = {
    "Hanoi": "vietnam/hanoi",
    "Ho Chi Minh": "vietnam/ho-chi-minh",
    "Da Nang": "vietnam/da-nang"
}

def scrape_and_save():
    print("🚀 Démarrage Scraper Stable")
    db_url = os.environ.get('POSTGRES_URL')
    if not db_url: return

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    data_list = []
    for city, slug in LOCATIONS_SLUGS.items():
        try:
            params = {'slug': slug, 'type': '2', 'source': 'web'}
            resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                if isinstance(data, list) and data: data = data[0]
                
                aqi = data.get('iaqi', {}).get('aqi') or data.get('aqi')
                if aqi:
                    print(f"✅ {city}: {aqi}")
                    data_list.append((datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')), city, int(aqi)))
        except: pass
        time.sleep(1)

    if data_list:
        try:
            with psycopg.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.executemany("INSERT INTO aqi_data (timestamp, location, aqi) VALUES (%s, %s, %s)", data_list)
                    conn.commit()
            print("💾 Sauvegarde OK")
        except Exception as e: print(e)

if __name__ == "__main__":
    scrape_and_save()
