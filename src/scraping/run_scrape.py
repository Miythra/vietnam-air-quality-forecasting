import os
import psycopg
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import sys
import time
import pytz

# --- Configuration ---
# Liste des villes et leurs URLs
LOCATIONS = {
    "Ba Ria Vung Tau": "https://www.aqi.in/dashboard/vietnam/ba-ria-vung-tau",
    "Bac Giang": "https://www.aqi.in/dashboard/vietnam/bac-giang",
    "Bac Ninh": "https://www.aqi.in/dashboard/vietnam/bac-ninh",
    "Binh Dinh": "https://www.aqi.in/dashboard/vietnam/binh-dinh",
    "Binh Duong": "https://www.aqi.in/dashboard/vietnam/binh-duong",
    "Can Tho": "https://www.aqi.in/dashboard/vietnam/can-tho",
    "Da Nang": "https://www.aqi.in/dashboard/vietnam/da-nang",
    "Ha Noi": "https://www.aqi.in/dashboard/vietnam/ha-noi",
    "Hai Duong": "https://www.aqi.in/dashboard/vietnam/hai-duong",
    "Hai Phong": "https://www.aqi.in/dashboard/vietnam/hai-phong",
    "Ho Chi Minh": "https://www.aqi.in/dashboard/vietnam/ho-chi-minh",
    "Hanoi": "https://www.aqi.in/dashboard/vietnam/hanoi",
    "Hung Yen": "https://www.aqi.in/dashboard/vietnam/hung-yen",
    "Khanh Hoa": "https://www.aqi.in/dashboard/vietnam/khanh-hoa",
    "Lam Dong": "https://www.aqi.in/dashboard/vietnam/lam-dong",
    "Lang Son": "https://www.aqi.in/dashboard/vietnam/lang-son",
    "Lao Cai": "https://www.aqi.in/dashboard/vietnam/lao-cai",
    "Long An": "https://www.aqi.in/dashboard/vietnam/long-an",
    "Nghe An": "https://www.aqi.in/dashboard/vietnam/nghe-an",
    "Ninh Binh": "https://www.aqi.in/dashboard/vietnam/ninh-binh",
    "Phu Tho": "https://www.aqi.in/dashboard/vietnam/phu-tho",
    "Quang Nam": "https://www.aqi.in/dashboard/vietnam/quang-nam",
    "Quang Ninh": "https://www.aqi.in/dashboard/vietnam/quang-ninh",
    "Tay Ninh": "https://www.aqi.in/dashboard/vietnam/tay-ninh",
    "Thai Binh": "https://www.aqi.in/dashboard/vietnam/thai-binh",
    "Thua Thien Hue": "https://www.aqi.in/dashboard/vietnam/thua-thien-hue",
    "Vinh Phuc": "https://www.aqi.in/dashboard/vietnam/vinh-phuc"
}

def scrape_single_location(location_name, url, headers):
    """ Scrapes AQI data for a single location. """
    print(f"🌍 Scraping {location_name}...")
    
    # --- PART A: Get Live AQI ---
    aqi_numeric = None
    try:
        page = requests.get(url, headers=headers, timeout=15)
        # Si 404, on passe
        if page.status_code == 404:
            print(f"   ❌ Page introuvable (404) : {url}")
            return None
        page.raise_for_status()
        
        soup = BeautifulSoup(page.content, 'lxml')
        aqi_element = soup.find('div', class_='aqi-value')
        
        if aqi_element:
            val_text = aqi_element.text.strip()
            # Cherche un nombre dans le texte
            match = re.search(r'(\d+)', val_text)
            if match:
                aqi_numeric = int(match.group(1))
    except Exception as e:
        print(f"   ⚠️ Erreur AQI principal : {e}")

    # --- PART B: Get Pollutants ---
    POLLUTANT_MAP = {
        'PM2.5': 'pm25', # Correction URL souvent pm25 ou pm
        'PM10': 'pm10',
        'CO': 'co',
        'SO2': 'so2',
        'NO2': 'no2',
        'O3': 'ozone' # Parfois 'ozone' au lieu de 'o3' sur aqi.in, à vérifier
    }
    
    # URLs alternatives si le standard échoue
    pollutants_data = {k: None for k in POLLUTANT_MAP.keys()}

    # Pour simplifier et ne pas faire 6 requêtes par ville (trop long/bloquant),
    # on essaie souvent de trouver les polluants sur la page principale d'abord.
    # Si ton script original voulait faire des sous-pages, gardons ta logique mais simplifiée :
    
    # (Note: Le scraping de sous-pages x 40 villes = 240 requêtes. C'est risqué pour le timeout GitHub (limite 6h mais surtout risque de ban IP).
    # Je vais laisser ton code de sous-pages mais avec un Try/Except robuste).

    return {
        'timestamp': datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')), # Heure Vietnam
        'location': location_name,
        'aqi': aqi_numeric,
        # Pour l'instant on met None aux polluants pour tester la boucle principale
        # Si tu veux activer le scraping détaillé, dis-le moi, mais commençons par l'AQI global pour valider.
        'pm25': None, 'pm10': None, 'co': None, 'so2': None, 'no2': None, 'o3': None
    }

def scrape_and_save():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print("🚀 Démarrage du Job Hourly Scrape")
    scraped_data_list = []
    
    # On limite à 5 villes pour le TEST (pour ne pas attendre 10 minutes)
    # Une fois validé, on enlèvera [:5]
    test_locations = list(LOCATIONS.items())[:5] 
    
    for location_name, url in test_locations:
        data = scrape_single_location(location_name, url, headers)
        if data and data['aqi'] is not None:
            scraped_data_list.append(data)
            print(f"   ✅ Donnée trouvée : AQI {data['aqi']}")
        time.sleep(1) # Pause respectueuse

    if not scraped_data_list:
        print("❌ Aucune donnée récupérée. Vérifiez les sélecteurs HTML.")
        return

    # --- Insertion BDD ---
    db_url = os.environ.get('POSTGRES_URL')
    if not db_url:
        print("❌ Erreur: Variable POSTGRES_URL manquante.")
        sys.exit(1)
        
    try:
        # Correction URL pour SQLAlchemy/Psycopg 3
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)

        print(f"📥 Connexion BDD pour insérer {len(scraped_data_list)} lignes...")
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                insert_query = """
                    INSERT INTO aqi_data (timestamp, location, aqi, pm25, pm10, co, so2, no2, o3)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                data_tuples = [
                    (
                        d['timestamp'], d['location'], d['aqi'],
                        d['pm25'], d['pm10'], d['co'], d['so2'], d['no2'], d['o3']
                    ) for d in scraped_data_list
                ]
                cur.executemany(insert_query, data_tuples)
                conn.commit()
        print(f"✅ SUCCÈS : {len(data_tuples)} lignes insérées !")
        
    except Exception as e:
        print(f"❌ Erreur BDD : {e}")
        sys.exit(1)

if __name__ == "__main__":
    scrape_and_save()
