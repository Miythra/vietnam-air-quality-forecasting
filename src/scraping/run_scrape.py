import os
import psycopg
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import sys
import time # Import time for adding delays

# --- List of locations to scrape ---
# We map the display name (e.g., "Hanoi") to its URL slug (e.g., "hanoi")
LOCATIONS = {
    "Ba Ria Vung Tau": "https://www.aqi.in/dashboard/vietnam/ba-ria-vung-tau",
    "Bac Giang": "https://www.aqi.in/dashboard/vietnam/bac-giang",
    "Bac Ninh": "https://www.aqi.in/dashboard/vietnam/bac-ninh",
    "Binh Dinh": "https://www.aqi.in/dashboard/vietnam/binh-dinh",
    "Binh Duong": "https://www.aqi.in/dashboard/vietnam/binh-duong",
    "Can Tho": "https://www.aqi.in/dashboard/vietnam/can-tho",
    "Cao Bang": "https://www.aqi.in/dashboard/vietnam/cao-bang",
    "Da Nang": "https://www.aqi.in/dashboard/vietnam/da-nang",
    "Gia Lai": "https://www.aqi.in/dashboard/vietnam/gia-lai",
    "Ha Nam": "https://www.aqi.in/dashboard/vietnam/ha-nam",
    "Ha Nam Province": "https://www.aqi.in/dashboard/vietnam/ha-nam-province",
    "Ha Noi": "https://www.aqi.in/dashboard/vietnam/ha-noi",
    "Hai Duong": "https://www.aqi.in/dashboard/vietnam/hai-duong",
    "Hai Phong": "https://www.aqi.in/dashboard/vietnam/hai-phong",
    "Ho Chi Minh": "https://www.aqi.in/dashboard/vietnam/ho-chi-minh",
    "Hanoi": "https://www.aqi.in/dashboard/vietnam/hanoi",
    "Hung Yen": "https://www.aqi.in/dashboard/vietnam/hung-yen",
    "Hoa Binh Province": "https://www.aqi.in/dashboard/vietnam/hoa-binh-province",
    "Khanh Hoa": "https://www.aqi.in/dashboard/vietnam/khanh-hoa",
    "Lam Dong": "https://www.aqi.in/dashboard/vietnam/lam-dong",
    "Lang Son": "https://www.aqi.in/dashboard/vietnam/lang-son",
    "Lang Son Province": "https://www.aqi.in/dashboard/vietnam/lang-son-province",
    "Lao Cai": "https://www.aqi.in/dashboard/vietnam/lao-cai",
    "Long An": "https://www.aqi.in/dashboard/vietnam/long-an",
    "Nghe An": "https://www.aqi.in/dashboard/vietnam/nghe-an",
    "Ninh Binh": "https://www.aqi.in/dashboard/vietnam/ninh-binh",
    "Ninh Thuan": "https://www.aqi.in/dashboard/vietnam/ninh-thuan",
    "Phu Tho": "https://www.aqi.in/dashboard/vietnam/phu-tho",
    "Quang Binh Province": "https://www.aqi.in/dashboard/vietnam/quang-binh-province",
    "Quang Nam": "https://www.aqi.in/dashboard/vietnam/quang-nam",
    "Quang Ngai": "https://www.aqi.in/dashboard/vietnam/quang-ngai",
    "Quang Ninh": "https://www.aqi.in/dashboard/vietnam/quang-ninh",
    "Quang Ninh Province": "https://www.aqi.in/dashboard/vietnam/quang-ninh-province",
    "Quang Tri": "https://www.aqi.in/dashboard/vietnam/quang-ninh-province",
    "Son La": "https://www.aqi.in/dashboard/vietnam/son-la",
    "Tay Ninh": "https://www.aqi.in/dashboard/vietnam/tay-ninh",
    "Thai Binh": "https://www.aqi.in/dashboard/vietnam/thai-binh",
    "Thai Binh Province": "https://www.aqi.in/dashboard/vietnam/thai-binh-province",
    "Thua Thien Hue": "https://www.aqi.in/dashboard/vietnam/thua-thien-hue",
    "Tra Vinh": "https://www.aqi.in/dashboard/vietnam/tra-vinh",
    "Tuyen Quang Province": "https://www.aqi.in/dashboard/vietnam/tuyen-quang-province",
    "Vinh Long": "https://www.aqi.in/dashboard/vietnam/vinh-long",
    "Vinh Phuc": "https://www.aqi.in/dashboard/vietnam/vinh-phuc"
    

    # You can add more locations here.
    # Find the slug from the URL on aqi.in
}

def scrape_single_location(location_name, url, headers):
    """
    Scrapes AQI data for a *single* location.
    This now scrapes the main page for the overall AQI,
    and then visits sub-pages for each pollutant.
    """
    print(f"Scraping {location_name} from {url}...")
    
    # --- PART A: Get the main "Live AQI" value from the base page ---
    aqi_numeric = None
    try:
        page = requests.get(url, headers=headers, timeout=15)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'lxml')

        aqi_element = soup.find('div', class_='aqi-value')
        aqi_val_messy_string = aqi_element.text.strip() if aqi_element else None
        
        if aqi_val_messy_string:
            aqi_match = re.search(r'Live AQI(\d+)', aqi_val_messy_string)
            if aqi_match:
                aqi_numeric = int(aqi_match.group(1))
        
    except Exception as e:
        print(f"Error scraping main AQI for {location_name} at {url}: {str(e)}")
        # We can still continue to try and get pollutant data
    
    # --- PART B: Get specific pollutant data from sub-pages ---
    
    # Map for pollutant name -> URL suffix
    POLLUTANT_MAP = {
        'PM2.5': 'pm',
        'PM10': 'pm10',
        'CO': 'co',
        'SO2': 'so2',
        'NO2': 'no2',
        'O3': 'o3'
    }
    
    pollutants_data = {name: None for name in POLLUTANT_MAP.keys()}

    for pollutant_name, url_suffix in POLLUTANT_MAP.items():
        pollutant_url = f"{url}/{url_suffix}"
        print(f"  -> Scraping {pollutant_name} from {pollutant_url}")
        
        try:
            # Wait 1 second between pollutant requests for the *same* location
            time.sleep(1) 
            
            pollutant_page = requests.get(pollutant_url, headers=headers, timeout=10)
            pollutant_page.raise_for_status()
            pollutant_soup = BeautifulSoup(pollutant_page.content, 'lxml')
            
            # Find the 'pollutant-info' div based on your HTML snippet
            info_div = pollutant_soup.find('div', class_='pollutant-info')
            
            if info_div:
                # Find the value span (e.g., <span class="text-[6rem] font-bold">18</span>)
                value_span = info_div.find('span', class_='text-[6rem]')
                if value_span:
                    value_str = value_span.text.strip()
                    try:
                        pollutants_data[pollutant_name] = float(value_str)
                    except (ValueError, TypeError):
                        print(f"    -> Could not parse value '{value_str}' for {pollutant_name}")
                else:
                    print(f"    -> Could not find value span for {pollutant_name}")
            else:
                # This page might not exist or might not have data (e.g., no CO data for this city)
                print(f"    -> Could not find 'pollutant-info' div for {pollutant_name} (Page may be 404 or missing data)")

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                print(f"    -> Page not found for {pollutant_name} (404). Skipping.")
            else:
                print(f"    -> HTTPError scraping {pollutant_name}: {str(http_err)}")
        except Exception as e:
            print(f"    -> General error scraping {pollutant_name}: {str(e)}")
        
        # We 'pass' on errors to continue to the next pollutant

    # --- PART C: Combine and return ---
    
    # Check if we got *any* data at all
    if aqi_numeric is None and all(v is None for v in pollutants_data.values()):
        print(f"No data parsed for {location_name}.")
        return None

    # Return a dictionary with all data
    print(f"  -> Scraped data for {location_name}: AQI={aqi_numeric}, Pollutants={pollutants_data}")
    return {
        'timestamp': datetime.now(),
        'location': location_name,
        'aqi': aqi_numeric,
        'pm25': pollutants_data.get('PM2.5'),
        'pm10': pollutants_data.get('PM10'),
        'co': pollutants_data.get('CO'),
        'so2': pollutants_data.get('SO2'),
        'no2': pollutants_data.get('NO2'),
        'o3': pollutants_data.get('O3')
    }

def scrape_and_save():
    """
    Scrapes AQI data for *all* locations in the LOCATIONS list
    and saves them to Vercel Postgres.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.aqi.in/dashboard/vietnam',
        'Upgrade-Insecure-Requests': '1'
    }
    
    print("--- Starting Hourly Scrape Job ---")
    scraped_data_list = []
    
    for location_name, url_slug in LOCATIONS.items():
        data = scrape_single_location(location_name, url_slug, headers)
        if data:
            scraped_data_list.append(data)
        
        # Be polite to the server: wait 2 seconds between *different locations*
        # time.sleep(2) 

    # --- PART C: Save all collected data to Vercel Postgres ---
    if not scraped_data_list:
        print("No data was successfully scraped from any location. Exiting.")
        return # Exit function

    db_url = os.environ.get('POSTGRES_URL')
    if not db_url:
        print("Error: POSTGRES_URL env var not set.")
        sys.exit(1)
        
    try:
        print(f"Connecting to database to insert {len(scraped_data_list)} rows...")
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Prepare the data for a batch insert
                insert_query = """
                    INSERT INTO aqi_data (
                        timestamp, location, aqi, 
                        pm25, pm10, co, so2, no2, o3
                    ) 
                    VALUES (
                        %s, %s, %s, 
                        %s, %s, %s, %s, %s, %s
                    )
                """
                
                # Create a list of tuples from our list of dictionaries
                data_tuples = [
                    (
                        d['timestamp'], d['location'], d['aqi'],
                        d['pm25'], d['pm10'], d['co'],
                        d['so2'], d['no2'], d['o3']
                    ) for d in scraped_data_list
                ]
                
                # Use executemany for an efficient batch insert
                cur.executemany(insert_query, data_tuples)
                conn.commit() # Commit the transaction
                
        print(f"Successfully inserted {len(data_tuples)} rows into database.")
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        sys.exit(1)
            
    print("--- Scrape Job Finished ---")

# (Your if __name__ == "__main__": block stays here)
if __name__ == "__main__":
    scrape_and_save()