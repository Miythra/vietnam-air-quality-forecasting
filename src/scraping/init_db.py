import os
import psycopg
from dotenv import load_dotenv

# Load .env file for local development (contains POSTGRES_URL)
# In Vercel, this will be set automatically.
load_dotenv() 

def create_table():
    """
    Connects to the Vercel Postgres database and creates the 'aqi_data' table
    if it doesn't already exist.
    """
    db_url = os.environ.get('POSTGRES_URL')
    if not db_url:
        print("Error: POSTGRES_URL environment variable not found.")
        print("Please create a Vercel Postgres database, add the URL to a .env file,")
        print("and then run this script again.")
        return

    try:
        # Connect to the database
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # SQL command to create the table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS aqi_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        location VARCHAR(255) NOT NULL,
                        aqi INT,
                        pm25 FLOAT,
                        pm10 FLOAT,
                        co FLOAT,
                        so2 FLOAT,
                        no2 FLOAT,
                        o3 FLOAT
                    );
                """)
                conn.commit() # Commit the transaction
                print("Table 'aqi_data' checked/created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_table()