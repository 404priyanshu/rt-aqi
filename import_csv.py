"""
Script to import CSV data into the AQI database for training.
"""
import pandas as pd
import sys
from backend.database. connection import SessionLocal, init_db
from backend.models. aqi_model import AQIReading
from backend.utils.helpers import calculate_aqi_category
from datetime import datetime

def import_csv_to_db(csv_file_path):
    """
    Import CSV data into the database.

    Expected CSV columns:
    - pm25, pm10, co, no2, so2, o3 (required)
    - aqi (optional - will be calculated if missing)
    - aqi_category (optional - will be calculated if missing)
    - city, location, country (optional)
    - timestamp (optional - will use current time if missing)
    """
    # Initialize database
    init_db()

    # Read CSV
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    print(f"Found {len(df)} rows in CSV")
    print(f"Columns: {df.columns.tolist()}")

    # Create database session
    db = SessionLocal()

    try:
        imported_count = 0

        for idx, row in df.iterrows():
            # Calculate AQI if not provided
            if 'aqi' not in df.columns or pd.isna(row. get('aqi')):
                # Simplified AQI calculation from PM2.5
                pm25 = row.get('pm25', 0)
                aqi = min(500, max(0, int(pm25 * 2)))
            else:
                aqi = int(row['aqi'])

            # Calculate category if not provided
            if 'aqi_category' not in df. columns or pd.isna(row.get('aqi_category')):
                aqi_category = calculate_aqi_category(aqi)
            else:
                aqi_category = row['aqi_category']

            # Get timestamp
            if 'timestamp' in df.columns and not pd.isna(row.get('timestamp')):
                timestamp = pd.to_datetime(row['timestamp'])
            else:
                timestamp = datetime.now()

            # Create reading
            reading = AQIReading(
                timestamp=timestamp,
                location=row.get('location', 'Unknown'),
                city=row.get('city', row.get('location', 'Unknown')),
                country=row. get('country', 'Unknown'),
                latitude=row.get('latitude'),
                longitude=row.get('longitude'),
                aqi=aqi,
                aqi_category=aqi_category,
                pm25=float(row.get('pm25', 0)),
                pm10=float(row.get('pm10', 0)),
                co=float(row.get('co', 0)),
                no2=float(row.get('no2', 0)),
                so2=float(row.get('so2', 0)),
                o3=float(row.get('o3', 0)),
                source='csv_import'
            )

            db.add(reading)
            imported_count += 1

            # Commit in batches
            if imported_count % 100 == 0:
                db.commit()
                print(f"Imported {imported_count} rows...")

        # Final commit
        db.commit()
        print(f"\n✅ Successfully imported {imported_count} rows!")

    except Exception as e:
        print(f"\n❌ Error importing data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_csv.py <path_to_csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    import_csv_to_db(csv_path)
