import pandas as pd
import requests
import time
import os
from datetime import datetime

class RAWGDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.rawg.io/api/games"
        self.df = pd.DataFrame()
        self.request_count = 0
        self.max_requests_per_minute = 30  # RAWGs rate limit is 40 requests/minute
    
    def fetch_games(self, pages=10, page_size=40, save_interval=5):
        """
        Fetch games data from RAWG API with pagination
        
        Args:
            pages (int): Number of pages to fetch (default 10)
            page_size (int): Number of games per page (max 40)
            save_interval (int): Save progress every N pages
        """
        all_games = []
        
        print(f"Starting data fetch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Fetching {pages} pages with {page_size} games per page...")
        
        try:
            for page in range(1, pages + 1):
                # Rate limiting
                if self.request_count >= self.max_requests_per_minute:
                    print("Approaching rate limit, sleeping for 60 seconds...")
                    time.sleep(60)
                    self.request_count = 0
                
                params = {
                    "key": self.api_key,
                    "page_size": page_size,
                    "page": page
                }
                
                print(f"Fetching page {page}/{pages}...", end=" ", flush=True)
                response = requests.get(self.base_url, params=params)
                self.request_count += 1
                
                if response.status_code == 200:
                    data = response.json()
                    games = data.get("results", [])
                    all_games.extend(games)
                    print(f"Success! Got {len(games)} games.")
                    
                    # Save progress periodically
                    if page % save_interval == 0 or page == pages:
                        self._save_progress(all_games, page)
                else:
                    print(f"Failed with status code {response.status_code}")
                    if response.status_code == 429:  # Too Many Requests
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"Rate limited. Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        page -= 1  # Retry this page
                    continue
                
                # Small delay between requests to be polite
                time.sleep(0.5)
            
            print("\nData fetch completed successfully!")
            self.df = pd.DataFrame(all_games)
            return self.df
            
        except Exception as e:
            print(f"\nError during fetch: {str(e)}")
            self._save_progress(all_games, page, is_error=True)
            return None
    
    def _save_progress(self, games, current_page, is_error=False):
        """Save progress to CSV file"""
        if not games:
            return
        
        temp_df = pd.DataFrame(games)
        
        # Create output directory if it doesn't exist
        os.makedirs("rawg_data", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rawg_data/rawg_games_page_{current_page}_{timestamp}.csv"
        
        try:
            temp_df.to_csv(filename, index=False)
            status = "Error backup" if is_error else "Progress"
            print(f"\n{status} saved to {filename} ({len(temp_df)} games)")
        except Exception as e:
            print(f"\nFailed to save progress: {str(e)}")
    
    def save_to_csv(self, filename="rawg_games.csv"):
        """Save the complete dataset to CSV"""
        if self.df.empty:
            print("No data to save. Fetch data first.")
            return False
        
        try:
            self.df.to_csv(filename, index=False)
            print(f"Successfully saved {len(self.df)} games to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save CSV: {str(e)}")
            return False
    
    def flatten_nested_columns(self):
        """Flatten nested JSON columns in the dataframe"""
        if self.df.empty:
            print("No data to process. Fetch data first.")
            return

        print("Processing nested columns...")

        # Example: Flatten genres
        if 'genres' in self.df.columns:
            self.df['genres'] = self.df['genres'].apply(
                lambda x: ', '.join([g.get('name', '') for g in x]) if isinstance(x, list) else ''
            )

        # Example: Flatten platforms
        if 'platforms' in self.df.columns:
            self.df['platforms'] = self.df['platforms'].apply(
                lambda x: ', '.join([p.get('platform', {}).get('name', '') for p in x]) if isinstance(x, list) else ''
            )

        # Example: Flatten developers (optional)
        if 'developers' in self.df.columns:
            self.df['developers'] = self.df['developers'].apply(
                lambda x: ', '.join([d.get('name', '') for d in x]) if isinstance(x, list) else ''
            )

        print("Nested columns processed.")


def main():
    # Initialize with your RAWG API key
    API_KEY = "711187b0527641ab907077c4c278a39c"  # Replace with your key
    
    fetcher = RAWGDataFetcher(API_KEY)
    
    print("RAWG API Data Fetcher")
    print("---------------------")
    
    try:
        # Get user input for fetch parameters
        pages = int(input("Enter number of pages to fetch (1-100): ") or 10)
        page_size = int(input(f"Enter games per page (1-40, default 40): ") or 40)
        
        # Fetch data
        df = fetcher.fetch_games(pages=pages, page_size=page_size)
        
        if df is not None and not df.empty:
            # Process nested data
            fetcher.flatten_nested_columns()
            
            # Save final CSV
            output_file = input("Enter output filename (default: rawg_games.csv): ") or "rawg_games.csv"
            fetcher.save_to_csv(output_file)
    except ValueError:
        print("Please enter valid numbers for pages and page size.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()