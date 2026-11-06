import yfinance as yf
import time
from typing import Optional

def get_ticker_from_name(name: str) -> Optional[str]:
    """
    Get the ticker symbol for a given company name using yfinance.

    Args:
        name (str): The name of the company.
    Returns:
        str: The ticker symbol of the company, or None if not found.
    """
    
    try:
        # yf.Search handles fuzzy matching natively through Yahoo's backend
        start = time.time()
        search = yf.Search(name, max_results=3, enable_fuzzy_query=True)
        
        # Ensure we received valid quotes back
        if not search.quotes:
            return None
            
        # Filter for equities to avoid accidentally returning options or mutual funds
        for quote in search.quotes:
            if quote.get('quoteType') == 'EQUITY':
                end = time.time()
                print(f"Search completed in {end - start:.2f} seconds.")
                return quote.get('symbol')
                
        # Fallback to the very first result if no specific equity filter matches
        end = time.time()
        print(f"Search completed in {end - start:.2f} seconds.")
        return None
        
    except Exception as e:
        print(f"Error searching for {name}: {e}")
        return None

if __name__ == "__main__":
    company_name = input("Enter the company name: ")
    ticker = get_ticker_from_name(company_name)
    if ticker:
        print(f"The ticker symbol for {company_name} is: {ticker}")
    else:
        print(f"No ticker symbol found for {company_name}.")