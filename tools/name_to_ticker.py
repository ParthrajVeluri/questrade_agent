"""Resolve company names to Yahoo Finance equity symbols."""

import yfinance as yf
import time
from typing import Optional, Tuple
from langchain_core.tools import tool


@tool("resolve_company_ticker")
def get_ticker_from_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a company name to an equity ticker and canonical name.

    Args:
        name: Company name or ticker to search for.

    Returns:
        A `(ticker, company_name)` pair, or `(None, None)` when no equity is found.
    """
    
    try:
        # yf.Search handles fuzzy matching natively through Yahoo's backend
        start = time.time()
        search = yf.Search(name, max_results=3, enable_fuzzy_query=True)
        
        # Ensure we received valid quotes back
        if not search.quotes:
            return None, None
        # Filter for equities to avoid accidentally returning options or mutual funds
        for quote in search.quotes:
            if quote.get('quoteType') == 'EQUITY':
                end = time.time()
                print(f"Search completed in {end - start:.2f} seconds.")
                company_name = quote.get('shortname') or quote.get('longname')
                symbol = quote.get('symbol')
                return symbol, company_name
                
        # Fallback to the very first result if no specific equity filter matches
        end = time.time()
        print(f"Search completed in {end - start:.2f} seconds.")
        return None, None
        
    except Exception as e:
        print(f"Error searching for {name}: {e}")
        return None, None

if __name__ == "__main__":
    # Standalone lookup utility for quick terminal testing.
    company_name = input("Enter the company name: ")
    ticker = get_ticker_from_name(company_name)
    if ticker:
        print(f"The ticker symbol for {company_name} is: {ticker}")
    else:
        print(f"No ticker symbol found for {company_name}.")