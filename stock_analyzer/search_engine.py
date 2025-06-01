from googlesearch import search
import time
import random

class AISearch:
        
    def search_engine(self, ticker):
        """Search for latest news about a ticker with robust error handling"""
        query = f"latest news about {ticker}."
        
        try:
            # Add small random delay to avoid hitting rate limits
            time.sleep(random.uniform(0.5, 1.5))  # Reduced delay
            
            # Try to get search results with limited number and timeout
            results = search(query, num_results=2, advanced=True)  # Reduced from 4 to 2
            
            formatted_results = []
            if results is not None:
                # Limit processing to avoid hanging
                count = 0
                for result in results:
                    if count >= 2:  # Only process first 2 results
                        break
                    try:
                        formatted_result = {
                            'title': result.title if hasattr(result, 'title') else 'N/A',
                            'description': result.description if hasattr(result, 'description') else 'N/A',
                            'url': result.url if hasattr(result, 'url') else 'N/A'
                        }
                        formatted_results.append(formatted_result)
                        count += 1
                    except Exception as inner_e:
                        print(f"Error processing search result: {str(inner_e)}")
                        continue
            else:
                formatted_results = None
                
        except Exception as e:
            # If search fails (rate limit, network, etc.), return None quickly
            print(f"Search failed: {str(e)}")
            formatted_results = None
            
        return formatted_results

    def serch_prompt_generate(self, user_input, search_mode):
        """Generate search prompt with fallback handling"""
        if search_mode == True:
            try:
                searched = self.search_engine(user_input)
                
                if searched is not None and len(searched) > 0:
                    search_prompt = f"Additional search results: {searched}.\nGenerate in markdown format if possible."
                else:
                    search_prompt = "No recent news available. Focus on technical analysis only."
            except Exception as e:
                print(f"Search prompt generation failed: {str(e)}")
                search_prompt = "No recent news available. Focus on technical analysis only."
        else:
            search_prompt = ""
            print(f"Search prompt: {search_prompt}")
       
        return search_prompt
        
    def search_patterns(self,query):
        #print(f" Debug: screening search patterns")
        import re

        query_patterns = [
            r"\b(internet|web|online|search|google|bing|yahoo|duckduckgo)\b",
            r"\b(latest|news|now|update|today|breaking|current)\b",
            r"\b(according to|source of|quoted by|cited by|do you know)\b",
            r"\b(research|study|article|report|statistics|evidence|tutorial|guide)\b"
        ]

        if any(re.search(pattern, query, re.IGNORECASE) for pattern in query_patterns):
            print("System: This query likely requires an internet search.")
            search_mode = True
        else:
            print("System: This query might be answered with existing knowledge.")
            search_mode = False
        return search_mode
