from googlesearch import search

class AISearch:
        
    def search_engine(self, ticker):
        #print(f" Debug: searching {query} on the web.")
        query = f"latest news about {ticker}."
 


        results = search(query, num_results=4, advanced=True)
        
        
        formatted_results = []
        if results is not None:
            for result in results:

                formatted_result = {
                    'title': result.title,
                    'description': result.description,
                    'url': result.url
                }
                formatted_results.append(formatted_result)

        else:
            formatted_results = None
        return formatted_results


    def serch_prompt_generate(self,user_input,search_mode):
        if search_mode == True:
            searched = self.search_engine(user_input)
            
            search_prompt = f"Additional search results :{searched}.\nGenerate in markdown format if possible."
        else:
            search_prompt =""
       
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
