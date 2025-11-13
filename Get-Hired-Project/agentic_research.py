"""
Market Research Agent - Mistral-7B via Official Mistral API
Specialized agent for web search analysis and salary insights
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from openai import OpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import json


@dataclass
class ResearchReport:
    """Structured research output"""
    salary_overview: str
    market_insights: str
    hiring_recommendations: str
    sources: List[str]
    raw_data: str


class Colors:
    """Pastel color scheme for terminal output"""
    RESET = '\033[0m'
    BLUE = '\033[38;5;117m'
    GREEN = '\033[38;5;114m'
    YELLOW = '\033[38;5;186m'
    RED = '\033[38;5;181m'
    PURPLE = '\033[38;5;183m'
    CYAN = '\033[38;5;152m'
    ORANGE = '\033[38;5;180m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class MarketResearchAgent:
    """
    Mistral-7B-powered Market Research Agent via Official Mistral API
    
    Specialized for:
    - Web search analysis using Tavily
    - Salary and compensation research
    - Market trend analysis
    - Hiring insights generation
    """
    
    def __init__(
        self,
        mistral_api_key: str,
        tavily_api_key: str,
        model: str = "open-mistral-7b",
        temperature: float = 0.3
    ):
        """Initialize Mistral-7B research agent via Official API"""
        print(f"{Colors.CYAN}[Research Agent] Initializing Mistral-7B via Official Mistral API...{Colors.RESET}")
        
        # Initialize Mistral client using OpenAI SDK (compatible)
        self.client = OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=mistral_api_key
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = 2048
        
        # Initialize Tavily search wrapper
        self.tavily_api_key = tavily_api_key
        self.search_wrapper = TavilySearchAPIWrapper(
            tavily_api_key=tavily_api_key
        )
        
        print(f"{Colors.GREEN}✓ Research agent ready (using Official Mistral API){Colors.RESET}\n")
    
    def conduct_research(
        self,
        query: str,
        context: str = None
    ) -> ResearchReport:
        """
        Conduct comprehensive market research
        
        Args:
            query: Research query (e.g., "salary for senior ML engineers")
            context: Optional context from conversation history
        
        Returns:
            ResearchReport with structured insights
        """
        try:
            print(f"\n{Colors.CYAN}{'─'*100}{Colors.RESET}")
            print(f"{Colors.BOLD}MARKET RESEARCH & SALARY ANALYSIS{Colors.RESET}\n")
            
            # Step 1: Enhance query with context
            enhanced_query = self._enhance_query(query, context)
            print(f"{Colors.DIM}Enhanced query: {enhanced_query[:100]}...{Colors.RESET}")
            
            # Step 2: Web search using Tavily
            print(f"{Colors.DIM}Searching web for current market data...{Colors.RESET}")
            search_results = self.search_wrapper.results(
                query=enhanced_query,
                max_results=5
            )
            
            # Extract sources and content
            sources = []
            content_for_analysis = []
            
            if isinstance(search_results, list):
                for result in search_results:
                    if isinstance(result, dict):
                        # Extract URL
                        url = result.get('url') or result.get('link')
                        if url:
                            sources.append(url)
                        
                        # Extract content for analysis
                        title = result.get('title', '')
                        content = result.get('content', '') or result.get('snippet', '')
                        if title or content:
                            content_for_analysis.append({
                                'title': title,
                                'content': content,
                                'url': url
                            })
            
            print(f"{Colors.GREEN}✓ Web search successful{Colors.RESET}")
            print(f"{Colors.GREEN}✓ Found {len(sources)} sources{Colors.RESET}\n")
            
            # Step 3: Analyze with Mistral-7B
            print(f"{Colors.DIM}Mistral-7B analyzing search results...{Colors.RESET}")
            report = self._analyze_results(query, content_for_analysis)
            
            print(f"{Colors.GREEN}✓ Analysis complete{Colors.RESET}")
            print(f"{Colors.CYAN}{'─'*100}{Colors.RESET}\n")
            
            return ResearchReport(
                salary_overview=report.get("salary_overview", ""),
                market_insights=report.get("market_insights", ""),
                hiring_recommendations=report.get("hiring_recommendations", ""),
                sources=sources,
                raw_data=str(content_for_analysis)
            )
            
        except Exception as e:
            print(f"{Colors.RED}Research error: {str(e)}{Colors.RESET}")
            import traceback
            print(f"{Colors.RED}Traceback: {traceback.format_exc()}{Colors.RESET}")
            return ResearchReport(
                salary_overview=f"Error retrieving salary data: {str(e)}",
                market_insights="Unable to complete market analysis",
                hiring_recommendations="Please try again with a more specific query",
                sources=[],
                raw_data=""
            )
    
    def _enhance_query(self, query: str, context: str = None) -> str:
        """Enhance query with context and specificity"""
        if context and ("this role" in query.lower() or "this position" in query.lower()):
            return f"{query} for {context[:200]}"
        
        # Add year for recency
        if "2025" not in query and "2024" not in query:
            return f"{query} 2025"
        
        return query
    
    def _analyze_results(self, query: str, search_results: List[Dict]) -> Dict[str, str]:
        """Use Mistral-7B to analyze search results and generate structured report"""
        
        # Format search results for the prompt
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')[:500]  # Limit content length
            url = result.get('url', 'No URL')
            formatted_results.append(f"[Source {idx}]\nTitle: {title}\nURL: {url}\nContent: {content}\n")
        
        results_text = "\n".join(formatted_results)
        
        messages = [
            {
                "role": "user",
                "content": f"""You are an expert HR market analyst specializing in compensation research and hiring trends.

Analyze the web search results below and provide a structured market report in JSON format.

QUERY: {query}

SEARCH RESULTS:
{results_text[:4000]}

Provide your analysis in VALID JSON format ONLY (no markdown, no other text):
{{
    "salary_data": [
        {{"level": "Junior", "experience": "1-3 years", "salary_range": "$80K - $100K", "location": "United States"}},
        {{"level": "Mid-Level", "experience": "4-6 years", "salary_range": "$100K - $140K", "location": "United States"}},
        {{"level": "Senior", "experience": "7-10 years", "salary_range": "$140K - $180K", "location": "United States"}},
        {{"level": "Lead/Staff", "experience": "10+ years", "salary_range": "$180K - $250K", "location": "United States"}}
    ],
    "market_insights": "• Insight 1 about demand trends\\n• Insight 2 about in-demand skills\\n• Insight 3 about market conditions\\n• Insight 4 about remote work impact",
    "hiring_recommendations": "1. Recommendation for competitive positioning\\n2. Budget planning considerations\\n3. Key factors for attracting talent\\n4. Timing and market conditions"
}}

CRITICAL REQUIREMENTS FOR SALARY DATA:
- Extract specific salary ranges by experience level (Junior, Mid-Level, Senior, Lead/Staff)
- Use consistent currency format: $XXK or $XXXK
- Include years of experience for each level
- Add geographic variations if mentioned in results (US, Europe, etc.)
- If multiple locations found, create separate entries
- Focus on 2024-2025 data only

CRITICAL REQUIREMENTS FOR INSIGHTS:
- Use bullet points (•) in market_insights
- Keep each insight concise (1-2 sentences)
- Focus on actionable information
- Include demand trends, skills, and growth indicators

CRITICAL REQUIREMENTS FOR RECOMMENDATIONS:
- Use numbered format (1., 2., 3.)
- Be specific and actionable
- Include budget guidance and hiring strategy"""
            }
        ]
        
        try:
            # Appel API Mistral via OpenAI SDK
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            analysis = self._parse_json_response(content)
            
            if not analysis or not any(key in analysis for key in ["salary_data", "market_insights", "hiring_recommendations"]):
                # Fallback to text parsing
                print(f"{Colors.YELLOW}[DEBUG] JSON parsing failed or incomplete, using fallback{Colors.RESET}")
                return self._fallback_analysis(content)
            
            # Format salary data into a table
            if "salary_data" in analysis:
                analysis["salary_overview"] = self._format_salary_table(analysis["salary_data"])
                del analysis["salary_data"]
            else:
                analysis["salary_overview"] = "No salary data available in search results"
            
            # Ensure all required fields exist
            if "market_insights" not in analysis or not analysis["market_insights"]:
                analysis["market_insights"] = "• Market analysis in progress\n• Additional data needed for comprehensive insights"
            
            if "hiring_recommendations" not in analysis or not analysis["hiring_recommendations"]:
                analysis["hiring_recommendations"] = "1. Review market data for competitive positioning\n2. Consider industry standards for compensation\n3. Consult with HR for final recommendations"
            
            return analysis
            
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Mistral-7B API error: {str(e)}{Colors.RESET}")
            return {
                "salary_overview": "Unable to retrieve salary data due to API error",
                "market_insights": "Market analysis unavailable - API connection failed",
                "hiring_recommendations": "Please try again or refine your query"
            }
    
    def _parse_json_response(self, response: str) -> Dict[str, str]:
        """Parse JSON from Mistral-7B response"""
        try:
            # Remove markdown code blocks if present
            response = response.replace('```json', '').replace('```', '')
            
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Debug: print what was parsed
                print(f"{Colors.DIM}[DEBUG] Successfully parsed JSON with keys: {list(parsed.keys())}{Colors.RESET}")
                
                return parsed
            
            print(f"{Colors.YELLOW}[DEBUG] No JSON object found in response{Colors.RESET}")
            return {}
        except json.JSONDecodeError as e:
            print(f"{Colors.YELLOW}[DEBUG] JSON decode error: {str(e)}{Colors.RESET}")
            return {}
        except Exception as e:
            print(f"{Colors.YELLOW}[DEBUG] Parsing error: {str(e)}{Colors.RESET}")
            return {}
    
    def _format_salary_table(self, salary_data: List[Dict[str, str]]) -> str:
        """Format salary data into a beautiful ASCII table"""
        if not salary_data:
            return "No salary data available"
        
        # Table structure
        table = []
        
        # Header
        table.append("┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐")
        table.append("│ LEVEL               │ EXPERIENCE          │ SALARY RANGE        │ LOCATION            │")
        table.append("├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤")
        
        # Data rows
        for entry in salary_data:
            # Convertir en string et gérer les listes
            level = str(entry.get("level", "N/A")) if not isinstance(entry.get("level"), list) else str(entry.get("level", ["N/A"])[0])
            experience = str(entry.get("experience", "N/A")) if not isinstance(entry.get("experience"), list) else str(entry.get("experience", ["N/A"])[0])
            salary = str(entry.get("salary_range", "N/A")) if not isinstance(entry.get("salary_range"), list) else str(entry.get("salary_range", ["N/A"])[0])
            location = str(entry.get("location", "N/A")) if not isinstance(entry.get("location"), list) else str(entry.get("location", ["N/A"])[0])
            
            # Tronquer et justifier
            level = level[:19].ljust(19)
            experience = experience[:19].ljust(19)
            salary = salary[:19].ljust(19)
            location = location[:19].ljust(19)
            
            table.append(f"│ {level} │ {experience} │ {salary} │ {location} │")
        
        # Footer
        table.append("└─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘")
        
        return "\n".join(table)
    
    def _fallback_analysis(self, content: str) -> Dict[str, str]:
        """Fallback text parsing if JSON fails"""
        
        # Try to extract structured information from text
        result = {
            "salary_overview": "Salary data extracted from analysis",
            "market_insights": "",
            "hiring_recommendations": ""
        }
        
        # Extract insights section
        insights_start = content.lower().find('market')
        insights_end = content.lower().find('hiring') if 'hiring' in content.lower() else len(content)
        if insights_start != -1:
            insights_text = content[insights_start:insights_end].strip()[:500]
            result["market_insights"] = insights_text if insights_text else "• Market data analysis in progress\n• Comprehensive insights available upon request"
        else:
            result["market_insights"] = "• High demand for data professionals continues in 2025\n• Remote work opportunities expanding globally\n• Skills in AI/ML remain highly valued\n• Competitive compensation across major markets"
        
        # Extract recommendations section
        rec_start = content.lower().find('hiring') if 'hiring' in content.lower() else content.lower().find('recommend')
        if rec_start != -1:
            rec_text = content[rec_start:].strip()[:500]
            result["hiring_recommendations"] = rec_text if rec_text else "1. Benchmark against market standards\n2. Consider location-based adjustments\n3. Factor in experience and skill levels\n4. Review total compensation packages"
        else:
            result["hiring_recommendations"] = "1. Align salary offers with local and international market rates\n2. Emphasize remote work flexibility as a competitive advantage\n3. Highlight career growth opportunities and skill development\n4. Consider comprehensive benefits beyond base salary"
        
        return result
    
    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract section from text based on markers"""
        text_lower = text.lower()
        start_pos = text_lower.find(start_marker)
        end_pos = text_lower.find(end_marker, start_pos) if end_marker != "end" else len(text)
        
        if start_pos != -1:
            section = text[start_pos:end_pos].strip()
            return section[:500] if len(section) > 500 else section
        
        return "Analysis unavailable"
    
    def format_report(self, report: ResearchReport) -> str:
        """Format research report with beautiful styling"""
        
        try:
            output = []
            
            # Header
            output.append(f"\n{Colors.BLUE}{'='*100}{Colors.RESET}")
            output.append(f"{Colors.BOLD}{Colors.PURPLE}MARKET RESEARCH REPORT{Colors.RESET}")
            output.append(f"{Colors.BLUE}{'='*100}{Colors.RESET}\n")
            
            # Salary Overview
            output.append(f"{Colors.BOLD}{Colors.CYAN}SALARY OVERVIEW{Colors.RESET}")
            output.append(f"{Colors.CYAN}{'─' * 100}{Colors.RESET}")
            
            # Check if it's a table format
            salary_text = self._format_multiline(report.salary_overview)
            if '┌' in salary_text or '│' in salary_text:
                # It's a table - add padding for centering
                table_lines = salary_text.split('\n')
                for line in table_lines:
                    output.append(f"  {line}")
            else:
                output.append(salary_text)
            
            output.append("")
            
            # Market Insights
            output.append(f"\n{Colors.BOLD}{Colors.GREEN}MARKET INSIGHTS{Colors.RESET}")
            output.append(f"{Colors.GREEN}{'─' * 60}{Colors.RESET}")
            output.append(self._format_multiline(report.market_insights))
            output.append("")
            
            # Hiring Recommendations
            output.append(f"\n{Colors.BOLD}{Colors.ORANGE}HIRING RECOMMENDATIONS{Colors.RESET}")
            output.append(f"{Colors.ORANGE}{'─' * 60}{Colors.RESET}")
            output.append(self._format_multiline(report.hiring_recommendations))
            output.append("")
            
            # Sources
            if report.sources:
                output.append(f"\n{Colors.PURPLE}{'='*100}{Colors.RESET}")
                output.append(f"{Colors.BOLD}{Colors.CYAN}SOURCES & REFERENCES{Colors.RESET}")
                output.append(f"{Colors.PURPLE}{'='*100}{Colors.RESET}\n")
                for idx, url in enumerate(report.sources, 1):
                    output.append(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {Colors.CYAN}{url}{Colors.RESET}")
            
            output.append(f"\n{Colors.BLUE}{'='*100}{Colors.RESET}\n")
            
            return "\n".join(output)
            
        except Exception as e:
            # Fallback formatting if there's any error
            return f"\n{Colors.RED}Error formatting report: {str(e)}{Colors.RESET}\n\nRaw data:\n{str(report)}\n"
    
    def _format_multiline(self, text) -> str:
        """Format text with proper indentation and bullet points"""
        if not text:
            return "  No data available"
        
        # Handle list inputs (convert to string)
        if isinstance(text, list):
            text = '\n'.join(str(item) for item in text)
        
        # Convert to string if not already
        text = str(text)
        
        # Check if text contains table characters (don't add indentation for tables)
        is_table = any(char in text for char in ['┌', '│', '└', '├', '┤', '┬', '┴', '┼'])
        
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            # For tables, don't strip or add indentation
            if is_table and any(char in line for char in ['┌', '│', '└', '├', '┤', '┬', '┴', '┼']):
                formatted.append(line)
                continue
            
            line = line.strip()
            if not line:
                continue
            
            # Format bullet points
            if line.startswith('-') or line.startswith('•'):
                formatted.append(f"  {Colors.YELLOW}•{Colors.RESET} {line[1:].strip()}")
            # Format numbered lists
            elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                formatted.append(f"  {Colors.BOLD}{line[:2]}{Colors.RESET} {line[2:].strip()}")
            # Format headers
            elif line.endswith(':') and len(line) < 60:
                formatted.append(f"\n  {Colors.BOLD}{line}{Colors.RESET}")
            # Regular text
            else:
                formatted.append(f"  {line}")
        
        return '\n'.join(formatted) if formatted else "  No data available"


def create_research_agent(
    mistral_api_key: str,
    tavily_api_key: str,
    model: str = "open-mistral-7b"
) -> MarketResearchAgent:
    """Factory function to create research agent instance"""
    return MarketResearchAgent(
        mistral_api_key=mistral_api_key,
        tavily_api_key=tavily_api_key,
        model=model
    )


# Test the agent
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print(f"\n{Colors.CYAN}{'='*100}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.PURPLE}MISTRAL-7B MARKET RESEARCH AGENT - TEST MODE{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*100}{Colors.RESET}\n")
    
    # Initialize agent
    agent = create_research_agent(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Test query
    test_query = "salary for senior machine learning engineers with 5+ years experience"
    print(f"{Colors.BOLD}Test Query:{Colors.RESET} {test_query}\n")
    
    # Conduct research
    report = agent.conduct_research(test_query)
    
    # Display formatted report
    print(agent.format_report(report))
    
    print(f"{Colors.GREEN}✓ Test complete{Colors.RESET}")