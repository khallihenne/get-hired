"""
Dual-LLM Agentic RAG System - Integrated Research Agent
Mistral routes queries, Llama handles candidate analysis, Mistral-7B handles market research
"""

import os
from typing import List, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_tool_calling_agent

from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

import weaviate
from weaviate.classes.init import Auth

# Import dual-LLM scorer (candidate analysis - Llama)
from agentic_scorer import (
    DualLLMAgenticScorer,
    create_dual_llm_scorer,
    ScoringDecision
)

# Import research agent (market research - Mistral-7B)
from agentic_research import (
    MarketResearchAgent,
    create_research_agent,
    ResearchReport
)


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


class TripleLLMAgenticRAG:
    """
    Enhanced RAG with Three Specialized LLMs
    
    Architecture:
    - Mistral Large: Query routing and conversation management
    - Llama-3.1-70B: Candidate analysis and ranking
    - Mistral-7B: Market research and salary insights
    """
    
    def __init__(
        self,
        weaviate_client,
        mistral_api_key: str,
        huggingface_api_key: str,
        tavily_api_key: str,
        collection_name: str = "Resume",
        enable_detailed_reasoning: bool = False
    ):
        print(f"\n{Colors.PURPLE}{'='*100}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}Initializing Triple-LLM HR Recruitment System{Colors.RESET}")
        print(f"{Colors.DIM}Routing: Mistral Large • Analysis: Llama-3.1-70B • Research: Mistral-7B{Colors.RESET}")
        print(f"{Colors.PURPLE}{'='*100}{Colors.RESET}\n")
        
        # Mistral for routing
        print(f"{Colors.CYAN}[1/5] Initializing Mistral Large for routing...{Colors.RESET}")
        self.conversation_llm = ChatMistralAI(
            api_key=mistral_api_key,
            model="mistral-large-latest",
            temperature=0.1,
            max_tokens=1024
        )
        print(f"{Colors.GREEN}✓ Mistral Large ready{Colors.RESET}\n")
        
        # Embeddings and vector store
        print(f"{Colors.CYAN}[2/5] Loading embeddings model...{Colors.RESET}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=collection_name,
            text_key="preprocessed_text",
            embedding=self.embeddings
        )
        print(f"{Colors.GREEN}✓ Vector store connected{Colors.RESET}\n")
        
        # Llama-3.1-70B for candidate analysis
        print(f"{Colors.CYAN}[3/5] Initializing Llama-3.1-70B for candidate analysis...{Colors.RESET}")
        self.analytical_scorer = create_dual_llm_scorer(
            huggingface_api_key=huggingface_api_key,
            temperature=0.2
        )
        print(f"{Colors.GREEN}✓ Llama-3.1-70B analytical engine ready{Colors.RESET}\n")
        
        # Mistral-7B for market research via Official Mistral API
        print(f"{Colors.CYAN}[4/5] Initializing Mistral-7B for market research...{Colors.RESET}")
        self.research_agent = create_research_agent(
            mistral_api_key=mistral_api_key,  # ✅ CORRIGÉ: utilise Mistral API
            tavily_api_key=tavily_api_key,
            model="open-mistral-7b"  # ✅ CORRIGÉ: modèle pour API Mistral
        )
        print(f"{Colors.GREEN}✓ Mistral-7B research agent ready{Colors.RESET}\n")
        
        self.enable_detailed_reasoning = enable_detailed_reasoning
        
        # Setup tools and agent
        print(f"{Colors.CYAN}[5/5] Creating agent tools and memory...{Colors.RESET}")
        self.tools = self._create_tools()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent = self._create_agent()
        
        print(f"{Colors.GREEN}✓ System fully initialized!{Colors.RESET}")
        print(f"\n{Colors.DIM}Architecture Summary:{Colors.RESET}")
        print(f"{Colors.DIM}  • Mistral Large: Query routing{Colors.RESET}")
        print(f"{Colors.DIM}  • Llama-3.1-70B: Candidate evaluation and ranking{Colors.RESET}")
        print(f"{Colors.DIM}  • Mistral-7B: Market research and salary analysis (via Official Mistral API){Colors.RESET}\n")
    
    def _create_tools(self) -> List[Tool]:
        """Create specialized agent tools"""
        
        def search_and_rank_candidates(query: str) -> str:
            """Llama-powered candidate search and ranking"""
            try:
                print(f"\n{Colors.CYAN}{'─'*100}{Colors.RESET}")
                print(f"{Colors.BOLD}CANDIDATE SEARCH & ANALYSIS PIPELINE{Colors.RESET}\n")
                
                # Extract requirements
                print(f"{Colors.DIM}[1/4] Llama extracting job requirements...{Colors.RESET}")
                requirements = self.analytical_scorer.extract_requirements(query)
                print(f"{Colors.GREEN}✓ Requirements identified{Colors.RESET}\n")
                
                # Semantic search
                print(f"{Colors.DIM}[2/4] Searching vector database...{Colors.RESET}")
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=10)
                
                if not docs_with_scores:
                    return f"{Colors.YELLOW}No matching candidates found in database.{Colors.RESET}"
                
                print(f"{Colors.GREEN}✓ Found {len(docs_with_scores)} potential candidates{Colors.RESET}\n")
                
                # Llama evaluation
                print(f"{Colors.DIM}[3/4] Llama-3.1-70B analyzing candidates...{Colors.RESET}")
                print(f"{Colors.DIM}    (Multi-phase reasoning: Assessment → Analysis → Reflection → Decision){Colors.RESET}\n")
                
                decisions: List[ScoringDecision] = []
                for idx, (doc, score) in enumerate(docs_with_scores, 1):
                    print(f"{Colors.DIM}    Analyzing candidate {idx}/{len(docs_with_scores)}...{Colors.RESET}", end='\r')
                    
                    decision = self.analytical_scorer.evaluate_candidate(
                        candidate_doc=doc,
                        semantic_score=score,
                        requirements=requirements
                    )
                    decisions.append(decision)
                
                print(f"{Colors.GREEN}✓ All candidates analyzed                    {Colors.RESET}\n")
                
                # Sort and format
                print(f"{Colors.DIM}[4/4] Ranking and formatting results...{Colors.RESET}")
                decisions.sort(key=lambda x: x.final_score, reverse=True)
                
                llama_output = self.analytical_scorer.format_decision_output(
                    decisions[:5],
                    colors_class=Colors,
                    show_reasoning=self.enable_detailed_reasoning
                )
                
                print(f"{Colors.GREEN}✓ Analysis complete{Colors.RESET}")
                print(f"{Colors.CYAN}{'─'*100}{Colors.RESET}\n")
                
                return llama_output
            
            except Exception as e:
                return f"{Colors.RED}Error in candidate evaluation: {str(e)}{Colors.RESET}"
        
        def market_research(query: str) -> str:
            """Mistral-7B-powered market research and salary insights via Official Mistral API"""
            try:
                # Get conversation context for query enhancement
                context = None
                history = self.memory.load_memory_variables({})
                if history and 'chat_history' in history:
                    messages = history['chat_history']
                    for msg in reversed(messages):
                        if hasattr(msg, 'content'):
                            content_lower = msg.content.lower()
                            if any(kw in content_lower for kw in ['engineer', 'developer', 'data scientist', 'analyst', 'manager', 'designer']):
                                context = msg.content
                                break
                
                # Conduct research using Mistral-7B via Official API
                report = self.research_agent.conduct_research(query, context)
                
                # Format and return
                formatted_report = self.research_agent.format_report(report)
                
                return formatted_report
            
            except Exception as e:
                return f"{Colors.RED}Error in market research: {str(e)}{Colors.RESET}"
        
        return [
            Tool(
                name="search_and_rank_candidates",
                func=search_and_rank_candidates,
                description="Search and rank candidates using Llama-3.1-70B multi-phase reasoning. Use for job descriptions or candidate searches."
            ),
            Tool(
                name="market_insights",
                func=market_research,
                description="Get market insights and salary data using Mistral-7B research agent. Use for compensation, salary, market trends, or hiring cost questions."
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create Mistral-powered routing agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing assistant for an HR recruitment system with specialized AI agents.

YOUR ROLE:
1. Understand user queries
2. Route to the appropriate specialized agent
3. Return agent outputs exactly as provided

AVAILABLE AGENTS:
- search_and_rank_candidates: Llama-3.1-70B for candidate evaluation
- market_insights: Mistral-7B for market research and salary data

ROUTING RULES:
- Candidate searches → search_and_rank_candidates
  Keywords: "find", "search", "candidates", "resume", "who", "engineer", "developer", "designer"
  
- Market/salary queries → market_insights
  Keywords: "salary", "compensation", "pay", "market", "wage", "cost", "budget", "typical", "average"

CRITICAL RULES:
- Call the appropriate tool ONCE with the user's exact query
- Return tool output EXACTLY as is - NO modifications, summaries, or additions
- After receiving tool output, your job is DONE - stop immediately
- NEVER add introductory text like "Here are the results:"
- DO NOT attempt to validate or improve the tool's response

The tools provide complete, beautifully formatted outputs. Your only job is routing."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_tool_calling_agent(self.conversation_llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=1,
            max_execution_time=180,
            return_intermediate_steps=False,
            early_stopping_method="force"
        )
    
    def chat(self, user_input: str) -> str:
        """Process user query through routing agent"""
        try:
            # Handle detailed reasoning command
            if user_input.lower().startswith('explain:'):
                self.enable_detailed_reasoning = True
                user_input = user_input[8:].strip()
            
            response = self.agent.invoke({"input": user_input})
            
            # Reset flag
            self.enable_detailed_reasoning = False
            
            return response["output"]
        except Exception as e:
            return f"{Colors.RED}Error: {str(e)}{Colors.RESET}"
    
    def toggle_detailed_reasoning(self, enabled: bool = None):
        """Toggle detailed reasoning traces"""
        if enabled is None:
            self.enable_detailed_reasoning = not self.enable_detailed_reasoning
        else:
            self.enable_detailed_reasoning = enabled
        
        status = "enabled" if self.enable_detailed_reasoning else "disabled"
        print(f"{Colors.PURPLE}Detailed reasoning traces: {status}{Colors.RESET}")


# Main execution
if __name__ == "__main__":
    print(f"\n{Colors.CYAN}{'='*100}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.PURPLE}TRIPLE-LLM HR RECRUITMENT ASSISTANT{Colors.RESET}")
    print(f"{Colors.DIM}Mistral Large (routing) • Llama-3.1-70B (analysis) • Mistral-7B (research){Colors.RESET}")
    print(f"{Colors.CYAN}{'='*100}{Colors.RESET}\n")
    
    # Connect to Weaviate
    print(f"{Colors.DIM}Connecting to Weaviate Cloud...{Colors.RESET}")
    client = weaviate.connect_to_wcs(
        cluster_url="",
        auth_credentials=Auth.api_key("")
    )
    
    if not client.is_ready():
        print(f"{Colors.RED}✗ Weaviate connection failed!{Colors.RESET}")
        exit(1)
    
    print(f"{Colors.GREEN}✓ Weaviate connected{Colors.RESET}\n")
    
    # Initialize system
    rag = TripleLLMAgenticRAG(
        weaviate_client=client,
        mistral_api_key="",
        huggingface_api_key="",
        tavily_api_key="",
        enable_detailed_reasoning=False
    )
    
    # Instructions
    print(f"\n{Colors.BLUE}{'─'*100}{Colors.RESET}")
    print(f"{Colors.BOLD}CHAT INTERFACE{Colors.RESET} {Colors.DIM}(type 'exit' to quit){Colors.RESET}\n")
    print(f"{Colors.DIM}Commands:{Colors.RESET}")
    print(f"  • 'clear' - Reset conversation")
    print(f"  • 'explain: <query>' - Show detailed Llama reasoning for one query")
    print(f"  • 'reasoning on/off' - Toggle detailed reasoning mode")
    print(f"\n{Colors.DIM}Example Queries:{Colors.RESET}")
    print(f"  • Find senior ML engineers with 5+ years Python and NLP experience")
    print(f"  • What's the typical salary for senior ML engineers?")
    print(f"  • What are the hiring trends for data scientists in 2025?")
    print(f"  • explain: Who are the best candidates for [job description]?")
    print(f"\n{Colors.BLUE}{'─'*100}{Colors.RESET}\n")
    
    # Chat loop
    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.CYAN}You: {Colors.RESET}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print(f"\n{Colors.PURPLE}Goodbye! Session ended.{Colors.RESET}")
                break
            
            if user_input.lower() == 'clear':
                rag.memory.clear()
                print(f"{Colors.GREEN}✓ Memory cleared{Colors.RESET}\n")
                continue
            
            if user_input.lower() == 'reasoning on':
                rag.toggle_detailed_reasoning(True)
                continue
            
            if user_input.lower() == 'reasoning off':
                rag.toggle_detailed_reasoning(False)
                continue
            
            # Process query
            response = rag.chat(user_input)
            
            # Print response
            print(f"\n{response}\n")
            print(f"{Colors.DIM}{'─'*100}{Colors.RESET}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.PURPLE}Session interrupted. Goodbye!{Colors.RESET}")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}\n")
    
    client.close()
    print(f"{Colors.GREEN}✓ Connection closed{Colors.RESET}")