import asyncio
import functools
import operator
import sys
import os
import json
import logging
from typing import TypedDict, Annotated, List, Any

# Load environment
from dotenv import load_dotenv
load_dotenv()

# LangChain / LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- IMPORT LOCAL MODULES ---
# Assuming you saved the feature extraction code in analyzer.py
from FeatureExtraction import FinancialAnalyzer 
from tools import calculate_runway, analyze_spending_spike, detect_predatory_loan, check_cash_velocity_risk

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger("RunwayOrchestrator")

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# --- 2. LLM SETUP ---
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY is missing. Check your .env file.")
    sys.exit(1)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- 3. CONTEXT LOADING (THE MISSING PIECE) ---
def load_and_analyze_data(filepath: str):
    """
    Loads JSON, runs the FinancialAnalyzer, and returns the DNA Dictionary.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Initialize the Analyzer Class we created earlier
        analyzer = FinancialAnalyzer(data)
        dna = analyzer.get_financial_dna()
        
        logger.info(f"Financial DNA Loaded. Status: {dna['survival_metrics']['survival_state']}")
        return dna
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        # Return a safe fallback if data fails
        return None

# --- 4. MCP CLIENT SETUP ---
async def setup_mcp_tools():
    # ... (Same MCP code as before) ...
    # For brevity, assuming this connects to your friend's server
    return [] 

# --- 5. AGENT FACTORY (Now with DYNAMIC PROMPTS) ---
def create_agent_executor(tools: List[Any], system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def agent_node(state: AgentState, agent_executor: AgentExecutor, name: str) -> dict:
    result = agent_executor.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name=name)]}

# --- 6. MAIN WORKFLOW ---
async def main():
    logger.info("Initializing Agent System...")

    # A. LOAD FINANCIAL CONTEXT
    # This is where we inject the "Brain"
    dna_report = load_and_analyze_data("user_data.json")
    
    if not dna_report:
        logger.error("Failed to load financial context. Exiting.")
        return

    # B. GENERATE PERSONA PROMPTS
    # We construct specific prompts using the extracted data
    
    # 1. Budget Context
    budget_prompt = f"""
    You are the Budget Guardian. 
    CURRENT USER STATUS:
    - Runway: {dna_report['survival_metrics']['current_runway_days']} DAYS
    - Survival State: {dna_report['survival_metrics']['survival_state']}
    - Daily Burn: ₹{dna_report['survival_metrics']['daily_burn_rate']}
    
    YOUR RULE:
    If the Survival State is CRITICAL or WARNING, you must aggressively reject non-essential spending.
    Always quote the 'Runway' days to the user to make them realize the reality.
    """

    # 2. Shark Context
    shark_prompt = f"""
    You are the Loan Shark Detective.
    RISK PROFILE:
    - Cash Reliance: {dna_report['risk_assessment']['cash_reliance_percent']}%
    - Risk Level: {dna_report['risk_assessment']['risk_level']}
    - Suspicious Credits: {dna_report['risk_assessment']['suspicious_loan_credits_count']}
    
    YOUR RULE:
    If Cash Reliance is > 50%, warn the user that they are 'Invisible' to banks.
    If they ask about a loan, check for predatory terms immediately.
    """

    # 3. Investment Context
    invest_prompt = f"""
    You are the Investment Growth Expert.
    FINANCIAL FREEDOM CHECK:
    - Investable Surplus: ₹{dna_report['growth_potential']['investable_surplus']}
    - Income Type: {dna_report['income_profile']['income_type']}
    
    YOUR RULE:
    If Surplus is 0, do NOT recommend stocks. Recommend 'Emergency Funds' only.
    If Income Type is GIG/VARIABLE, recommend 'Liquid Funds' (flexible withdrawal).
    """

    # C. CREATE EXECUTORS WITH CONTEXT
    # Define Tool Sets
    budget_tools = [calculate_runway, analyze_spending_spike]
    sharky_tools = [detect_predatory_loan, check_cash_velocity_risk]
    investment_tools = await setup_mcp_tools()

    # Create Executors with the DYNAMIC prompts
    budget_executor = create_agent_executor(budget_tools, budget_prompt)
    sharky_executor = create_agent_executor(sharky_tools, shark_prompt)
    investment_executor = create_agent_executor(investment_tools, invest_prompt)

    # D. CREATE NODES
    budget_node = functools.partial(agent_node, agent_executor=budget_executor, name="BudgetAgent")
    sharky_node = functools.partial(agent_node, agent_executor=sharky_executor, name="SharkyAgent")
    investment_node = functools.partial(agent_node, agent_executor=investment_executor, name="InvestmentAgent")

    # E. SUPERVISOR
    agent_names = ["BudgetAgent", "SharkyAgent", "InvestmentAgent"]
    
    # The Supervisor also needs a bit of context to know urgency
    supervisor_prompt = f"""
    You are the Runway Manager. Manage the conversation.
    Global Status: {dna_report['survival_metrics']['survival_state']}
    
    - 'BudgetAgent': For spending, balance, and survival questions.
    - 'SharkyAgent': For loans, debt, and risk questions.
    - 'InvestmentAgent': For stocks, savings, and growth questions.
    
    Return ONLY the agent name or 'FINISH'.
    """

    supervisor_chain = create_supervisor(
        llm=llm,
        agents=agent_names,
        system_prompt=supervisor_prompt
    )

    # F. BUILD GRAPH
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_chain)
    workflow.add_node("BudgetAgent", budget_node)
    workflow.add_node("SharkyAgent", sharky_node)
    workflow.add_node("InvestmentAgent", investment_node)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_edge("BudgetAgent", "supervisor")
    workflow.add_edge("SharkyAgent", "supervisor")
    workflow.add_edge("InvestmentAgent", "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "BudgetAgent": "BudgetAgent",
            "SharkyAgent": "SharkyAgent",
            "InvestmentAgent": "InvestmentAgent",
            "FINISH": END,
        },
    )

    app = workflow.compile()
    logger.info("Graph Compiled with User Financial DNA.")

    # --- 7. TEST RUN ---
    user_query = "Can I afford a vacation?"
    print(f"\nUser: {user_query}\n" + "-"*40)
    
    inputs = {"messages": [HumanMessage(content=user_query)]}
    
    async for output in app.astream(inputs):
        for key, value in output.items():
            if "messages" in value:
                print(f"[{key}]: {value['messages'][-1].content}\n")

if __name__ == "__main__":
    asyncio.run(main())