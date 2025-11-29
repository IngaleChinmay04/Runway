import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- AGNO IMPORTS ---
try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.google import Gemini
except ImportError:
    print("Error: Agno not installed. Run: pip install agno google-generativeai pandas python-dotenv")
    exit(1)

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RunwayAI")

# Check API Key
if not os.getenv("GOOGLE_API_KEY"):
    logger.warning("GOOGLE_API_KEY not found in environment variables. Please set it in .env or export it.")

# ==========================================
# 1. EMBEDDED SAMPLE DATA (The Bank Statement)
# ==========================================
SAMPLE_JSON_DATA = """
{
    "Account": {
        "Summary": { "currentBalance": "761.41", "currency": "INR" },
        "Transactions": {
            "Transaction": [
                { "type": "DEBIT", "mode": "CARD", "amount": "100.0", "transactionTimestamp": "2023-06-27T09:40:19", "narration": "FUEL STATION" },
                { "type": "DEBIT", "mode": "CARD", "amount": "170.0", "transactionTimestamp": "2023-06-28T09:51:57", "narration": "FUEL STATION" },
                { "type": "DEBIT", "mode": "CARD", "amount": "500.0", "transactionTimestamp": "2023-07-26T10:04:00", "narration": "FUEL STATION" },
                { "type": "DEBIT", "mode": "ATM", "amount": "1000.0", "transactionTimestamp": "2023-08-07T17:13:13", "narration": "ATM CASH WITHDRAWAL" },
                { "type": "CREDIT", "mode": "UPI", "amount": "15000.0", "transactionTimestamp": "2023-08-01T10:00:00", "narration": "SALARY CREDIT" },
                { "type": "DEBIT", "mode": "UPI", "amount": "8000.0", "transactionTimestamp": "2023-08-02T11:00:00", "narration": "LOAN EMI" },
                { "type": "DEBIT", "mode": "UPI", "amount": "50.0", "transactionTimestamp": "2023-08-05T12:00:00", "narration": "Coffee Shop" },
                { "type": "CREDIT", "mode": "IMPS", "amount": "2000.0", "transactionTimestamp": "2023-08-10T14:00:00", "narration": "Swift Loan App Pvt Ltd" },
                { "type": "CREDIT", "mode": "IMPS", "amount": "1500.0", "transactionTimestamp": "2023-08-12T14:00:00", "narration": "EasyMoney Finance" }
            ]
        }
    }
}
"""

# ==========================================
# 2. FINANCIAL ANALYZER (Feature Extraction)
# ==========================================
class FinancialAnalyzer:
    def __init__(self, json_data: Dict[str, Any]):
        try:
            self.summary = json_data['Account']['Summary']
            self.current_balance = float(self.summary['currentBalance'])
            txns = json_data['Account']['Transactions']['Transaction']
            self.df = pd.DataFrame(txns)
            
            # Pre-processing
            self.df['amount'] = pd.to_numeric(self.df['amount'])
            self.df['timestamp'] = pd.to_datetime(self.df['transactionTimestamp'])
            self.debits = self.df[self.df['type'] == 'DEBIT'].copy()
            self.credits = self.df[self.df['type'] == 'CREDIT'].copy()
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            self.df = pd.DataFrame() # Fallback

    def get_burn_metrics(self):
        """Calculates Survival State"""
        if self.df.empty: return {"survival_state": "UNKNOWN", "runway_days": 0, "daily_burn": 0}
        
        # Calculate daily burn (Total debits / approx days)
        total_spend = self.debits['amount'].sum()
        days_active = max(1, (self.df['timestamp'].max() - self.df['timestamp'].min()).days)
        daily_burn = total_spend / days_active
        
        runway_days = int(self.current_balance / daily_burn) if daily_burn > 0 else 999
        
        state = "SAFE"
        if runway_days < 7: state = "CRITICAL"
        elif runway_days < 30: state = "WARNING"
            
        return {
            "daily_burn": round(daily_burn, 2),
            "runway_days": runway_days,
            "survival_state": state
        }

    def get_risk_metrics(self):
        """Detects Shark Behavior"""
        if self.df.empty: return {"risk_level": "UNKNOWN", "cash_ratio": 0}

        # Cash Velocity
        atm_withdrawals = self.debits[self.debits['mode'] == 'ATM']['amount'].sum()
        total_debits = self.debits['amount'].sum()
        cash_ratio = (atm_withdrawals / total_debits * 100) if total_debits > 0 else 0
        
        # Shark Keywords
        shark_keywords = ['finance', 'loan', 'pvt', 'credit', 'money']
        sus_credits = self.credits[self.credits['narration'].str.contains('|'.join(shark_keywords), case=False, na=False)]
        
        risk_level = "LOW"
        if cash_ratio > 40: risk_level = "HIGH (Cash Trap)"
        if len(sus_credits) > 1: risk_level = "CRITICAL (Loan Sharks Detected)"
        
        return {
            "cash_ratio": round(cash_ratio, 1),
            "risk_level": risk_level,
            "suspicious_count": len(sus_credits)
        }

    def get_growth_metrics(self, burn_metrics):
        """Calculates Investable Surplus"""
        # Safety Buffer: Keep 15 days of burn
        safe_buffer = burn_metrics['daily_burn'] * 15
        investable = max(0, self.current_balance - safe_buffer)
        
        return {
            "investable_surplus": round(investable, 2),
            "buffer_required": round(safe_buffer, 2)
        }

    def get_full_dna(self):
        burn = self.get_burn_metrics()
        risk = self.get_risk_metrics()
        growth = self.get_growth_metrics(burn)
        return {**burn, **risk, **growth, "balance": self.current_balance}

# ==========================================
# 3. AGENT TOOLS (Python Functions)
# ==========================================
def tool_calculate_runway(current_balance: float, daily_burn: float) -> str:
    """Returns the exact number of days money will last."""
    if daily_burn <= 0: return "Infinite"
    days = int(current_balance / daily_burn)
    return f"{days} Days"

def tool_analyze_loan(interest_rate: float, tenure_months: int) -> str:
    """Analyzes a loan offer. Warns if interest > 20%."""
    if interest_rate > 20:
        return "DANGER: High Interest Rate (>20%). This is predatory."
    return "SAFE: Standard terms."

# ==========================================
# 4. MAIN ORCHESTRATION
# ==========================================
def main():
    print("--- 1. LOADING FINANCIAL DATA ---")
    data = json.loads(SAMPLE_JSON_DATA)
    analyzer = FinancialAnalyzer(data)
    dna = analyzer.get_full_dna()
    
    print(f"User DNA Extracted:")
    print(f"Status: {dna['survival_state']} | Runway: {dna['runway_days']} Days")
    print(f"Risk Level: {dna['risk_level']}")
    print("-" * 40)

    # --- DEFINE AGENTS WITH DYNAMIC CONTEXT ---
    
    # 1. Budget Guardian (Strict)
    budget_agent = Agent(
        name="BudgetGuardian",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[tool_calculate_runway],
        instructions=[
            f"You are the Budget Guardian. User Status: {dna['survival_state']}.",
            f"User has only {dna['runway_days']} days of money left.",
            f"Daily Burn Rate is {dna['daily_burn']}.",
            "If Status is CRITICAL or WARNING, reject non-essential spending aggressively.",
            "Always mention 'Runway Days' in your final answer."
        ],
        # show_tool_calls=True,

        markdown=True
    )

    # 2. Shark Detective (Protective)
    shark_agent = Agent(
        name="SharkDetective",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[tool_analyze_loan],
        instructions=[
            f"You are the Loan Shark Detective. Risk Level: {dna['risk_level']}.",
            f"User withdraws {dna['cash_ratio']}% of income as CASH.",
            "If Cash Ratio > 50%, warn user they are 'Invisible to Banks'.",
            "If user asks about a loan app, act skeptical."
        ],
        # show_tool_calls=True,
        markdown=True
    )

    # 3. Investment Expert (Growth)
    investment_agent = Agent(
        name="InvestmentExpert",
        model=Gemini(id="gemini-2.0-flash"),
        instructions=[
            f"You are the Investment Strategist.",
            f"Investable Surplus: {dna['investable_surplus']} INR.",
            "If Surplus is 0, do NOT recommend stocks/crypto. Suggest 'Emergency Fund' only.",
            "If Surplus > 0, suggest safe Liquid Funds."
        ],
        markdown=True
    )

    # --- SUPERVISOR TEAM ---
    runway_team = Team(
        name="RunwaySupervisor",
        members=[budget_agent, shark_agent, investment_agent],
        model=Gemini(id="gemini-2.0-flash"),
        instructions=[
            "You are the Manager of 'Runway AI'.",
            f"Global User Health: {dna['survival_state']}.",
            "Route the user's query to the correct expert:",
            "- BudgetGuardian: Spending, balance, affordability.",
            "- SharkDetective: Loans, debt, suspicious apps.",
            "- InvestmentExpert: Savings, stocks, growing money.",
            "If the query is generic (like 'Hi'), introduce yourself and the User's Financial Health."
        ],
        # show_tool_calls=True,
        markdown=True
    )

    print("\n--- RUNWAY AI SYSTEM READY ---")
    print("Sample Queries: 'Can I buy a watch?', 'Is this 25% loan safe?', 'Where to invest?'")
    
    # Interactive Loop
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Run Agno Team
            runway_team.print_response(user_input, stream=True)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()