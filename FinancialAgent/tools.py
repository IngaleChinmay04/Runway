import logging
from langchain_core.tools import tool
from typing import List

logger = logging.getLogger(__name__)

# --- BUDGET AGENT TOOLS ---

@tool
def calculate_runway(current_balance: float, daily_burn_rate: float) -> str:
    """
    Calculates how many days the user can survive.
    Use this when the user asks 'How long will my money last?'
    """
    if daily_burn_rate <= 0:
        return "Runway is infinite (Burn rate is 0 or negative)."
    
    days = int(current_balance / daily_burn_rate)
    logger.info(f"Calculated Runway: {days} days")
    
    if days < 7:
        return f"CRITICAL: Only {days} days remaining."
    elif days < 30:
        return f"WARNING: {days} days remaining."
    return f"SAFE: {days} days remaining."

@tool
def analyze_spending_spike(expenses: List[float], threshold: float = 1000) -> str:
    """
    Identifies abnormal high-value transactions from a list of expense amounts.
    """
    spikes = [e for e in expenses if e > threshold]
    if spikes:
        return f"Found {len(spikes)} suspicious large transactions: {spikes}"
    return "No spending spikes detected."

# --- SHARKY AGENT TOOLS ---

@tool
def detect_predatory_loan(interest_rate_annual: float, processing_fee_percent: float) -> str:
    """
    Analyzes a loan offer for 'Shark' signals. 
    Shark signals: Interest > 20% or Processing Fee > 2%.
    """
    risk_score = 0
    reasons = []
    
    if interest_rate_annual > 20:
        risk_score += 50
        reasons.append(f"High Interest Rate ({interest_rate_annual}%)")
    
    if processing_fee_percent > 2:
        risk_score += 30
        reasons.append(f"High Processing Fee ({processing_fee_percent}%)")
        
    if risk_score > 40:
        return f"DANGER: This looks like a Shark Loan. Reasons: {', '.join(reasons)}"
    return "SAFE: This loan offer looks standard."

@tool
def check_cash_velocity_risk(atm_withdrawal_total: float, total_income: float) -> str:
    """
    Checks if the user is withdrawing too much cash (hiding money from banks).
    """
    if total_income == 0: return "No income data."
    
    ratio = (atm_withdrawal_total / total_income) * 100
    if ratio > 50:
        return f"HIGH RISK: You withdraw {ratio:.1f}% of income as cash. Banks verify you as 'High Risk'."
    return f"Cash usage is normal ({ratio:.1f}%)."