import json
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    def __init__(self, json_data):
        """
        Initializes the analyzer with raw JSON data.
        """
        try:
            # 1. Extract Balance and Holder Info
            self.summary = json_data['Account']['Summary']
            self.current_balance = float(self.summary['currentBalance'])
            
            # 2. Extract Transactions
            txns = json_data['Account']['Transactions']['Transaction']
            self.df = pd.DataFrame(txns)
            
            # 3. Data Cleaning
            self.df['amount'] = pd.to_numeric(self.df['amount'])
            self.df['timestamp'] = pd.to_datetime(self.df['transactionTimestamp'])
            self.df['date'] = self.df['timestamp'].dt.date
            self.df['day'] = self.df['timestamp'].dt.day
            
            # 4. Separate Credits (Income) and Debits (Spend)
            self.credits = self.df[self.df['type'] == 'CREDIT'].copy()
            self.debits = self.df[self.df['type'] == 'DEBIT'].copy()
            
            logger.info(f"Data Loaded: {len(self.df)} transactions processed.")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _get_burn_metrics(self):
        """Calculates Burn Rate, Runway, and Survival State."""
        # Calculate total spend over the period available
        total_spend = self.debits['amount'].sum()
        
        # Calculate number of days in the data
        if not self.df.empty:
            days_diff = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            days_active = max(1, days_active) if 'days_active' in locals() else max(1, days_diff)
        else:
            days_active = 1

        daily_burn_rate = total_spend / days_active
        
        # Calculate Runway (Days to Zero Balance)
        runway_days = int(self.current_balance / daily_burn_rate) if daily_burn_rate > 0 else 999
        
        # Determine Survival State
        if runway_days < 7:
            state = "CRITICAL (Panic Mode)"
        elif runway_days < 30:
            state = "WARNING (Budget Tight)"
        else:
            state = "SAFE (Stable)"
            
        return {
            "current_runway_days": runway_days,
            "daily_burn_rate": round(daily_burn_rate, 2),
            "survival_state": state
        }

    def _analyze_income_pattern(self):
        """Analyzes Income Volatility, Frequency, and Payday Ranges."""
        if self.credits.empty:
            return {"income_status": "No Income Detected"}

        # 1. Volatility (Stability of Amount)
        mean_income = self.credits['amount'].mean()
        std_dev = self.credits['amount'].std()
        # CV: Coefficient of Variation. > 0.3 usually means irregular income
        volatility_score = (std_dev / mean_income) if mean_income > 0 else 0
        
        income_type = "GIG/VARIABLE" if volatility_score > 0.3 else "SALARIED/FIXED"

        # 2. Major Credit Range (When do they usually get paid?)
        # We group by day of month and find the day with max incoming money
        pay_day_stats = self.credits.groupby('day')['amount'].sum().idxmax()
        
        return {
            "income_type": income_type,
            "income_volatility_score": round(volatility_score, 2), # Higher = More erratic
            "avg_income_per_credit": round(mean_income, 2),
            "likely_payday_date": f"Around day {pay_day_stats} of the month"
        }

    def _detect_recurring_and_fixed(self):
        """Identifies Fixed Costs (EMIs, Rent) vs Variable."""
        # Heuristic: Recurring payments often have specific keywords or exact repeating amounts
        keywords = ['EMI', 'LOAN', 'RENT', 'INSURANCE', 'NETFLIX', 'SIP']
        
        # Filter debits that match keywords
        fixed_ops = self.debits[self.debits['narration'].str.contains('|'.join(keywords), case=False, na=False)]
        
        # Also check for exact duplicate amounts (often subscription fees)
        dup_amounts = self.debits[self.debits.duplicated(subset=['amount'], keep=False)]
        
        # Combine unique identified fixed costs
        combined = pd.concat([fixed_ops, dup_amounts]).drop_duplicates()
        fixed_total = combined['amount'].sum()
        
        return {
            "estimated_fixed_monthly_cost": round(fixed_total, 2),
            "fixed_vs_variable_ratio": round(len(combined) / len(self.debits), 2),
            "identified_recurring_items": combined['narration'].unique().tolist()[:3] # Show top 3
        }

    def _analyze_spending_behavior(self):
        """Analyzes 'Timeout' gap and Major Debit timing."""
        # 1. Major Debit Timing
        # On which day of the month is the most money spent?
        spend_day_stats = self.debits.groupby('day')['amount'].sum().idxmax()
        
        # 2. Timeout (Gap between Income and Spending)
        # This calculates "How long does money sit in the account?"
        # Logic: Find largest credit, then find first subsequent debit > 10% of that credit
        max_credit = self.credits.nlargest(1, 'amount')
        timeout_days = "N/A"
        
        if not max_credit.empty:
            credit_time = max_credit.iloc[0]['timestamp']
            credit_amt = max_credit.iloc[0]['amount']
            
            # Find debits AFTER this credit
            future_debits = self.debits[
                (self.debits['timestamp'] > credit_time) & 
                (self.debits['amount'] > (credit_amt * 0.10)) # Looking for big spend
            ]
            
            if not future_debits.empty:
                first_big_spend = future_debits.iloc[0]['timestamp']
                timeout = (first_big_spend - credit_time).total_seconds() / 3600 # in hours
                timeout_days = round(timeout / 24, 1) # Convert to days

        return {
            "major_spending_date": f"Around day {spend_day_stats}",
            "money_retention_gap_days": timeout_days, # Low number = Living paycheck to paycheck
            "spending_velocity": "HIGH (Impulsive)" if isinstance(timeout_days, float) and timeout_days < 2 else "MODERATE"
        }

    def _detect_risks_and_sharks(self):
        """Identifies suspicious patterns and loan traps."""
        # 1. Cash Velocity (ATM Usage)
        # High cash usage is a red flag for credit scoring and shark dependency
        atm_txns = self.debits[self.debits['mode'] == 'ATM']
        total_debit = self.debits['amount'].sum()
        cash_amt = atm_txns['amount'].sum()
        
        cash_ratio = (cash_amt / total_debit * 100) if total_debit > 0 else 0
        
        # 2. Shark Keywords in Credits
        shark_keywords = ['FINANCE', 'LENDING', 'PVT', 'CREDIT']
        sus_credits = self.credits[self.credits['narration'].str.contains('|'.join(shark_keywords), case=False, na=False)]
        
        risk_level = "LOW"
        if cash_ratio > 50: risk_level = "HIGH (Cash Trap)"
        if not sus_credits.empty: risk_level = "CRITICAL (Loan Sharks Detected)"

        return {
            "risk_level": risk_level,
            "cash_reliance_percent": round(cash_ratio, 1),
            "suspicious_loan_credits_count": len(sus_credits)
        }

    def _calculate_savings_potential(self):
        """Calculates Investable Surplus."""
        burn_data = self._get_burn_metrics()
        daily_burn = burn_data['daily_burn_rate']
        
        # Logic: You need 15 days of emergency fund. Rest is investable.
        emergency_buffer = daily_burn * 15
        investable_amount = max(0, self.current_balance - emergency_buffer)
        
        suggestion = "Keep saving."
        if investable_amount > 500:
            suggestion = "Consider Liquid Funds (Safe, Withdraw anytime)"
        if investable_amount > 5000:
            suggestion = "Consider Index Funds (Long term growth)"

        return {
            "emergency_buffer_needed": round(emergency_buffer, 2),
            "investable_surplus": round(investable_amount, 2),
            "investment_suggestion": suggestion
        }

    def get_financial_dna(self):
        """Aggregates all analysis into a single DNA Dictionary."""
        logger.info("Generating Financial DNA...")
        
        return {
            "survival_metrics": self._get_burn_metrics(),
            "income_profile": self._analyze_income_pattern(),
            "recurring_commitments": self._detect_recurring_and_fixed(),
            "spending_behavior": self._analyze_spending_behavior(),
            "risk_assessment": self._detect_risks_and_sharks(),
            "growth_potential": self._calculate_savings_potential()
        }

# --- MAIN EXECUTION BLOCK (Usage Example) ---
if __name__ == "__main__":
    # SAMPLE DATA (Pasted from your prompt for testing)
    sample_json_str = """
    {
        "Account": {
            "Summary": { "currentBalance": "761.41" },
            "Transactions": {
                "Transaction": [
                    { "type": "DEBIT", "mode": "CARD", "amount": "100.0", "transactionTimestamp": "2023-06-27T09:40:19", "narration": "FUEL STATION" },
                    { "type": "DEBIT", "mode": "CARD", "amount": "170.0", "transactionTimestamp": "2023-06-28T09:51:57", "narration": "FUEL STATION" },
                    { "type": "DEBIT", "mode": "CARD", "amount": "500.0", "transactionTimestamp": "2023-07-26T10:04:00", "narration": "FUEL STATION" },
                    { "type": "CREDIT", "mode": "OTHERS", "amount": "15.0", "transactionTimestamp": "2023-08-06T11:10:38", "narration": "Interest" },
                    { "type": "DEBIT", "mode": "ATM", "amount": "1000.0", "transactionTimestamp": "2023-08-07T17:13:13", "narration": "ATM CASH" },
                    { "type": "DEBIT", "mode": "UPI", "amount": "1.0", "transactionTimestamp": "2023-08-22T08:05:06", "narration": "UPI PAYMENT" },
                    { "type": "CREDIT", "mode": "UPI", "amount": "15000.0", "transactionTimestamp": "2023-08-01T10:00:00", "narration": "SALARY CREDIT" },
                    { "type": "DEBIT", "mode": "UPI", "amount": "8000.0", "transactionTimestamp": "2023-08-02T11:00:00", "narration": "LOAN EMI" }
                ]
            }
        }
    }
    """
    
    # 1. Load Data
    data = json.loads(sample_json_str)
    
    # 2. Initialize Analyzer
    analyzer = FinancialAnalyzer(data)
    
    # 3. Get Full Report
    dna_report = analyzer.get_financial_dna()
    
    # 4. Pretty Print Output
    print("\n" + "="*50)
    print(" FINANCIAL DNA REPORT (Ready for AI Agent)")
    print("="*50)
    print(json.dumps(dna_report, indent=4))
    
    # 5. Save JSON to file in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"financial_dna_report_{timestamp}.json"
    output_path = os.path.join(script_dir, output_filename)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(dna_report, f, indent=4)
        logger.info(f"Financial DNA report saved to: {output_path}")
        print(f"\n📁 Report saved as: {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    # 6. Access specific points easily
    print("\n" + "-"*20)
    print(f"Runway: {dna_report['survival_metrics']['current_runway_days']} Days")
    print(f"Risk Level: {dna_report['risk_assessment']['risk_level']}")
    print(f"Investable Amount: ₹{dna_report['growth_potential']['investable_surplus']}")