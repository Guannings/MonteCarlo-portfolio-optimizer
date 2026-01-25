import numpy as np
import matplotlib.pyplot as plt


def compare_deposit_frequencies():
    print("--- FEE IMPACT ANALYSIS ($4.79 per Deposit) ---")

    # Parameters
    monthly_savings = 185
    fee = 4.79
    annual_return = 0.10  # Assume 10% average market growth
    years = 10
    months = years * 12
    monthly_rate = annual_return / 12

    # STRATEGY 1: MONTHLY DEPOSIT
    # Invest ($185 - Fee) every single month
    wealth_monthly = 0
    total_fees_monthly = 0

    for i in range(months):
        wealth_monthly = wealth_monthly * (1 + monthly_rate)  # Growth
        investment = monthly_savings - fee
        wealth_monthly += investment
        total_fees_monthly += fee

    # STRATEGY 2: QUARTERLY DEPOSIT (Every 3 Months)
    # Hold cash for 2 months, invest in month 3
    wealth_quarterly = 0
    total_fees_quarterly = 0
    cash_pile = 0

    for i in range(1, months + 1):
        wealth_quarterly = wealth_quarterly * (1 + monthly_rate)  # Existing money grows
        cash_pile += monthly_savings  # Add savings to cash pile

        if i % 3 == 0:  # Every 3rd month
            investment = cash_pile - fee
            wealth_quarterly += investment
            total_fees_quarterly += fee
            cash_pile = 0  # Reset cash

    # STRATEGY 3: SEMI-ANNUAL DEPOSIT (Every 6 Months)
    wealth_semi = 0
    total_fees_semi = 0
    cash_pile = 0

    for i in range(1, months + 1):
        wealth_semi = wealth_semi * (1 + monthly_rate)
        cash_pile += monthly_savings

        if i % 6 == 0:  # Every 6th month
            investment = cash_pile - fee
            wealth_semi += investment
            total_fees_semi += fee
            cash_pile = 0

    print(f"Total Savings (10 Years): ${monthly_savings * months:,.0f}")
    print("-" * 40)
    print(f"1. MONTHLY Investing:")
    print(f"   Final Value:  ${wealth_monthly:,.0f}")
    print(f"   Fees Paid:    ${total_fees_monthly:.0f} (High!)")
    print("-" * 40)
    print(f"2. QUARTERLY Investing:")
    print(f"   Final Value:  ${wealth_quarterly:,.0f}")
    print(f"   Fees Paid:    ${total_fees_quarterly:.0f}")
    print(f"   Difference:   ${wealth_quarterly - wealth_monthly:+.0f} vs Monthly")
    print("-" * 40)
    print(f"3. SEMI-ANNUAL Investing:")
    print(f"   Final Value:  ${wealth_semi:,.0f}")
    print(f"   Fees Paid:    ${total_fees_semi:.0f}")
    print(f"   Difference:   ${wealth_semi - wealth_monthly:+.0f} vs Monthly")
    print("=" * 40)

    # Recommendation Logic
    best_strategy = max(wealth_monthly, wealth_quarterly, wealth_semi)
    if best_strategy == wealth_monthly:
        winner = "MONTHLY"
    elif best_strategy == wealth_quarterly:
        winner = "QUARTERLY"
    else:
        winner = "SEMI-ANNUAL"

    print(f"WINNER: {winner} DEPOSITS")
    print(f"Why? The fee savings outweigh the lost market time.")


compare_deposit_frequencies()