import pandas as pd

df = pd.read_csv("data/raw/hotel_bookings.csv")


def rate(mask, label):
    s = df[mask]
    if len(s) == 0:
        print(f"  {label:56s} n=0")
        return
    rate_val = s["is_canceled"].mean()
    print(f"  {label:56s} n={len(s):6d}  rate={rate_val:.4f}")


print("=== actual cancellation rates for the test booking profile ===")
rate(
    (df.hotel == "Resort Hotel")
    & (df.country == "GBR")
    & (df.market_segment == "Online TA")
    & (df.deposit_type == "No Deposit")
    & (df.lead_time > 300),
    "Resort/GBR/OnlineTA/NoDep/lead>300",
)
rate(
    (df.hotel == "Resort Hotel") & (df.country == "GBR") & (df.deposit_type == "No Deposit"),
    "Resort/GBR/NoDeposit",
)
rate(
    (df.hotel == "Resort Hotel")
    & (df.market_segment == "Online TA")
    & (df.deposit_type == "No Deposit")
    & (df.lead_time > 300),
    "Resort/OnlineTA/NoDep/lead>300",
)
rate(
    (df.country == "GBR") & (df.deposit_type == "No Deposit") & (df.lead_time > 300),
    "GBR/NoDep/lead>300",
)
rate((df.deposit_type == "No Deposit") & (df.lead_time > 300), "NoDep/lead>300")
rate(
    (df.booking_changes >= 3) & (df.deposit_type == "No Deposit"),
    "booking_changes>=3/NoDep",
)
rate((df.customer_type == "Transient") & (df.deposit_type == "No Deposit"), "Transient/NoDep")
