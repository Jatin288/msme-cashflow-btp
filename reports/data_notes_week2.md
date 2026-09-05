# Data Notes — Week 2 (EDA)

## Dataset 1: IBM Watson Analytics (WA_Fn-UseC_-Accounts-Receivable.csv)
- 2,466 invoices, zero missing values (unusually clean — likely a curated/tutorial dataset)
- DaysLate: only captures lateness, floored at 0 (min=0, median=0, max=45)
- Right-skewed — consistent with a gamma distribution
- InvoiceAmount: mean (59.90) ≈ median (60.56) — essentially symmetric, NOT right-skewed.
  Lognormal would be the wrong fit here; normal distribution fits better.

## Dataset 2: SAP-style export (dataset.csv)
- 50,000 raw invoices → 46,081 USD → 36,818 after excluding 10,000 open (unpaid) invoices
- Open invoices (isOpen=1) have no clear_date — these are censored observations,
  relevant to RQ1 (detection horizon) later, excluded from delay calculation for now
- days_late genuinely spans negative to positive: min=-89, max=204
- Three-way split: 38.7% early, 22.2% on-time, 39.1% late — NOT a one-sided lateness distribution

## Key implication for the synthetic generator (relevant later, not this week)
A single one-sided distribution (e.g. gamma alone) cannot represent real payment behavior,
since a large fraction of invoices are paid early, not just on-time or late. The generator
needs a two-part model: (1) classify early/on-time/late, (2) model the magnitude within
each category separately. This is a more realistic design than originally assumed, and is
grounded in actual data rather than a modeling convenience.

## Customer-level payment behavior
Customers show consistent, individual payment personalities, not random noise:
- Earliest payer (100054980): mean -24.5 days, std 3.56, n=6 — consistently early, low variance
- Latest payer (CCU013): mean +42.3 days, std 14.68, n=539 — consistently late, large sample confirms this is real behavior, not noise

Implication: the synthetic generator should assign each simulated customer/business relationship
its own payment-delay parameters (not one global distribution shared by everyone), to reflect
that real payment behavior is customer-specific.