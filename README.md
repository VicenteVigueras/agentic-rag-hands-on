# Google ADK + Semantic Search 

- I want to see how useful RAG can actually be, and under which circumstances.
- But the real-life goal is how can I achieve accurate interaction with my data in natural language.

Useful commands:
- pkill ollama
- ollama serve
- ollama run llama3

## Comparison Results

Output from running `python compare_agents.py`:

```
Question: What columns store information about when a customer made a purchase?

--------------------------------------------------------------------------------
1. SENTENCE TRANSFORMERS ALONE (RAG Retrieval)
--------------------------------------------------------------------------------
Relevant database columns:
- order_date (date): order_date column storing Date when the customer completed the purchase
- delivery_date (date): delivery_date column storing Date when the order was delivered to the customer
- return_date (date): return_date column storing Date when the customer returned the product

--------------------------------------------------------------------------------
2. AGENT WITHOUT RAG CONTEXT
--------------------------------------------------------------------------------
Response:
I'd be happy to help!

As an e-commerce platform, the columns that typically store information about when a customer made a purchase are:

* Order Date: This column stores the date and time when the order was placed.
* Purchase Date: This column stores the date and time when the order was fulfilled or shipped.
* Last Activity Date: This column stores the date and time of the last interaction with the customer, which could be an update to their order status.

Please note that the specific column names may vary depending on the e-commerce platform or database management system being used.

--------------------------------------------------------------------------------
3. AGENT WITH RAG CONTEXT
--------------------------------------------------------------------------------
Response:
Based on the provided database context, I can answer your question!

According to the information you've shared, the columns that store information about when a customer made a purchase are:

* `order_date`: This column stores the date when the customer completed the purchase.

Let me know if you have any further questions or concerns!

================================================================================
COMPARISON COMPLETE
================================================================================

Notice:
- Sentence Transformers alone just lists matching columns
- Agent without context gives generic advice
- Agent WITH context leverages the retrieved data for intelligent answers
```
