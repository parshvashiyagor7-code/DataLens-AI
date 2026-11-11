# DataLens AI — Phase 1 Test Dataset

## Files
- `Orders.csv` — 800 order-line records
- `Customers.csv` — 120 customer records
- `Products.csv` — 48 product records

## Relationships
- `Orders.Customer_ID` → `Customers.Customer_ID` (many orders to one customer)
- `Orders.Product_ID` → `Products.Product_ID` (many orders to one product)
- Each Orders row represents one order line and has a unique `Order_ID`.
- `Sales_Amount = Gross_Amount - Discount_Amount`; `Gross_Amount = Quantity × Unit_Price`.
- Dates range from 2025-01-01 through 2026-10-31.
- This is clean baseline data: IDs are valid, required fields are populated, and no intentional duplicates/missing values are inserted.

## Upload instructions
1. Extract this ZIP.
2. In DataLens AI → Data Sources, upload all three CSV files together.
3. Open Ask Your Data and run the questions below one at a time.
4. Open Reports & History to verify the query and result snapshot are retained in the current session.

## Test questions

### 01. How many orders are there?
- Tables: Orders only
- Check: Expected: 800 rows / 800 unique Order_ID values.

### 02. What is the total sales amount?
- Tables: Orders only
- Check: Expected: sum of Orders.Sales_Amount.

### 03. Which payment method is used most often?
- Tables: Orders only
- Check: Expected: group by Payment_Method and count Order_ID.

### 04. How many orders are there for each order status?
- Tables: Orders only
- Check: Expected: group by Order_Status and count Order_ID; good pie-chart candidate.

### 05. What is the monthly sales trend?
- Tables: Orders only
- Check: Expected: group Sales_Amount by Order_Date month, chronologically; good line-chart candidate.

### 06. Which product category has the highest total sales?
- Tables: Orders + Products
- Check: Expected: join on Product_ID, group by Category, sum Sales_Amount.

### 07. 
- Tables: Orders + Products
- Check: Expected: join on Product_ID, group by Product_Name, sum Sales_Amount, descending, limit 5; good bar-chart candidate.

### 08. Which region generates the highest total sales?
- Tables: Orders + CustomersWhat are the top     5 products by total sales?
- Check: Expected: join on Customer_ID, group by Region, sum Sales_Amount.

### 09. Which customer segment has the highest total sales?
- Tables: Orders + Customers
- Check: Expected: join on Customer_ID, group by Segment, sum Sales_Amount.

### 10. Which category has the highest average quantity per order?
- Tables: Orders + Products
- Check: Expected: join on Product_ID, group by Category, mean Quantity.

### 11. Show the 10 orders with the highest Sales_Amount.
- Tables: Orders only
- Check: Expected: 10 rows sorted by Sales_Amount descending.

### 12. Which 5 customers generated the most sales?
- Tables: Orders + Customers
- Check: Expected: join on Customer_ID, group by Customer_Name, sum Sales_Amount, descending, limit 5.

### 13. Compare total sales across product categories and customer segments.
- Tables: Orders + Products + Customers
- Check: Expected: multi-table join; group by Category and Segment. Advanced join test.

### 14. Which region and product category combination generated the highest sales?
- Tables: Orders + Customers + Products
- Check: Expected: multi-table join; group by Region and Category. Advanced join test.

## Baseline checks
- Orders: 800 rows, 800 unique Order_IDs.
- Customers: 120 rows, unique Customer_IDs.
- Products: 48 rows, unique Product_IDs.
- All Orders.Customer_ID values exist in Customers.
- All Orders.Product_ID values exist in Products.
- Sales_Amount equals Gross_Amount minus Discount_Amount for every row.

## Notes
Synthetic data created for application testing. Names, orders, products, prices, and metrics are fictional and should not be treated as real business data.
