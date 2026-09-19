# Data

`Streaming.csv` is a streaming-service customer dataset obtained from Kaggle. It is small, so it
is included directly in this folder and the notebook runs without any extra download.

## Columns

`Customer_ID`, `Age`, `Gender`, `Subscription_Length`, `Region`, `Payment_Method`,
`Support_Tickets_Raised`, `Satisfaction_Score`, `Discount_Offered`, `Last_Activity`,
`Monthly_Spend` (regression target), and `Churned` (classification target, 1 = churned).

## Cleaning applied in the notebook

Age and Satisfaction_Score have missing values, filled with KDE imputation after a two-sample
t-test confirmed the missingness was unrelated to spend. Two rows with negative Monthly_Spend are
dropped rather than imputed, since spend is the target.

The dataset is provided for coursework and is used here only for that purpose.
