library(tidyverse)
library(InvariantCausalPrediction)
library(MLmetrics)

churn_df <- read_csv('../../data/churn.csv')
churn_df %>% head()

df <- churn_df %>%
  mutate(
    PhoneService = recode(PhoneService, "Yes" = 1, "No" = 0),
    MultipleLines = recode(MultipleLines, "Yes" = 1, "No" = 0, "No phone service" = 0),
    DSL = recode(InternetService, "DSL" = 1, "No" = 0, "Fiber optic" = 0),
    FiberOptic = recode(InternetService, "Fiber optic" = 1, "No" = 0, "DSL" = 0),
    OnlineSecurity = recode(OnlineSecurity, "Yes" = 1, "No" = 0, "No internet service" = 0),
    OnlineBackup = recode(OnlineBackup, "Yes" = 1, "No" = 0, "No internet service" = 0),
    DeviceProtection = recode(DeviceProtection, "Yes" = 1, "No" = 0, "No internet service" = 0),
    TechSupport = recode(TechSupport, "Yes" = 1, "No" = 0, "No internet service" = 0),
    StreamingTV = recode(StreamingTV, "Yes" = 1, "No" = 0, "No internet service" = 0),
    StreamingMovies = recode(StreamingMovies, "Yes" = 1, "No" = 0, "No internet service" = 0),
    MonthlyBilling = recode(Contract, "Month-to-month" = 1, .default = 0),
    OneYearBilling = recode(Contract, "One year" = 1, .default = 0),
    TwoYearBilling = recode(Contract, "Two year" = 1, .default = 0),
    PaperlessBilling = recode(PaperlessBilling,  "Yes" = 1, "No" = 0),
    PaymentElectronicCheck = recode(PaymentMethod, "Electronic check" = 1, .default = 0),
    PaymentMailedCheck = recode(PaymentMethod, "Mailed check" = 1, .default = 0),
    PaymentCreditCard = recode(PaymentMethod, "Credit card (automatic)" = 1, .default = 0),
    PaymentBankTransfer = recode(PaymentMethod, "Bank transfer (automatic)" = 1, .default = 0),
    Churn = recode(Churn, "Yes" = 1, "No" = 0)
  ) %>% select (
    -one_of(c('InternetService', 'Contract', 'PaymentMethod'))
  )

train_df <- df %>% filter(Partner == 'No' | Dependents == 'No')

test_df <- df %>% filter(Partner == 'Yes' & Dependents == 'Yes')

train_X <- train_df %>% select(
  PhoneService,
  MultipleLines,
  DSL,
  FiberOptic,
  OnlineSecurity,
  OnlineBackup,
  DeviceProtection,
  TechSupport,
  StreamingTV,
  StreamingMovies,
  MonthlyBilling,
  OneYearBilling,
  TwoYearBilling,
  MonthlyCharges,
  tenure
) %>% as.matrix()

train_Y <- train_df$Churn

train_E <- train_df %>%
  mutate(env = as.factor(paste0(Partner, Dependents))) %>%
  select(env) %>% as_vector()

churn_icp <- ICP(
  X = train_X,
  Y = train_Y,
  ExpInd = train_E
)

plot(churn_icp)

correlative_model <- glm(
  Churn ~ PhoneService + MultipleLines + DSL + FiberOptic + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies + MonthlyBilling + OneYearBilling + TwoYearBilling + MonthlyCharges + tenure - 1,
  data = train_df,
  family = binomial(link='logit'))

causal_model <- glm(
  Churn ~ PhoneService + MultipleLines + FiberOptic + OnlineSecurity + TechSupport + MonthlyBilling + MonthlyCharges + tenure - 1,
  data = train_df,
  family = binomial
)

correlative_predictions <- predict(correlative_model, newdata = test_df, type = "response")
causal_predictions <- predict(causal_model, newdata = test_df, type = "response")

AUC(correlative_predictions, test_df$Churn)
AUC(causal_predictions, test_df$Churn)

Accuracy(ifelse(correlative_predictions > 0.5, 1, 0), test_df$Churn)
Accuracy(ifelse(causal_predictions > 0.5, 1, 0), test_df$Churn)
Accuracy(ifelse(correlative_predictions < 0, 1, 0), test_df$Churn)

LogLoss(correlative_predictions, test_df$Churn)
LogLoss(causal_predictions, test_df$Churn)
