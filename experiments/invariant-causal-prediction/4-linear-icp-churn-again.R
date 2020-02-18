library(tidyverse)
library(InvariantCausalPrediction)
library(MLmetrics)

churn_df = read_csv('../../data/churn.csv')
head(churn_df)

# First, we define some environments. We'll do that with demographics.
# Our assumption here is that demographics do not cause churn, or cause it
# only through our listed potential causes, i.e. there is nothing confounding
# churn and demographics.
# Of course, all assumptions should be questionned.
# This makes expressing them explicitly more important.

# Then, we do some binary and one-hot encoding, as a practical matter to
# prepare the data set for learning.

df <- churn_df %>%
  # defining environments
  mutate(
    Partner = recode(Partner, "Yes" = "Partnered", "No" = "Single"),
    Dependents = recode(Dependents, "Yes" = "Dependents", "No" = "NoDependents"),
    SeniorCitizen = recode(SeniorCitizen, `1` = "Senior", `0` = "Junior"),
    env = paste(gender, Partner, Dependents, SeniorCitizen, sep = "-")
  ) %>%
  # binary encoding
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
    PaymentMailedCheck = recode(PaymentMethod, "Electronic check" = 1, .default = 0),
    PaymentCreditCard = recode(PaymentMethod, "Credit card (automatic)" = 1, .default = 0),
    PaymentBankTransfer = recode(PaymentMethod, "Bank transfer (automatic)" = 1, .default = 0),
    Churn = recode(Churn, "Yes" = 1, "No" = 0)
  ) %>% select (
    -one_of(c('InternetService', 'Contract', 'PaymentMethod'))
  )

# Now we hold out some environments to test on later.
# Let's imagine the business case that the telco has previously targeted single
# people, and hold out Dependents = 1 and Partner = 1 for testing.

holdout <- df %>% filter((Dependents == "Dependents") & (Partner == "Partnered"))
train <- df %>% filter((Dependents == "NoDependents") | (Partner == "Single"))

train_X <- train %>%
  select(
    PhoneService,
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

train_Y <- train$Churn

train_E <- as.factor(train$env)

churn_icp <- ICP(
  X = train_X,
  Y = train_Y,
  ExpInd = train_E
)

churn_icp
plot(churn_icp)

# Here, we throw in all the variables, as we usually do in ML.
# Of course, we might use some variable selection technique too, focussed on
# predictive power in the train set (but not accounting for environments).
correlative_model <- glm(
  Churn ~ PhoneService + MultipleLines + DSL + FiberOptic + OnlineSecurity +
    OnlineBackup + DeviceProtection + TechSupport + StreamingTV +
    StreamingMovies + MonthlyBilling + OneYearBilling + TwoYearBilling +
    PaperlessBilling + PaymentElectronicCheck + PaymentMailedCheck +
    PaymentCreditCard + PaymentBankTransfer,
  data = train,
  family = binomial(link = "logit")
  )

# Here we include only those predictors that have an effect with non-1.0 p-value
# (ie. an effect at _any_ significance, which is very permissive) under the ICP
# procedure.
invariant_model <- glm(
  Churn ~ PhoneService + DSL + FiberOptic + OnlineSecurity +
    DeviceProtection + TechSupport + StreamingTV + StreamingMovies +
    MonthlyBilling  + MonthlyCharges + tenure,
  data = train,
  family = binomial(link = "logit")
)

correlative_predictions <- ifelse(predict(correlative_model, newdata = holdout, type = "response") < 0.5, 0, 1)
correlative_predictions

invariant_predictions <- ifelse(predict(invariant_model, newdata = holdout, type = "response") < 0.5, 0, 1)
invariant_predictions

# training AUC
AUC(train$Churn, ifelse(correlative_model$fitted.values < 0.5, 0, 1))
AUC(train$Churn, ifelse(invariant_model$fitted.values < 0.5, 0, 1))

# test AUC
AUC(holdout$Churn, correlative_predictions)
AUC(holdout$Churn, invariant_predictions)
