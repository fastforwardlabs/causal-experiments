library(tidyverse)
library(InvariantCausalPrediction)

churn_df <- read_csv('../../data/churn.csv')
churn_df %>% head()

df <- churn_df %>%
  mutate(
#    gender = as.factor(gender), #recode(gender, "Female" = 1, "Male" = 0),
#    SeniorCitizen = as.factor(SeniorCitizen),
#    Partner = as.factor(Partner), #recode(Partner, "Yes" = 1, "No" = 0),
#    Dependents = as.factor(Dependents), #recode(Dependents, "Yes" = 1, "No" = 0),
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

X <- df %>% select(
  PhoneService,
  #MultipleLines,
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
  #PaperlessBilling,
  #PaymentElectronicCheck,
  #PaymentMailedCheck,
  #PaymentCreditCard,
  #PaymentBankTransfer,
  MonthlyCharges,
  tenure
) %>% as.matrix()

Y <- df$Churn

E <- df %>%
  mutate(env = as.factor(paste0(Partner, Dependents))) %>%
  select(env) %>% as_vector()

churn_icp <- ICP(
  X = X,
  Y = Y,
  ExpInd = E
)

plot(churn_icp)

linear_model <- glm(Y ~ X-1, family = binomial(link='logit'))
linear_model

glm(
  Churn ~ PhoneService + DSL + FiberOptic + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + StreamingTV + StreamingMovies + MonthlyBilling + OneYearBilling + TwoYearBilling + MonthlyCharges + tenure - 1,
  data = df,
  family = binomial
)

glm(
  Churn ~ PhoneService + DSL + FiberOptic + OnlineSecurity + TechSupport + MonthlyBilling + MonthlyCharges + tenure - 1,
  data = df,
  family = binomial
)

churn_icp
