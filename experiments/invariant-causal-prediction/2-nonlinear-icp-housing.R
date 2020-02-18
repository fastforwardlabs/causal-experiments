library(tidyverse)
library(nonlinearICP)
library(CondIndTests)

housing_df <- read_csv('../../data/housing.csv') %>% drop_na() %>% sample_n(200)
housing_df %>% head()

X <- housing_df %>%
  select(housing_median_age,
         total_rooms,
         total_bedrooms,
         population,
         households,
         median_income) %>%
  as.matrix()

Y <- housing_df$median_house_value %>% as.vector()

E <- housing_df %>% select(longitude, latitude) %>% as.matrix()

start_time <- Sys.time()
nicp <- nonlinearICP(
  X = X,
  Y = Y,
  environment = E,
  condIndTest = ResidualPredictionTest,
  condIndTestNames = 'RP',
  verbose = TRUE
)
end_time <- Sys.time()

run_time <- end_time - start_time


housing_df %>% ggplot() +
  geom_point(aes(x=longitude, y=latitude))

run_time
nicp$retrievedCausalVars
