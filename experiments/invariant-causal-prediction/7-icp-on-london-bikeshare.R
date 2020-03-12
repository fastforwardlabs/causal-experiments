library(tidyverse)
library(fastDummies)
library(nonlinearICP)
library(CondIndTests)

bikes <- read_csv('../../data/london-bikes.csv') %>%
  mutate(
    season = as.factor(season),
    weather_code = as.factor(weather_code),
    hr = as.numeric(format(timestamp, '%H'))
  ) %>%
  sample_n(5000)

X <- bikes %>%
  select(-c(timestamp, cnt, season)) %>%
  dummy_columns(
    remove_selected_columns = TRUE,
    remove_first_dummy = TRUE
  ) %>% as.matrix()

Y <- bikes$cnt

ExpInd <- bikes$season

icp <- nonlinearICP(
  X, Y, ExpInd,
  condIndTest = InvariantEnvironmentPrediction,
  argsCondIndTest = c(ntree=3)
)

icp$definingSets
icp$acceptedSets

