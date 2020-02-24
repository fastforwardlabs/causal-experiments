library(tidyverse)
library(nonlinearICP)
library(InvariantCausalPrediction)

bike_df <- read_csv('../../data/Bike-Sharing-Dataset/day.csv') %>% filter(yr = 0)
head(bike_df)

X <- bike_df %>%
  mutate(
    season = as.factor(season),
    weekday = as.factor(weekday),
    clearweather = as.factor(weathersit)
  ) %>%
  select(
#    mnth,
#    holiday,
#    weekday,
#    workingday,
#    weathersit,
    temp,
    atemp,
    hum,
    windspeed
  ) %>% as.matrix()

Y <- bike_df$cnt

E <- as.factor(bike_df$season)

bike_nicp <- nonlinearICP(
  X = X,
  Y = Y,
  environment = E,
  verbose = TRUE
)

bike_icp <- ICP(
  X = X,
  Y = Y,
  ExpInd = E
)

bike_icp$pvalues

bike_df %>% ggplot() + geom_line(aes(x=dteday, y=cnt))
