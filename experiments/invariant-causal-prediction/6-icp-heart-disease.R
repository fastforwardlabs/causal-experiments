library(tidyverse)
library(fastDummies)
library(InvariantCausalPrediction)

# hd = heart disease
cols <- c(
  "age",
  "sex",
  "chest_pain",
  "blood_pressure",
  "cholesterol",
  "blood_sugar",
  "ecg",
  "heart_rate",
  "angina",
  "st_depression",
  "peak_st_segment",
  "major_vessels",
  "thalassemia",
  "diagnosis")

hd <- read_csv('../../data/processed.cleveland.data', col_names = cols) %>%
  mutate(
    diagnosis = fct_lump(as.factor(diagnosis), other_level = "1"),
    sex = as.factor(sex),
    chest_pain = as.factor(chest_pain),
    blood_sugar = as.factor(blood_sugar),
    ecg = as.factor(ecg),
    angina = as.factor(angina),
    peak_st_segment = as.factor(peak_st_segment),
    major_vessels = as.factor(major_vessels),
    thalassemia = as.factor(thalassemia)
  )

X <- hd %>% select(-c(sex, diagnosis)) %>%
  dummy_cols(remove_selected_columns = TRUE,
             remove_first_dummy = TRUE)

Y <- hd$diagnosis

ExpInd <- hd$sex

icp <- ICP(as.matrix(X), Y, ExpInd)
icp

plot(icp)
