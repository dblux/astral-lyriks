library(limma)
library(sva)
source('R/subset.R')

### Load data ###

file <- 'data/astral/reprocessed-data-renamed.csv'
data <- read.csv(file, row.names = 1)
data[data == 0] <- NA

lyriks <- data[, startsWith(colnames(data), 'L')] 
lyriks_full <- log2(na.omit(lyriks))

file <- 'data/astral/metadata-psy_602_16-v1.csv'
metadata <- read.csv(file, row.names = 1)

### Preprocess data ###

# Subset data
outliers_sn <- c('L0626C', 'L0018C')
# Psychosis signature (cvt, mnt, ctrl)
cmc <- subset_cols(
  lyriks_full,
  metadata,
  group %in% c("Convert", "Maintain", "Healthy control") &
  !(sn %in% outliers_sn)
)

metadata_cmc <- metadata[colnames(cmc), ]
# Relevel with convert batch as reference
metadata_cmc$extraction_date <- factor(metadata_cmc$extraction_date)
metadata_cmc$extraction_date <- relevel(
  metadata_cmc$extraction_date,
  ref = '5/9/24'
)
# Convert to days since earliest and center
metadata_cmc$run_datetime <- as.POSIXct(
  metadata_cmc$run_datetime,
  format = "%d/%m/%y %H:%M",
  tz = "UTC"
)
metadata_cmc$run_datetime_days <- as.numeric(difftime(
  metadata_cmc$run_datetime,
  min(metadata_cmc$run_datetime, na.rm = TRUE),
  units = "days"
))
metadata_cmc$run_datetime_centered <- 
  metadata_cmc$run_datetime_days - mean(metadata_cmc$run_datetime_days)
mod <- model.matrix(~group, data = metadata_cmc)


# Limma correction (linear model, no Bayesian shrinkage)
# Assumption: No interaction between batch effects due to run_datetime and extraction_date
limma_cmc <- removeBatchEffect(
  cmc,
  batch = metadata_cmc$extraction_date,
  covariates = metadata_cmc$rundatetime_centered,
  design = model.matrix(~ group, metadata_cmc)
)

file <- 'data/tmp/cmc_265_306-limma.csv'
write.csv(limma_cmc, file)

# ComBat correction
combat_cmc <- ComBat(
  cmc_102,
  batch = metadata_cmc$extraction_date,
  mod = mod,
  par.prior = TRUE,
  ref.batch = '5/9/24'
)
dim(cmc_102)

file <- 'data/tmp/cmc_265_312-combat.csv'
write.csv(combat_cmc, file)
