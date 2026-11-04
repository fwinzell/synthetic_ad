# WARNING: This script uses the old ADNIMERGE package
# For usage it should be updated to use the new ADNIMERGE2

merge_adni <- function() {
  library(ADNIMERGE)

  adni_data <- adnimerger()

  adni_data$ABETA[which(adni_data$ABETA == "<200")] <- 200
  adni_data$ABETA[which(adni_data$ABETA == ">1700")] <- 1700
  adni_data$PTAU[which(adni_data$PTAU == "<8")] <- 8
  adni_data$PTAU[which(adni_data$PTAU == ">120")] <- 120
  adni_data$TAU[which(adni_data$TAU == "<80")] <- 80
  adni_data$TAU[which(adni_data$TAU == ">1300")] <- 1300
  
  adni_data$ABETA <- as.numeric(adni_data$ABETA)
  adni_data$PTAU <- as.numeric(adni_data$PTAU)
  adni_data$TAU <- as.numeric(adni_data$TAU)
  adni_data$AV45 <- as.numeric(adni_data$AV45)
  
  return(adni_data)
}
