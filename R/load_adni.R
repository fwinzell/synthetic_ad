# WARNING: This script uses the old ADNIMERGE package
# For usage it should be updated to use the new ADNIMERGE2

load_adni <- function() {
  library(ADNIMERGE)

  adni_data <- adnimerger()
  
  # Replace ABETA CSF values with original values from UPENNBIOMK9
  upenn9 <- upennbiomk9
  
  rids <- unique(adni_data$RID[which(!is.na(adni_data$ABETA))])
  matched_rids <- unique(upenn9$RID[which(upenn9$RID %in% rids)])
  
  upenn9$ABETA <- as.numeric(as.character(upenn9$ABETA))
  
  na_idxs <- which(is.na(upenn9$ABETA))
  upenn9$ABETA[na_idxs] <- as.numeric(gsub("\\D", "", upenn9$COMMENT))[na_idxs]
  
  upenn <- upenn9[which(upenn9$RID %in% matched_rids), c("RID", "VISCODE", "ABETA")]
  abeta <- merge(adni_data[, c("RID", "VISCODE")], upenn, by=c("RID", "VISCODE"), all.x=TRUE)
  
  adni_data$ABETA <- abeta$ABETA 
  
  # Set to numeric, NAs for string values
  adni_data$ABETA <- as.numeric(adni_data$ABETA)
  adni_data$PTAU <- as.numeric(adni_data$PTAU)
  adni_data$TAU <- as.numeric(adni_data$TAU)
  
  return(adni_data)
}



