getLongDX <- function(subjects, adni_data) {
  dx.codes <- c("CN", "MCI", "Dementia")
  rids <- unique(subjects$RID)
  dx.groups <- rep(NA, length(rids))
  i = 1
  for (rid in rids) {
    #print(rid)
    dxs <- adni_data$DX[which(adni_data$RID == rid)]
    dxs <- na.omit(dxs)
    if (length(dxs) == 0) {
      print(sprintf("No DX data for %s", rid))
    } else if ( dxs[1] != dxs[length(dxs)] & which(dx.codes == dxs[1]) < which(dx.codes == dxs[length(dxs)]) ) {
      dx.groups[i] <- sprintf("Transition %s to %s", dxs[1], dxs[length(dxs)])
    } else {
      dx.groups[i] <- sprintf("%s stable", dxs[1])
    }
    i = i + 1
  }
  
  # Add patient labels to the data frame
  subjects <- subjects %>%
    mutate(DX = dx.groups[match(RID, rids)])
  return(subjects)
}