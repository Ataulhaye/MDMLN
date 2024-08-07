#install.packages("mice")

# library to read matlab data formats into R
# install.packages("R.matlab")
library(R.matlab)
library(mice)

stg <- readMat("C:/Users/ataul/source/Uni/BachelorThesis/poc/fmri_data/left_STG_MTG_AALlable_ROI.rex.mat")
ifg <- readMat("C:/Users/ataul/source/Uni/BachelorThesis/poc/fmri_data/ROI_aal_wfupick_left44_45.rex.mat")
# check out data structure
stgm <- stg$R
str(stg)

ifgm <- ifg$R

summary(stg)
str(stg)
md.pattern(stgm)

rows = head(ifgm, 5)

testmatrix <- matrix(c(rows), nrow = 5, ncol = 523)

dataframetest = as.data.frame(ifgm) 

# Multiple Imputation mit dem mice Paket
#=========================================

ifg_imput <- mice(data = testmatrix)

pred_mat <- quickpred(dataframetest)
ifg_frameImpute <- mice(data = dataframetest, predictorMatrix=pred_mat, method = "norm")

ifg_imput <- mice(data = testmatrix)
long <- complete(ifg_imput, "long")
# Prüfen auf Konvergenz der Schätzung
plot(ifg_imput)
plot(long)
# Prüfen, ob Imputation gültige Werte ergibt
stripplot(ifg_imput)
stripplot(ifg_imput)

ifg_imput <- mice(data = ifgm, m = 2, maxit = 5)

# 1. Imputation
imp.data1 <- mice(data = stgm, m = 5, maxit = 2, seed = 12345)

a <- mice(data = stgm, m = 1, maxit = 1)
b <- mice(data = stgm, m = 50, maxit = 10)

imp.data <- mice(data = stgm, m = 50, maxit = 10, seed = 12345)

# Which Methoden wurden zum Imputieren genutzt?
imp.data

#=====================================================================
# (Wenn man sich die vervollständigten Datensätze ansehen möchte:




#' "long"' produces a data set where imputed data sets are stacked vertically.
#' The columns are added: 1) '.imp', integer, referring the imputation number, and 2) '.id', character, the row names of 'data$data'
long <- complete(b, "long")
#' "stacked"' same as '"long"' but without the two additional columns;
stacked <- complete(b, "stacked")
#' "broad"' produces a data set with where imputed data sets are stacked horizontally. Columns are ordered as in the original data. The imputation number is appended to each column name;
broad <- complete(b, "broad")

long
imp.datasets <- complete(imp.data, "long")
imp.datasets
#)

#=====================================================================

# Prüfen auf Konvergenz der Schätzung
plot(imp.data)

# Prüfen, ob Imputation gültige Werte ergibt
stripplot(imp.data)



