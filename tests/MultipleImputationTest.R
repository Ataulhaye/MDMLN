#install.packages("mice")

library(mice)
mice(nhanes2)


# Verwendung des Datensatzes nhanes2 aus dem mice Package
#nhanes2
summary(nhanes2)
str(nhanes2)
md.pattern(nhanes2)

# Multiple Imputation mit dem mice Paket
#=========================================

# 1. Imputation

imp.data <- mice (data = nhanes2, m = 50, maxit = 10, seed = 12345, print=FALSE)

# Which Methoden wurden zum Imputieren genutzt?
imp.data

#=====================================================================
# (Wenn man sich die vervollständigten Datensätze ansehen möchte:
imp.datasets <- complete(imp.data, "long")
imp.datasets
#)

#=====================================================================

# Prüfen auf Konvergenz der Schätzung
plot(imp.data)

# Prüfen, ob Imputation gültige Werte ergibt
stripplot(imp.data)

