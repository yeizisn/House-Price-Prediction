rm(list = ls())
traindata <- read.csv("train.csv",header = TRUE)
print(traindata["SalePrice"])
a = attributes(traindata)
col = a['names']

for (i in col$names) {
  if (i != "id" & i != "SalePrice") {
    setwd("/Users/Song/Downloads/house_price/picture")
    future = paste(i,".jpg")
    jpeg(file = future)
    plot(traindata[[i]],traindata[["SalePrice"]],main = paste(i,'& SalePrice'),xlab = i,ylab = "SalePrice")
    dev.off()
  }
}
setwd("/Users/Song/Downloads/house_price/")

