setwd("~/Desktop/impala_data/")

files<-list.files("Ti64_Flyer_TrainingSets/Ti64_Church2002_Manganin_591_612/Results/")

files<-files[which(substr(files, nchar(files)-6+1, nchar(files))=="Sigma1")]
ff<-do.call("rbind",strsplit(files,'.',fixed=T))[,1]
nn<-as.numeric(substr(ff,nchar(ff)-4+1, nchar(ff)))

datx<-daty<-matrix(nrow=length(files),ncol=8000)
for(i in 1:nrow(dat)){
  dd<-as.matrix(read.csv(paste0("Ti64_Flyer_TrainingSets/Ti64_Church2002_Manganin_591_612/Results/",files[i])))[1:8000,2:3]
  datx[i,]<-dd[,1]
  daty[i,]<-dd[,2]
}

matplot(t(datx)[seq(1,8000,10),],t(daty)[seq(1,8000,10),],type='l')




files<-list.files("TaylorTi64_Partial/taylor_mcd2007_fig4/Results/")

ff<-do.call("rbind",strsplit(files,'.',fixed=T))[,1]
nn<-as.numeric(substr(ff,nchar(ff)-4+1, nchar(ff)))

datxt<-datyt<-matrix(nrow=length(files),ncol=781)
for(i in 1:nrow(datxt)){
  dd<-as.matrix(read.csv(paste0("TaylorTi64_Partial/taylor_mcd2007_fig4/Results/",files[i])))[1:781,]
  datxt[i,]<-dd[,1]
  datyt[i,]<-dd[,2]
}

matplot(t(datxt),t(datyt),type='l')

yy<-apply(datyt,1,max)


x<-read.table('TaylorTi64_Partial/Ti64.design.ptw.1000.txt',head=T)

library(BASS)
mod<-bass(x[nn,],yy)
plot(mod)

plot(sobol(mod))
