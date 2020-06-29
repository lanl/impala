############################################################
## get data

setwd('~/Desktop/immpala_data/TaylorTrial3/')
files<-list.files('Results/')
runnum<-substr(files,17,20)
use<-which(substr(files,30,30)=='.')
runnum[use]

#daty<-datx<-matrix(nrow=1000,ncol=481)
daty<-datx<-matrix(nrow=1000,ncol=783)
for(i in 1:length(runnum[use])){
  dat.top<-read.table(paste('Results/Cu_Taylor_Allen_',runnum[use][i],'.deformed_top.xyz',sep=''))*10
  dat<-read.table(paste('Results/Cu_Taylor_Allen_',runnum[use][i],'.deformed.xyz',sep=''))*10 # * 10 to make mm
  dat.bot<-read.table(paste('Results/Cu_Taylor_Allen_',runnum[use][i],'.deformed_bot.xyz',sep=''))*10
  #print(nrow(dat))
  dat<-rbind(dat.top,dat[nrow(dat):1,],dat.bot[nrow(dat.bot):1,])
  
  datx[as.numeric(runnum[use][i]),]<-dat[,2]
  daty[as.numeric(runnum[use][i]),]<-dat[,3] - min(dat[,3])
}



inputs<-read.table('OFHC-Cu.design.ptw.1000.txt',head=T)
use.run<-as.numeric(runnum[use])



r<-3.935 # undeformed radius, mm
h<-39.35 # undeformed height, mm
dat.exp<-read.table('~/git/immpala/taylor_cylinder/allen-ofhc-cu-taylor.txt',skip=10)[,2:1]
int.exp<-integrate(approxfun(dat.exp[,2],dat.exp[,1]),min(dat.exp[,2]),max(dat.exp[,2]))$val
dat.exp<-rbind(c(0,0),dat.exp,c(0,dat.exp[43,2]))
integrate(approxfun(dat.exp[,2],dat.exp[,1]),min(dat.exp[,2]),max(dat.exp[,2]))$val

plot(approx(dat.exp[,2],dat.exp[,1],xout=seq(0,28,length.out=500)))


mult<-.7
pdf('../Docs/sky_talk/condHB_closeSimTC.pdf',height=10*mult,width=6*mult)
plot(datx[use.run[769],],daty[use.run[769],],col=2,type='l',xlim=c(0,8.5),xlab='position (mm)',ylab='position (mm)')
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey90")
abline(h=axTicks(2),col='white')
abline(v=axTicks(1),col='white')
lines(datx[use.run[769],],daty[use.run[769],],col=2,lwd=3)
lines(dat.exp[,1],dat.exp[,2],lwd=3,lty=3)
legend('topright',c('model','experiment'),lty=c(1,3),lwd=c(3,3),col=c(2,1),cex=.9,bg="white")
dev.off()

############################################################
## plot

col<-BASS:::scale.range(inputs[use.run,1])
#rbPal <- colorRampPalette(c('red','blue'))
#col <- rbPal(100)[as.numeric(cut(col,breaks = 100))]
col <- rev(rainbow(100,end=.7))[as.numeric(cut(col,breaks = 100))]

pdf('../Docs/sky_talk/tc_model_runs.pdf',height=10*mult,width=6*mult)
plot(1,type='n',xlab='position (mm)',ylab='position (mm)',xlim=c(0,11),ylim=c(0,40),yaxs="i",xaxs="i",main='Deformed Taylor Cylinder Profiles')
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey88")
abline(h=c(5,10,15,20,25,30,35,40),col='white')
abline(v=c(0,2,4,6,8,10),col='white')
matplot(t(datx[use.run,]),t(daty[use.run,]),type='l',add=T,col=col,lty=1)
#abline(h=0,col=1)
#segments(0,daty[use.run,481],r,daty[use.run,481],col=col)
segments(r,0,r,h,lty=2,col=1)
segments(r,h,0,h,lty=2,col=1)
lines(dat.exp,lwd=3,col=1,lty=4)
#segments(0,dat.exp[1,43],r,dat.exp[43,1],lwd=3,lty=4)
#segments(dat.exp[2,1],0,0,0,lwd=3,lty=4)
legend('topright',c('undeformed cylinder','model runs (color varying by A)','experiment'),lty=c(2,1,4),lwd=c(1,1,3),col=c(1,2,1),cex=.6,bg="white")
dev.off()


col=rep(2,1000)
col[use.run]=1
pairs(inputs,col=col,cex=.2)

nn<-names(inputs)

pdf('../docs/tc_fails.pdf',height=10,width=14)
par(mfrow=c(ncol(inputs),ncol(inputs)),oma=c(5,5,5,5),mar=c(1.5,1.5,1.5,1.5))
for(i in 1:ncol(inputs)){
  for(j in 1:ncol(inputs)){
    if(i==j){
      hist(inputs[use.run,i],breaks=50,col=rgb(1,.5,0,0.5),main='',border=NA)
      hist(inputs[-use.run,i],add=T,col=rgb(0,0,1,0.5),breaks=50,border=NA)
      mtext(nn[i],1,3)
      #plot(d1,ylim=range(c(d1$y,d2$y)),xlim=range(d1$x,d2$x))
      #lines(d2,col=2)
    }
    if(i<j){
      plot(inputs[use.run,j],inputs[use.run,i],ylab='',xlab='',col=rgb(1,.5,0,0.5),cex=.75,cex.axis=.5,pch=16)
      points(inputs[-use.run,j],inputs[-use.run,i],col=rgb(0,0,1,0.5),cex=.75,pch=16)
    }
    if(i>j)
      plot.new()
    if(i==2 & j==1)
      legend('bottomleft',c('completed','failed'),fill=c(rgb(1,.5,0,0.5),rgb(0,0,1,0.5)),bty='n')
  }
}
dev.off()

############################################################
## get features


foot.exp<-max(dat.exp[,1])
top.exp<-max(dat.exp[,2])
#int.exp<-integrate(approxfun(dat.exp[,2],dat.exp[,1]),min(dat.exp[,2]),max(dat.exp[,2]))$val

foot<-apply(datx[use.run,],1,max) # foot size
top<-apply(daty[use.run,],1,max) # top height

library(geometry)
int<-NA # integral of 2D profile (akin to volume)
for(i in 1:length(use.run))
  int[i]<-polyarea(c(datx[use.run[i],]),c(daty[use.run[i],]))#integrate(approxfun(daty[use.run[i],],datx[use.run[i],]),min(daty[use.run[i],]),max(daty[use.run[i],]))$val

hist(int)

int[1]
polyarea(c(datx[1,]),c(daty[1,]))
plot(datx[1,],daty[1,],xlim=c(0,9))
points(dat.exp,col=2)

plot(approx(daty[use.run[i],],datx[use.run[i],],xout=seq(0,35,.01)))
points(daty[use.run[i],],datx[use.run[i],],col=2,cex=.1)


polyarea(dat.exp[,1],dat.exp[,2])


hist(foot)
abline(v=foot.exp,col=2)

hist(top)
abline(v=top.exp,col=2)

hist(int)
abline(v=int.exp,col=2)

sim.tc<-cbind(foot,top,int)
dat.tc<-c(foot.exp,top.exp,int.exp)
inputs.sim.tc<-inputs[use.run,]

save(sim.tc,dat.tc,inputs.sim.tc,file='features.rda')

############################################################
## emulators

library(BASS)

mod.foot<-bass(inputs[use.run,],foot)
plot(mod.foot)

mod.top<-bass(inputs[use.run,],top)
plot(mod.top)

mod.int<-bass(inputs[use.run,],int)
plot(mod.int)

## sensitivity

ss.foot<-sobol(mod.foot)
ss.top<-sobol(mod.top)
ss.int<-sobol(mod.int)
plot(ss.foot)
plot(ss.top)
plot(ss.int)

## joint emulator
pairs(cbind(foot,top,int))

ho<-sample(use.run,100)

mod<-bassPCA(inputs[use.run[-ho],],rbind(foot,top,int)[,-ho],scale=T,n.pc=3,n.cores = 3)
pred<-apply(predict(mod,inputs[use.run[ho],],mcmc.use=1:1000),2:3,mean)
plot(pred[,1],foot[ho]); abline(a=0,b=1,col=2)
plot(pred[,2],top[ho]); abline(a=0,b=1,col=2)
plot(pred[,3],int[ho]); abline(a=0,b=1,col=2)

sqrt(mean((pred[,1]-foot)^2))
sqrt(mean((colMeans(predict(mod.foot,inputs[use.run,]))-foot)^2))

sqrt(mean((pred[,2]-top)^2))
sqrt(mean((colMeans(predict(mod.top,inputs[use.run,]))-top)^2))

sqrt(mean((pred[,3]-int)^2))
sqrt(mean((colMeans(predict(mod.int,inputs[use.run,]))-int)^2))

## joint emulator sensitivity

ss<-sobolOB(mod,int.order = 2,ncores = 20)
plot(ss)
