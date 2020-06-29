
#burn<-1:9500
burn<-1:29500

use<-seq(1,500*8,40)

#dat.tc[3]<-125
#dat.tc[3]<-118.2933

# x.rr<-matrix(c(.0001,.01,
#                .0001,.01,
#                .0002,.3,
#                .0001,1.5,
#                .002,3,
#                .03,.0315, # flyer plate vel
#                .03,.0315, # tc vel
#                .3,.6 # sm_0
# ),ncol=2,byrow=T)
# 
# j=0
# j=j+1
# plot(1,ylim=c(0,1),xlim=c(0,nrow(ps[[1]]$u)),main=j)
# for(i in 1:length(ps)){
#   lines(ps[[i]]$u[,j],col=i)
# }

u<-ps[[1]]$u[-burn,]
for(j in 2:length(ps))
  u<-rbind(u,ps[[j]]$u[-burn,])

# inputs.sim.tc.st<-do.call(cbind,lapply(1:5,function(col) BASS:::scale.range(inputs.sim.tc[,col],x.rr[col,])))
# which.min(as.matrix(dist(rbind(colMeans(u)[1:5],inputs.sim.tc.st)))[-1,1])
# 
# unstandardize(colMeans(u))[1:5]
# inputs.sim.tc[769,1:5]


udf<-data.frame(u)
nn<-c('A','B','C','n','m','velFP','velTC','sm_0')

u.unst<-t(unstandardize(t(u)))

#u.unst<-u.unst[1:7000,]
u.unst<-u.unst[,1:5]

mult<-.65

png(paste('../Docs/sky_talk_temper/cond',cond,'_parameters.png',sep=''),height=12*mult,width=14*mult,units='in',res=350/mult)
par(mfrow=c(ncol(u.unst),ncol(u.unst)),oma=c(5,5,5,5),mar=c(1,1,1,1))
for(i in 1:ncol(u.unst)){
  for(j in 1:ncol(u.unst)){
    if(i==j){
      hist(u.unst[,i],col='cornflowerblue',main='',xlim=x.rr[i,],border='cornflowerblue')
      mtext(nn[i],1,3)
    }
    if(i<j){
      plot(u.unst[,j],u.unst[,i],ylab='',xlab='',col='slategray3',cex=.5,cex.axis=.5,pch=16,xlim=x.rr[j,],ylim=x.rr[i,])
    }
    if(i>j)
      plot.new()
    #if(i==2 & j==1)
    #  legend('bottomleft',c('completed','failed'),fill=c(rgb(1,.5,0,0.5),rgb(0,0,1,0.5)),bty='n')
  }
}
dev.off()




pred<-list()
for(ee in 1:length(dat)){
  pred[[ee]]<-do.call(cbind,mclapply(1:nrow(u),function(i) f(unstandardize(u[i,]),temps[ee],edots[ee])[,2],mc.cores = nc))
}

pred.vel<-do.call(rbind,mclapply(1:nrow(u),function(i) c(predict(mod.vel,t(unstandardize(u[i,])[c(1:5,6,8)]),n.cores = 1,mcmc.use = sample(1000,1))),mc.cores=nc))


pred.tc<-do.call(rbind,mclapply(1:nrow(u),function(i) c(predict(mod.tc,t(unstandardize(u[i,])[c(1:5,7,8)]),n.cores = 1,mcmc.use = sample(1000,1))),mc.cores=nc))

col<-c(6:2,'burlywood4','darkgoldenrod1')
xx<-f(unstandardize(runif(p)),temps[1],edots[1])[,1]
xx1<-f(unstandardize(runif(p)),temps[2],edots[2])[,1]
ymax<-.008#max(unlist(pred),dat[[4]][,2],na.rm = T)
ss<-colMeans(sqrt(ps[[j]]$s2.hop[-burn,]))[1]

mult<-.65
png(paste('../Docs/sky_talk_temper/cond',cond,'_predHB.png',sep=''),height=5*mult,width=8*mult,units='in',res=350/mult)
par(mfrow=c(2,3),mar=c(0,0,0,0),oma=c(5,5,5,5))
for(i in 1:length(dat)){
  plot(xx,xx,ylim=c(0,ymax),type='n',xlab='',ylab='',xaxt='n',yaxt='n')
  rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey90")
  abline(h=axTicks(2),col='white')
  abline(v=axTicks(1),col='white')
  matplot(xx,pred[[i]][,use],type='l',ylim=c(0,ymax),xlab='',ylab='',xaxt='n',yaxt='n',add=T)
  lines(dat[[i]],lwd=3,col=1,lty=3)
  mult<-1
  #segments(x0=dat[[i]][,1],y0=dat[[i]][,2]-ss*mult,y1=dat[[i]][,2]+ss*mult,col=col[i],lwd=.5)
  #qq<-apply(pred[[i]],1,quantile,probs=c(.025,.975))
  #mm<-rowMeans(pred[[i]])
  #lines(xx,mm,lwd=2,col=col[i],lty=3)
  #lines(xx,qq[1,],col=col[i],lty=2)
  #lines(xx,qq[2,],col=col[i],lty=2)
  if(i==1)
    legend('topright',c(paste('temperature (C):',temps[i]-273),paste('strain rate (1/s):',edots[i])),bty='n')
  else
    legend('topright',legend=c(temps[i]-273,edots[i]),bty='n')
  if(i %in% c(1,4))
    axis(2)
  if(i %in% 4:8)
    axis(1)
  if(i==4)
    legend('topleft',c('experiemnt','posterior predictions'),lty=c(3,1),col=c(1,4),lwd=c(3,1),bty='n',cex=.75)
}
#mtext('JC fit',3,outer = T,cex=2,line=2)
mtext('strain',1,outer = T,cex=1,line=3)
mtext('stress',2,outer = T,cex=1,line=3)
dev.off()



mult<-.7
png(paste('../Docs/sky_talk_temper/cond',cond,'_predFP.png',sep=''),height=8*mult,width=8*mult,units = 'in',res=200/mult)
obs<-read.table('~/Desktop/Cu-Cu_Thomas_PlateImpact/ofhc-cu-symmetric-impact.txt')
plot(obs,type='b',xlim=c(.8,1.3),xlab='time',ylab='velocity',lwd=2)#,main='Cu Flyer Plate')

rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey90")
abline(h=axTicks(2),col='white')
abline(v=axTicks(1),col='white')

use2<-apply(pred.vel[,c(1,3,7,9,5)],1,function(x) all(diff(x)>0) & all(x>0))
use2<-apply(pred.vel[,c(1,3,7,9,5)+1],1,function(x) all(x>0)) & use2

#matplot(dat.vel.arr[1,c(1,3,7,9,5),use],dat.vel.arr[1,c(1,3,7,9,5)+1,use],type='l',col='lightgrey',add=T)

matplot(t(pred.vel[use2,c(1,3,7,9,5)]),t(pred.vel[use2,c(1,3,7,9,5)+1]),type='l',add=T,lty=1)

lines(obs,type='b',lwd=2)
legend('bottomright',c('velocimetry measurements','posterior velocimetry features'),pch=c(1,NA),lty=c(3,1),lwd=c(3,1),col=c(1,4),bty='n')
dev.off()


png(paste('../Docs/sky_talk_temper/cond',cond,'_predTC.png',sep=''),height=4*mult,width=8*mult,units = 'in',res=200/mult)
par(mfrow=c(1,3),mar=c(5,4,5,1))
ymax<-max(density(sim.tc[,1])$y,density(pred.tc[,1])$y)
hist(sim.tc[,1],freq=F,ylim=c(0,ymax),main='',xlab='max radius (mm)')
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey90")
abline(h=axTicks(2),col='white')
abline(v=axTicks(1),col='white')
hist(sim.tc[,1],freq=F,ylim=c(0,ymax),add=T,col='slategray1')
abline(v=dat.tc[1],col=1,lwd=3,lty=3)
abline(v=mean(pred.tc[,1]),col=3,lwd=1,lty=2)
lines(density(pred.tc[,1]),col=3,lwd=1)
legend('topleft',c('experiment','simulations','posterior prediction'),fill=c(1,'slategray1',3),bty='n',cex=.7)

ymax<-max(density(sim.tc[,2])$y,density(pred.tc[,2])$y)
hist(sim.tc[,2],freq=F,ylim=c(0,ymax),main='',xlab='height (mm)')
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey90")
abline(h=axTicks(2),col='white')
abline(v=axTicks(1),col='white')
hist(sim.tc[,2],freq=F,ylim=c(0,ymax),add=T,col='slategray1')
abline(v=dat.tc[2],col=1,lwd=3,lty=3)
abline(v=mean(pred.tc[,2]),col=3,lwd=1,lty=2)
lines(density(pred.tc[,2]),col=3,lwd=1)

ymax<-max(density(sim.tc[,3])$y,density(pred.tc[,3])$y)
hist(sim.tc[,3],freq=F,ylim=c(0,ymax),main='',xlab='volume')
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "grey90")
abline(h=axTicks(2),col='white')
abline(v=axTicks(1),col='white')
hist(sim.tc[,3],freq=F,ylim=c(0,ymax),add=T,col='slategray1')
abline(v=dat.tc[3],col=1,lwd=3,lty=3)
abline(v=mean(pred.tc[,3]),col=3,lwd=1,lty=2)
lines(density(pred.tc[,3]),col=3,lwd=1)
dev.off()

