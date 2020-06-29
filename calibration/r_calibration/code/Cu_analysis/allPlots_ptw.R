rm(list=ls())

load('~/Desktop/immpala_data/calib_start_ptw.rda')
source('~/git/immpala/code/strength_setup_ptw.R')

load('~/Desktop/immpala_data/calib_hop_indepSampler_ptw.rda')
cond<-'HB'
source('~/git/immpala/code/plots_ptw.R')

load('~/Desktop/immpala_data/calib_vel_indepSampler_ptw.rda')
cond<-'FP'
source('~/git/immpala/code/plots_ptw.R')

load('~/Desktop/immpala_data/calib_tc_indepSampler_ptw.rda')
cond<-'TC'
source('~/git/immpala/code/plots_ptw.R')

load('~/Desktop/immpala_data/calib_hopvel_indepSampler_ptw.rda')
cond<-'HB+FP'
source('~/git/immpala/code/plots_ptw.R')

load('~/Desktop/immpala_data/calib_equalWeight_indepSampler_ptw.rda')
cond<-'All'
source('~/git/immpala/code/plots_ptw.R')

g<-function(ps){
  do.call(rbind,lapply(ps,function(x) x$u[seq(5000,10000,10),]))
}

theta.list<-list()
load('~/Desktop/immpala_data/calib_hop_indepSampler_ptw.rda')
theta.list[[1]]<-g(ps)
load('~/Desktop/immpala_data/calib_vel_indepSampler_ptw.rda')
theta.list[[2]]<-g(ps)
load('~/Desktop/immpala_data/calib_tc_indepSampler_ptw.rda')
theta.list[[3]]<-g(ps)
load('~/Desktop/immpala_data/calib_hopvel_indepSampler_ptw.rda')
theta.list[[4]]<-g(ps)
load('~/Desktop/immpala_data/calib_equalWeight_indepSampler_ptw.rda')
theta.list[[5]]<-g(ps)

names<-c('Split-Hopkinson pressure bar + quasi-static','flyer plate','Taylor cylinder','HB+FP','all')



use<-seq(1,nrow(theta.list[[1]]),1)
for(i in 1:length(theta.list))
  my.pd(theta.list[[i]][use,],nlevels = 10)#,pt=X[jj,])


theta.list.burn<-lapply(theta.list,function(mat) mat[use,])
#PairsDensity(theta.list.burn)



pnames<-c('theta0', 'p', 's0','sinf',       'y0',      'yinf','y1', 'y2','kappa', 'gamma', 'vel')

x.rr<-matrix(c(.001,.1,
               0,10,
               .001,.05,
               .0001,.05,
               0,.05,
               0,.05, 
               0,.1, 
               0,1,
               .01,1,
               10^-6,10^-1,
               0.030184,0.031316 # vel
),ncol=2,byrow=T)


rr<-t(x.rr)

rr.true<-t(x.rr)
theta.list.burn.df<-lapply(theta.list.burn,as.data.frame)
for(i in 1:length(theta.list.burn.df))
  names(theta.list.burn.df[[i]])<-pnames

my.pd(theta.list.burn.df[[1]],nlevels = 10)#,pt=X[jj,])

p<-11


theta.list.burn.us<-theta.list.burn
for(k in 1:length(theta.list)){
  for(i in 1:p)
    theta.list.burn.us[[k]][,i]<-BASS:::unscale.range(theta.list.burn[[k]][,i],rr[,i])
}


library(ks)
mat<-matrix(nrow=length(theta.list),ncol=p)
for(i in 1:length(theta.list)){
  mat[i,]<-apply(theta.list.burn.us[[i]],2,function(x) max(density(x)$y))
}
ymax<-apply(mat,2,max)
dr<-apply(rr,2,diff)

ymax<-rep(1,p)

cols<-1:5#rep(1:6,each=6)
ltys<-rep(1,5)
lwds<-rep(2,5)

use.plot<-1:3
p<-10

ptwPaperVals<-c(.025,2,.0085,.00055,.0001,.0001,.094,.575,.11,.00001)
ptwPaperVals.scaled<-apply(t(1:p),2,function(i) BASS:::scale.range(ptwPaperVals[i],rr[,i]))

par(mfrow=c(p,p),oma=c(5,5,5,5),mar=c(.1,.1,.1,.1),xpd=T)
for(i in 1:p){
  for(j in 1:p){
    if(i==j){
      plot(1,type='n',xlim=rr[,i],ylim=c(0,ymax[i]),yaxt='n')
      mtext(pnames[i],1,3,cex=1)
      for(k in use.plot){
        dens<-density(theta.list.burn.us[[k]][,i])
        lines(dens$x,dens$y/max(dens$y),col=cols[k],lty=ltys[k],lwd=lwds[k])
      }
      abline(v=ptwPaperVals[i],lty=3)
    }
    if(i<j){
      plot(1,type='n',xlim=rr[,j],ylim=rr[,i],xaxt='n',yaxt='n')
      for(k in use.plot){
        mat<-theta.list.burn.us[[k]][,c(j,i)]
        kd <- kde(mat, compute.cont=TRUE,H=matrix(c(dr[j]^2,0,0,dr[i]^2),2)*.004)
        contour_95 <- with(kd, contourLines(x=eval.points[[1]], y=eval.points[[2]],z=estimate, levels=cont["5%"]))
        lapply(contour_95,function(x) lines(x,type='l',col=cols[k],lty=ltys[k],lwd=lwds[k]))
      }
      points(ptwPaperVals[j],ptwPaperVals[i],pch=4,cex=2)
    }
    if(i>j){
      plot.new()
    }
    if(i==4 & j==1)
      legend('topright',names[use.plot],col=cols[use.plot],lty=ltys[use.plot],lwd=lwds[use.plot],cex=1,xpd=NA)
  }
}


ltys<-c(1,2,3)
pdf('~/Desktop/ptw_compare2.pdf',height=7,width=9)
dr<-c(.0005,.007,.007)
par(mfrow=c(2,2),oma=c(5,5,5,5),mar=c(1,1,1,1))
for(i in c(2,7)){
  for(j in c(2,7)){
    if(i==j){
      plot(1,type='n',xlim=c(0,1),ylim=c(0,ymax[i]),yaxt='n',xaxt='n')
      axis(1,at=seq(0,1,length.out=length(pretty(rr[,i]))),labels = pretty(rr[,i]))
      mtext(pnames[i],1,3,cex=1)
      for(k in use.plot){
        dens<-density(theta.list.burn[[k]][,i])
        lines(dens$x,dens$y/max(dens$y),col=cols[k],lty=ltys[k],lwd=lwds[k])
      }
      abline(v=ptwPaperVals.scaled[i],lty=3)
    }
    if(i<j){
      plot(1,type='n',xlim=c(0,1),ylim=c(0,1),xaxt='n',yaxt='n')
      for(k in use.plot){
        mat<-theta.list.burn[[k]][,c(j,i)]
        kd <- kde(mat, compute.cont=TRUE,H=diag(2)*dr[k])
        contour_95 <- with(kd, contourLines(x=eval.points[[1]], y=eval.points[[2]],z=estimate, levels=cont["5%"]))
        lapply(contour_95,function(x) lines(x,type='l',col=cols[k],lty=ltys[k],lwd=lwds[k]))
      }
      points(ptwPaperVals.scaled[j],ptwPaperVals.scaled[i],pch=4,cex=2)
    }
    if(i>j){
      plot.new()
      legend('left',c(names[use.plot],'original nominal values'),col=c(cols[use.plot],1),lty=c(ltys[use.plot],NA),lwd=c(lwds[use.plot],NA),pch=c(-1,-1,-1,4),pt.cex=2,xpd=NA,cex=1,bty='n')
    }
    if(i==1 & j==2)
      legend('center',c(names[use.plot],'original nominal values'),col=c(cols[use.plot],1),lty=c(ltys[use.plot],NA),lwd=c(lwds[use.plot],NA),pch=c(-1,-1,-1,4),pt.cex=2,xpd=NA)
  }
}
dev.off()

library(lattice)

dd<-predict(kd,x=expand.grid(seq(0,1,.01),seq(0,1,.01)))
#image(matrix(dd,nrow=length(seq(0,1,.01))))
wireframe(matrix(dd,nrow=length(seq(0,1,.01))),colorkey=T,drape=T,zlab='posterior',xlab='y1',ylab='p')

mm<-matrix(.5,nrow=length(seq(0,1,.01)),ncol=length(seq(0,1,.01)))
mm[1,1]<-.55
mm[1,2]<-.45
wireframe(mm,colorkey=T,drape=T,zlab='prior',xlab='y1',ylab='p')


# temps<-c(-196,125,125,-196,25,100,100,25) + 273.15 # +273.15 takes from C to K
# edots<-c(.001,.001,.1,2000,2500,3000,3500,7000)# * 1e-6 # strain per second, using time units of microsecond
# plot(f(ptwPaperVals,temps[1],edots[1]))
# X<-as.matrix(inputs.sim.tc)
# y<-apply(X,1,function(x) f(as.numeric(x),temps[1],edots[1])[,2])
# 
# matplot(y,type='l')
# 
# mod<-bassPCA(X,y,n.cores = 10,n.pc=10)
# 
# sens<-sobolBasis(mod,3,mcmc.use=1,n.cores = 20)


plot(theta.list[[1]][,2:3])
