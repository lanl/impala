rm(list=ls())

load('~/Desktop/immpala_data/calib_start.rda')
source('~/git/immpala/code/strength_setup.R')

load('~/Desktop/immpala_data/calib_hop_temper.rda')
cond<-'HB'
source('~/git/immpala/code/plots.R')

load('~/Desktop/immpala_data/calib_vel_temper.rda')
cond<-'FP'
source('~/git/immpala/code/plots.R')

load('~/Desktop/immpala_data/calib_tc_temper.rda')
cond<-'TC'
source('~/git/immpala/code/plots.R')

load('~/Desktop/immpala_data/calib_hopvel_temper.rda')
cond<-'HB+FP'
source('~/git/immpala/code/plots.R')

load('~/Desktop/immpala_data/calib_equalWeight_temper.rda')
cond<-'All'
source('~/git/immpala/code/plots.R')




g<-function(ps){
  do.call(rbind,lapply(ps,function(x) x$u[seq(25000,30000,10),]))
}

theta.list<-list()
load('~/Desktop/immpala_data/calib_hop_temper.rda')
theta.list[[1]]<-g(ps)
load('~/Desktop/immpala_data/calib_vel_temper.rda')
theta.list[[2]]<-g(ps)
load('~/Desktop/immpala_data/calib_tc_temper.rda')
theta.list[[3]]<-g(ps)
load('~/Desktop/immpala_data/calib_hopvel_temper.rda')
theta.list[[4]]<-g(ps)
load('~/Desktop/immpala_data/calib_equalWeight_temper.rda')
theta.list[[5]]<-g(ps)

names<-c('HB/QS','FP','TC','HB+FP','all')



use<-seq(1,nrow(theta.list[[1]]),1)
for(i in 1:length(theta.list))
  my.pd(theta.list[[i]][use,],nlevels = 10)#,pt=X[jj,])


theta.list.burn<-lapply(theta.list,function(mat) mat[use,])
#PairsDensity(theta.list.burn)



pnames<-c("A",'B','C','m','n','vFP','vTC','sm_0')

x.rr<-matrix(c(.0001,.01,
               .0001,.01,
               .0002,.3,
               .0001,1.5,
               .002,3,
               .03,.0315, # flyer plate vel
               .03,.0315, # tc vel
               .3,.6 # sm_0
),ncol=2,byrow=T)

rr<-t(x.rr)

rr.true<-t(x.rr)
theta.list.burn.df<-lapply(theta.list.burn,as.data.frame)
for(i in 1:length(theta.list.burn.df))
  names(theta.list.burn.df[[i]])<-pnames

my.pd(theta.list.burn.df[[1]],nlevels = 10)#,pt=X[jj,])

p<-8


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
ltys<-1:5#rep(1,5)
lwds<-rep(2,5)

use.plot<-c(1,2,3)
p<-5

pdf('~/Desktop/jc_posts.pdf',width=12,height=10)
par(mfrow=c(p,p),oma=c(5,5,5,5),mar=c(.1,.1,.1,.1),xpd=F)
for(i in 1:p){
  for(j in 1:p){
    if(i==j){
      plot(1,type='n',xlim=rr[,i],ylim=c(0,ymax[i]),yaxt='n')
      mtext(pnames[i],1,3,cex=1)
      for(k in use.plot){
        dens<-density(theta.list.burn.us[[k]][,i])
        lines(dens$x,dens$y/max(dens$y),col=cols[k],lty=ltys[k],lwd=lwds[k])
      }
      #abline(v=djvals.us[i],lty=3)
    }
    if(i<j){
      plot(1,type='n',xlim=rr[,j],ylim=rr[,i],xaxt='n',yaxt='n')
      for(k in use.plot){
        mat<-theta.list.burn.us[[k]][,c(j,i)]
        kd <- kde(mat, compute.cont=TRUE,H=matrix(c(dr[j]^2,0,0,dr[i]^2),2)*.004)
        contour_95 <- with(kd, contourLines(x=eval.points[[1]], y=eval.points[[2]],z=estimate, levels=cont["5%"]))
        lapply(contour_95,function(x) lines(x,type='l',col=cols[k],lty=ltys[k],lwd=lwds[k]))
      }
      #points(djvals.us[j],djvals.us[i],pch=4,cex=2)
    }
    if(i>j){
      plot.new()
    }
    if(i==2 & j==1)
      legend(x=.5,y=.5,names[use.plot],col=cols[use.plot],lty=ltys[use.plot],lwd=lwds[use.plot],cex=1,xpd=NA)
  }
}
dev.off()