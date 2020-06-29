
mcmc.block<-function(y1,y2,ss1,ss2,w,nmcmc=10000,itemp.ladder=1){
  start.temper=1000
  p<-3 # number of x's
  n<-2 # number of y types
  nz<-length(which(w==0))
  n<-n-nz
  w.n<-w*n
  require(mnormt)
  theta.list<-list()
  ntemp<-length(itemp.ladder)
  for(tt in 1:ntemp){
    theta.list[[tt]]<-matrix(nrow=nmcmc,ncol=p)
    theta.list[[tt]][1,]<-.5
  }
  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  lpost<-function(th)
    -.5*( (y1-f(th))^2*w.n[1]/ss1^2 + (y2-g(th))^2*w.n[2]/ss2^2 )
  
  count.swap<-0
  swap.vals<-matrix(nrow=nmcmc,ncol=2)
  count<-rep(0,ntemp)
  
  for(i in 2:nmcmc){

    for(tt in 1:ntemp){
      
      theta.list[[tt]][i,]<-theta.list[[tt]][i-1,]
      
      if(i>300){# & i<nburn){
        mi<-max(1,i-300)
        S<-cov(theta.list[[tt]][mi:(i-1),])*cc+diag(eps*cc,p)
      }
    
    
    
      theta.cand<-rmnorm(1,theta.list[[tt]][i-1,],S)
      if(any(theta.cand>1) | any(theta.cand<0)){
        alpha<- -9999
      } else{
        alpha<- (lpost(theta.cand) - lpost(theta.list[[tt]][i-1,]))*itemp.ladder[tt]
      }
        
      if(log(runif(1)) < alpha){
        theta.list[[tt]][i,]<-theta.cand
        count[tt]<-count[tt]+1
      }
    
    }
    
    if(i>start.temper & ntemp>1){
      swap<-sample(1:ntemp,size=2) # candidate swap
      alpha<-(itemp.ladder[swap[2]]-itemp.ladder[swap[1]])*( lpost(theta.list[[swap[1]]][i,]) - lpost(theta.list[[swap[2]]][i,]) )
      if(log(runif(1))<alpha){ # swap states
        temp<-theta.list[[swap[1]]][i,]
        theta.list[[swap[1]]][i,]<-theta.list[[swap[2]]][i,]
        theta.list[[swap[2]]][i,]<-temp
        
        count.swap<-count.swap+1
        swap.vals[i,]<-sort(swap)
      }
    }
    

  }
  return(list(theta=theta.list[[1]],count=count,count.swap=count.swap,swap.vals=swap.vals,theta.list=theta.list))
}

mcmc.block.hier<-function(y1,y2,ss1,ss2,w,a=0,b=0,nmcmc=10000,itemp.ladder=1){
  start.temper=1000
  p<-3 # number of x's
  n<-2 # number of y types
  nz<-length(which(w==0))
  n<-n-nz
  w.n<-w*n
  require(mnormt)
  theta0.list<-thetai.list<-t2.list<-list()
  ntemp<-length(itemp.ladder)
  for(tt in 1:ntemp){
    theta0.list[[tt]]<-matrix(nrow=nmcmc,ncol=p)
    theta0.list[[tt]][1,]<-.5
    thetai.list[[tt]]<-array(dim=c(nmcmc,p,n))
    thetai.list[[tt]][1,,]<-.5
    t2.list[[tt]]<-matrix(nrow=nmcmc,ncol=p)
    t2.list[[tt]][1,]<-1
  }
  theta.new<-matrix(nrow=nmcmc,ncol=p)
  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  lpost<-function(th1,th2,th0,t2)
    -.5*( (y1-f(th1))^2/ss1^2 + (y2-g(th2))^2/ss2^2 + sum((th1-th0)^2/t2) + sum((th2-th0)^2/t2) ) - 2*sum(pnorm(1,th0,sqrt(t2))-pnorm(0,th0,sqrt(t2)))
  
  count.swap<-0
  swap.vals<-matrix(nrow=nmcmc,ncol=2)
  count1<-count2<-rep(0,ntemp)
  

  
  for(i in 2:nmcmc){
    
    for(tt in 1:ntemp){
      
      thetai.list[[tt]][i,,]<-thetai.list[[tt]][i-1,,]
      
      if(i>300){# & i<nburn){
        mi<-max(1,i-300)
        S<-cov(thetai.list[[tt]][mi:(i-1),,1])*cc+diag(eps*cc,p)
      }
      
      theta.cand<-rmnorm(1,thetai.list[[tt]][i-1,,1],S)
      if(any(theta.cand>1) | any(theta.cand<0)){
        alpha<- -9999
      } else{
        alpha<- (lpost(theta.cand,thetai.list[[tt]][i-1,,2],theta0.list[[tt]][i-1,],t2.list[[tt]][i-1,]) - lpost(thetai.list[[tt]][i-1,,1],thetai.list[[tt]][i-1,,2],theta0.list[[tt]][i-1,],t2.list[[tt]][i-1,]))*itemp.ladder[tt]
      }
      
      if(log(runif(1)) < alpha){
        thetai.list[[tt]][i,,1]<-theta.cand
        count1[tt]<-count1[tt]+1
      }
    
    
    
    
    if(i>300){# & i<nburn){
      mi<-max(1,i-300)
      S<-cov(thetai.list[[tt]][mi:(i-1),,2])*cc+diag(eps*cc,p)
    }
    
    theta.cand<-rmnorm(1,thetai.list[[tt]][i-1,,2],S)
    if(any(theta.cand>1) | any(theta.cand<0)){
      alpha<- -9999
    } else{
      alpha<- (lpost(thetai.list[[tt]][i,,1],theta.cand,theta0.list[[tt]][i-1,],t2.list[[tt]][i-1,]) - lpost(thetai.list[[tt]][i,,1],thetai.list[[tt]][i-1,,2],theta0.list[[tt]][i-1,],t2.list[[tt]][i-1,]))*itemp.ladder[tt]
    }
    
    if(log(runif(1)) < alpha){
      thetai.list[[tt]][i,,2]<-theta.cand
      count2[tt]<-count2[tt]+1
    }
  
  
  
    for(j in 1:p){
      theta0.list[[tt]][i,j]<-rtruncnorm(1,0,1,mean(thetai.list[[tt]][i,j,]),sqrt(t2.list[[tt]][i-1,j]/n))
      t2.list[[tt]][i,j]<-1/rgamma(1,a+n/2,b+.5*sum((theta0.list[[tt]][i,j]-thetai.list[[tt]][i,j,])^2))
    }

    }
    
    
    if(i>start.temper & ntemp>1){
      swap<-sample(1:ntemp,size=2) # candidate swap
      alpha<-(itemp.ladder[swap[2]]-itemp.ladder[swap[1]])*( lpost(thetai.list[[swap[1]]][i,,1],thetai.list[[swap[1]]][i,,2],theta0.list[[swap[1]]][i,],t2.list[[swap[1]]][i,]) - lpost(thetai.list[[swap[2]]][i,,1],thetai.list[[swap[2]]][i,,2],theta0.list[[swap[2]]][i,],t2.list[[swap[2]]][i,]) )
      if(log(runif(1))<alpha){ # swap states
        temp<-thetai.list[[swap[1]]][i,,]
        thetai.list[[swap[1]]][i,,]<-thetai.list[[swap[2]]][i,,]
        thetai.list[[swap[2]]][i,,]<-temp
        
        temp<-theta0.list[[swap[1]]][i,]
        theta0.list[[swap[1]]][i,]<-theta0.list[[swap[2]]][i,]
        theta0.list[[swap[2]]][i,]<-temp
        
        temp<-t2.list[[swap[1]]][i,]
        t2.list[[swap[1]]][i,]<-t2.list[[swap[2]]][i,]
        t2.list[[swap[2]]][i,]<-temp
        
        count.swap<-count.swap+1
        swap.vals[i,]<-sort(swap)
      }
    }
    
    theta.new[i,]<-rtruncnorm(p,0,1,theta0.list[[1]][i,],sqrt(t2.list[[1]][i,]))
  }
  #browser()
  return(list(theta.new=theta.new,thetai=thetai.list[[1]],theta0=theta0.list[[1]],t2=t2.list[[1]],count1=count1,count2=count2,count.swap=count.swap,swap.vals=swap.vals))
}


mcmc.block.l1<-function(y1,y2,ss1,ss2,w,nmcmc=10000,itemp.ladder=1){
  start.temper=1000
  p<-3 # number of x's
  n<-2 # number of y types
  nz<-length(which(w==0))
  n<-n-nz
  w.n<-w*n
  require(mnormt)
  theta.list<-list()
  ntemp<-length(itemp.ladder)
  for(tt in 1:ntemp){
    theta.list[[tt]]<-matrix(nrow=nmcmc,ncol=p)
    theta.list[[tt]][1,]<-.5
  }
  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  lpost<-function(th)
    -1/sqrt(2)*( abs(y1-f(th))*w.n[1]/ss1 + abs(y2-g(th))*w.n[2]/ss2 )
  
  count.swap<-0
  swap.vals<-matrix(nrow=nmcmc,ncol=2)
  count<-rep(0,ntemp)
  
  for(i in 2:nmcmc){
    
    for(tt in 1:ntemp){
      
      theta.list[[tt]][i,]<-theta.list[[tt]][i-1,]
      
      if(i>300){# & i<nburn){
        mi<-max(1,i-300)
        S<-cov(theta.list[[tt]][mi:(i-1),])*cc+diag(eps*cc,p)
      }
      
      
      
      theta.cand<-rmnorm(1,theta.list[[tt]][i-1,],S)
      if(any(theta.cand>1) | any(theta.cand<0)){
        alpha<- -9999
      } else{
        alpha<- (lpost(theta.cand) - lpost(theta.list[[tt]][i-1,]))*itemp.ladder[tt]
      }
      
      if(log(runif(1)) < alpha){
        theta.list[[tt]][i,]<-theta.cand
        count[tt]<-count[tt]+1
      }
      
    }
    
    if(i>start.temper & ntemp>1){
      swap<-sample(1:ntemp,size=2) # candidate swap
      alpha<-(itemp.ladder[swap[2]]-itemp.ladder[swap[1]])*( lpost(theta.list[[swap[1]]][i,]) - lpost(theta.list[[swap[2]]][i,]) )
      if(log(runif(1))<alpha){ # swap states
        temp<-theta.list[[swap[1]]][i,]
        theta.list[[swap[1]]][i,]<-theta.list[[swap[2]]][i,]
        theta.list[[swap[2]]][i,]<-temp
        
        count.swap<-count.swap+1
        swap.vals[i,]<-sort(swap)
      }
    }
    
    
  }
  return(list(theta=theta.list[[1]],count=count,count.swap=count.swap,swap.vals=swap.vals,theta.list=theta.list))
}


################################################################################################################
################################################################################

# line and inverted cross posteriors, where they can overlap

y1<-1
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1) # use (x[1]-.5)*(x[2]*2-1) or -(x[1]-.5)*(x[2]*2-1) for just two

itemps<-c(1,.5,.2,.01)
m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

m1$count.swap
m2$count.swap
m3$count.swap


burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

plot(m2$theta.list[[3]][-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta.list[[3]][-burn,1:2],cex=.3,col=2)
points(m3$theta.list[[3]][-burn,1:2],cex=.3,col=3)

m2$count/50000

m2$count.swap/50000
table(apply(m2$swap.vals,1,paste,collapse = '-'))/50000*10
plot(as.numeric(as.factor(apply(m2$swap.vals,1,paste,collapse = '-'))))

################################################################################
# line and inverted cross posteriors, no overlap

y1<-1.9
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# same as above, but increase inverted cross weight

y1<-1.9
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.05,.95),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# line and butterfly wings posteriors, no overlap, smaller variance on wings

y1<-1
y2<-.1

ss1<-.04
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  (x[1]-.5)*(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 50000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# line and butterfly wings posteriors, no overlap, same variance

y1<-1
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  (x[1]-.5)*(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# opposing lines posterior, one with larger variance

y1<-1
y2<-0

ss1<-.01
ss2<-.1

f<-function(x)
  x[1]+x[2]
g<-function(x)
  x[1]-x[2]

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# opposing lines posterior, one with larger weight

y1<-1
y2<-0

ss1<-.01
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  x[1]-x[2]

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.95,.05),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)


################################################################################
# line and uniform posteriors

y1<-1
y2<-0

ss1<-.01
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  10000

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# same, increasing uniform weight

y1<-1
y2<-0

ss1<-.01
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  10000

c(.01,.99)*2/c(.01,.01)^2

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.01,.99),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################
# equal weights but smaller uniform variance

y1<-1
y2<-0

ss1<-.01
ss2<-.00001

f<-function(x)
  x[1]+x[2]
g<-function(x)
  10000

c(.5,.5)*2/c(.01,.00001)^2

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 20000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 20000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)


################################################################################
# stingray posteriors, no overlap

y1<-.5
y2<-.1

ss1<-.004
ss2<-.004

f<-function(x)
  x[1]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 20000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 50000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 50000,itemps)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)


################################################################################
# simple example

y1<-.55
y2<-.45

ss1<-.01
ss2<-.04

f<-function(x)
  x[1]
g<-function(x)
  x[1]

hist(sqrt(1/rgamma(100000,50,.1)))
mh<-mcmc.block.hier(y1,y2,ss1,ss2,w=c(.5,.5),a=50,b=.1,nmcmc = 50000,itemps)
hist(mh$thetai[-burn,1,1])
hist(mh$thetai[-burn,1,2])
hist(mh$theta0[-burn,1])
hist(mh$theta.new[-burn,1])
hist(sqrt(mh$t2[-burn,1]))


m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 50000,itemps)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 50000,itemps)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 50000,itemps)

m4<-mcmc.block.l1(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 50000,itemps)
m5<-mcmc.block.l1(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 50000,itemps)
m6<-mcmc.block.l1(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 50000,itemps)

pdf('calib_ex.pdf',height=5,width=8)
burn<-1:15000
plot(density(m1$theta[-burn,1]),xlim=c(.25,.75),lwd=3,xlab='parameter',main='Gaussian Likelihood (L2 norm)',bty='n',ylim=c(0,60))
lines(density(m2$theta[-burn,1]),col=1,lwd=3)
lines(density(m3$theta[-burn,1]),col=3,lwd=3)
#rug(c(y1+c(-ss1,ss1),y2+c(-ss2,ss2)),lty=2,col=4)
#rug(c(y1,y2),col=4)
points(c(y1,y2),c(0,0),pch=19,col=4)
segments(c(y1-ss1,y2-ss2),0,c(y1+ss1,y2+ss2),col=4)
legend('topleft',c('parameter posteriors while calibrating \n individually',' ','parameter posterior while calibrating \n jointly (product likelihood)',' ','targets (with sd)'),col=c(1,NA,3,NA,4),lty=c(1,NA,1,NA,1),lwd=c(3,NA,3,NA,1),bty='n',pch=c(NA,NA,NA,NA,19),cex=.7)

plot(density(m4$theta[-burn,1]),xlim=c(.25,.75),lwd=3,xlab='parameter',main='Double Exponential Product Likelihood (L1 norm)',bty='n')
lines(density(m5$theta[-burn,1]),col=1,lwd=3)
lines(density(m6$theta[-burn,1]),col=3,lwd=3)
points(c(y1,y2),c(0,0),pch=19,col=4)
segments(c(y1-ss1,y2-ss2),0,c(y1+ss1,y2+ss2),col=4)
legend('topleft',c('parameter posteriors while calibrating \n individually',' ','parameter posterior while calibrating \n jointly (product likelihood)',' ','targets (with sd)'),col=c(1,NA,3,NA,4),lty=c(1,NA,1,NA,1),lwd=c(3,NA,3,NA,1),bty='n',pch=c(NA,NA,NA,NA,19),cex=.7)

plot(density(mh$thetai[-burn,1,1]),xlim=c(.25,.75),lwd=3,xlab='parameter',main='Gaussian Hierarchical Likelihood',bty='n')
lines(density(mh$thetai[-burn,1,2]),col=1,lwd=3)
lines(density(mh$theta.new[-burn,1]),col=3,lwd=3)
points(c(y1,y2),c(0,0),pch=19,col=4)
segments(c(y1-ss1,y2-ss2),0,c(y1+ss1,y2+ss2),col=4)
legend('topleft',c('parameter posteriors while calibrating \n individually',' ','parameter posterior while calibrating \n jointly (product likelihood)',' ','targets (with sd)'),col=c(1,NA,3,NA,4),lty=c(1,NA,1,NA,1),lwd=c(3,NA,3,NA,1),bty='n',pch=c(NA,NA,NA,NA,19),cex=.7)
dev.off()

pdf('like_effect.pdf',height=5,width=5)
plot(density(m3$theta[-burn,1]),xlim=c(.25,.75),col=3,lwd=3,xlab='parameter',main='Gaussian Likelihood (L2 norm)',bty='n',lty=2)
lines(density(m6$theta[-burn,1]),col=3,lwd=3,lty=3)
lines(density(mh$theta.new[-burn,1]),col=3,lwd=3,lty=1)
#rug(c(y1+c(-ss1,ss1),y2+c(-ss2,ss2)),lty=2,col=4)
#rug(c(y1,y2),col=4)
points(c(y1,y2),c(0,0),pch=19,col=4)
segments(c(y1-ss1,y2-ss2),0,c(y1+ss1,y2+ss2),col=4)
legend('topleft',c('Gaussian','Double Exp','Hierarchical','targets (with sd)'),col=c(3,3,3,4),lty=c(2,3,1,1),lwd=c(3,3,3,1),bty='n',pch=c(NA,NA,NA,19),cex=.7)
dev.off()





pdf('llike.pdf',height=5,width=9)
par(mfrow=c(1,2))
curve(dnorm(x,y1,ss1),n=1000,col=1,log = 'y',xlim=c(.25,.75),xlab='parameter',ylab='log likelhood (Gaussian)')
curve(dnorm(x,y2,ss2),n=1000,add=T,col=2)
curve(dnorm(x,y1,ss1)*dnorm(x,y2,ss2),n=1000,add=T,col=3,lwd=3)

library(smoothmest)
curve(ddoublex(x,y1,ss1/sqrt(2)),n=1000,col=1,log='y',xlim=c(.25,.75),xlab='parameter',ylab='log likelhood (Double Exponential)')
curve(ddoublex(x,y2,ss2/sqrt(2)),n=1000,add=T,col=2)
curve(ddoublex(x,y1,ss1/sqrt(2))*ddoublex(x,y2,ss2/sqrt(2)),n=1000,add=T,col=3,lwd=3)

legend(x=.45,y=1e-13,c('individual likelihoods','product likelihood'),col=c(1,3),lwd=c(1,3),bty='n',cex=.5)
dev.off()

plot(seq(.25,.75,.001),dnorm(seq(.25,.75,.001),y1,ss1),xlim=c(.25,.75),type='l',ylim=c(0,60))
lines(seq(.25,.75,.001),dnorm(seq(.25,.75,.001),y2,ss2),col=2)
lines(seq(.25,.75,.001),dnorm(seq(.25,.75,.001),y1,ss1)*dnorm(seq(.25,.75,.001),y2,ss2)/sum(dnorm(seq(.25,.75,.001),y1,ss1)*dnorm(seq(.25,.75,.001),y2,ss2))*1000,col=3,lwd=3)

################################################################################
################################################################################

y1<-1.9
y2<-.1

ss1<-.4
ss2<-.4

f<-function(x){
  if(abs(abs(x[1]-.5)*abs(x[2]*2-1)-y2)<.03)
    return(abs(abs(x[1]-.5)*abs(x[2]*2-1)-y2))
  if(abs(x[1]+x[2]-y1)<.03)
    return(x[1]+x[2])
  return(3)
}
g<-function(x){
  if(abs(abs(x[1]-.5)*abs(x[2]*2-1)-y2)<.03)
    return(abs(x[1]-.5)*abs(x[2]*2-1))
  if(abs(x[1]+x[2]-y1)<.03)
    return((x[1]+x[2]))
  return(3)
}

test1<-apply(expand.grid(seq(.0001,1,.01),seq(.0001,1,.01)),1,f)#+rnorm(10201,0,ss2)
test2<-apply(expand.grid(seq(.0001,1,.01),seq(.0001,1,.01)),1,g)


library(fields)
image.plot(matrix(-.5*(log(abs(y1-test1))/ss1^2+log(abs(y2-test2))/ss2^2),100))
#image.plot(matrix(-.5*(log(abs(y1-test1))/ss2^2+log(abs(y2-test2))/ss1^2),100))
contour(matrix(test1,100),levels=y1,add=T)
contour(matrix(test2,100),levels=y2,add=T)



#









## notes
# in real code, calculation of alpha is wrong, depends on previous state rather than current state and candidate