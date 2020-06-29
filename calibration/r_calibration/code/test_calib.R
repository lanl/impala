mcmc.step<-function(y1,y2,ss1,ss2,w,nmcmc=10000){
  p<-3 # number of x's
  n<-2 # number of y types
  w.n<-w*n
  theta<-matrix(nrow=nmcmc,ncol=p)
  s2<-matrix(nrow=nmcmc,ncol=p)
  
  theta[1,]<-.5
  s2[1,]<-1
  
  astep<-function(sz,ct){
    for(s in 1:length(sz)){
      if(ct[s]<20)
        sz[s]<-sz[s]*.5
      if(ct[s]>30)
        sz[s]<-sz[s]*1.1
    }
    return(sz)
  }
  stepsize<-rep(.5,p)
  count<-count100<-rep(0,p)
  a1<-b1<-a2<-b2<-1
  for(i in 2:nmcmc){
    s2[i,1]<-ss1^2#1/rgamma(1,.5+a1+w.n[1]/2,b1+(y1-f(theta[i-1,]))^2*w.n[1]/2)
    s2[i,2]<-ss2^2#1/rgamma(1,.5+a2+w.n[2]/2,b2+(y2-g(theta[i-1,]))^2*w.n[2]/2)
    
    theta[i,]<-theta[i-1,]
    for(jj in 1:p){
      theta.cand<-theta[i,]
      theta.cand[jj]<-runif(1,max(theta.cand[jj]-stepsize[jj],0),min(theta.cand[jj]+stepsize[jj],1))
      alpha<- -.5*( (y1-f(theta.cand))^2*w.n[1]/s2[i,1] + (y2-g(theta.cand))^2*w.n[2]/s2[i,2] - (y1-f(theta[i,]))^2*w.n[1]/s2[i,1] - (y2-g(theta[i,]))^2*w.n[2]/s2[i,2] )
      
      if(log(runif(1)) < alpha){
        theta[i,]<-theta.cand
        count[jj]<-count[jj]+1
        count100[jj]<-count100[jj]+1
      }
    }
    
    if(i%%100==0){
      if(i<3000)
        stepsize<-astep(stepsize,count100)
      #cat('it:',i,'acc:',count100,timestamp(quiet=T),'\n')
      count100<-rep(0,p)
    }
  }
  return(list(theta=theta,s2=s2,count=count))
}
mcmc.block<-function(y1,y2,ss1,ss2,w,nmcmc=10000){
  p<-3 # number of x's
  n<-2 # number of y types
  w.n<-w*n
  require(mnormt)
  theta<-matrix(nrow=nmcmc,ncol=p)
  s2<-matrix(nrow=nmcmc,ncol=p)
  
  theta[1,]<-.5
  s2[1,]<-1
  
  cc<-2.4^2/p
  eps<-1e-10
  S<-diag(p)*eps
  
  count<-0
  a1<-b1<-a2<-b2<-1
  for(i in 2:nmcmc){
    s2[i,1]<-ss1^2#1/rgamma(1,.5+a1+w.n[1]/2,b1+(y1-f(theta[i-1,]))^2*w.n[1]/2)
    s2[i,2]<-ss2^2#1/rgamma(1,.5+a2+w.n[2]/2,b2+(y2-g(theta[i-1,]))^2*w.n[2]/2)
    
    theta[i,]<-theta[i-1,]
    
    if(i>300){# & i<nburn){
      mi<-max(1,i-300)
      S<-cov(theta[mi:(i-1),])*cc+diag(eps*cc,p)
    }
    
    theta.cand<-rmnorm(1,theta[i-1,],S)
    if(any(theta.cand>1) | any(theta.cand<0)){
      alpha<- -9999
    } else{
      alpha<- -.5*( (y1-f(theta.cand))^2*w.n[1]/s2[i,1] + (y2-g(theta.cand))^2*w.n[2]/s2[i,2] - (y1-f(theta[i,]))^2*w.n[1]/s2[i,1] - (y2-g(theta[i,]))^2*w.n[2]/s2[i,2] )
    }
      
    if(log(runif(1)) < alpha){
      theta[i,]<-theta.cand
      count<-count+1
    }
    
    

  }
  return(list(theta=theta,s2=s2,count=count))
}

################################################################################################################

# two different sources of information, one is multimodal, but they can agree (with multimodal result); decrease s2 to make modes more distinct

y1<-1
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1) # use (x[1]-.5)*(x[2]*2-1) or -(x[1]-.5)*(x[2]*2-1) for just two


m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

# two different sources of information, one is multimodal, and they can't agree (with multimodal result); decrease s2 to make modes more distinct

y1<-1.9
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)


y1<-1.9
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  abs(x[1]-.5)*abs(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.05,.95),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

# two different sources of information, one is multimodal, and they can't agree (with multimodal result); decrease s2 to make modes more distinct

y1<-1
y2<-.1

ss1<-.04
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  (x[1]-.5)*(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

y1<-1
y2<-.1

ss1<-.04
ss2<-.04

f<-function(x)
  x[1]+x[2]
g<-function(x)
  (x[1]-.5)*(x[2]*2-1)

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)


# two different sources of information, one is multimodal, and they can't agree (with multimodal result); decrease s2 to make modes more distinct

y1<-1
y2<-0

ss1<-.01
ss2<-.1

f<-function(x)
  x[1]+x[2]
g<-function(x)
  x[1]-x[2]

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)



y1<-1
y2<-0

ss1<-.01
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  10000

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)



y1<-1
y2<-0

ss1<-.01
ss2<-.01

f<-function(x)
  x[1]+x[2]
g<-function(x)
  10000

c(.01,.99)*2/c(.01,.01)^2

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.01,.99),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)


y1<-1
y2<-0

ss1<-.01
ss2<-.00001

f<-function(x)
  x[1]+x[2]
g<-function(x)
  10000

c(.5,.5)*2/c(.01,.00001)^2

m1<-mcmc.block(y1,y2,ss1,ss2,w=c(1,0),nmcmc = 100000)
m2<-mcmc.block(y1,y2,ss1,ss2,w=c(0,1),nmcmc = 100000)
m3<-mcmc.block(y1,y2,ss1,ss2,w=c(.5,.5),nmcmc = 100000)

burn<-1:15000
plot(m2$theta[-burn,1:2],xlim=c(0,1),ylim=c(0,1),cex=.3)
points(m1$theta[-burn,1:2],cex=.3,col=2)
points(m3$theta[-burn,1:2],cex=.3,col=3)

################################################################################

g<-function(x)
  (x[1])+(1-x[2])

test<-apply(expand.grid(seq(0,1,.01),seq(0,1,.01)),1,g)
image(matrix(test,101))







m2<-mcmc.block(y1,y2,ss1,ss2,w,nmcmc = 100000)
m2$count

plot(m2$theta[,1],type='l')
plot(m2$theta[,2],type='l')
plot(m2$theta[,3],type='l')


hist(apply(m2$theta[-burn,],1,f))
abline(v=y1,col=2)
hist(apply(m2$theta[-burn,],1,g))
abline(v=y2,col=2)


m1<-mcmc.step(y1,y2,ss1,ss2,w,100000)
m1$count

plot(m1$theta[,1],type='l')
plot(m1$theta[,2],type='l')
burn<-1:5000
plot(m1$theta[-burn,])

hist(apply(m1$theta[-burn,],1,f))
abline(v=y1,col=2)
hist(apply(m1$theta[-burn,],1,g))
abline(v=y2,col=2)


library(MASS)
contour(kde2d(m2$theta[-burn,1],m2$theta[-burn,2],n = 200),xlim=c(0,1),ylim=c(0,1))
contour(kde2d(m1$theta[-burn,1],m1$theta[-burn,2],n=200),col=2,add=T)
plot(m2$theta[-burn,1:2])
points(m1$theta[-burn,1:2],col=2)




## notes
# in real code, calculation of alpha is wrong, depends on previous state rather than current state and candidate