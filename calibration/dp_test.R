xx<-seq(0,1,length.out=100)
B<-splines2::bSpline(xx,df=6,intercept = T,order=3)

B2<-splines2::bSpline(xx,df=8,intercept = T,order=3)
matplot(B,type='l')
matplot(B2,type='l')

p<-ncol(B)
true.theta1<-rnorm(ncol(B2))
true.theta2<-rnorm(ncol(B2))

nexp<-5

y<-matrix(nrow=length(xx),ncol=nexp)
for(i in 1:3){
  y[,i]<-B2%*%rnorm(ncol(B2),true.theta1,sd=.05)+rnorm(length(xx))*.01
}
for(i in 4:5){
  y[,i]<-B2%*%rnorm(ncol(B2),true.theta2,sd=.05)+rnorm(length(xx))*.01
}

BtBinv<-solve(crossprod(B))

matplot(y,type = 'l')

######################################
## independent calibration

p<-ncol(B)
nmcmc<-10000
theta<-array(dim = c(nmcmc,p,nexp))
theta[1,,]<-rnorm(p*nexp)
s2<-NA
N<-length(xx)*nexp

for(m in 2:nmcmc){

  s2[m]<-1/rgamma(1,N/2,sum((y-B%*%theta[m-1,,])^2)/2)

  for(i in 1:nexp){
    theta[m,,i]<-mnormt::rmnorm(1,BtBinv%*%crossprod(B,y[,i]),s2[m]*BtBinv)
  }

}

plot(sqrt(s2[-c(1:1000)]),type='l')
theta.hat<-apply(theta[-c(1:1000),,],2:3,mean)

matplot(y,type = 'l',lty=1)
matplot(B%*%theta.hat,add=T,type='l',lty=2)


######################################
## clustering calibration
BtB<-crossprod(B)

# initialize
nmcmc<-10000
theta<-theta.star<-array(dim = c(nmcmc,p,nexp))
theta[1,,]<-rnorm(p*nexp)
theta.star[1,,]<-rnorm(p*nexp)
gamma<-matrix(nrow=nmcmc,ncol=nexp)
gamma[1,]<-1:nexp
theta0<-matrix(nrow=nmcmc,ncol=p)
theta0[1,]<-rnorm(p)
S.theta<-array(dim=c(nmcmc,p,p))
S.theta[1,,]<-diag(p)
u<-matrix(nrow=nmcmc,ncol=nexp)
u[1,]<-runif(nexp)
alpha<-NA
alpha[1]<-1
s2<-NA

# useful quantities
N<-length(xx)*nexp
nclust<-NA
nclust<-nexp
lwk<-cumsum(log(1-u[1,]))
K<-nexp
S.theta.inv<-mnormt::pd.solve(S.theta[1,,])

# priors
S<-diag(p)*10
Sinv<-mnormt::pd.solve(S)
mm<-rep(0,p)
v<-2*p
U<-BtB
a.alpha<-1
b.alpha<-5
hist(rgamma(1000000,a.alpha,b.alpha))


for(m in 2:nmcmc){

  # sample sigma^2
  s2[m]<-1/rgamma(1,N/2,sum((y-B%*%theta[m-1,,])^2)/2)

  # sample gamma, cluster membership indicator
  lpmat<-matrix(nrow=nexp,ncol=K)
  for(i in 1:nexp){
    for(k in 1:K){
      lpmat[i,k]<-lwk[k] + sum(dnorm(y[,i],B%*%theta[m-1,,k],sqrt(s2[m]),log=T)) + sum(mnormt::dmnorm(theta.star[m-1,,k],theta0[m-1,],S.theta[m-1,,],log = T))
    }

    lweights=lpmat[i,]-max(lpmat[i,])
    probs<-exp(lweights)/sum(exp(lweights))
    gamma[m,i]<-sample(1:K,size=1,prob=probs)
  }
  nclust<-length(unique(gamma[m,]))

  # sample cluster parameters, theta.star
  for(k in 1:K){
    kind<-which(gamma[m,]==k)
    nk<-length(kind)
    ys<-rep(0,length(xx))
    if(nk>0)
      ys<-rowSums(y[,kind,drop=F])
    Ainv<- mnormt::pd.solve(nk/s2[m]*BtB + S.theta.inv)
    b<-t(B)%*%ys/s2[m]+S.theta.inv%*%theta0[m-1,]
    theta.star[m,,k]<-mnormt::rmnorm(1,Ainv%*%b,Ainv)

    kind<-which(gamma[m,]==k)
    if(length(kind)>0)
      theta[m,,kind]<-theta.star[m,,k]
  }

  # sample hyper mean
  Ainv<-mnormt::pd.solve(nexp*S.theta.inv+Sinv)
  b<-S.theta.inv%*%rowSums(theta[m,,])+Sinv%*%mm
  theta0[m,]<-mnormt::rmnorm(1,Ainv%*%b,Ainv)

  # sample hyper covariance matrix
  TT<-tcrossprod(theta[m,,1]-theta0[m,])
  for(i in 2:nexp)
    TT<-TT+tcrossprod(theta[m,,i]-theta0[m,])
  S.theta[m,,]<-MCMCpack::riwish(v+nexp,U+TT)
  S.theta[m,,][upper.tri(S.theta[m,,])]<-t(S.theta[m,,])[upper.tri(S.theta[m,,])]
  S.theta.inv<-mnormt::pd.solve(S.theta[m,,])

  # sample u's (from stick breaking)
  nk<-NA
  for(k in 1:K){
    kind<-which(gamma[m,]==k)
    nk[k]<-length(kind)
  }
  u[m,]<-rbeta(K,1+nk,alpha[m-1]+cumsum(rev(nk)))
  u[m,K]<-1
  # get log weights
  lwk<-rep(0,K)
  lwk[1]<-log(u[m,1])
  for(k in 2:K)
    lwk[k]<-log(u[m,k])+sum(log(1-u[m,1:(k-1)]))

  # sample alpha
  alpha[m]<-rgamma(1,nclust+a.alpha,b.alpha-log(sum(1-u[m,])))

}

plot(sqrt(s2[-c(1:1000)]),type='l')
gamma[m,]

nburn=2000
heat.mat<-matrix(0,nrow=nexp,ncol=nexp)
for(i in 1:nexp){
  for(j in 1:nexp){
    heat.mat[i,j]<-mean(gamma[nburn:nmcmc,i]==gamma[nburn:nmcmc,j])
  }
}
fields::image.plot(heat.mat,col=rev(heat.colors(100)))


theta.hat<-apply(theta[nburn:nmcmc,,],2:3,mean)
matplot(y,type = 'l',lty=1)
matplot(B%*%theta.hat,add=T,type='l',lty=2)
