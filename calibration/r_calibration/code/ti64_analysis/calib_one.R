emax   = 0.65
Nhist  = 100
source('~/git/immpala/calibration/r_calibration/code/strength_setup_ptw.R')

calib<-function(experiment,nchains=1,nmcmc=3000,nburn=2000){
  
  ###################################
  ## get data
  
  dat.header<-list() # header data
  dat.chem<-list() # chemistry data
  dat.ss<-list() # stress strain data
  
  dirs<-list.dirs('~/git/immpala/data/ti-6al-4v/Data/',recursive = F)
  for(i in 1:length(dirs)){
    dat.ss[[i]]<-dat.header[[i]]<-dat.chem[[i]]<-list()
    ff<-list.files(dirs[i])
    csv.files<-which(substr(ff,nchar(ff)-3,nchar(ff))=='.csv')
    for(j in 1:length(csv.files)){
      dat.ss[[i]][[j]]<-read.csv(paste0(dirs[i],'/',ff[csv.files[j]]),skip=17,colClasses=rep("numeric",2))
      dat.chem[[i]][[j]]<-read.csv(paste0(dirs[i],'/',ff[csv.files[j]]),head=F,skip=5,nrows=11,colClasses=rep("character",2))
      dat.header[[i]][[j]]<-read.csv(paste0(dirs[i],'/',ff[csv.files[j]]),head=F,nrows=3,colClasses="character")
    }
  }
  
  dat<-list()
  for(i in 1:length(dat.ss)){
    dat[[i]]<-list()
    for(j in 1:length(dat.ss[[i]])){
      dat[[i]][[j]]<-list()
      dat[[i]][[j]]$strainType<-substr(names(dat.ss[[i]][[j]])[1],1,1) # true or plastic
      dat[[i]][[j]]$paper<-strsplit(dat.header[[i]][[j]][1,],' ')[[1]][2]
      dat[[i]][[j]]$temp<-as.numeric(strsplit(dat.header[[i]][[j]][2,],' ')[[1]][3])
      dat[[i]][[j]]$strainRate<-as.numeric(strsplit(dat.header[[i]][[j]][3,],' ')[[1]][4])
      dat[[i]][[j]]$ss<-dat.ss[[i]][[j]]
      dat[[i]][[j]]$chem<-dat.chem[[i]][[j]]
      suppressWarnings(class(dat[[i]][[j]]$chem[,2])<-'numeric')
    }
  }
  
  
  dat.info.true<-dat.info.plastic<-data.frame(paper=character(),temp=numeric(),strainRate=numeric(),chem_Y=numeric(),chem_N=numeric(),chem_C=numeric(),chem_H=numeric(),chem_Fe=numeric(),chem_0=numeric(),chem_Al=numeric(),chem_V=numeric(),chem_Mn=numeric(),chem_Si=numeric(),stringsAsFactors=F)
  dat.true<-dat.plastic<-list()
  k<-h<-0
  for(i in 1:length(dat)){
    for(j in 1:length(dat[[i]])){
      if(dat[[i]][[j]]$strainType=='T'){
        k<-k+1
        dat.true[[k]]<-dat[[i]][[j]]$ss
        dat.info.true[k,]<-c(dat[[i]][[j]]$paper,dat[[i]][[j]]$temp,dat[[i]][[j]]$strainRate,dat[[i]][[j]]$chem[-1,2])
      } else{
        h<-h+1
        dat.plastic[[h]]<-dat[[i]][[j]]$ss
        dat.info.plastic[h,]<-c(dat[[i]][[j]]$paper,dat[[i]][[j]]$temp,dat[[i]][[j]]$strainRate,dat[[i]][[j]]$chem[-1,2])
      }
    }
  }
  

  
  
  
  
  model='PTW'
  x.rr<-matrix(c(.001,.1,
                 0,10,
                 .001,.05,
                 .001,.05,
                 0,.05,
                 0,.05, 
                 0,.1, 
                 0,1,
                 .01,1,
                 10^-6,10^-1,
                 0,1 # test
  ),ncol=2,byrow=T)
  p=nrow(x.rr)
  

  
  unstandardize<-function(u){
    u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
  }
  
  dat.calib<-dat.plastic[[experiment]]
  dat.calib[,2]<-dat.calib[,2]*1e-5
  temp.calib<-as.numeric(dat.info.plastic$temp[experiment])
  strainRate.calib<-as.numeric(dat.info.plastic$strainRate[experiment])
  
  
  ###################################
  ## functions
  
  astep.mat<-function(sz.mat,ct.mat){
    for(s in 1:ncol(sz.mat)){
      sz.mat[,s]<-astep(sz.mat[,s],ct.mat[,s])
    }
    sz.mat
  }
  
  astep<-function(sz,ct){
    for(s in 1:length(sz)){
      if(ct[s]<20)
        sz[s]<-sz[s]*.5
      if(ct[s]>30)
        sz[s]<-sz[s]*1.1
    }
    return(sz)
  }
  
  
  
  get.se<-function(uu){
    #se<-NA
    #  unlist(mclapply(1:length(dat.plastic),function(i){
    #for(i in 1:length(dat.plastic)){
    mm<-f(unstandardize(uu),temp.calib,strainRate.calib)
    mm.dat<-approx(mm[,1],mm[,2],xout=dat.calib[,1],rule=c(2,2))$y
    sum((mm.dat-dat.calib[,2])^2)
    # },mc.cores=4))
  }
  
  # R<-exp(-as.matrix(dist(dat.calib[,1]))^2/.01^2)+diag(length(dat.calib[,1]))*1e-10
  # R[1:10,1:10]
  # Rinv<-solve(R)
  # get.se<-function(uu){
  #   #se<-NA
  #   #  unlist(mclapply(1:length(dat.plastic),function(i){
  #   #for(i in 1:length(dat.plastic)){
  #   mm<-f(unstandardize(uu),temp.calib,strainRate.calib)
  #   mm.dat<-approx(mm[,1],mm[,2],xout=dat.calib[,1],rule=c(2,2))$y
  #   c(t(mm.dat-dat.calib[,2])%*%Rinv%*%(mm.dat-dat.calib[,2]))
  #   # },mc.cores=4))
  # }
  
  system.time(aa<-get.se(rep(.5,11)))
  
  get.alpha<-function(uu,s2.hop,se.hop.curr,itemp){
    se.hop.cand<-get.se(uu)
    alpha<- -.5*(sum(se.hop.cand/s2.hop) - sum(se.hop.curr/s2.hop) )*itemp
    return(list(alpha=alpha,se.hop=se.hop.cand))
  }
  
  
  constraints<-function(model,uu){
    if(any(uu<0 | uu>1))
      return(T)
    
    uu<-unstandardize(uu)
    if(model=='PTW'){
      if(uu[4]>uu[3] | uu[5]>uu[3] | uu[5]<max(uu[6],0) | uu[6]>min(uu[4],uu[5]) | uu[7]<uu[3])#if(uu[4]>uu[3] | uu[7]<uu[8] | uu[7]>uu[3] | uu[8]>min(uu[7],uu[4]) )#| uu[9]<uu[3] | uu[10]<uu[11])
        return(T)
    }
    return(F)
  }
  
  ###################################
  ## MCMC
  
  mcmc<-function(){
    require(BASS)
    require(mnormt)
    
    itemp.ladder<-1/(1.5^(0:8))#c(1,.8,.6,.4,.2,.1,.05,1/20)
    start.temper=100
    
    ntemp<-ntemps<-length(itemp.ladder)
    
    u<-array(dim=c(nmcmc,p,ntemp))
    
    for(tt in 1:ntemps){
      start.u<-runif(p)
      cont<-constraints(model,start.u)
      while(cont){
        start.u<-runif(p)
        cont<-constraints(model,start.u)
      }
      u[1,,tt]<-start.u
    }
    
    n<-length(dat.plastic[[experiment]][,1])
    n.hop=1
    s2.hop<-matrix(nrow=nmcmc,ncol=tt)
    se.hop<-matrix(nrow=nmcmc,ncol=tt)
    
    a.hop<-rep(0,n.hop)
    b.hop<-rep(0,n.hop)
    
    # # hop variance: mean: .0003, sd=.0003 a<-100, b<-1e-5
    # hist(sqrt(1/rgamma(100000,8,1e-7)))
    a.hop<-rep(8,n.hop)
    b.hop<-rep(1e-7,n.hop)
    
    for(tt in 1:ntemp){
      cand<-get.alpha(t(u[1,,tt]),rep(1,n.hop),rep(1,n.hop),itemp.ladder[tt])
      se.hop[1,tt]<-cand$se.hop
    }
    
    
    stepsize<-matrix(.5,nrow=p,ncol=ntemps)
    
    count<-count100<-matrix(0,nrow=p,ncol=ntemps)
    count.swap<-0
    swap.vals<-matrix(nrow=nmcmc,ncol=2)
    
    lpost<-function(s2.hop,se.hop){
      (-n/2-a.hop-1)*log(s2.hop) +
        -1/s2.hop*(se.hop/2 + b.hop)
    }
    
    for(i in 2:nmcmc){
      
      for(tt in 1:ntemps)
        s2.hop[i,tt]<-BASS:::rigammaTemper(1,n/2+a.hop,b.hop+.5*sum(se.hop[i-1,tt]),itemper = itemp.ladder[tt])
      
      u[i,,]<-u[i-1,,]
      se.hop[i,]<-se.hop[i-1,]
      
      
      for(tt in 1:ntemps){
        
        for(jj in 1:p){
          u.cand<-u[i,,tt]
          u.cand[jj]<-runif(1,max(u.cand[jj]-stepsize[jj,tt],0),min(u.cand[jj]+stepsize[jj,tt],1))
          
          if(constraints("PTW",u.cand))
            alpha<- -9999
          else{
            cand<-get.alpha(t(u.cand),s2.hop[i,tt],se.hop[i-1,tt],itemp.ladder[tt])
            alpha<-cand$alpha
            if(is.na(alpha))
              alpha<- -9999
          }
          
          if(log(runif(1)) < alpha){
            u[i,,tt]<-u.cand
            se.hop[i,tt]<-cand$se.hop
            count[jj,tt]<-count[jj,tt]+1
            count100[jj,tt]<-count100[jj,tt]+1
          }
        }
        
      }
      
      if(i>start.temper & ntemp>1){
        #browser()
        swap<-sample(1:ntemp,size=2) # candidate swap
        alpha<-(itemp.ladder[swap[2]]-itemp.ladder[swap[1]])*(lpost(s2.hop[i,swap[1]],se.hop[i,swap[1]])-lpost(s2.hop[i,swap[2]],se.hop[i,swap[2]]))
        if(log(runif(1))<alpha){ # swap states
          temp<-u[i,,swap[1]]
          u[i,,swap[1]]<-u[i,,swap[2]]
          u[i,,swap[2]]<-temp
          
          temp<-se.hop[i,swap[1]]
          se.hop[i,swap[1]]<-se.hop[i,swap[2]]
          se.hop[i,swap[2]]<-temp
          
          temp<-s2.hop[i,swap[1]]
          s2.hop[i,swap[1]]<-s2.hop[i,swap[2]]
          s2.hop[i,swap[2]]<-temp
          
          
          count.swap<-count.swap+1
          swap.vals[i,]<-sort(swap)
        }
      }
      
      
      
      if(i%%100==0){
        if(i<nburn & i>100)
          stepsize<-astep(stepsize,count100)
        cat('it:',i,'acc:',count.swap,timestamp(quiet=T),'\n')
        count100<-matrix(0,nrow=p,ncol=ntemps)
      }
      
      
    }
    
    return(list(u=u,count=count,s2.hop=s2.hop,se.hop=se.hop,count=count,count.swap=count.swap,swap.vals=swap.vals))
  }
  

  
  library(parallel)
  ps<-mclapply(1:nchains,function(x) mcmc(),mc.cores = 1)
  
  save(ps,experiment,file = paste0('~/git/immpala/calibration/r_calibration/results/ps_',experiment,'.rda'))
  
  #return(list(ps=ps,experiment=experiment))
  return(NULL)
  
}

#aa=calib(1,nchains=2)
#aa<-mclapply(1:nexp,function(i) calib(i,nchains=2),mc.cores = 2,mc.preschedule = F)

library(parallel)

nexp<-174
aa<-mclapply(1:nexp,function(i) calib(i,nchains=1),mc.cores = 20,mc.preschedule = F)

save.image('~/git/immpala/calibration/r_calibration/results/inversion_trial1.rda')
