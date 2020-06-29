setwd("~/git/immpala/")

dat.header<-list() # header data
dat.chem<-list() # chemistry data
dat.ss<-list() # stress strain data
files<-list()

dirs<-list.dirs('data/ti-6al-4v/Data/',recursive = F)
for(i in 1:length(dirs)){
  dat.ss[[i]]<-dat.header[[i]]<-dat.chem[[i]]<-list()
  files[[i]]<-NA
  ff<-list.files(dirs[i])
  csv.files<-which(substr(ff,nchar(ff)-3,nchar(ff))=='.csv')
  for(j in 1:length(csv.files)){
    dat.ss[[i]][[j]]<-read.csv(paste0(dirs[i],'/',ff[csv.files[j]]),skip=17,colClasses=rep("numeric",2))
    dat.chem[[i]][[j]]<-read.csv(paste0(dirs[i],'/',ff[csv.files[j]]),head=F,skip=5,nrows=11,colClasses=rep("character",2))
    dat.header[[i]][[j]]<-read.csv(paste0(dirs[i],'/',ff[csv.files[j]]),head=F,nrows=3,colClasses="character")
    files[[i]][j]<-ff[csv.files[j]]
  }
}


plot(1,xlim=c(0,.65),ylim=c(0,2000),type='n')
for(i in 1:length(dat.ss)){
  for(j in 1:length(dat.ss[[i]])){
    lines(dat.ss[[i]][[j]])
  }
}

as.numeric(as.character(dat.chem[[1]][[1]][,2]))

dat<-list()
for(i in 1:length(dat.ss)){
  dat[[i]]<-list()
  for(j in 1:length(dat.ss[[i]])){
    dat[[i]][[j]]<-list()
    dat[[i]][[j]]$filename<-files[[i]][j]
    dat[[i]][[j]]$strainType<-substr(names(dat.ss[[i]][[j]])[1],1,1) # true or plastic
    dat[[i]][[j]]$paper<-substring(strsplit(dat.header[[i]][[j]][1,],' Fig')[[1]][1],3)
    dat[[i]][[j]]$temp<-as.numeric(strsplit(dat.header[[i]][[j]][2,],' ')[[1]][3])
    dat[[i]][[j]]$strainRate<-as.numeric(strsplit(dat.header[[i]][[j]][3,],' ')[[1]][4])
    dat[[i]][[j]]$ss<-dat.ss[[i]][[j]]
    dat[[i]][[j]]$chem<-dat.chem[[i]][[j]]
    suppressWarnings(class(dat[[i]][[j]]$chem[,2])<-'numeric')
  }
}


plot(dat[[1]][[1]]$ss)
str(dat[[1]][[1]])

dat.info.true<-dat.info.plastic<-data.frame(filename=character(),paper=character(),temp=numeric(),strainRate=numeric(),chem_Y=numeric(),chem_N=numeric(),chem_C=numeric(),chem_H=numeric(),chem_Fe=numeric(),chem_0=numeric(),chem_Al=numeric(),chem_V=numeric(),chem_Mn=numeric(),chem_Si=numeric(),stringsAsFactors=F)
dat.true<-dat.plastic<-list()
k<-h<-0
for(i in 1:length(dat)){
  for(j in 1:length(dat[[i]])){
    if(dat[[i]][[j]]$strainType=='T'){
      k<-k+1
      dat.true[[k]]<-dat[[i]][[j]]$ss
      dat.info.true[k,1:2]<-c(dat[[i]][[j]]$filename,dat[[i]][[j]]$paper)
      dat.info.true[k,-c(1:2)]<-c(dat[[i]][[j]]$temp,dat[[i]][[j]]$strainRate,dat[[i]][[j]]$chem[-1,2])
    } else{
      h<-h+1
      dat.plastic[[h]]<-dat[[i]][[j]]$ss
      dat.info.plastic[k,1:2]<-c(dat[[i]][[j]]$filename,dat[[i]][[j]]$paper)
      dat.info.plastic[k,-c(1:2)]<-c(dat[[i]][[j]]$temp,dat[[i]][[j]]$strainRate,dat[[i]][[j]]$chem[-1,2])
    }
  }
}

source('~/git/immpala/code/strength_setup_ptw.R')


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

emax   = 0.65
Nhist  = 100

unstandardize<-function(u){
  u*(x.rr[,2]-x.rr[,1])+x.rr[,1]
}

ii=1
dat.calib<-dat.plastic[[ii]]
dat.calib[,2]<-dat.calib[,2]*1e-5
temp.calib<-as.numeric(dat.info.plastic$temp[ii])
strainRate.calib<-as.numeric(dat.info.plastic$strainRate[ii])

# experiment
temp.calib<-293
strainRate.calib<-.01
dat.calib<-f(unstandardize(rep(.5,11)),temp.calib,strainRate.calib)
dat.calib[,2]<-(dat.calib[,2]+rnorm(100,0,.00005))#*1e5

#write.table(dat.calib,file = '../py_calibration/ti64_test.txt',row.names = F,col.names = F)


############################################################################################################
# write results

load('calibration/r_calibration/results/ti64/ps_1.rda')

dat.info.plastic<-data.frame(label=1:nrow(dat.info.plastic),dat.info.plastic)

results<-matrix(0,nrow=200*175,ncol=174+10)
for(i in 1:174){
  results[1:200+(i-1)*200,1:10]<-ps[[1]]$u[seq(2001,3000,5),1:10,1]
  results[1:200+(i-1)*200,10+i]<-1
}

i=175
results[1:200+(i-1)*200,1:10]<-ps[[1]]$u[seq(2001,3000,5),1:10,1]
results[1:200+(i-1)*200,-c(1:10)]<-1
results<-data.frame(results)
nn<-c('theta0', 'p', 's0','sinf',       'y0',      'yinf','y1', 'y2','kappa', 'gamma',paste0('useData_',1:174))
names(results)<-nn


write.table(dat.info.plastic,file='~/git/immpala/data/ti-6al-4v/results/datasets.csv',row.names = F,sep=',')

write.table(results,file='~/git/immpala/data/ti-6al-4v/results/posterior_samples.csv',row.names = F,sep=',')

## how to remove big file from git history, do this before committing a new posterior_samples.csv
# git filter-branch --force --index-filter \
# "git rm --cached --ignore-unmatch ~/git/immpala/data/ti-6al-4v/results/posterior_samples.csv" \
# --prune-empty --tag-name-filter cat -- --all
