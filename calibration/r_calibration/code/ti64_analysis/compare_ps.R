ii=1
load(paste0('ps_',ii,'.rda'))
dat.calib<-dat.plastic[[ii]]
dat.calib[,2]<-dat.calib[,2]*1e-5
temp.calib<-as.numeric(dat.info.plastic$temp[ii])
strainRate.calib<-as.numeric(dat.info.plastic$strainRate[ii])


par(mfrow=c(6,2),mar=c(1,3,1,3))
for(kk in 1:11){
  plot(ps[[1]]$u[,kk,1],type='l',ylim=c(0,1))
}

jj<-3000
plot(dat.calib[,1],dat.calib[,2])
lines(f(unstandardize(ps[[1]]$u[jj,,1]),temp.calib,strainRate.calib))


uu.common<-ps[[1]]$u[jj,,1]




par(mfrow=c(6,3),mar=c(1,3,1,3))
for(ii in 1:170){

  dat.calib<-dat.plastic[[ii]]
  dat.calib[,2]<-dat.calib[,2]*1e-5
  temp.calib<-as.numeric(dat.info.plastic$temp[ii])
  strainRate.calib<-as.numeric(dat.info.plastic$strainRate[ii])
    
  plot(dat.calib[,1],dat.calib[,2])
  lines(f(unstandardize(ps[[1]]$u[jj,,1]),temp.calib,strainRate.calib))
}

files<-list.files()
files<-files[substr(files,1,2)=='ps']
enum<-as.numeric(unlist(strsplit(do.call(rbind,strsplit(files,'_'))[,2],'.rda')))
u.list<-list()
for(i in 1:length(enum)){
  load(files[i])
  u.list[[i]]<-cbind(ps[[1]]$u[seq(2000,3000,10),-11,1],i)
}
u.mat<-do.call(rbind,u.list)


pairs(u.mat[,1:10],xlim=c(0,1),ylim=c(0,1),cex=.1,col=u.mat[,11])


as.numeric(dat.info.plastic$chem_0)[enum]

pdf('ti-6al-4v-analysis/parameters2.pdf',height=10,width=6)
for(kk in 2:ncol(dat.info.plastic)){

  par(mfrow=c(5,2),oma=c(3,3,3,3),mar=c(2,2,2,2)) 
  for(cc in 1:10){
    at<-as.numeric(dat.info.plastic[,kk])[enum]
    boxplot.matrix(matrix(u.mat[,cc],nrow=101,ncol=length(enum)),at=jitter(at),boxwex=.02*diff(range(at,na.rm = T)),xlim=range(at,na.rm = T),border=1:length(enum),names=at,main=nn[cc])
  }
    
  mtext(names(dat.info.plastic)[kk],3,outer = T)
}
dev.off()
#plot(as.numeric(dat.info.plastic$chem_0)[enum],colMeans(matrix(u.mat[,1],nrow=101,ncol=length(enum))))

nn<-c('theta0', 'p', 's0','sinf',       'y0',      'yinf','y1', 'y2','kappa', 'gamma')
png('ti-6al-4v-analysis/parameters.png',height=12,width=14,units='in',res=350)
par(mfrow=c(10,10),oma=c(5,5,5,5),mar=c(1,1,1,1))
for(i in 1:10){
  for(j in 1:10){
    if(i==j){
      plot(1,xlim=c(0,1),ylim=c(0,1),type='n')
      for(kk in 1:length(enum)){
        dd<-density(u.mat[u.mat[,11]==kk,i])
        lines(dd$x,dd$y/max(dd$y),col=kk,main='',xlim=c(0,1))
      }
      mtext(nn[i],1,3)
    }
    if(i<j){
      plot(u.mat[,j],u.mat[,i],ylab='',xlab='',cex=.5,cex.axis=.5,pch=16,xlim=c(0,1),ylim=c(0,1),col=u.mat[,11])
    }
    if(i>j)
      plot.new()
    #if(i==2 & j==1)
    #  legend('bottomleft',c('completed','failed'),fill=c(rgb(1,.5,0,0.5),rgb(0,0,1,0.5)),bty='n')
  }
}
dev.off()
