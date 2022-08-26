from sklearn import cluster
from sklearn import neighbors
import numpy as np
import copy


class pointillism:
    
    def __init__(self):
        self.threshold = 5
        
    def get_centroids_dbscan(self,input_pc,eps,min_samples):
        data = copy.deepcopy(input_pc)
        if not data.size:
            return np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        clustering = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(data[:,:3])
        labels = np.array(clustering.labels_,dtype=np.float32)
        
        unique_cluster = np.unique(labels)
        if unique_cluster[0]==-1:
            unique_cluster=unique_cluster[1:]
        
        centroids = []
        point_list = []
        self_probability = []
        for label in unique_cluster:
            cluster_points = data[labels==label,:]
            centroids.append(np.mean(cluster_points,axis=0))
            point_list.append(cluster_points)
            self_probability.append(len(cluster_points))
        if len(self_probability)!=0:    
            self_probability = np.array(self_probability)/np.max(np.array(self_probability))
        centroids = np.array(centroids)
        noise = np.array(data[labels==-1,:])
        return centroids,noise,point_list,np.array(self_probability),labels
    
    def find_potential(self,pc0,pc1):
        pntcloud0 = copy.deepcopy(pc0)
        pntcloud1 = copy.deepcopy(pc1)
        if not pntcloud0.size or not pntcloud1.size:
            return np.ones(pntcloud0.shape[0]),np.ones(pntcloud1.shape[0])
        pntcloud0_tree = neighbors.KDTree(pntcloud0, metric='euclidean')
        pntcloud1_tree = neighbors.KDTree(pntcloud1, metric='euclidean')
        dist_1, nbr_1 = pntcloud0_tree.query(pntcloud1, k=1)
        dist_0, nbr_0 = pntcloud1_tree.query(pntcloud0, k=1)
        
        neighbours_1 = np.squeeze(pntcloud0[nbr_1]).reshape((len(nbr_1),-1))
        neighbours_0 = np.squeeze(pntcloud1[nbr_0]).reshape((len(nbr_0),-1))
        
       
        den = 1
        power = 2
        theta = 0
        x_coeff = 1/5
        y_coeff = 1/2
        potential_0 = 1/((np.sqrt((x_coeff*abs(pntcloud0[:,0]-neighbours_0[:,0]))**2+(y_coeff*abs(pntcloud0[:,1]-neighbours_0[:,1]))**2)/den)**power+1);
        potential_1 = 1/((np.sqrt((x_coeff*abs(pntcloud1[:,0]-neighbours_1[:,0]))**2+(y_coeff*abs(pntcloud1[:,1]-neighbours_1[:,1]))**2)/den)**power+1);
        
        return np.reshape(potential_0,(np.size(potential_0,0))),np.reshape(potential_1,(np.size(potential_1,0)))

    def find_log_likelihood(self, pntcloud0,pntcloud1,distThreshold):
        if not pntcloud0.size or not pntcloud1.size:
            return np.ones(pntcloud0.shape[0]),np.ones(pntcloud1.shape[0])
        pntcloud0_tree = neighbors.KDTree(pntcloud0, metric='euclidean')
        pntcloud1_tree = neighbors.KDTree(pntcloud1, metric='euclidean')
        dist_1, nbr_1 = pntcloud0_tree.query(pntcloud1, k=1)
        dist_0, nbr_0 = pntcloud1_tree.query(pntcloud0, k=1)
        potential_1 = 1/(dist_1+1)
        potential_0 = 1/(dist_0+1)
        
        distThreshMask_0 = (2<dist_0)*(dist_0<distThreshold)
        distThreshMask_1 = (2<dist_1)*(dist_1<distThreshold)
        
        log_likelihood_0 = np.log(potential_0/(1-potential_0))
        log_likelihood_1 = np.log(potential_1/(1-potential_1))
        
        log_likelihood_0 = log_likelihood_1[nbr_0[:,0]]*distThreshMask_0 + log_likelihood_0*np.invert(distThreshMask_0)
        log_likelihood_1 = log_likelihood_0[nbr_1[:,0]]*distThreshMask_1 + log_likelihood_1*np.invert(distThreshMask_1)
        
        probability_0 = np.exp(log_likelihood_0)/(1+np.exp(log_likelihood_0))
        probability_1 = np.exp(log_likelihood_1)/(1+np.exp(log_likelihood_1))
        
        return probability_0[:,0],probability_1[:,0]

    def find_llpc(self, pc0,pc1,eps,min_samples,takeProjection=False):
        pntcloud0 = copy.deepcopy(pc0)
        pntcloud1 = copy.deepcopy(pc1)
        
        if takeProjection:
            pntcloud0[:,2] = 0
            pntcloud1[:,2] = 0
        centroids_0, noise_0, pointlist_0, self_probability_0,labels_0 = self.get_centroids_dbscan(pntcloud0[:,:3], eps, min_samples)
        centroids_1, noise_1, pointlist_1, self_probability_1,labels_1 = self.get_centroids_dbscan(pntcloud1[:,:3], eps, min_samples)
        [pot_0,pot_1] = self.find_potential(centroids_0,centroids_1)
        llpc_0 = np.log(self_probability_0/(1-self_probability_0+1e-10))+np.log(pot_0/(1-pot_0+1e-10))
        llpc_1 = np.log(self_probability_1/(1-self_probability_1+1e-10))+np.log(pot_1/(1-pot_1+1e-10))
        likelihoods_0 = np.zeros(len(labels_0))
        likelihoods_1 = np.zeros(len(labels_1))
        for i in range(len(centroids_0)):
            likelihoods_0[labels_0==i]=pot_0[i]
        for j in range(len(centroids_1)):
            likelihoods_1[labels_1==j]=pot_1[j]
        return likelihoods_0,likelihoods_1
    
def filter_pts(pc,radius,confidence=None):
    temp = copy.deepcopy(pc)
    dist = np.sqrt(pc[:,0]**2 +  pc[:,1]**2 + pc[:,2]**2)
    temp = pc[dist>radius]
    if confidence is not None:
        cnf = confidence[dist<radius]
        return temp,cnf
    else:
        return temp   

