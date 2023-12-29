import numpy as np

class ourLDA:
    def get_Sw(self, X, y):
        N = X.shape[1] #number of features
        S_w = np.zeros((N,N))
        class_labels = np.unique(y)
        c = class_labels.shape[0] #number of classes
        #calculate scatter matrix for each class
        for class_ in range(c):
            S_i = np.zeros((N,N))
            class_subset = X[y == class_] #get rows which are a part of the current class
            mean_vector = (np.mean(class_subset, axis=0)).reshape(N, 1) #vector m_i containing
            #means of all features in class i
            for row_idx in range(class_subset.shape[0]):
                x = (class_subset[row_idx, :]).reshape(N, 1)
                S_i += (np.dot((x - mean_vector), np.transpose(x - mean_vector))) #apply formula for within class scatter matrix
            S_w += S_i
        return S_w
    
    def get_Sb(self, X, y):
        N = X.shape[1] #number of features
        m = (np.mean(X, axis=0)).reshape(N,1) #overall mean
        S_b = np.zeros((N,N))
        class_labels = np.unique(y)
        c = class_labels.shape[0] #number of classes
        for class_ in range(c):
            class_subset = X[y == class_]
            n_rows = class_subset.shape[0] #get number of rows which are a part of the current class
            mean_vector = (np.mean(class_subset, axis=0)).reshape(N, 1) #vector m_i containing
            #means of all features in class i
            S_b += n_rows * ((mean_vector - m).dot((mean_vector - m).T)) #apply formula for between class scatter matrix
        return S_b
    
    def get_linear_discriminants(self, S_w, S_b):
        # calculate the eigenvectors and eigenvalues of the matrix ((S_w)^-1)(S_b)
        eig_vals, eig_vecs = np.linalg.eig((np.linalg.inv(S_w)).dot(S_b))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))] #create a list of corresponding
        #eigenvectors and eigenvalues
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        #sort the list by the eigenvalues in decreasing order
        return eig_pairs
    
    def set_lda_components(self, X, y):
        N = X.shape[1] #get number of features
        S_w = self.get_Sw(X, y) #get within class scatter matrix
        S_b = self.get_Sb(X, y) #get between class scatter matrix
        sorted_eigenvecs = self.get_linear_discriminants(S_w, S_b) #get linear discriminants sorted by
        #variance explained in descending order (most descriptive first)
        #get first 2 linear discriminants
        self.W = np.hstack((sorted_eigenvecs[0][1].reshape(N,1), sorted_eigenvecs[1][1].reshape(N,1)))
    
    def get_two_discriminants(self, X):
        return X.dot(self.W)
