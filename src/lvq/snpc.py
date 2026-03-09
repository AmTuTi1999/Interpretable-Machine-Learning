import numpy as np
class SNPC:
    def __init__(self, num_prototypes_per_class, initialization_type = 'mean', sigma = 1, learning_rate = 0.05,max_iter = 100, cat_full = False, test_data = None, test_labels = None):

        self.max_iter = max_iter 
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_prototypes = num_prototypes_per_class
        self.cat_full = cat_full
        self.sigma = sigma
        self.initialization_type = initialization_type
        self.alpha = learning_rate
    
    
    

    def initialization(self, train_data, train_labels):
        if self.initialization_type == 'mean':
            """Prototype initialization: if number of prototypes is 1, prototype initialised is the mean
            if prototype is n>1, prototype initilised is the mean plus n-1 points closest to mean"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    list1.append(mu)#.astype(int))
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                if self.cat_full == True:
                    P = np.array(prototypes)
                else:
                    P = np.array(prototypes) + (0.02 *self.sigma*self.num_prototypes*np.random.uniform(low = -1.0, high = 1.0, size = 1))
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    if self.cat_full == True:
                        mu = mu#.astype(int)
                        distances = [self.indicator_dist(mu, c) for c in class_data]
                    else:
                        distances = [(mu-c)@(mu-c).T for c in class_data]
                    index = np.argsort(distances)
                    indices = index[1:self.num_prototypes]
                    prototype = class_data[indices]
                    r = np.vstack((mu, prototype))
                    list2.append(r)
                    ind = []
                    for j in range(self.num_prototypes ):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)#.flatten()   
                prototypes = M.reshape(num_protos,num_dims)
                if self.cat_full == True:
                    P = np.array(prototypes)
                else:
                    P = np.array(prototypes) + (0.02 *self.sigma*self.num_prototypes*np.random.uniform(low = -1.0, high = 1.0, size = 1))
            return np.array(new_labels).flatten(), P
        
        elif self.initialization_type == 'random':
            """Prototype initialization random: randomly chooses n points per class"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            proto_labels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    random_int = np.random.choice(np.array(index))
                    prototype = train_data[random_int]
                    list1.append(prototype)
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                #regulate the prototypes, could also be done with GMM
                P = np.array(prototypes) + (0.02 *self.sigma*self.num_prototypes*np.random.uniform(low = -1.0, high = 1.0, size = 1))
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    random_integers = np.random.choice(np.array(index), size=self.num_prototypes)
                    prototype = train_data[random_integers]
                    list2.append(prototype)
                    ind = []
                    for j in range(self.num_prototypes):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)  
                prototypes = M.reshape(num_protos,num_dims)
                P = np.array(prototypes) + (0.02 *self.sigma*self.num_prototypes*np.random.uniform(low = -1.0, high = 1.0, size = 1))
            return np.array(new_labels).flatten(), P



    def inner_f(self, x, p):
       

        coef = -1/(2*(self.sigma *self.sigma))
        dist = (x -p)@(x- p).T
        return coef*dist

    def inner_derivative(self, x, p):
    
        coef = 1/(self.sigma *self.sigma)

        diff = (x -p) 
        return coef*diff
        
    def Pl(self, x, index):
        inner = np.exp(np.array([self.inner_f(x, p) for p in  self.prototypes]))# + 1e-10
        numerator = np.exp(np.array(self.inner_f(x, self.prototypes[index])))# +1e-10
        denominator = inner.sum()
        return numerator/(denominator) 

    def lst(self, x, x_label):
        u = np.exp(np.array([self.inner_f(x, self.prototypes[i]) for i in range(len(self.prototypes)) if x_label != self.proto_labels[i]]))
        inner = np.exp(np.array([self.inner_f(x, p) for p in  self.prototypes])) #+ 1e-10
        den = inner.sum()
        num = u.sum()
        return num/den

    def gradient_descent(self, data,labels,prototypes, proto_labels):
        
        # = init(data, labels, num_prototypes)
        for i in range(len(data)):
            xi = data[i]
            x_label = labels[i]
            for j in range(prototypes.shape[0]):
                d = (xi - prototypes[j])
                c = 1/(self.sigma*self.sigma) 
                if self.proto_labels[j] == x_label:
                    self.prototypes[j] += self.alpha*(self.Pl(xi, j)*self.lst(xi, x_label))*c*d
                else:
                    self.prototypes[j] -= self.alpha*(self.Pl(xi,j)*(1 - self.lst(xi,x_label)))*c*d
            
        return self.prototypes
    def Error_function(self,prototypes, data, labels):
    
        numerator = []
        denominator = len(data)
        
        
        for i in range(len(data)):
            #prototypes = gradient_ascent(data, labels, epochs)
            xi = data[i]
            x_label = labels[i]
            for j in range(len(prototypes)):
                if x_label != self.proto_labels[j]:
                    numerator.append(self.Pl(xi, j))
            
                
                
        a = np.sum(np.array(numerator))
    

        
                
        return a/denominator
    
    def fit(self, train_data, train_labels, show_plot = False):
        self.proto_labels, self.prototypes = self.initialization(train_data, train_labels)
        self.prototypes = self.prototypes.astype(float)
        import matplotlib.pyplot as plt
        loss =[]
        iter = 0

        while iter < self.max_iter:
            self.prototypes = self.gradient_descent(train_data, train_labels, self.prototypes, self.proto_labels)
            predicted = []
            for i in range(len(train_data)):
                predicted.append(self.predict(train_data[i]))
            val_acc = (np.array(predicted) == np.array(train_labels).flatten()).mean() * 100  
            lr = self.Error_function(self.prototypes, train_data, train_labels)
            print(f'Acc.......{val_acc}, loss......{lr}')
            loss.append(lr)
            iter += 1
            
        if show_plot  == True:
            plt.plot(loss)
            plt.ylabel('log likelihood ratio')
            plt.xlabel(' number of iterations')
        return self.prototypes, self.proto_labels
    

    def predict_all(self, data, return_scores = False):

        """predict an array of instances""" 
        label = []
        #prototypes, _ = RSLVQ(data, labels, num_prototypes, max_iter)
        if return_scores == False:
            for i in range(data.shape[0]):
                xi = data[i]
                distances = np.array([np.linalg.norm(xi - p) for p in self.prototypes])
                index = np.argwhere(distances == distances.min())
                x_label = self.proto_labels[index]
                label.append(x_label)
            return np.array(label).flatten()
        else:
            predicted = []
            for i in range(len(data)):
                predicted.append(self.proba_predict(data[i]))
            return predicted 

    


    def predict(self, input):
        """predicts only one output at the time, numpy arrays only, 
        might want to convert"""
        
   


       
         
   
        distances = np.array([np.linalg.norm(input - p) for p in self.prototypes])
        index = np.argmin(distances)
        x_label = self.proto_labels[index]
        
        return x_label
    
    def proba_predict(self, input):
        """probabilistic prediction of a point by approximation of distances of a point to closest prototypes
        the argmin is the desired class"""
        scores = []
        closest_prototypes = []
        for i in np.unique(self.proto_labels):
            label_prototypes = self.prototypes[np.flatnonzero(self.proto_labels == i)]
            distances = np.array([np.linalg.norm(input - label_prototypes[j]) for j in range(label_prototypes.shape[0])])
            closest_prototype = label_prototypes[np.argmin(distances)]
            closest_prototypes.append(closest_prototype)
        dists = np.array([np.linalg.norm(input - prototype) for prototype in closest_prototypes])
        scores = np.array([d/dists.sum() for d in dists])
        return scores 
    


    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = []
        for i in range(len(test_data)):
            predicted.append(self.predict(test_data[i]))
        val_acc = (np.array(predicted) == np.array(test_labels).flatten()).mean() * 100 
        return val_acc

    
    def Pl_loss(self, unit, target_class):
        #updated_prototypes = self.fit()
        index = np.flatnonzero(self.proto_labels != target_class)[0]

        u = []
        for i in range(len(self.prototypes)):
            if target_class == self.proto_labels[i]:
                u.append(np.exp(self.inner_f(unit, self.prototypes[i])))
            else:
                u.append(0)
            numerator = np.array(u).sum()
        denominator = np.sum(np.array([np.exp(self.inner_f(unit, p)) for p in self.prototypes]))
        
        return numerator/denominator 


    


