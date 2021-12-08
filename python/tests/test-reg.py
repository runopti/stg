print("__name__: {}".format(__name__))
print("__package__: {}".format(__package__))
print("__file__: {}".format(__file__))

import unittest, sys
sys.path.insert(0, "/Users/yutaro/code/stg")
from stg import STG
import stg.utils as utils
from examples.dataset import create_twomoon_dataset, create_sin_dataset

from sklearn.model_selection import train_test_split

import torch
import unittest, sys
import numpy as np
import time

class Test(unittest.TestCase):
    def setUp(self):
        n_size = 1000 #Number of samples
        #p_size = 20   #Number of features
        #X_data, y_data=create_twomoon_dataset(n_size,p_size)
        p_size = 2 #Number of features
        X_data, y_data=create_sin_dataset(n_size, p_size)
        print(X_data.shape)
        print(y_data.shape)
        np.random.seed(123)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, train_size=0.3)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, train_size=0.8)

        self.params = {'hidden_layers_node': [60, 20, 2], 'param_search': False, 
                        'display_step': 1000, 'activation': 'tanh', 'lam': 0.02, 
                        'sigma': 0.5, 'feature_selection': True, 'learning_rate': 0.01, 
                        'output_node': 2}
        # Adjust params
        self.params['lam'] = 0.5
        self.params['learning_rate'] = 0.1
        self.params['input_node'] = self.X_train.shape[1]
        self.params['batch_size'] = self.X_train.shape[0]

    def tearDown(self):
        pass

    def test_torch(self):
        args_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if args_cuda else "cpu")
        device='cpu'
        #print("what is devcie: {}".format(device))
        #torch.backends.cudnn.benchmark = True
        feature_selection = True
        report_maps = {'l2_norm_1st_neuron': lambda x : torch.norm(x.mlp[0][0].weight[:, 0]).item(),
                        'l2_norm_2nd_neuron': lambda x : torch.norm(x.mlp[0][0].weight[:, 1]).item()}
        model = STG(task_type='classification',
                    input_dim=self.X_train.shape[1], 
                    output_dim=2, 
                    hidden_dims=[60, 20], 
                    activation='tanh',
                    optimizer='SGD', 
                    learning_rate=0.1, 
                    batch_size=self.X_train.shape[0], 
                    feature_selection=feature_selection, 
                    sigma=0.5, lam=0.5, random_state=1, 
                    device=device, report_maps=report_maps)
        now = time.time()
        model.fit(self.X_train, self.y_train, nr_epochs=100, valid_X=self.X_valid, valid_y=self.y_valid, print_interval=10)
        print("Passed time: {}".format(time.time() - now))
        if feature_selection:
            print(model.get_gates(mode='prob'))

    def test_lspin(self):
        args_cuda = torch.cuda.is_available()
        device = 'cpu' #torch.device("cuda" if args_cuda else "cpu")
        #torch.backends.cudnn.benchmark = True
        feature_selection = True
        model = STG(task_type='regression',
                input_dim=self.X_train.shape[1],
                output_dim=1, hidden_dims=[60, 20], 
                activation='tanh',
                extra_args={'gating_net_hidden_dims': [30, 30]},
                optimizer='SGD', learning_rate=0.1, batch_size=self.X_train.shape[0], feature_selection=feature_selection, sigma=0.5, lam=0.5, random_state=1, device=device)
        now = time.time()
        model.fit(self.X_train, self.y_train, nr_epochs=3, valid_X=self.X_valid, valid_y=self.y_valid, print_interval=10)
        print("Passed time: {}".format(time.time() - now))
        if feature_selection:
            gates = model.get_gates(mode='prob', x=self.X_train[0].reshape(1, -1))

        import ipdb; ipdb.set_trace()
    
    def test_lspin_train(self):
        np.random.seed(34)
        n_sample = 300
        Xs1 = np.random.normal(loc=1,scale=0.5,size=(n_sample,5))
        Ys1 = -2*Xs1[:,0]+1*Xs1[:,1]-0.5*Xs1[:,2]

        Xs2 = np.random.normal(loc=-1,scale=0.5,size=(n_sample,5))
        Ys2 = -0.5*Xs2[:,2]+1*Xs2[:,3]-2*Xs2[:,4]

        X_data = np.concatenate((Xs1,Xs2),axis=0)
        Y_data = np.concatenate((Ys1.reshape(-1,1),Ys2.reshape(-1,1)),axis=0)

        Y_data = Y_data-Y_data.min()
        Y_data=Y_data/Y_data.max()

        # The ground truth group label of each sample
        case_labels = np.concatenate((np.array([1]*n_sample),np.array([2]*n_sample)))

        Y_data = np.concatenate((Y_data,case_labels.reshape(-1,1)),axis=1)

        # 10% for validation, 10% for test 
        X_train,X_remain,yc_train,yc_remain = train_test_split(X_data,Y_data,train_size=0.8,shuffle=True,random_state=34)
        X_valid,X_test,yc_valid,yc_test = train_test_split(X_remain,yc_remain,train_size=0.5,shuffle=True,random_state=34)

        # Only 10 samples used for training
        X_train,_,yc_train,_ = train_test_split(X_train,yc_train,train_size=10,shuffle=True,random_state=34)

        print("Sample sizes:")
        print(X_train.shape[0],X_valid.shape[0],X_test.shape[0])

        y_train = yc_train[:,0].reshape(-1,1)
        y_valid = yc_valid[:,0].reshape(-1,1)
        y_test = yc_test[:,0].reshape(-1,1)

        train_label = yc_train[:,1]
        valid_label = yc_valid[:,1]
        test_label= yc_test[:,1]

        #dataset = DataSet(**{'_data':X_train, '_labels':y_train,
        #        '_valid_data':X_valid, '_valid_labels':y_valid,
        #        '_test_data':X_test, '_test_labels':y_test})

        # reference ground truth feature matrix (training/test)
        ref_feat_mat_train = np.array([[1,1,1,0,0] if label == 1 else [0,0,1,1,1] for label in train_label])
        ref_feat_mat_test = np.array([[1,1,1,0,0] if label == 1 else [0,0,1,1,1] for label in test_label])

        # params = {     
        #     "input_node" : X_train.shape[1],       # input dimension for the prediction network
        #     "hidden_layers_node" : [100,100,10,1], # number of nodes for each hidden layer of the prediction net
        #     "output_node" : 1,                     # number of nodes for the output layer of the prediction net
        #     "feature_selection" : True,            # if using the gating net
        #     "gating_net_hidden_layers_node": [10], # number of nodes for each hidden layer of the gating net
        #     "display_step" : 500                   # number of epochs to output info
        # }

        args_cuda = torch.cuda.is_available()
        device = 'cpu' #torch.device("cuda" if args_cuda else "cpu")
        #torch.backends.cudnn.benchmark = True
        feature_selection = True
        model = STG(task_type='regression',
                input_dim=X_train.shape[1],
                output_dim=1, 
                hidden_dims=[100, 100, 10, 1], 
                activation='tanh',
                extra_args={'gating_net_hidden_dims': [30]},
                optimizer='SGD', 
                learning_rate=0.2, 
                batch_size=X_train.shape[0], 
                feature_selection=feature_selection, 
                sigma=0.5, 
                lam=0.001, 
                random_state=1, 
                device=device)
        now = time.time()
        model.fit(X_train, y_train, 
                  nr_epochs=20000,
                  valid_X=X_valid,
                  valid_y=y_valid,
                  print_interval=1000)
        print("Passed time: {}".format(time.time() - now))
        if feature_selection:
            #gates = model.get_gates(mode='prob', x=X_train[0].reshape(1, -1))
            sorted_order = np.concatenate((np.where(train_label == 1)[0],np.where(train_label == 2)[0]))
            gates = model.get_gates(mode='prob', x=X_train[sorted_order, :])
            sorted_order_test = np.concatenate((np.where(test_label == 1)[0],np.where(test_label == 2)[0]))
            gates_test = model.get_gates(mode='prob', x=X_test[sorted_order_test, :]) 

        import ipdb; ipdb.set_trace()


if __name__=='__main__':
    unittest.main()  
