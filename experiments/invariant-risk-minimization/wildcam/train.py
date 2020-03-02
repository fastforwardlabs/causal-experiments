import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import autograd
from torch.utils.data import DataLoader
from collections import Counter

class Train:
    def __init__(self, X, Y, X_te, Y_te, net, handler, args):
        self.X = X
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.class_distribution = {}
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def get_distribution(self):
        return self.class_distribution
    
    # Define loss function helpers
    def mean_nll(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(self, logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def _train_irm(self, step, loader_tr, optimizer):
        self.clf.train()
        total_loss = 0 
        nll = 0 
        acc = 0 
        penalty = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            
            y.resize_((y.shape[0], 1))
            
            train_nll = self.mean_nll(out, y.float())
            train_acc = self.mean_accuracy(out, y.float())
            train_penalty = self.penalty(out, y.float())
            
            nll += train_nll.detach().cpu().numpy()
            acc += train_acc.detach().cpu().numpy()
            penalty += train_penalty.detach().cpu().numpy()
                        
            #loss = F.binary_cross_entropy_with_logits(out, y.float())
            loss = train_nll.clone()
            
            weight_norm = torch.tensor(0.).cuda()
            # since feature extraction
            for w in self.clf.fc.parameters():
                weight_norm += w.norm().pow(2)

            #loss = train_nll.clone()
            loss += self.args['optimizer_args']['l2_regularizer_weight'] * weight_norm
            penalty_weight = (self.args['optimizer_args']['penalty_weight'] if step >= self.args['optimizer_args']['penalty_anneal_iters'] else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
            
            total_loss += loss.cpu().item()            
            loss.backward()
            optimizer.step()
            
        return total_loss/len(loader_tr), nll/len(loader_tr), acc/len(loader_tr), penalty/len(loader_tr)
    
    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y,
                                            transform=self.args['transform']['test']),
                               shuffle=True, **self.args['loader_te_args'])
        self.clf.eval()
        total_loss = 0
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                """
                print('prediction x shape {}'.format(x.shape))
                print('prediction y shape {}'.format(y.shape))
                """
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                y.resize_((y.shape[0], 1))
                loss = F.binary_cross_entropy_with_logits(out, y.float())
                total_loss += loss.cpu().item()
                pred = out.max(1)[1]
                if str(self.device) == 'cuda':
                    P[idxs] = pred.cpu()
                else:
                    P[idxs] = pred
    
        return P, total_loss/len(loader_te)

    def check_accuracy(self, X, Y):
        loader = DataLoader(self.handler(X, Y,
                                            transform=self.args['transform']['test']),
                               shuffle=True, **self.args['loader_te_args'])
        self.clf.eval()
        accuracy = 0
        num_batches = 0.0
        
        for x, y, idxs in loader:
            x, y = x.to(self.device), y.to(self.device)
            
            scores, e1 = self.clf(x)
            #print("scores: ", scores)
            y.resize_((y.shape[0], 1))
            #print("y: ", y.float())
            acc = self.mean_accuracy(scores, y.float())
            accuracy += acc.detach().cpu().numpy()
            num_batches += 1.0
        return accuracy/num_batches        
        
    def train(self):        
        n_classes = self.args['n_classes']
        self.clf = self.net(n_classes=n_classes).to(self.device)
        if self.args['fc_only']:
            # for feature extraction using transfer learn
            print("feature extraction")
            optimizer = optim.SGD(self.clf.fc.parameters(), self.args['optimizer_args']['lr'])
            #optimizer = optim.Adam(self.clf.fc.parameters(), betas=(0.9,0.99), lr=0.00005)
        else:
            optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
            
        loader_tr = DataLoader(self.handler(self.X,
                                            self.Y,
                                            transform=self.args['transform']['train']),
                               shuffle=True,
                               **self.args['loader_tr_args'])
        print("step\ttrain loss\ttrain nll\ttrain penalty\ttrain acc\ttest loss\ttest acc")
        train_acc = 0.0
        test_acc = 0.0
        for step in range(self.args['steps']):   
            train_loss, train_nll, train_acc, train_penalty = self._train_irm(step, loader_tr, optimizer)           
            _, test_loss = self.predict(self.X_te, self.Y_te)            
            
            test_acc = self.check_accuracy(self.X_te, self.Y_te)
            
            # calculate mean value - accuracy, nll, penalty
            if step % 10 == 0:                
                print("{}\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(step, round(train_loss, 4), 
                                                                    round(train_nll, 4),
                                                                    round(train_penalty, 4), 
                                                                    round(train_acc, 4),
                                                                    round(test_loss, 4), 
                                                                    round(test_acc, 4)))
        return train_acc, test_acc