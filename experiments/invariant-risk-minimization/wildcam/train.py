import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import autograd
from torch.utils.data import DataLoader
from collections import Counter


class Train:
    def __init__(self, X, Y, X_te, Y_te, net, handler, args):
        """
        """
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
   
    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        total_loss = 0 
        nll = 0 
        acc = 0 
        penalty = 0
        num_batches = 0.0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            """
            print('_train x shape {}'.format(x.shape))
            print('_train y shape {}'.format(y.shape))
            """
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            #print('output shape {}'.format(out.shape))
            #print('output[0] {}'.format(out[0]))
            #print('target shape {}'.format(y.shape))
            #print('target[0] {}'.format(y[0].float()))
            
            y.resize_((y.shape[0], 1))
            train_nll = self.mean_nll(out, y.float())
            train_acc = self.mean_accuracy(out, y.float())
            train_penalty = self.penalty(out, y.float())
            
            nll += train_nll.detach().cpu().numpy()
            acc += train_acc.detach().cpu().numpy()
            penalty += train_penalty.detach().cpu().numpy()
            num_batches += 1.0
            #print('nll {}'.format(nll), 'acc {}'.format(acc), 'penalty {}'.format(penalty))
            
            loss = F.binary_cross_entropy_with_logits(out, y.float())
            total_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()
            
            #print(type(nll.mean().detach().cpu().numpy()))
        #print("train acc: ", acc)
        #print("train num_batches: ", num_batches)  
        
        return total_loss/len(loader_tr), nll/len(loader_tr), acc/num_batches, penalty/len(loader_tr)

    def _train_irm(self, epoch, loader_tr, optimizer):
        self.clf.train()
        total_loss = 0 
        nll = 0 
        acc = 0 
        penalty = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            """
            print('_train x shape {}'.format(x.shape))
            print('_train y shape {}'.format(y.shape))
            """
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            #print('output shape {}'.format(out.shape))
            #print('output[0] {}'.format(out[0]))
            #print('target shape {}'.format(y.shape))
            #print('target[0] {}'.format(y[0].float()))
            
            y.resize_((y.shape[0], 1))
            
            train_nll = self.mean_nll(out, y.float())
            train_acc = self.mean_accuracy(out, y.float())
            train_penalty = self.penalty(out, y.float())
            
            nll += train_nll.detach().cpu().numpy()
            acc += train_acc.detach().cpu().numpy()
            penalty += train_penalty.detach().cpu().numpy()
            
            #print('nll {}'.format(nll), 'acc {}'.format(acc), 'penalty {}'.format(penalty))
            
            #loss = F.binary_cross_entropy_with_logits(out, y.float())
            loss = train_nll.clone()
            
            weight_norm = torch.tensor(0.).cuda()
            # since feature extraction
            for w in self.clf.fc.parameters():
                weight_norm += w.norm().pow(2)

            #loss = train_nll.clone()
            loss += self.args['optimizer_args']['l2_regularizer_weight'] * weight_norm
            penalty_weight = (self.args['optimizer_args']['penalty_weight'] if epoch >= self.args['optimizer_args']['penalty_anneal_iters'] else 1.0)
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
        # Return the fraction of datapoints that were correctly classified.
        # print("acc: ", accuracy)
        # print("num_batches: ", num_batches)  

        return accuracy/num_batches        
        
    def train(self):
        n_epoch = self.args['n_epoch']
        n_classes = self.args['n_classes']
        self.clf = self.net(n_classes=n_classes).to(self.device)
        #print(self.clf)
        if self.args['fc_only']:
            # for feature extraction using transfer learn
            print("feature extraction")
            optimizer = optim.SGD(self.clf.fc.parameters(), self.args['optimizer_args']['lr'])
            #optimizer = optim.Adam(self.clf.fc.parameters(), betas=(0.9,0.99), lr=0.00005)
        else:
            optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
            
        # compute distribution of labels
        all_y = self.Y
        self.class_distribution = dict(Counter(all_y.numpy()))
        loader_tr = DataLoader(self.handler(self.X,
                                            self.Y,
                                            transform=self.args['transform']['train']),
                               shuffle=True,
                               **self.args['loader_tr_args'])
        print("epoch\ttrain loss\ttrain nll\ttrain penalty\ttrain acc\ttest loss\ttest acc")
        for epoch in range(1, n_epoch+1):
            if self.args['mode'] == 'IRM':
                train_loss, train_nll, train_acc, train_penalty = self._train_irm(epoch, loader_tr, optimizer)
            else:
                train_loss, train_nll, train_acc, train_penalty = self._train(epoch, loader_tr, optimizer)
            
            _, test_loss = self.predict(self.X_te, self.Y_te)
            
            #train_acc = self.check_accuracy(self.X, self.Y)
            test_acc = self.check_accuracy(self.X_te, self.Y_te)
            print("{}\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(epoch, round(train_loss, 4), round(train_nll, 4),
                                                                round(train_penalty, 4), round(train_acc, 4), 
                                                                round(test_loss, 4), round(test_acc, 4)))

    def sample_embeddings(self, q_idxs):
        # extract embeddings for samples indexed by q_idxs
        loader_sample = DataLoader(self.handler(self.X[q_idxs],
                                                self.Y[q_idxs],
                                                transform=self.args['transform']['test']),
                                   shuffle=True,
                                   **self.args['loader_sample_args'])
        # get embeddings
        self.clf.eval()
        emb = np.zeros((len(q_idxs), self.clf.get_embedding_dim()))
        with torch.no_grad():
            for x, y, idxs in loader_sample:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                emb[idxs] = e1

        return emb

    def sample_images(self, q_idxs, n_images):
        # extract embeddings for these new samples
        sample_images = self.handler(self.X[q_idxs],
                                     self.Y[q_idxs])
        images = [x for x, y, idxs in sample_images if idxs < n_images]
        return images

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y,
                                            transform=self.args['transform']['test']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        # probs = torch.zeros([len(Y), len(np.unique(Y))])
        # corner case for caltech dataset, the remaining training data after multiple rounds
        # of active learning comes from less than 10 classes (class 6 for example, does not have
        # training data left to pick
        probs = torch.zeros([len(Y), self.args['n_classes']])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                # get probabilities by computing softmax of the output
                prob = F.softmax(out, dim=1)
                if str(self.device) == 'cuda':
                    # print("predict proba index {}".format(idxs.shape))
                    # print("predict proba prob shape {}".format(prob.shape))
                    # print("predict proba probs shape {}".format(probs.shape))
                    probs[idxs] = prob.cpu()
                else:
                    probs[idxs] = prob

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        # each n_drop is a mask
        # run multiple mask to estimate uncertainty
        loader_te = DataLoader(self.handler(X, Y,
                                            transform=self.args['transform']['test']),
                               shuffle=False, **self.args['loader_te_args'])
        # set to train mode to get dropout masks
        self.clf.train()
        probs = torch.zeros([len(Y), self.args['n_classes']])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    # add prob across n_drop
                    if str(self.device) == 'cuda':
                        probs[idxs] += prob.cpu()
                    else: 
                        probs[idxs] += prob
        probs /= n_drop

        return probs
                    
    def set_clf(self, path):
        # model trained on cpu, load to either CPU or GPU
        n_classes = self.args['n_classes']
        # self.clf = self.net(n_classes=n_classes).to(self.device)
        self.clf = self.net(n_classes=n_classes)
        print("loading model")
        self.clf.load_state_dict(torch.load(path, map_location=self.device))
        self.clf.to(self.device)