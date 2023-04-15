import numpy as np, pandas as pd, os, torch
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import timm


class Densenet(torch.nn.Module):
    
    def __init__(self, outputs=2):

        super(Densenet, self).__init__()

        self.DenseNet = timm.create_model('densenet121', pretrained=True, num_classes=0, global_pool='')
        self.post_dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(1048576, outputs))

        self.loss_criterion = torch.nn.CrossEntropyLoss() 

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)

    def forward(self, x):

        x_proj = self.DenseNet(x)
        feat_subset1 = self.post_dropout(x_proj)
        feat_subset2 = self.post_dropout(x_proj)

        # compute and normalize compatibility scores for each pair of non-masked features (by dropout) on each subset
        score = torch.einsum('bkij,bmij->bkm', feat_subset1, feat_subset2)
        score = score.view(score.size(0), -1)
        score = torch.sign(score) * torch.sqrt(torch.abs(score) + 1e-12)
        score = torch.nn.functional.normalize(score)

        y_hat = self.classifier(score)

        return y_hat

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def makeOptimizer(self, epoches, steps_per_epoch, lower_lr, upper_lr):

        if upper_lr is None:
            params = [{'params': self.parameters(), 'lr': lower_lr}]
        else:
            groups = len(self.DenseNet._modules['features']._modules.keys()) 
            lrs = [lower_lr + step*(upper_lr - lower_lr )/groups for step in range(groups)] 
            params = []

            for step, (layer, block) in enumerate(self.DenseNet._modules['features']._modules.items()):
                params += [{'params': block.parameters(), 'lr': lrs[step], 'name':layer}]

        params += [{'params': self.classifier.parameters(), 'lr': upper_lr, 'name':'classifier'}]
        opt = torch.optim.Adam(params, upper_lr)
        
        
        return opt #, scheduler

    def unfreeze(self):
        for parm in self.DenseNet.parameters():
            parm.requires_grad = True
        print('Unfreezing DenseNet')

    def freeze(self, optm, lr, remaining_epoches, steps_per_epoch):
        for parm in self.DenseNet.parameters():
            parm.requires_grad = False

        print('Freezing DenseNet')
        
        for group in range(len(optm.param_groups)):
            if optm.param_groups[group]['name'] == 'classifier':
                optm.param_groups[group]['lr'] = lr
            else:
                optm.param_groups[group]['lr'] = 0
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optm, lr, epochs=remaining_epoches, 
        steps_per_epoch=steps_per_epoch)
        
        return optm , scheduler

    def computeLoss(self, outputs, data):
        return self.loss_criterion(outputs, data['label'].to(self.device) )

def get_lr(optimizer): 
    for group in range(len(optimizer.param_groups)):
            if optimizer.param_groups[group]['name'] == 'classifier':
                return optimizer.param_groups[group]['lr']

def train_model( trainloader, devloader, epoches, batch_size, output, lower_lr, upper_lr=None, freeze_at=25):

    eerror, ef1, edev_error, edev_f1, eloss, dev_loss= [], [], [], [], [], []
    best_f1 = None
    print(f'Change on epoch {freeze_at}')

    SS = []

    model = Densenet(outputs=20)
    
    params_amount = sum([np.prod(param.shape) for param in model.parameters()])
    print('Total number of parameters: ', params_amount)

    optimizer = model.makeOptimizer(epoches=epoches, steps_per_epoch=len(trainloader), lower_lr=lower_lr, upper_lr=upper_lr)
    scheduler_steping = False

    for epoch in range(epoches):
        running_stats = {'preds': [], 'label': [], 'loss': 0.}

        if epoch == freeze_at + 1:
            stw1 = [i for i in model.DenseNet._modules['features']._modules['denseblock1']._modules['denselayer4']._modules['conv2'].parameters()][0].detach().cpu().numpy()
            print(f"Space Shift : {np.linalg.norm(stw - stw1):.3f}")

        model.train()
        if epoch == freeze_at:
            optimizer, scheduler = model.freeze(optm = optimizer, lr=upper_lr, remaining_epoches=epoches-epoch+1, steps_per_epoch = len(trainloader))
            scheduler_steping = True
            stw = [i for i in model.DenseNet._modules['features']._modules['denseblock1']._modules['denselayer4']._modules['conv2'].parameters()][0].detach().cpu().numpy()


        iter = tqdm(enumerate(trainloader, 0))
        iter.set_description(f'Epoch: {epoch:3d}')
        for j, data_batch in iter:

            torch.cuda.empty_cache()         
            inputs, labels = data_batch['imgs'], data_batch['label']

            outputs = model(inputs.to(model.device))
            loss = model.loss_criterion(outputs, labels.type(torch.LongTensor).to(model.device))

            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()
            optimizer.zero_grad()

            if scheduler_steping:
                scheduler.step() 
            SS += [get_lr(optimizer)]
            # print statistics
            with torch.no_grad():

                running_stats['preds'] += torch.max(outputs, 1)[1].detach().cpu().numpy().tolist()
                running_stats['label'] += labels.detach().cpu().numpy().tolist()
                running_stats['loss'] += loss.item()

                f1 = f1_score(running_stats['label'], running_stats['preds'], average='macro')
                error = 1. - accuracy_score(running_stats['label'], running_stats['preds'])
                loss = running_stats['loss'] / (j+1)

                iter.set_postfix_str(f'loss:{loss:.3f} f1:{f1:.3f}, error:{error:.3f}') 

            if j == len(trainloader) - 1:

                model.eval()
                eerror += [error]
                ef1 += [f1]
                eloss += [loss]

                with torch.no_grad():

                    running_stats = {'preds': [], 'label': [], 'loss': 0.}
                    for k, data_batch_dev in enumerate(devloader, 0):
                        torch.cuda.empty_cache() 

                        inputs, labels = data_batch_dev['imgs'], data_batch_dev['label']
                        outputs = model(inputs.to(model.device))

                        running_stats['preds'] += torch.max(outputs, 1)[1].detach().cpu().numpy().tolist()
                        running_stats['label'] += labels.detach().cpu().numpy().tolist()

                        loss = model.loss_criterion(outputs, labels.type(torch.LongTensor).to(model.device))
                        running_stats['loss'] += loss.item()


                f1 = f1_score(running_stats['label'], running_stats['preds'], average='macro')
                error = 1. - accuracy_score(running_stats['label'], running_stats['preds'])
                loss  = running_stats['loss'] / len(devloader)

                edev_error += [error]
                edev_f1 += [f1]
                dev_loss += [loss]

                if best_f1 is None or best_f1 < edev_error:
                    torch.save(model.state_dict(), output) 
                    best_f1 = edev_error

                iter.set_postfix_str(f'loss:{eloss[-1]:.3f} f1:{ef1[-1]:.3f} error:{eerror[-1]:.3f} dev_loss: {loss:.3f} f1_dev:{f1:.3f} dev_error:{error:.3f}') 

    return {'error': eerror, 'f1': ef1, 'dev_error': edev_error, 'dev_f1': edev_f1}, model, SS