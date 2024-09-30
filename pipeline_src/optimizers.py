import gc
import torch
import random
from copy import deepcopy
from collections import defaultdict


from collections import defaultdict

import multiprocessing
import time


def dict_to_device(d, device='cuda'):
    out = {}
    for k, val in d.items():
        if isinstance(val, dict):
            out[k] = dict_to_device(val)
        else:
            out[k] = val.to(device)

    return out


class MeritFedBaseSGD(torch.optim.Optimizer):
    def __init__(self, params, config, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=config.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.device = device
        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.config = config
        self.i = 0
        # self.device = device
        self.weights = torch.ones(self.config.npeers, requires_grad=False, device=device) / self.config.npeers
        self.weights_grad = torch.zeros_like(self.weights)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(config.npeers):
                    state[w_id] = torch.zeros_like(p.data, requires_grad=False)

                # state['step'] = 0
                state['p'] = torch.zeros_like(p.data, requires_grad=False)
                state['g'] = torch.zeros_like(p.data, requires_grad=False)

    def update(self, p, group, weights=None):
        pass

    def step(self, w_id, model, criterion, loader):
        # j = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    # state[w_id].data.copy_(state[w_id] + float('nan'))
                    state[w_id].mul_(0)
                    # j += 1
                else:
                    state[w_id].data.copy_(p.grad.data)
        self.grads_received += 1
        # print(w_id, j)

        if self.grads_received != self.config.npeers: 
            return
        # print()

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'].data.copy_(p.data)

        for i, batch in enumerate(loader):
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    out = self.update(p, group)
                    if out is None:
                        continue
                    p.data = out

            model_input = dict_to_device(batch['model_input'], device)
            output = model.forward(**model_input)
            self.zero_grad()
            # if hasattr(output, "logits"):
            #     output = output.logits
            try:
                loss = output["loss"]
            except IndexError:
                loss = criterion(output, batch[-1].to(self.device))
            loss.backward()

            # self.weights_grad.mul_(0)
            # for group in self.param_groups:
            #     for p in group['params']:
            #         state = self.state[p]
            #         for w_id in range(self.config.npeers):
            #             self.weights_grad[w_id] = self.weights_grad[w_id].add(torch.sum(p.grad.data * -group['lr']*state[w_id]))
            # print(self.weights_grad)

            self.weights_grad.mul_(0)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]

                    # try:
                    #     for w_id in range(self.config.npeers):
                    #         if torch.isnan(state[w_id]).any():
                    #             raise ValueError()
                    # except ValueError:
                    #     continue
                    
                    # def f(weights):
                    #     state['g'].data.mul_(0)
                    #     for w_id in range(self.config.npeers):
                    #         state['g'].add_(state[w_id].data * weights[w_id])
                    #     grad = state['g']
                    #     return state['p'] - group['lr'] * grad

                    f = lambda w: self.update(p, group, w)                  
                    _, dp_dw = torch.autograd.functional.vjp(func=f, inputs=self.weights, v=p.grad.data, strict=True)
                    dp_dw.detach_()
                    # self.weights.detach_()
                    # if not torch.isnan(dp_dw).any():
                    #     self.weights_grad.add_(dp_dw)
                    # else:
                    #     raise RuntimeError('dp_dw must not contain nans')
            # print(self.weights_grad)
            # print()

            step = -self.config.mdlr_ * self.weights_grad
            if self.config.mdnorm_:
                step /= torch.norm(self.weights_grad)
            # step = -self.config.mdlr_ * self.weights_grad
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = vec / torch.sum(vec)
            # if i == self.config.mdniters_-1:
            #     break
            # print(self.weights_grad)

        self.grads_received = 0
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                state = self.state[p]
                p.data.copy_(state['p'].data)

                out = self.update(p, group, fair=True)
                if out is None:
                    continue
                p.data = out
                # state['m'].data.copy_(m.data)
                # state['v'].data.copy_(v.data)

    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict



class MeritFedSGD(MeritFedBaseSGD):
    def update(self, p, group, weights=None, fair=None):
        state = self.state[p]
        w = self.weights if weights is None else weights
        state['g'].data.mul_(0)
        for w_id in range(self.config.npeers):
            # if weights is None and torch.isnan(state[w_id]).any():
            #     return None
            state['g'].add_(state[w_id].data * w[w_id])
        grad = state['g']
        return state['p'] - group['lr'] * grad


class MeritFedBase(torch.optim.Optimizer):
    def __init__(self, params, config, device, weight_name_map):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=config.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.device = device
        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.config = config
        self.i = 0
        self.weight_name_map = weight_name_map
        # self.device = device
        self.weights = torch.ones(self.config.npeers, requires_grad=False, device=device) / self.config.npeers
        self.weights_grad = torch.zeros_like(self.weights)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(config.npeers):
                    state[w_id] = torch.zeros_like(p.data, requires_grad=False)

                # state['step'] = 0
                state['m'] = torch.zeros_like(p.data, requires_grad=False)
                state['v'] = torch.zeros_like(p.data, requires_grad=False)
                state['p'] = torch.zeros_like(p.data, requires_grad=False)
                state['g'] = torch.zeros_like(p.data, requires_grad=False)

                state['f'] = lambda w: self.update(p, group, w)[0]

        # Auxiliary optimizer
        self.beta_1 = self.config.fl_beta_1
        self.beta_2 = 0.999
        self.eps = 1e-8
        self.momentum = torch.zeros_like(self.weights)
        self.second_momentum = torch.zeros_like(self.weights)
        self.cur_step = 1

    def update(self, p, group, weights=None):
        pass

    @torch.no_grad()
    def adam_update(self, group, p):
        state = self.state[p]
        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']
        beta1, beta2 = group['betas']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        m = torch.mul(m, beta1) + (1 - beta1)*grad
        v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']
        step = self.i
        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        denom = torch.sqrt(vhat + group['eps'])
        p.data.copy_(state['p'] - group['lr'] * mhat / denom)
        state['m'].data.copy_(m.data)
        state['v'].data.copy_(v.data)

        self.grads_received = 0
        for group in self.param_groups:
            for p in group['params']:
                self.adam_update(group, p)

    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (self.weight_name_map[i])
            self.metrics_dict[key] = w.item()
        return self.metrics_dict


class MeritFedAdam(MeritFedBase):

    def step(self, w_id, model, criterion, loader):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    # state[w_id].data.copy_(state[w_id] + float('nan'))
                    state[w_id].mul_(0)
                else:
                    state[w_id].data.copy_(p.grad.data)
        self.grads_received += 1

        if self.grads_received != self.config.npeers: 
            return

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'].data.copy_(p.data)

        for i, batch in enumerate(loader):
            for group in self.param_groups:
                for p in group['params']:
                    self.update(group, p)
                    
            model_input = dict_to_device(batch['model_input'], self.device)
            output = model.forward(**model_input)
            self.zero_grad()
            # if hasattr(output, "logits"):
            #     output = output.logits
            try:
                loss = output["loss"]
                if loss.dim() != 0:
                    loss = loss.mean()
            except IndexError:
                loss = criterion(output, batch[-1].to(self.device))
            loss.backward()

            self.weights_grad.mul_(0)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    dp_dw = self.update(group, p, p.grad.data)
                    self.weights_grad.add_(dp_dw.detach_())

            self.momentum = self.beta_1 * self.momentum + (1 - self.beta_1) * self.weights_grad
            #print(f'momentum {self.momentum}')
            self.second_momentum = self.beta_2 * self.second_momentum + (1 - self.beta_2) * (self.weights_grad ** 2) 
            #print(f'second momentum {self.second_momentum}')
            # bias correction
            self.second_momentum_hat = self.second_momentum / (1 - self.beta_2 ** (self.cur_step))
            self.momentum_hat = self.momentum / (1 - self.beta_1 ** (self.cur_step))
            #print(f'momentum after BC {self.momentum_hat}')
            #print(f'second momentum after BC {self.second_momentum_hat}')
            self.adaptive_lr = self.lr / ((1 - self.beta_1 ** (self.cur_step)) * (torch.sqrt(self.second_momentum_hat) + self.eps))
            #print(f'adaptive lr {self.adaptive_lr}')
            # mdlr = gamma, self.weights_grad = g
            step = -self.adaptive_lr * self.momentum_hat
            # if self.config.mdnorm_:
            #     step /= torch.norm(self.weights_grad)
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = vec / torch.sum(vec)
            self.cur_step += 1
            if i == self.config.mdniters_-1:
                break
            # print(self.weights_grad)

        self.grads_received = 0
        for group in self.param_groups:
            for p in group['params']:
                self.adam_update(group, p)

    @torch.no_grad()
    def update(self, group, p, p_new_grad=None):
        state = self.state[p]
        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']
        beta1, beta2 = group['betas']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        m = torch.mul(m, beta1) + (1 - beta1)*grad
        v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']
        step = self.i
        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        denom = torch.sqrt(vhat + group['eps'])
        if p_new_grad is None:
            p.data.copy_(state['p'] - group['lr'] * mhat / denom)
            return

        vjp = list()
        for w_id in range(self.config.npeers):
            tmp = (1-beta1) / (1 - beta1 ** step) * state[w_id].data * denom - mhat * (1-beta2) / (1 - beta2 ** step) / denom * grad * state[w_id].data
            tmp /= denom*denom
            vjp.append(-group['lr']*torch.tensordot(tmp, p_new_grad, len(tmp.shape)))
        dp_dw = torch.stack(vjp)
        return dp_dw

class MeritFedMD(MeritFedBase):

    def step(self, w_id, model, criterion, loader):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    # state[w_id].data.copy_(state[w_id] + float('nan'))
                    state[w_id].mul_(0)
                else:
                    state[w_id].data.copy_(p.grad.data)
        self.grads_received += 1

        if self.grads_received != self.config.npeers: 
            return

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'].data.copy_(p.data)

        for i, batch in enumerate(loader):
            for group in self.param_groups:
                for p in group['params']:
                    self.update(group, p)
                    
            model_input = dict_to_device(batch['model_input'], self.device)
            output = model.forward(**model_input)
            self.zero_grad()
            # if hasattr(output, "logits"):
            #     output = output.logits
            try:
                loss = output["loss"]
                if loss.dim() != 0:
                    loss = loss.mean()
            except IndexError:
                loss = criterion(output, batch[-1].to(self.device))
            loss.backward()

            self.weights_grad.mul_(0)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    dp_dw = self.update(group, p, p.grad.data)
                    self.weights_grad.add_(dp_dw.detach_())

            # mdlr = gamma, self.weights_grad = g
            step = -self.config.mdlr_ * self.weights_grad
            # if self.config.mdnorm_:
            #     step /= torch.norm(self.weights_grad)
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = vec / torch.sum(vec)
            if i == self.config.mdniters_-1:
                break
            # print(self.weights_grad)

        self.grads_received = 0
        for group in self.param_groups:
            for p in group['params']:
                self.adam_update(group, p)

    @torch.no_grad()
    def update(self, group, p, p_new_grad=None):
        state = self.state[p]
        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']
        beta1, beta2 = group['betas']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        m = torch.mul(m, beta1) + (1 - beta1)*grad
        v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']
        step = self.i
        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        denom = torch.sqrt(vhat + group['eps'])
        if p_new_grad is None:
            p.data.copy_(state['p'] - group['lr'] * mhat / denom)
            return

        vjp = list()
        for w_id in range(self.config.npeers):
            tmp = (1-beta1) / (1 - beta1 ** step) * state[w_id].data * denom - mhat * (1-beta2) / (1 - beta2 ** step) / denom * grad * state[w_id].data
            tmp /= denom*denom
            vjp.append(-group['lr']*torch.tensordot(tmp, p_new_grad, len(tmp.shape)))
        dp_dw = torch.stack(vjp)
        return dp_dw


class MeritFedParallelMD(torch.optim.Optimizer):
    def __init__(self, params, config, val_loader, model, accelerator):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.device = accelerator.device
        defaults = dict(device=self.device, lr=config.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.metrics_dict = defaultdict(float)
        self.config = config
        self.i = 0
        self.weight_name_map = config.weight_name_map
        self.weights = torch.ones(self.config.npeers, requires_grad=False, device=self.device) / self.config.npeers
        # self.weights[0] -= 0.05
        # self.weights[1] += 0.05
        self.grads_received = torch.zeros_like(self.weights)
        self.weights_grad = torch.zeros_like(self.weights)
        self.val_loader = val_loader
        self.model = model
        self.accelerator = accelerator
        self.drop_threshold = config.drop_threshold
        self.active_weight_ids = torch.ones_like(self.weights).to(torch.bool)
        self.enabled_fl = True

        

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(config.npeers):
                    state[w_id] = torch.zeros_like(p.data, requires_grad=False)

                # state['step'] = 0
                state['m'] = torch.zeros_like(p.data, requires_grad=False)
                state['v'] = torch.zeros_like(p.data, requires_grad=False)
                state['p'] = torch.zeros_like(p.data, requires_grad=False)
                state['g'] = torch.zeros_like(p.data, requires_grad=False)

                state['f'] = lambda w: self.update(p, group, w)[0]

        # Auxiliary optimizer
        self.beta_1 = self.config.fl_beta_1
        self.beta_2 = 0.999
        self.eps = 1e-8
        self.momentum = torch.zeros_like(self.weights)
        self.second_momentum = torch.zeros_like(self.weights)
        self.cur_step = 1

    @torch.no_grad()
    def adam_update(self, group, p):
        state = self.state[p]
        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            if self.active_weight_ids[w_id]:
                state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']
        beta1, beta2 = group['betas']


        m = torch.mul(m, beta1) + (1 - beta1)*grad
        v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']
        step = self.i
        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        denom = torch.sqrt(vhat + group['eps'])
        p.data.copy_(state['p'] - group['lr'] * mhat / denom)
        state['m'].data.copy_(m.data)
        state['v'].data.copy_(v.data)

    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (self.weight_name_map[i])
            self.metrics_dict[key] = w.item()
        return self.metrics_dict

    def all_grads_recieved(self):
        return (self.grads_received[self.active_weight_ids] == 1).all()

    def register_worker_grad(self, w_id):

        assert self.active_weight_ids[w_id], "Registering weight for dropped worker"

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    # state[w_id].data.copy_(state[w_id] + float('nan'))
                    state[w_id].mul_(0)
                else:
                    state[w_id].data.copy_(p.grad.data)

        self.grads_received[w_id] = 1
        
    def step(self, closure=None):
        assert self.all_grads_recieved()

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'].data.copy_(p.data)
        
        # around 5-6 sec
        if self.enabled_fl:
            self.auxiliary_optimizer_step()

        self.grads_received = torch.zeros_like(self.weights)
        for group in self.param_groups:
            for p in group['params']:
                self.adam_update(group, p)

    def auxiliary_optimizer_step(self):
        
        self.model.eval()
        for i, batch in enumerate(self.val_loader):

            #around 0.15 sec, can make faster although
            for group in self.param_groups:
                for p in group['params']:
                    self.fast_update(group, p)

            model_input = dict_to_device(batch['model_input'], self.device)
            model_input.pop('src')
            model_input.pop('src_att_mask')
            # around 0.2 sec
            output = self.model.forward(**model_input)

            self.zero_grad()
            loss = output["loss"]

            # around 0.1-0.2 sec
            self.accelerator.backward(loss)
            
            # around 0.6 sec, too long
            self.weights_grad.mul_(0)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    dp_dw = self.fast_update(group, p, p.grad.data)
                    assert (dp_dw[~self.active_weight_ids] == 0).all(), "Gradient is non zero for dropped weight"
                    self.weights_grad.add_(dp_dw.detach_())


            # mdlr = gamma, self.weights_grad = g
            step = -self.config.mdlr_ * self.weights_grad
            # if self.config.mdnorm_:
            #     step /= torch.norm(self.weights_grad)
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = vec / torch.sum(vec)

            if i == self.config.mdniters_-1:
                self.model.train()
                break

    @torch.no_grad()
    def fast_update(self, group, p, p_new_grad=None):
        state = self.state[p]
        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            if self.active_weight_ids[w_id]:
                state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']
        beta1, beta2 = group['betas']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        m = torch.mul(m, beta1) + (1 - beta1)*grad
        v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']
        step = self.i
        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        denom = torch.sqrt(vhat + group['eps'])
        if p_new_grad is None:
            p.data.copy_(state['p'] - group['lr'] * mhat / denom)
            return

        vjp = list()
        for w_id in range(self.config.npeers):
            if self.active_weight_ids[w_id]:
                tmp = (1-beta1) / (1 - beta1 ** step) * state[w_id].data * denom - mhat * (1-beta2) / (1 - beta2 ** step) / denom * grad * state[w_id].data
                tmp /= denom*denom
                vjp.append(-group['lr']*torch.tensordot(tmp, p_new_grad, len(tmp.shape)))
            else:
                vjp.append(torch.tensor(0., dtype=denom.dtype, device=self.device))
        dp_dw = torch.stack(vjp)
        return dp_dw

    @torch.no_grad()
    def drop_weights(self):
        new_active_weight_ids = torch.tensor(self.weights > self.drop_threshold)
        if (new_active_weight_ids != self.active_weight_ids).any():
            # zero weights below threshold
            self.weights[~new_active_weight_ids] = 0
            # Calculate the sum of non-zero elements
            non_zero_sum = self.weights[new_active_weight_ids].sum()
            # Rescale non-zero elements to sum to 1
            self.weights = torch.where(self.weights != 0, self.weights / non_zero_sum, self.weights)
            self.active_weight_ids = new_active_weight_ids
