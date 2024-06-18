import gc
import torch
import random
from copy import deepcopy
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

class MeritFedZOBase(torch.optim.Optimizer):
    def __init__(self, params, cfg, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.device = device
        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.cfg = cfg
        self.i = 0
        # self.device = device
        self.weights = torch.ones(self.cfg.npeers, requires_grad=False, device=device) / self.cfg.npeers
        self.weights_grad = torch.zeros_like(self.weights)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(cfg.npeers):
                    state[w_id] = torch.zeros_like(p.data, requires_grad=False)

                # state['step'] = 0
                state['m'] = torch.zeros_like(p.data, requires_grad=False)
                state['v'] = torch.zeros_like(p.data, requires_grad=False)
                state['p'] = torch.zeros_like(p.data, requires_grad=False)
                state['g'] = torch.zeros_like(p.data, requires_grad=False)

    def update(self, p, group, weights=None, fair=None):
        state = self.state[p]
        w = self.weights if weights is None else weights
        state['g'].data.mul_(0)
        for w_id in range(self.cfg.npeers):
            state['g'].add_(state[w_id].data * w[w_id])
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
        return state['p'] - group['lr'] * mhat / denom, m.data, v.data
    # def update(self, p, group, weights=None):
    #     pass

    @staticmethod
    def sample_hypersphere(d, n=1):
        rnd = torch.randn(d)
        samples = rnd / torch.norm(rnd, dim=-1, keepdim=True)
        return samples

    @torch.no_grad()
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

        if self.grads_received != self.cfg.npeers: 
            return

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['p'].data.copy_(p.data)

        for t, batch in enumerate(loader):
            # sample a unit vector to determine weights step
            # do finite difference and find step
            # update weights

            d = self.weights.shape[0]
            e = self.sample_hypersphere(d)

            imin = torch.argmin(self.weights).item()
            imax = torch.argmax(self.weights).item()

            h = 1e-5
            tol = 0.05
            for i in range(d):
                if e[i] < 0:
                    h = min(h, -tol * self.weights[i] / e[i])
                if e[i] > 0:
                    h = min(h, tol * self.weights[i] / e[i])

            e = e.to(self.device)
            ws = [self.weights + e*h, self.weights - e*h]
            # print(torch.all(ws[0] > 0))
            # print(torch.all(ws[1] > 0))
            # print()

            losses = list()
            for w in ws:
                for group in self.param_groups:
                    for p in group['params']:
                        state = self.state[p]

                        out = self.update(p, group, weights=w)
                        if out is None:
                            continue
                        p.data.copy_(out[0].data) 

                model_input = dict_to_device(batch['model_input'], self.device)
                output = model.forward(**model_input)
                self.zero_grad()
                # if hasattr(output, "logits"):
                #     output = output.logits
                try:
                    loss = output["loss"]
                except IndexError:
                    loss = criterion(output, batch[-1].to(self.device))
                losses.append(loss.item())
            
            self.weights_grad.copy_(e*(losses[0] - losses[1])*d/2) # /h) use h as md step size factor


            step = -self.cfg.mdlr_ * self.weights_grad# * (h / 1e-5)
            if self.cfg.mdnorm_:
                step /= torch.norm(self.weights_grad)
            # step = -self.cfg.mdlr_ * self.weights_grad
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = self.weights * t / (t+1) + vec / torch.sum(vec) / (t+1)
            # self.weights = vec / torch.sum(vec)
            # if t == self.cfg.mdniters_-1:
            #     break
            # print(self.weights_grad)
        print(self.weights)


        self.grads_received = 0
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                state = self.state[p]
                # p.data.copy_(state['p'].data)

                out = self.update(p, group, fair=True)
                if out is None:
                    continue
                # p.data, state['m'].data, state['v'].data = out
                p.data.copy_(out[0].data)
                state['m'].data.copy_(out[1].data)
                state['v'].data.copy_(out[2].data)

    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict


class SGD(torch.optim.Optimizer):
    def __init__(self, params, cfg, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.cfg = cfg
        self.i = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(cfg.npeers):
                    state[w_id] = torch.zeros_like(p.data)

                # state['step'] = 0
                state['g'] = torch.zeros_like(p.data)
                    

    def step(self, w_id, model, criterion, loader):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    # state[w_id].data.copy_(state[w_id] + float('nan'))
                    state['g'].mul_(0)
                else:
                    state[w_id].data.copy_(p.grad.data)
                    # if group['weight_decay'] != 0:
                        # state[w_id].add_(group['weight_decay'], p.data)
        self.grads_received += 1

        if self.grads_received != self.cfg.npeers:
            return

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                state = self.state[p]

                state['g'].mul_(0)
                # try:
                #     for w_id in range(self.cfg.npeers):
                #         if torch.isnan(state[w_id]).any():
                #             raise ValueError()
                #         state['g'].add_(state[w_id].data / self.cfg.npeers)
                # except ValueError:
                #     continue
                for w_id in range(self.cfg.npeers):
                    state['g'].add_(state[w_id].data / self.cfg.npeers)

                grad = state['g']                 
                p.data = p.data - group['lr'] * grad
        self.grads_received = 0

    @torch.no_grad()
    def metrics(self) -> float:
        return self.metrics_dict



class MeritFedBaseSGD(torch.optim.Optimizer):
    def __init__(self, params, cfg, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.device = device
        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.cfg = cfg
        self.i = 0
        # self.device = device
        self.weights = torch.ones(self.cfg.npeers, requires_grad=False, device=device) / self.cfg.npeers
        self.weights_grad = torch.zeros_like(self.weights)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(cfg.npeers):
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

        if self.grads_received != self.cfg.npeers: 
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
            #         for w_id in range(self.cfg.npeers):
            #             self.weights_grad[w_id] = self.weights_grad[w_id].add(torch.sum(p.grad.data * -group['lr']*state[w_id]))
            # print(self.weights_grad)

            self.weights_grad.mul_(0)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]

                    # try:
                    #     for w_id in range(self.cfg.npeers):
                    #         if torch.isnan(state[w_id]).any():
                    #             raise ValueError()
                    # except ValueError:
                    #     continue
                    
                    # def f(weights):
                    #     state['g'].data.mul_(0)
                    #     for w_id in range(self.cfg.npeers):
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

            step = -self.cfg.mdlr_ * self.weights_grad
            if self.cfg.mdnorm_:
                step /= torch.norm(self.weights_grad)
            # step = -self.cfg.mdlr_ * self.weights_grad
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = vec / torch.sum(vec)
            # if i == self.cfg.mdniters_-1:
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
        for w_id in range(self.cfg.npeers):
            # if weights is None and torch.isnan(state[w_id]).any():
            #     return None
            state['g'].add_(state[w_id].data * w[w_id])
        grad = state['g']
        return state['p'] - group['lr'] * grad




class Adam(torch.optim.Optimizer):
    def __init__(self, params, cfg, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.cfg = cfg
        self.i = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(cfg.npeers):
                    state[w_id] = torch.zeros_like(p.data)

                # state['step'] = 0
                state['m'] = torch.zeros_like(p.data)
                state['v'] = torch.zeros_like(p.data)
                state['g'] = torch.zeros_like(p.data)
                    

    def step(self, w_id, model, criterion, loader):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    # state[w_id].data.copy_(state[w_id] + float('nan'))
                    state[w_id].mul_(0)
                else:
                    state[w_id].data.copy_(p.grad.data)
                    # if group['weight_decay'] != 0:
                        # state[w_id].add_(group['weight_decay'], p.data)
        self.grads_received += 1

        if self.grads_received != self.cfg.npeers:
            return

        self.i += 1
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                state = self.state[p]

                state['g'].data.mul_(0)
                # try:
                #     for w_id in range(self.cfg.npeers):
                #         if torch.isnan(state[w_id]).any():
                #             raise ValueError()
                #         state['g'].add_(state[w_id].data / self.cfg.npeers)
                # except ValueError:
                #     continue

                grad = state['g']                
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                m = torch.mul(m, beta1) + (1 - beta1)*grad
                v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

                # step = state['step']
                step = self.i
                mhat = m / (1 - beta1 ** step)
                vhat = v / (1 - beta2 ** step)
                denom = torch.sqrt(vhat + group['eps']) 
                p.data = p.data - group['lr'] * mhat / denom
                state['m'].data.copy_(m.data)
                state['v'].data.copy_(v.data)
        self.grads_received = 0

    @torch.no_grad()
    def metrics(self) -> float:
        return self.metrics_dict


class MeritFedBase(torch.optim.Optimizer):
    def __init__(self, params, cfg, device, weight_name_map):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)
        super().__init__(params, defaults)

        self.device = device
        self.metrics_dict = defaultdict(float)
        self.grads_received = 0
        self.cfg = cfg
        self.i = 0
        self.weight_name_map = weight_name_map
        # self.device = device
        self.weights = torch.ones(self.cfg.npeers, requires_grad=False, device=device) / self.cfg.npeers
        self.weights_grad = torch.zeros_like(self.weights)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for w_id in range(cfg.npeers):
                    state[w_id] = torch.zeros_like(p.data, requires_grad=False)

                # state['step'] = 0
                state['m'] = torch.zeros_like(p.data, requires_grad=False)
                state['v'] = torch.zeros_like(p.data, requires_grad=False)
                state['p'] = torch.zeros_like(p.data, requires_grad=False)
                state['g'] = torch.zeros_like(p.data, requires_grad=False)

                state['f'] = lambda w: self.update(p, group, w)[0]

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

        if self.grads_received != self.cfg.npeers: 
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

            step = -self.cfg.mdlr_ * self.weights_grad
            # if self.cfg.mdnorm_:
            #     step /= torch.norm(self.weights_grad)
            step = torch.exp(step)
            vec = self.weights * step
            self.weights = vec / torch.sum(vec)
            if i == self.cfg.mdniters_-1:
                break
            # print(self.weights_grad)

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



class MeritFedA(MeritFedBase):
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
        for w_id in range(self.cfg.npeers):
            tmp = (1-beta1) / (1 - beta1 ** step) * state[w_id].data * denom - mhat * (1-beta2) / (1 - beta2 ** step) / denom * grad * state[w_id].data
            tmp /= denom*denom
            vjp.append(-group['lr']*torch.tensordot(tmp, p_new_grad, len(tmp.shape)))
        dp_dw = torch.stack(vjp)
        return dp_dw



class MeritFedB(MeritFedBase):
    @torch.no_grad()
    def update(self, group, p, p_new_grad=None):
        state = self.state[p]
        step = self.i
        beta1, beta2 = group['betas']
        if p_new_grad is not None:
            vjp = list()
            for w_id in range(self.cfg.npeers):
                tmp = (1-beta1) / (1 - beta1 ** step) * state[w_id].data 
                tmp /= torch.sqrt(state['v'] / (1 - beta2 ** step) + group['eps'])
                vjp.append(-group['lr']*torch.tensordot(tmp, p_new_grad, len(tmp.shape)))
            dp_dw = torch.stack(vjp)
            return dp_dw

        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        m = torch.mul(m, beta1) + (1 - beta1)*grad
        # v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']

        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        denom = torch.sqrt(vhat + group['eps'])
        p.data.copy_(state['p'] - group['lr'] * mhat / denom)



class MeritFedC(MeritFedBase):
    @torch.no_grad()
    def update(self, group, p, p_new_grad=None):
        state = self.state[p]
        step = self.i
        beta1, beta2 = group['betas']
        if p_new_grad is not None:
            vjp = list()
            for w_id in range(self.cfg.npeers):
                tmp = (1-beta1) / (1 - beta1 ** step) * state[w_id].data 
                vjp.append(-group['lr']*torch.tensordot(tmp, p_new_grad, len(tmp.shape)))
            dp_dw = torch.stack(vjp)
            return dp_dw

        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            state['g'].add_(state[w_id].data * w)
        grad = state['g']
        m, v = state['m'], state['v']

        # if group['weight_decay'] != 0:
        #     grad = grad.add(group['weight_decay'], p.data)

        m = torch.mul(m, beta1) + (1 - beta1)*grad
        # v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

        # state['step'] += 1
        # step = state['step']
        
        mhat = m / (1 - beta1 ** step)
        # vhat = v / (1 - beta2 ** step)
        # denom = torch.sqrt(vhat + group['eps'])
        p.data.copy_(state['p'] - group['lr'] * mhat)



class MeritFedD(MeritFedBase):
    @torch.no_grad()
    def update(self, group, p, p_new_grad=None):
        state = self.state[p]
        step = self.i
        beta1, beta2 = group['betas']

        if p_new_grad is not None:
            vjp = list()
            for w_id in range(self.cfg.npeers):
                vjp.append(-group['lr']*torch.tensordot(state[w_id], p_new_grad, len(state[w_id].shape)))
            dp_dw = torch.stack(vjp)
            return dp_dw

        state['g'].data.mul_(0)
        for w_id, w in enumerate(self.weights):
            state['g'].add_(state[w_id].data * w)
        grad = state['g']
        p.data.copy_(state['p'] - group['lr'] * grad)