import torch

from models import net_definitions


class RegressionModelWrapper():
    @staticmethod
    def build_model(net):
        net = net_definitions[net]()
        return net

    @staticmethod
    def build_optimizer(name, net, lr, parameters=None):
        if parameters is None:
            parameters = net.parameters()
        if name == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=lr)
        elif name == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=lr)
        elif name == "LBFGS":
            optimizer = torch.optim.LBFGS(parameters, max_iter=15, lr=lr)
        else:
            raise Exception(f'Not implemented optimizer: "{name}"')
        return optimizer

    def __init__(self, net, optimizer, device):
        self.device = device
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.loss = torch.nn.MSELoss().to(self.device)

    def set_train(self):
        self.net = self.net.train()

    def set_eval(self):
        self.net = self.net.eval()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        out, loss = self.forward_pass(batch)
        loss.mean().backward()
        self.optimizer.step()
        return out, loss

    def test_step(self, batch):
        out, loss = self.forward_pass(batch)
        return out, loss

    def forward_pass(self, batch):
        inputs = batch['input_pressure_wave']
        targets = batch['target_speed_wave']
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).float()

        outputs = self.net.forward(inputs)

        loss = self.loss(outputs, targets)

        return outputs, loss

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        new_state_dict = {}
        for k, v in self.net.state_dict().items():
            if k not in state_dict:
                print('WARNING: {} is missing in checkpoint.'.format(k))
            else:
                if v.shape != state_dict[k].shape:
                    print('WARNING: {} in checkpoint has different shape in net.'.format(k))
                else:
                    new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
        self.net.load_state_dict(state_dict, strict=False)
