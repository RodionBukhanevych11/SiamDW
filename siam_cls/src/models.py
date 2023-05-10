import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models   
import torchvision

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get model
        self.features = models.mobilenet_v2(pretrained=False, width_mult = 0.25)
        self.fc_in_features = self.features.classifier[1].in_features
        self.features = torch.nn.Sequential(*(list(self.features.children())[:-1]))
        self.features.add_module(module = nn.AdaptiveAvgPool2d(output_size=(1, 1)), name = 'avgpool')
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.features.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        
        return output
    
    
class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        # get model
        self.features = models.mobilenet_v2(pretrained=False, width_mult = 0.25)
        self.fc_in_features = self.features.classifier[1].in_features
        self.features = torch.nn.Sequential(*(list(self.features.children())[:-1]))
        self.features.add_module(module = nn.AdaptiveAvgPool2d(output_size=(1, 1)), name = 'avgpool')
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.sigmoid = nn.Sigmoid()
        self.features.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        
        return output1, output2, output3
    