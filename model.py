# Defining the Convolutional Neural Network model
import torch.nn as nn
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 24, 3, stride=2)
#         self.conv2 = nn.Conv2d(24, 48, 3, stride=2)
#         self.pool = nn.MaxPool2d(4, stride=4)
#         self.drop = nn.Dropout(p=0.25)
#         self.fc1 = nn.Linear(48*4*19,50)
#         self.fc2 = nn.Linear(50, 10)
#         self.fc3 = nn.Linear(10,1)
#
#     def forward(self, x):
#         x = x.view(x.size(0), 3, 70, 320)
#         x = F.elu(self.conv1(x))
#         x = self.drop(self.pool(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.elu(self.fc1(x))
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

#
#
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


# Defining the CNN-LSTM  model
# Defining the Convolutional Neural Network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2)
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2)
        self.pool = nn.MaxPool2d(4, stride=4)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.drop(self.pool(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        return x

class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = Net()
        self.lstm = nn.LSTM(input_size=3648, hidden_size=600, num_layers=1, batch_first = True)
        self.fc1 = nn.Linear(600,100)
        self.fc2 = nn.Linear(100,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1) # to make x of the form (batch_size, sequence length, input_size)
        _, (h_out, _) = self.lstm(x)
        h_out = h_out.view(-1, 600)
        x = self.fc1(h_out)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = Combine()
