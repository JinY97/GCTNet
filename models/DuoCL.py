import torch
import torch.nn as nn

class DuoCL(nn.Module):
    def __init__(self, data_num=512):
        super(DuoCL, self).__init__()
        self.model_7 = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(16, 32, 7, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(32, 128, 7, 1, "same"), nn.ReLU(inplace=True),  
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(128, 256, 7, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(256, 512, 7, 1, "same"), nn.ReLU(inplace=True),  
            nn.Dropout(0.1),
        )
        self.model_3 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(16, 32, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(32, 128, 3, 1, "same"), nn.ReLU(inplace=True),  
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(128, 256, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(256, 512, 3, 1, "same"), nn.ReLU(inplace=True),  
            nn.Dropout(0.1),
        )
        self.linear_7 = nn.Sequential(
            nn.Linear(32 * data_num, 1024),
            nn.Linear(1024, 512),  
        )
        self.linear_3 = nn.Sequential(
            nn.Linear(32 * data_num, 1024),
            nn.Linear(1024, 512),  
        )
        self.lstm_7 = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
    
        self.linear = nn.Linear(1024, data_num)

    def forward(self, x):
        x_7 = self.linear_7(self.model_7(x).view(x.shape[0], -1))
        x_3 = self.linear_3(self.model_3(x).view(x.shape[0], -1))
    
        x_7, _ = self.lstm_7(x_7.unsqueeze(dim=-1))  # _x is input, size (seq_len, batch, input_size)
        x_3, _ = self.lstm_3(x_3.unsqueeze(dim=-1))
        
        out = torch.cat((x_7, x_3), axis = -1).view(x.shape[0], -1)
        return self.linear(out)
    
if __name__ == '__main__':
    x = torch.rand((32, 1, 512))
    model = DuoCL()
    y = model(x)
    print(y.shape)