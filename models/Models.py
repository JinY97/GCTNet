'''
Author: JinYin
Date: 2022-07-01 20:36:29
LastEditors: JinYin
LastEditTime: 2023-04-24 18:50:17
FilePath: \GCTNet\models\Models.py
Description: 
'''
import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, data_num=512):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(data_num, data_num), nn.ReLU(inplace=True), nn.Dropout(0.1),
            
            nn.Linear(data_num, data_num),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x.view(x.shape[0], -1)
        
class SimpleCNN(nn.Module):
    def __init__(self, data_num=512):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.linear = nn.Linear(64 * data_num, data_num)

    def forward(self, x):
        t = self.model(x).view(x.shape[0], -1)
        return self.linear(t)

class ResCNN(nn.Module):
    def __init__(self, data_num=512):
        super(ResCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
            BasicBlockall(),
            nn.Conv1d(32 * 3, 32, 1, 1, "same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True), 
        )
        self.linear = nn.Linear(32 * data_num, data_num)

    def forward(self, x):
        t = self.model(x).view(x.shape[0], -1)
        return self.linear(t)

class NovelCNN(nn.Module):
    def __init__(self, data_num=512):
        super(NovelCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(32, 32, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(32, 64, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(64, 64, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(64, 128, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(128, 128, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2),
            nn.Conv1d(128, 256, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(256, 256, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2), nn.Dropout(0.5),
            nn.Conv1d(256, 512, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(512, 512, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2), nn.Dropout(0.5),
            nn.Conv1d(512, 1024, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(1024, 1024, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.AvgPool1d(2, stride=2), nn.Dropout(0.5),
            nn.Conv1d(1024, 2048, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Conv1d(2048, 2048, 3, 1, "same"), nn.ReLU(inplace=True), 
            nn.Dropout(0.5),

        )
        self.linear = nn.Linear(32 * data_num, data_num)

    def forward(self, x):
        t = self.model(x).view(x.shape[0], -1)
        return self.linear(t)
     
class Res_BasicBlock(nn.Module):
  def __init__(self, kernelsize, stride=1):
    super(Res_BasicBlock, self).__init__()
    self.bblock = nn.Sequential(
        nn.Conv1d(32, 32, kernel_size = kernelsize, stride=stride,padding="same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        nn.Conv1d(32, 16, kernel_size = kernelsize, stride=1,padding="same"), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
        nn.Conv1d(16, 32, kernel_size = kernelsize, stride=1,padding="same"), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
    )

  def forward(self, inputs):
    #Through the convolutional layer``
    out = self.bblock(inputs)
    identity = inputs

    output = torch.add(out, identity)  #layers下面有一个add，把这2个层添加进来相加。
    
    return output

class BasicBlockall(nn.Module):
  def __init__(self, stride=1):
    super(BasicBlockall, self).__init__()

    self.bblock3 = nn.Sequential(Res_BasicBlock(3),
                              Res_BasicBlock(3)
                              )                      
    
    self.bblock5 = nn.Sequential(Res_BasicBlock(5),
                              Res_BasicBlock(5)
                              )                      

    self.bblock7 = nn.Sequential(Res_BasicBlock(7),
                              Res_BasicBlock(7)
                              )


  def forward(self, inputs): 
 
    out3 = self.bblock3(inputs)
    out5 = self.bblock5(inputs)
    out7 = self.bblock7(inputs)

    out = torch.cat((out3,out5,out7) , axis = 1)
    return out
  

class BG(nn.Module):
    def __init__(self, data_num=512, embedding=1):
        super(BG, self).__init__()

        # encoder
        self.gru1_encoding = nn.GRU(input_size=embedding, hidden_size=embedding, bidirectional=True)

        self.MHA1_encoding = nn.MultiheadAttention(embed_dim=512, num_heads=2)

        #self.adding1_encoding = torch.add(self.gru1_encoding, self.MHA1_encoding)
        self.norm1_encoding = nn.LayerNorm(embedding*2)
        self.dense1_encoding = nn.Linear(embedding*2, embedding*2)
        #self.adding2_encoding = torch.add(self.norm1_encoding, self.dense1_encoding)
        self.norm2_encoding = nn.LayerNorm(embedding*2)


        # decoder
        self.gru1_decoding = nn.GRU(input_size=embedding*2, hidden_size=embedding*2, bidirectional=True)

        self.MHA1_decoding = nn.MultiheadAttention(embed_dim=512, num_heads=2)

        #self.adding1_decoding = torch.add(self.gru2_decoding, self.MHA1_decoding)
        self.norm1_decoding = nn.LayerNorm(embedding*4)
        self.dense1_decoding = nn.Linear(embedding*4, embedding*4)
        #self.adding2_decoding = torch.add(self.norm1_decoding, self.dense1_decoding)
        self.norm2_decoding = nn.LayerNorm(embedding*4)

        self.gru2_decoding = nn.GRU(input_size=embedding*4, hidden_size=embedding*4, bidirectional=True)

        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(data_num*2, data_num)


    def forward(self, inputs):
        # encoder
        print('inputs.shape:', inputs.shape)
        inputs = inputs.permute(2, 0, 1)
        print('inputs.shape:', inputs.shape)

        gru1_encoding, _ = self.gru1_encoding(inputs)
        
        print('gru1_encoding.shape:', gru1_encoding.shape)
        mha1_inputs = gru1_encoding.permute(2, 1, 0)
        print('mha1_inputs.shape:', mha1_inputs.shape)
        MHA1_encoding, _ = self.MHA1_encoding(mha1_inputs, mha1_inputs, mha1_inputs)
        
        MHA1_encoding = MHA1_encoding.permute(2, 1, 0)
        adding1_encoding = gru1_encoding + MHA1_encoding
        norm1_encoding = self.norm1_encoding(adding1_encoding)
        dense1_encoding = self.dense1_encoding(norm1_encoding)
        adding2_encoding = norm1_encoding + dense1_encoding
        norm2_encoding = self.norm2_encoding(adding2_encoding)


        # decoder
        print('norm2_encoding.shape:', norm2_encoding.shape)
        gru1_decoding, _ = self.gru1_decoding(norm2_encoding)
        print('gru1_decoding.shape:', gru1_decoding.shape)

        print('gru1_decoding.shape:', gru1_decoding.shape)
        mha2_inputs = gru1_decoding.permute(2, 1, 0)
        print('mha2_inputs.shape:', mha2_inputs.shape)
        MHA1_decoding, _ = self.MHA1_decoding(mha2_inputs, mha2_inputs, mha2_inputs)

        MHA1_decoding = MHA1_decoding.permute(2, 1, 0)
        adding1_decoding = gru1_decoding + MHA1_decoding
        norm1_decoding = self.norm1_decoding(adding1_decoding)
        dense1_decoding = self.dense1_decoding(norm1_decoding)
        adding2_decoding = norm1_decoding + dense1_decoding
        norm2_decoding = self.norm2_decoding(adding2_decoding)

        gru2_decoding, _ = self.gru2_decoding(norm2_decoding)

        flatten = self.flatten(gru2_decoding)
        output_layer = self.output_layer(flatten)
            
        return output_layer