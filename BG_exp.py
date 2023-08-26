import torch
import torch.nn as nn

class BG(nn.Module):
    def __init__(self, data_num=512, embedding=16):
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
        self.output_layer = nn.Linear(data_num*4, data_num)


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
    
    
"""
class BG(nn.Module):
    def __init__(self, data_num=512, embedding=16):
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
        self.output_layer = nn.Linear(embedding*4, data_num)


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
    """