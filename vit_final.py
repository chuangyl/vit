import torch
import torch.nn as nn
import copy
from torchsummary import summary

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,qkv_bias=False,qk_scale=None,
                    dropout=0.,attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim/num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim,self.all_head_dim *3,
                            bias=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim,embed_dim)

    def transpose_multi_head(self,x):
        # input shape: [batch,num_tokens,embed_dim]
        # reshape: [batch,num_tokens,num_heads,head_dim]
        # ps: embed_dim = num_heads * head_dim
        x = x.reshape([x.shape[0],x.shape[1],self.num_heads,self.head_dim])
        # permute: [batch,num_heads,num_tokens,head_dim]
        # computing self-attention based on per-head
        x = x.permute(0,2,1,3)
        return x

    def forward(self,x):
        # x---->: [batch,num_token,embed_dim]
        B,N,_ = x.shape
        # qkv --- >:[batch,num_token,embed_dim*3]
        qkv = self.qkv(x).chunk(3,-1)
        q,k,v = map(self.transpose_multi_head,qkv)
        # attn shape: [batch,num_heads,num_tokens,num_tokens]
        attn = torch.matmul(q,k.permute(0,1,3,2))
        attn = self.scale*attn
        attn = self.softmax(attn)
        #dropout
        # out shape: [batch,num_heads,num_tokens,head_dim]
        out = torch.matmul(attn,v)
        # out shape: [batch,num_tokens,num_heads,head_dim]
        out = out.permute(0,2,1,3)
        # out shape: [batch,num_tokens,embed_dim]
        out = out.reshape(B,N,-1)
        out = self.proj(out)
        return out

class PatchEmbedding_1(nn.Module):
    def __init__(self,image_size,patch_size,in_channels,embed_dim,dropout=0.):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        print(x.shape)
        x = self.patch_embed(x) # shape [batch,embed_dim,h',w']
        print(x.shape)
        x = x.flatten(2) # shape [batch,embed_dim,num_patchs]
        x = x.permute([0,2,1]) # shape[batch,num_patchs,embed_dim]
        x = self.dropout(x)
        return x
class PatchEmbedding(nn.Module):
    def __init__(self,image_size=224,patch_size=16,in_channels=3,embed_dim=768,dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size//patch_size)
        self.patch_embedding = nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
        self.dropout = nn.Dropout(dropout)
        # add class token
        self.class_token = nn.parameter.Parameter(torch.zeros([1,1,embed_dim],dtype=torch.float32),requires_grad=True)
        # add position embedding
        tensor_position = torch.empty(1,n_patches+1,embed_dim)
        nn.init.trunc_normal_(tensor_position,std=.02)
        self.position_embedding = nn.parameter.Parameter(tensor_position)
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        #[n,c,h,w]
        cls_token = self.class_token.expand(x.shape[0],1,self.embed_dim) # for batch
        x = self.patch_embedding(x) # [n,embed_dim,h',w']
        x = x.flatten(2) #[n,embed_dim,num_patches]
        x = x.permute([0,2,1])
        x = torch.cat([cls_token,x],axis=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x
class Mlp(nn.Module):
    def __init__(self,embed_dim,mlp_ratio,dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim,int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio),embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,embed_dim=768,num_heads=4,qkv_bias=True,mlp_ratio=4.0,dropout=0.):
        super().__init__()
        self.atten_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim,num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim,mlp_ratio)
    def forward(self,x):
        h = x
        x = self.atten_norm(x)
        x = self.attn(x)
        x = x+h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x+h
        return x

class Encoder(nn.Module):
    def __init__(self,embed_dim,depth):
        super().__init__()
        layer_list = [EncoderLayer() for i in range(depth)]
        self.layers = nn.Sequential(*layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class VisualTransformer(nn.Module):
    def __init__(self,image_size=224,patch_size=16,in_channels=3,num_classes=1000,embed_dim=768,depth=3,num_heads=8,mlp_ratio=4,qkv_bias=True,dropout=0.,attention_dropout=0.,droppath=0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size,patch_size,in_channels,embed_dim)
        self.encoder = Encoder(embed_dim,depth)
        self.classifier = nn.Linear(embed_dim,num_classes)
    def forward(self,x):
        # x shape: [patch,channels,h,w]
        x = self.patch_embedding(x) #[batch,num_tokens,embed_dim]
        x = self.encoder(x)
        x = self.classifier(x[:,0])
        return x

def main():
    data = torch.randn(8,3,224,224)
    model = VisualTransformer()
    result = model(data)
    print(result.shape)
    print(model)
if __name__ == "__main__":
    main()
