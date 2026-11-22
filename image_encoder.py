"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        #raise NotImplementedError('TODO-02: split the feature axis into heads')
        
        #获取输入的形状
        B,N,D=x.shape
        # 拆解特征维度：把 768 维拆成 12 个头，每个头 64 维
        # 形状从 [B, N, 768] 变为 [B, N, H, D/H]
        x = x.reshape(B, N, self.heads, self.head_dim)
        # 交换维度：为了让每个头独立并行计算，把 Head 维度提前
        # 形状变为 [B, H, N, D/H]
        x = x.transpose(0,2,1,3)
        #返回目标
        return x
        

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        #raise NotImplementedError('TODO-03a: calculate scores from q and k')

        #转置 k 的最后两个维度，让[B, H, N, 64]变为[B, H, 64, N]
        #不能直接用 k.T，因为 k 是 4 维的，.T 会把 Batch 和 Head 维度也翻转
        k_transposed = k.transpose(0, 1, 3, 2)
        #矩阵乘法并缩放：q @ k_transposed 得到 [B, H, N, N]
        scores = (q @ k_transposed) / np.sqrt(self.head_dim)
        #强制把 scores 转为 float32，防止因除法产生 float64 导致精度测试报错
        scores = scores.astype(np.float32) 

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        #raise NotImplementedError('TODO-03b: weighted values and merged heads')

        #权重乘 V: [B, H, N, N] @ [B, H, N, 64] 得到 context: [B, H, N, 64]
        context = coefficients @ v
        #换轴：把 H 和 N 换回来得到 [B, N, H, 64]
        context = context.transpose(0, 2, 1, 3)
        #合并：把 H 和 64 合并得到 merged: [B, N, 768]
        B, N, H, head_dim = context.shape
        merged = context.reshape(B, N, self.width)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        #raise NotImplementedError('TODO-03c: project merged attention outputs')

        #merged: [B, N, 768]，out_weight.T: [768, 768]，计算得到目标
        output = merged @ self.out_weight.T + self.out_bias
        #返回目标
        return output


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        #raise NotImplementedError('TODO-04a: add the attention update to x')

        #直接相加即可，形状都是 [B, N, D]
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        #raise NotImplementedError('TODO-04b: MLP and the second residual')

        #第一层线性层 + GELU 激活，形状变为 [B, N, 3072]
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        #第二层线性层，形状变回 [B, N, 768]
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        #第二次残差连接
        final_output = residual + mlp_update
        #强制转为 float32，防止因除法产生 float64 导致精度测试报错
        final_output = final_output.astype(np.float32)
        #返回目标
        return final_output


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        #raise NotImplementedError('TODO-01: turn each image into 49 patch rows')
        
        # 动态获取 Batch Size 以适应不同数量图片的输入
        B=images.shape[0]
        # 拆分维度：将高 224 拆成 7x32，宽 224 拆成 7x32
        # 形状从 [B, 3, 224, 224] 变为 [B, 3, 7, 32, 7, 32]
        x = images.reshape(B, 3, 7, 32, 7, 32)
        # 交换轴：把网格维度(7,7)放到前面，把颜色通道和像素放到后面
        # 形状变为 [B, 7, 7, 3, 32, 32]，确保每个 patch 内部像素是连续的
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # 拉平：合并空间维度 (7*7=49) 和特征维度 (3*32*32=3072)
        # 形状变为 [B, 49, 3072]
        x = x.reshape(B, 49, 3072)
        # 返回目标
        return x

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        #raise NotImplementedError('TODO-05a: apply the supplied patch projection')

        #矩阵乘法
        projected_patches = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        #raise NotImplementedError('TODO-05b: CLS, patch rows and positions')

        #沿着 axis=1拼接 CLS
        tokens = np.concatenate([cls, projected_patches], axis=1)
        #加上可学习的位置编码
        tokens = tokens + self.position
        #返回目标
        return tokens

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        #raise NotImplementedError('TODO-06: CLS selection, post-Norm and projection')

        #提取第 0 个 CLS Token
        cls_token = tokens[:, 0]
        #后置归一化
        normed_cls = self.post_norm(cls_token)
        #投影到 512 维 (不需要转置)
        output = normed_cls @ self.projection
        #强制把数据类型转换回 float32，防止因除法产生 float64 导致精度测试报错
        output = output.astype(np.float32)
        #L2 归一化，保证向量长度为 1，返回目标
        return unit_vector(output)