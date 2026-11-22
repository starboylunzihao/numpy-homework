import numpy as np
#1数组、形状与切片
A = np.array([[0.20, -0.50, 1.10, 0.40],
              [0.80, 0.30, -0.70, 0.60],
              [-0.40, 0.90, 0.10, -0.20]])
#1.1:输出 A 的 ndim、shape 和 size
print(A.ndim)
print(A.shape)
print(A.size)
#1.2:分别提取第 2 行（下标 1）和最后 2 列，并输出结果及 shape
row=A[1,:]
print(row)
print(row.shape)
last2=A[:,-2:]
print(last2)
print(last2.shape)
#1.3:复制为 B，将 B 的负数改成 0，保持 A 不变
B=A.copy()
B[B<0]=0
print(B)
print(A)
print()

#2逐元素乘法、点乘与长度
a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])
#2.1:计算 a * b 和 a @ b，输出结果及各自的 shape
c=a*b
d=a@b
print(c)
print(c.shape)
print(f'{d:.6f}')
print(d.shape)
#2.2:分别计算 a、b 的欧氏长度，使用平方、求和和开平方实现
len_a1=np.sqrt(np.sum(a**2))
len_b1=np.sqrt(np.sum(b**2))
print(f'{len_a1:.6f}')
print(f'{len_b1:.6f}')
#2.3用 np.linalg.norm 核对长度
len_a2=np.linalg.norm(a)
len_b2=np.linalg.norm(b)
print(f'{len_a2:.6f}')
print(f'{len_b2:.6f}')
print()

#3两个向量的夹角
a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])
#3.1:利用 cos(theta) = (a @ b) / (||a|| * ||b||) 计算夹角
#3.2:使用 np.clip 将余弦值限制到 [-1, 1]，再求反余弦并转换为度
#3.3:任一输入为零向量时抛出 ValueError；用给定输入调用函数
def F(a,b):
    len_a=np.linalg.norm(a)
    len_b=np.linalg.norm(b)
    ab=a@b
    if len_a == 0 or len_b == 0:
        raise ValueError("不可输入零向量")
    cos=ab/(len_a*len_b)
    cos1= np.clip(cos, -1.0, 1.0)
    rad = np.arccos(cos1)
    deg = np.degrees(rad)
    return deg
x=F(a,b)
print(f'a,b的夹角为{x:.6f}度')
print()

#4两个点的距离
p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])
#4.1输出差向量 p - q
diff=p-q
print("p - q:", diff)
#4.2计算欧氏距离：各坐标差的平方和，再开平方
jl1=np.sqrt(np.sum(diff ** 2))
print(f"欧氏距离:{jl1:.6f}")
#4.3计算曼哈顿距离：各坐标差的绝对值之和
jl2=np.sum(np.abs(diff))
print(f"曼哈顿距离:{jl2:.6f}")
print()

#5把向量变为单位长度
a = np.array([-1.20, 0.40, 0.80])
#5.1将向量除以自身长度，不修改原数组
#5.2输入为零向量时抛出 ValueError
#5.3输出单位向量，并用 np.allclose 核对其长度是否为 1
def F(a):
    n=np.linalg.norm(a)
    if n==0:
         raise ValueError("不可输入零向量")
    return a/n

u=F(a)
print(f"单位向量:{u}")
print(f"单位向量长度:{np.linalg.norm(u):.6f}")
print("长度是否为1:", np.allclose(np.linalg.norm(u), 1.0))
print(f"原数组是否未改变:{a}")
print()

#6对同一个二维数组，分别按列、按行和整体汇总
A = np.array([[0.20, -0.50, 1.10],
 [0.80, 0.40, -0.30]])
#6.1计算 A.sum(axis=0)，输出数值和 shape
s=A.sum(axis=0)
print(f"A.sum(axis=0):{s}")
print(f"shape:{s.shape}")
#6.2计算 A.mean(axis=1)，输出数值和 shape
m=A.mean(axis=1)
print(f"A.mean(axis=0):{m}")
print(f"shape:{m.shape}")
#6.3求整个数组的最大值，并解释前两个结果分别对应哪些输入位置
Amax=A.max()
print(f"A.max():{Amax}")
print("第一个结果输出的是按行维度进行压缩，即得到每列的总数；第二个结果输出的是按列维度进行压缩，即得到每行中的最大值，")
print()

#7一次广播，处理所有行
X = np.array([[0.80, -0.60, 0.70],
 [-0.40, 0.50, 0.20]])
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])
#7.1不写 Python 循环，用一条数组表达式计算 Y = X * scale + bias
Y = X * scale + bias
#7.2输出 Y 和 shape
print(f"Y:{Y}")
print(f"shape:{Y.shape}")
#7.3说明 scale、bias 如何与 X 的两行对齐
print("X的shape 为 (2,3)，scale和bias的shape 为 (3,)，广播从右往左对齐，广播时(3,)时先变为(1,3),然后变为(2,3),就可以进行计算了")
print()

#8把两张图像组成一个 batch
a = np.array([[[0.20, -0.40], [0.70, 0.10]],
 [[0.30, 0.80], [-0.20, 0.50]],
 [[-0.10, 0.60], [0.40, -0.30]]])
b = a + 0.10
#8.1用 np.stack 将 a、b 组成 batch，新增的第 0 轴表示图像编号
batch=np.stack([a, b], axis=0) 
#8.2输出 batch 的 shape，并提取两张图像的蓝色通道（C=2）
print(f"shape:{batch.shape}")
blue=batch[:, 2, :, :]
print(f"蓝色通道shape:{blue.shape}")
print(f"蓝色通道值:{blue}")
#8.3将 batch 转为 (B,H,W,C) 顺序，只输出变换后的 shape
batch1=np.transpose(batch, (0, 2, 3, 1))
print(f"变换后的shape:{batch1.shape}")
print()

#9批量计算夹角
query = np.array([0.60, -0.80, 0.00])
X = np.array([[0.30, -0.40, 0.00],
 [-0.60, 0.80, 0.00],
 [0.80, 0.60, 0.00],
 [0.00, 0.00, 0.00]])
#9.1实现 angles_to(query, X)，返回 shape 为 (4,) 的角度数组，单位为度
#9.2用矩阵乘法和按行求长度完成，不写遍历 X 各行的循环。
#9.3X 中的零向量行返回 np.nan；query 为零向量则抛出 ValueError
#9.4求反余弦前 clip 到 [-1,1]，并保持输出顺序与 X 的行顺序一致
def angles_to(query, X):
    qn=np.linalg.norm(query)
    if qn==0:
         raise ValueError("不可输入零向量")
    xn=np.linalg.norm(X,axis=1)
    dots=X@query
    out=np.full(X.shape[0], np.nan)
    valid=xn>0
    cos=dots[valid]/xn[valid]*qn
    cos=np.clip(cos, -1.0, 1.0)
    out[valid] = np.degrees(np.arccos(cos))
    return out
x=angles_to(query, X)
print(f"角度数组:{x}")
print(f"shape:{x.shape}")
print()

#10用余弦相似度做 Top-k 检索
query = np.array([0.60, -0.80, 0.00])
gallery = np.array([[0.60, -0.80, 0.00],
 [0.30, -0.40, 0.00],
 [0.80, 0.60, 0.00],
 [-0.60, 0.80, 0.00],
 [0.00, 0.00, 0.00]])
ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
#10.1实现 top_k_cosine(query, gallery, k)，返回行索引和相似度；本题 k=3
#10.2相似度是点乘除以两边长度。零向量行不参与排序，零查询抛出 ValueError
#10.3分数相同时保持原行顺序，可使用 np.argsort(..., kind="stable")
#10.4k 必须为正整数且不超过有效行数，否则抛出 ValueError；输出选中 ID
def top_k_cosine(query, gallery, k, ids):
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    ids = np.asarray(ids)

    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        raise ValueError("top_k_cosine: query 不能是零向量")

    row_norms = np.linalg.norm(gallery, axis=1)
    valid = row_norms > 0
    valid_indices = np.where(valid)[0]

    if not isinstance(k, int) or k <= 0 or k > len(valid_indices):
        raise ValueError("top_k_cosine: k 必须是正整数且不超过有效行数")

    scores = (gallery[valid] @ query) / (row_norms[valid] * q_norm)

    # 降序排序；同分时保持原行顺序
    order = np.argsort(-scores, kind="stable")[:k]
    selected_idx = valid_indices[order]
    selected_scores = scores[order]
    selected_ids = ids[selected_idx]
    return selected_idx, selected_scores, selected_ids

idx, sim, selected_ids = top_k_cosine(query, gallery, k=3, ids=ids)
print("Top-k 行索引:", idx)
print("Top-k 相似度:", sim)
print("Top-k ID:", selected_ids)
print()

#11所有点对之间的距离矩阵
A = np.array([[0.20, -0.40],
 [0.80, 0.30],
 [-0.50, 0.70]])
B = np.array([[0.10, 0.20],
 [-0.60, -0.20]])
#11.1不写双重循环，用广播构造差数组，说明其 shape
diff=A[:,None,:]-B[None,:,:]
print(f"差数组:{diff}")
print(f"shape:{diff.shape}")
#11.2计算距离矩阵 D，使 D[i,j] 表示 A[i] 到 B[j] 的距离
D=np.sqrt(np.sum(diff ** 2, axis=2))
#11.3输出 D 及 shape，并为 A 的每一行返回距离最近的 B 行索引
print(f"距离矩阵:{D}")
print(f"shape:{D.shape}")
nearest=np.argmin(D, axis=1)
print(f"A每行最近B行索引:{nearest}")
print()

#12用矩阵运算完成一次前向计算
X = np.array([[0.80, -0.60, 0.70],
 [-0.40, -0.50, 0.90]])
W1 = np.array([[-0.10, 0.20, -0.80],
 [0.30, -0.50, 0.10]])
b1 = np.array([-0.15, 0.05])
W2 = np.array([[0.20, -0.60], [-0.80, 0.52]])
b2 = np.array([0.10, -0.20])
#12.1W1、W2 按 (输出特征数, 输入特征数) 存储。计算 Z = X @ W1.T + b1
Z = X @ W1.T + b1
#12.2使用 H = np.maximum(Z, 0.0)，将每个负数替换为 0
H = np.maximum(Z, 0.0)
print(f"H:{H}")
#12.3计算 Y = H @ W2.T + b2，输出 Z、H、Y 及各自 shape
Y = H @ W2.T + b2
print(f"Z:{Z}")
print(f"Z的shape:{Z.shape}")
print(f"H:{H}")
print(f"H的shape:{H.shape}")
print(f"Y:{Y}")
print(f"Y的shape:{Y.shape}")
#12.4不写遍历样本的循环，不改变给定参数