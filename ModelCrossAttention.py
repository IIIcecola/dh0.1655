"""
Audio2Face with cross-attention alignment

核心思想：
1. 使用可学习的125个query向量
2. 通过cross-attention让query去attend 249帧的音频特征
3. 模型自动学习“哪几帧对应某个口型”

优势：
- 显示学习音频-面部对齐
- 不破坏时序信息
- 可以处理任意长度的输入音频
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """标准的位置编码"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, max_len=300, d_model=768):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
        
class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module
    使用深度可分离卷积 + pointwise 卷积
    """
    def __init__(self, d_model, kernal_size=15, dropout=0.1):
        super().__init__()
        assert kernal_size % 2 == 1, "kernal_size must be odd"
        padding = (kernal_size - 1) // 2

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernal_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernal_size=kernal_size,
            padding=padding, groups=d_model
        )
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, 2 * d_model, kernal_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GLU(dim=1)
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
        """
        residual = x
        x = self.layer_norm(x)

        # (B, T, D) > (B, D, T)
        x = x.transpose(1, 2)

        # pointwise conv with GLU
        x = self.pointwise_conv1(x)
        x = self.activation(x)

        # depthwise conv
        x = self.depthwise_conv(x)

        # pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # (B, D, T) > (B, T, D)
        x = x.transpose(1, 2)

        # layernorm
        x = self.norm(x)

        return residual + x




class AudioEncoder(nn.Module):
    """
    音频编码器：将Wav2Vec2特征编码为更丰富的表示
    支持三种编码方式：
    - transformer
    - cnn
    - conformer(结合cnn + transformer)
    """
    def __init__(self, d_model=768, nhead=8, num_layers=4, dim_feedforward=2048, 
                 encoder_type='transformer', dropout=0.1):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-LN for stability
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif encoder_type == 'cnn':
            self.conv_layers = nn.ModuleList([
                # 保留原始特征的跳跃连接
                nn.Identity(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model),
            ])
            self.fusion = nn.Conv1d(d_model * 4, d_model, kernal_size=1)
        elif encoder_type == 'conformer':
            self.encoder = nn.ModuleList([
                ConformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T_audio, 768)音频特征
        Returns:
            encoded: (B, T_audio, 768)编码后的音频特征
        """
        if encoder_type == 'transformer':
            x = self.pos_encoding(x)
            x = self.dropout(x)
            encoded = self.encoder(x)
        elif encoder_type == 'cnn':
            x_orig = x.transpose(1, 2) # (B, 768, T)
            features = []
            for conv in self.conv_layers:
                if isinstance(conv, nn.Identity):
                    feat = x_orig
                else:
                    feat = conv(x_orig)
                features.append(feat)
            concat = torch.cat(features, dim=1)
            fused = self.fusion(concat)
            encoded = fused.transpose(1, 2)
        elif encoder_type == 'conformer':
            x = self.pos_encoding(x)
            x = self.dropout(x)
            encoded = x
            for layer in self.encoder:
                encoded = layer(encoded)
        return encoded


class FaceQueryGenerator(nn.Module):
    """
    生成面部参数的query向量
    支持三种方式：
    - embedding: 使用可学习的embedding
    - mlp: 使用MLP从位置信息生成query
    - hybrid: 结合embedding和content_aware（基于音频内容）
      hybrid = alpha * embedding + (1 - alpha) * content_aware
    """
    def __init__(self, num_queries=125, d_model=768, query_type='embedding', hybrid_alpha=0.5):
        super().__init__()
        self.num_queries=num_queries
        self.query_type=query_type
        self.hybrid_alpha=hybrid_alpha
        if query_type == 'embedding':
            # 直接学习每个位置的query
            self.query_embed = nn.Embedding(num_queries, d_model)
            nn.init.normal_(self.query_embed.weight, mean=0, std=0.1)
        elif query_type == 'mlp':
            # 从位置信息生成query（更灵活，可以泛化到不同长度）
            self.mlp = nn.Sequential(
                nn.Linear(3, d_model // 4), # 输入：归一化的位置索引
                nn.ReLU(),
                nn.Linear(d_model // 4, d_model)
            )
        elif query_type == 'hybrid':
            # Embedding query （可学习）
            self.query_embed = nn.Embedding(num_queries, d_model)
            nn.init.normal_(self.query_embed.weight, mean=0, std=0.1)

            # Content-aware query （从音频内容生成）
            # 使用attention从音频中提取关键信息作为query
            self.content_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.content_query = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.1)
            self.content_norm = nn.LayerNorm(d_model)
            self.content_proj = nn.Linear(d_model, d_model)

        # 位置编码
        self.query_pos_encoding = LearnablePositionalEncoding(max_len=num_queries, d_model=d_model)
    def forward(self, batch_size, device, audio_features=None):
        """
        Args:
            batch_size: 批量大小
            device：设备
            audio_features: (B, T_audio, 768)音频特征（仅hybrid模式需要）
        Returns:
            queries: (B, num_queries, 768)
        """
        if self.query_type == 'embedding':
            # 生成query索引
            indices = torch.arange(self.num_queries, device=device)
            queries = self.query_embed(indices) # （num_queries, 768）
            queries = queries.unsqueeze(0).expand(batch_size, -1, -1) # (B, num_queries, 768)
        elif self.query_type == 'mlp':
            # 生成归一化的位置索引
            positions = torch.linspace(0, 1, self.num_queries, device=device)
            positions = positions.unsqueeze(0).unsqueeze(-1) # (1, num_queries, 1)
            # 扩展到3维
            positions = positions.expand(batch_size, -1, 3) # (B, num_queries, 3)
            queries = self.mlp(positions) #  (B, num_queries, 768)
        elif self.query_type == 'hybrid':
            # 1.embedding query 可学习
            indices = torch.arange(self.num_queries, device=device)
            queries_embed = self.query_embed(indices) # （num_queries, 768）
            queries_embed = queries.unsqueeze(0).expand(batch_size, -1, -1) # (B, num_queries, 768)
            # 2. content-aware query （从音频内容生成）
            if audio_features is not None:
                # 使用可学习的content query 去 attend 音频特征
                content_query = self.content_query.expand(batch_size, -1, -1) # (B, num_queries, 768)
                queries_content, _ = self.content_attn(
                    content_query,  # query
                    audio_features, # key
                    audio_features  # value
                ) # (B, num_queries, 768)
                queries_content = self.content_norm(queries_content)
                queries_content = self.content_proj(queries_content)
            else:
                # 如果没有音频特征，使用零初始化
                queries_content = torch.zeros_like(queries_embed)
            # 3. 加权融合：alpha * embedding + (1 - alpha) * content_aware
            alpha = self.hybrid_alpha
            queries = alpha * queries_embed + (1 - alpha) * queries_content
        # 添加位置编码
        queries = self.query_pos_encoding(queries)
        return queries


class CrossAttentionDecoder(nn.Module):
    """
    Cross-Attention解码器
    使用query通过cross-attention从音频特征中提取信息
    """
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    def forward(self, query, memory):
        """
        Args:
            query: (B, T_query, 768) 面部query
            memory: (B, T_memory, 768) 编码后的音频特征

        Returns:
            output: (B, T_query, 768)解码后的特征
        """  
        output = self.decoder(query, memory)
        return output

class Audio2FaceCrossAttention(nn.Module):
    """
    完整的Audio2Face模型（使用Cross-Attention对齐）

    架构：
        Audio Features (B, 249, 768) >
        AudioEncoder (Transformer / CNN / Conformer) >
        Encoded Audio (B, 249, 768) >
        FaceQueryGenerator, 125 Queries (B, 249, 768) >
        CrossAttentionDecoder (queries attend to audio) >
        Decoded Features (B, 249, 768) >
        Output Projection >
        Face Parameters (B, 249, 768)
    """
    def __init__(
        self,
        input_dim=768,
        output_dim=136,
        num_queries=125,
        audio_encoder_type='transformer',
        query_type='embedding',

        # Audio Encoder参数
        audio_encoder_layers=4,
        audio_encoder_nhead=8,
        audio_encoder_ff_dim=2048,

        # Decoder参数
        decoder_layers=6,
        decoder_nhead=8,
        decoder_ff_dim=2048,

        # Hybrid query 参数
        hybrid_alpha=0.5,

        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries

        # 音频编码器
        self.audio_encoder = AudioEncoder(
            d_model=input_dim,
            nhead=audio_encoder_nhead,
            num_layers=audio_encoder_layers,
            dim_feedforward=audio_encoder_ff_dim,
            encoder_type=audio_encoder_type,
            dropout=dropout
        )

        # 面部Query生成器
        self.query_generator = FaceQueryGenerator(
            num_queries=num_queries,
            d_model=input_dim,
            query_type=query_type,
            hybrid_alpha=hybrid_alpha
        )

        # Cross-Attention解码器
        self.decoder = CrossAttentionDecoder(
            d_model=input_dim,
            nhead=decoder_nhead,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ff_dim,
            dropout=dropout
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

        # Skip connection （从query直接映射到输出）
        self.skip_proj = nn.Linear(input_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
        # 输出层使用较小的初始化，匹配目标值范围（类似Transformer）
        # 目标值通常在 [0, 1] 或类似小范围
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.skip_proj.weight, gain=0.1)
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, T_audio, 768) Wav2Vec音频特征
            return_attention: 是否返回attention权重（用于可视化）
        Returns:
            output: (B, 125, 136) 面部参数
            attention_weights: （可选）attention权重
        """
        batch_size = x.size(0)

        # 1. 编码音频特征
        memory = self.audio_encoder(x) # (B, T_audio, 768)
        # 2. 生成面部query(hybrid 模式需要传递原始音频特征)
        if self.query_generator.query_type == 'hybrid':
            # hybrid 模式使用编码后的音频特征
            query = self.query_generator(batch_size, x.device, audio_features=memory)
        else:
            query = self.query_generator(batch_size, x.device) # (B, 125, 768)
        # 3. Cross-Attention解码
        decoded = self.decoder(query, memory)
        # 4. 输出投影
        output_main = self.output_proj(decoded)
        output_skip = self.skip_proj(query)
        # 5.融合主输出和skip connection
        output = output_main + output_skip
        if return_attention:
            # 获取最后一层的attention权重（用于可视化对齐）
            # 注意：这需要修改decoder来返回attention
            return output, None
        return output
    def get_attention_maps(self, x):
        """
        获取attention map用于可视化
        Returns:
            attention_maps: List of (B, num_heads, 125, T_audio)
        """
        # 这需要修改decoder来保存attention权重
        # 暂时返回None
        return None


















