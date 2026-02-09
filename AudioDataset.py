
# huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./wav2vec2-base-960h --local-dir-use-symlinks False


from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import os 
import json
import torch

from tqdm import tqdm
from typing import List, Tuple, Any
from torch.utils.data import Dataset, DataLoader

from PreProcess import ctrl_expressions as ctrl_expressions_list
import pickle
from matplotlib.ticker import MaxNLocator

def pack_up(exp, save_path):
    ''' '''
    N = len(exp)
    motion_pred = np.zeros((N, 55*3)).tolist()
    face_pred = exp
    transfer_motion = {
        "motion_pred": motion_pred,
        "face_pred": face_pred,
        "fps": 25,
        "frames": N,
    }

class UE_CurvesManager:
    def __init__(self,json_path):
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            datas = json.load(f) 
        
        self.curve_name = datas['AnimName']
        self.data = datas[self.curve_name]
        # self.time_long = datas['time_long']
    
    def get_match_data(self,key,t):
        """
        输入时间 t，返回对应插值的值
        times: list[float]
        values: list[float]
        """
        # t = frame_index/self.fps
        times = self.data[key]["time"]
        values = self.data[key]["value"]


        if not times:
            return None
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]

        # 遍历找到区间
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            if t0 <= t <= t1:
                v0, v1 = values[i], values[i + 1]
                alpha = (t - t0) / (t1 - t0)
                return v0 * (1 - alpha) + v1 * alpha


    def sample(self,time_point,seconds,fps:int=25):
        ''' '''
        sample_num = int(seconds*fps)
        exp_frames = []
        for frame_index in range(sample_num):
            exp_frame = []
            for i,exp_key in enumerate(ctrl_expressions_list):
                ''' '''
                t = time_point + frame_index/fps
                v = self.get_match_data(exp_key,t)
                exp_frame.append(v)
            exp_frames.append(exp_frame)
        
        return exp_frames
    
    def showChannel(self, exp_key_list, t_start=None, t_end=None):
        """
        每个表情通道一个子图（原始 keyframe，不采样、不插值）
        """
        n = len(exp_key_list)
        if n == 0:
            return

        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, exp_key in zip(axes, exp_key_list):
            curve = self.data.get(exp_key)
            if curve is None:
                ax.set_title(f"{exp_key} (missing)")
                continue

            t_list = curve['time']
            v_list = curve['value']
            if not t_list:
                ax.set_title(f"{exp_key} (empty)")
                continue

            start_t = t_start if t_start is not None else t_list[0]
            end_t   = t_end   if t_end   is not None else t_list[-1]

            t_clip = []
            v_clip = []
            for t, v in zip(t_list, v_list):
                if start_t <= t <= end_t:
                    t_clip.append(t)
                    v_clip.append(v)

            if t_clip:
                ax.plot(t_clip, v_clip)
            ax.set_title(exp_key)
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
        



class WavSample:
    def __init__(self, json_path, wav_path, processor, model):
        self.json_path = json_path
        self.wav_path = wav_path

        self.wave_data, self.sr = librosa.load(self.wav_path, sr=16000)
        self.time_long = len(self.wave_data) / self.sr

        self.curveManager = UE_CurvesManager(self.json_path)
        self.processor = processor
        self.model = model

    def randomSample(self, sample_num, seconds):
        """
        从音频中随机抽取 sample_num 段，每段长度为 seconds 秒
        返回值: list of np.ndarray，每个元素是一段音频数据
        """
        wavSamples = []
        expSamples = []
        segment_len = int(self.sr * seconds)
        
        if segment_len > len(self.wave_data):
            raise ValueError("指定的采样长度大于音频总长度")
        
        for _ in range(sample_num):
            start_idx = random.randint(0, len(self.wave_data) - segment_len)
            end_idx = start_idx + segment_len
            wavSample = self.wave_data[start_idx:end_idx]

            inputs = self.processor(wavSample, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            features = outputs.last_hidden_state[0]
            wavSamples.append(features)

            time_point = start_idx/self.sr
            expSample = self.curveManager.sample(time_point,seconds)

            expSample = torch.tensor(expSample, dtype=torch.float32)
            expSamples.append(expSample)

            # print("features.shape : ",features.shape)
            # print("expSample.shape: ",expSample.shape)


        return wavSamples, expSamples
    
    @torch.no_grad()
    def plot_compare(
        self,
        time_point,
        channel,              # ["CTRL_expressions_jawOpen", ..]
        seconds=5,
        decoder=None,         # 解码器
        device="cuda",
        max_xticks=10,        # x轴最大刻度数，自动间隔
        hspace=0.5,           # 子图纵向间距
        channels_per_fig=10,  # 每张图片画多少通道
        save_dir="./compare_plots"  # 图片保存目录
    ):
        """
        支持批量绘制多通道，保存为图片，不显示窗口
        """
        assert decoder is not None, "必须指定 decoder 模型"

        if isinstance(channel, str):
            channel = [channel]
        if not isinstance(channel, (list, tuple)):
            raise ValueError("channel 必须是 str 或 list[str]")

        # ---- 名字 → index ----
        channel_idx = []
        for ch in channel:
            if ch not in ctrl_expressions_list:
                raise ValueError(f"通道名 {ch} 不存在！")
            channel_idx.append(ctrl_expressions_list.index(ch))

        # ========= 1. 截取音频 ==========
        segment_len = int(self.sr * seconds)
        start_idx = int(time_point * self.sr)
        end_idx = start_idx + segment_len
        if end_idx > len(self.wave_data):
            raise ValueError("超出音频长度")
        wav = self.wave_data[start_idx:end_idx]
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        wav_feat = self.model(**inputs).last_hidden_state.to(device)

        # ========= 2. decoder 推理 ===========
        pred = decoder(wav_feat)
        pred = pred.squeeze(0).cpu().numpy()

        # ========= 3. Ground Truth ===========
        gt = np.array(self.curveManager.sample(time_point, seconds))

        # ========= 4. 创建保存目录 ===========
        os.makedirs(save_dir, exist_ok=True)

        # ========= 5. 分批绘制 ===========
        for start in range(0, len(channel), channels_per_fig):
            sub_channels = channel[start:start+channels_per_fig]
            sub_indices = channel_idx[start:start+channels_per_fig]
            n = len(sub_channels)

            fig, axes = plt.subplots(n, 1, figsize=(12, 4*n), sharex=True)
            if n == 1:
                axes = [axes]

            x = np.linspace(0, seconds, pred.shape[0])
            for i, (ch_name, idx) in enumerate(zip(sub_channels, sub_indices)):
                ax = axes[i]
                pred_curve = pred[:, idx]
                gt_curve = gt[:, idx]

                ax.plot(x, gt_curve, label="GT", linewidth=2)
                ax.plot(x, pred_curve, "--", label="Pred")

                ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
                ax.set_title(f"{ch_name} (index={idx}) @ t={time_point:.2f}s")
                ax.set_ylabel("Value")
                ax.grid(True)
                ax.legend()

            axes[-1].set_xlabel("Seconds")
            fig.subplots_adjust(hspace=hspace)
            plt.tight_layout()

            # 保存图片，不显示
            filename = f"{save_dir}/compare_{time_point:.2f}s_{start}-{start+n-1}.png"
            fig.savefig(filename)
            plt.close(fig)  # 关闭图，释放内存

        return pred, gt

    def paddingZero(self, seconds, mode="all"):
        """
        对音频进行零填充，使总长度达到 seconds 秒
        mode: "all" -> 前后各补一半
              "before" -> 只在开头补零
              "after" -> 只在末尾补零
        """
        target_len = int(self.sr * seconds)
        current_len = len(self.wave_data)

        if current_len >= target_len:
            return self.wave_data  # 已经够长，不补零

        pad_len = target_len - current_len

        if mode == "all":
            pad_before = pad_len // 2
            pad_after = pad_len - pad_before
        elif mode == "before":
            pad_before = pad_len
            pad_after = 0
        elif mode == "after":
            pad_before = 0
            pad_after = pad_len
        else:
            raise ValueError("mode 必须是 'all', 'before', 或 'after'")

        padded_wave = np.pad(self.wave_data, (pad_before, pad_after), mode='constant', constant_values=0)

        return padded_wave
    

class AudioDataset(Dataset):
    def __init__(self,processor, model, config):
        """
        多模块音频数据集：支持从多个模块路径加载数据，并记录样本所属模块
        :param module_mapping: 字典，格式为 {模块名: {"path": 模块数据路径, "sample_num": 样本数量}}
        :param processor: 原有的wav2vec2处理器
        :param model: 原有的wav2vec2模型
        """  
        self.processor = processor 
        self.model = model 
        self.cache_path = config.get('cache_path', None)
        self.datasets_path = config.get('datasets', None)
        if self.cache_path is None:
            try:
                self.json_path = './Dataset/zhuboshuolianbo/json/'
                self.wav_path = './Dataset/zhuboshuolianbo/wav/'
                self.video_list = []
                self.initializeVideoList()
                self.wavSampleManagerList = []
                if processor is not None:
                    for _video in self.video_list:
                        j_p = os.path.join(self.json_path,'CD_'+_video+'_1.json')
                        w_p = os.path.join(self.wav_path,_video+'.wav')
                        wavSample = WavSample(j_p, w_p, processor=processor, model=model)
                        self.wavSampleManagerList.append(wavSample)
            except ExceptionTtpe as e:
                raise AnotherException('载入数据集错误') from e
                
        self.input_list = [] 
        self.output_list = []
        self.datasets_list = config.datasets_list
        self.samples = []  # 存储所有模块的样本
        self.module_labels = []  # 存储每个样本的模块标签
        self.sample_weights = []
        self.loss_weights = []
        for dataset in config.datasets_list:
            module_samples = self._load_module_samples(
                cache_path=os.path.join(config.cache_path, dataset.file_name. 'temp'),
                sample_mode=dataset.sample_mode,
                sample_num=dataset.get('num_sample', -1)
            )
            if len(module_samples) == 0:
                raise ValueError(f"训练接{dataset.file_name}，载入的样本数量为{len(module_samples)}" )
            
            self.samples.extend(module_samples)
            self.module_labels.extend([dataset.file_name] * len(module_samples))
            self.sample_weights.extend([dataset.sample_weight] * len(module_samples))
            self.loss_weights.extend([dataset.loss_weight] * len(module_samples))
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float32)
        self.loss_weights = torch.tensor(self.loss_weights, dtype=torch.float32)

    def _load_module_samples(self, cache_path, sample_num):
        """
        加载单个模块的样本（适配load_flag=True，从缓存目录加载指定总数量的样本）
        :param cache_path: 模块缓存路径（存储xxx.pkl样本文件）
        :param sample_num: 该模块需要加载的总样本数量
        :return: list[(audio_feat, target)] 加载完成的样本列表
        """
        # 1. 遍历缓存目录下的pkl文件（复用原有_load_cache_from_dir逻辑）
        if not os.path.isdir(cache_path):
            raise FileNotFoundError(f"模块缓存路径{cache_path}不存在！")
        valid_modes = {"all", "sequential", "random"}
        if sample_mode not in valid_modes:
            raise ValueError(
                f"sample_mode 必须是{valid_modes}之一"
            )
        if sample_mode != "all":
            if not isinstance(sample_num, int) or sample_num <= 0:
                raise ValueError(
                    f"sample_num 必须是正整数"
                )
        # 获取缓存目录下所有pkl文件并排序
        cache_files = sorted([f for f in os.listdir(cache_path) if f.endswith(".pkl")])
        total_files = len(cache_files)
        loaded_samples: List[Tuple[Any, Any]] = []

        tqdm_desc = f"加载模块缓存"
        iterator = tqdm(cache_files, desc=tqdm_desc, unit="file")
        
        for file_name in iterator:
            file_path = os.path.join(cache_path, file_name)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                # 提取音频特征和目标特征（与原有缓存格式一致）
                audio_feat = data['input']  # 对应原有缓存的input（audio feature）
                target = data['output']     # 对应原有缓存的output（表情特征）
                loaded_samples.append((audio_feat, target))

                # 对于sequential模式，提前退出避免不必要的I/O
                if (
                    sample_mode=="sequential"
                    and len(loaded_samples) >= sample_num
                ):
                    break
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as e:
                raise RuntimeError(f"加载缓存文件{file_path}失败：{e}")
        if sample_mode == "all":
            return loaded_samples
        if sample_mode == "sequential":
            if len(loaded_samples) < sample_num:
                raise ValueError(
                    f"目录{cache_path}中仅有{len(loaded_samples)}条样本，"
                    f"不足以满足 sequential 模式要求的 {sample_num} 条"
                )
                return loaded_samples[:sample_num]
        if len(loaded_samples) < sample_num:
            raise ValueError(
                    f"目录{cache_path}中仅有{len(loaded_samples)}条样本，"
                    f"不足以满足 random 模式要求的 {sample_num} 条"
                )
            return random.sample(loaded_samples, sample_num)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ''' '''
        # 返回样本和对应的模块标签（结构不变）
        audio_feat, target = self.samples[idx]
        module_label = self.module_labels[idx]
        loss_weights = self.loss_weights[idx]
        # return self.input_list[idx], self.output_list[idx]
        
        return audio_feat, target, module_label, loss_weights
        

    def initializeVideoList(self):
        ''' '''
        # self.video_list = ["BV1A3H6z9Exx","BV1acosYBEJr"]
        path = './Dataset/zhuboshuolianbo/video'
        files_name = os.listdir(path)
        for file_name in files_name:
            self.video_list.append(file_name[:-4])

    
    def generateSample(self, seconds: int = 5, load_flag: bool = False):
        """
        每条 sample 保存为单独 pickle 文件，self.cache_path 是目录
        """
        # 确保目录存在
        os.makedirs(self.cache_path, exist_ok=True)

        if load_flag:
            self._load_cache_from_dir()
        else:
            self.input_list.clear()
            self.output_list.clear()

            print(f"[AudioDataset] Generating samples for {len(self.wavSampleManagerList)} videos...")

            for wavSample in tqdm(self.wavSampleManagerList, desc="Videos", unit="video"):
                time_long = wavSample.time_long
                sample_num = int(2 * (time_long / seconds))

                wavSamples, expSamples = wavSample.randomSample(sample_num, seconds)

                for idx, (feat, exp) in enumerate(tqdm(zip(wavSamples, expSamples),
                                                        total=len(wavSamples),
                                                        desc=f"Samples for {wavSample.wav_path}",
                                                        leave=False, unit="sample")):
                    # ---------- audio feature ----------
                    if isinstance(feat, torch.Tensor) and feat.ndim == 3:
                        feat = feat.squeeze(0)  # (249, 768)
                    feat = feat.float()

                    # ---------- target ----------
                    if not isinstance(exp, torch.Tensor):
                        exp = torch.tensor(exp)
                    exp = exp.float()  # (125, 136)

                    # 保存到内存列表
                    self.input_list.append(feat)
                    self.output_list.append(exp)

                    # 保存为单独文件
                    filename = f"{wavSample.wav_path.split('/')[-1][:-4]}_{idx}.pkl"
                    filepath = os.path.join(self.cache_path, filename)
                    with open(filepath, 'wb') as f:
                        pickle.dump({'input': feat, 'output': exp}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def clearInit(self):
        ''' '''
    
    def _load_cache_from_dir(self):
        """
        从 self.cache_path 目录加载所有 pickle 文件，遇到损坏文件会跳过
        """
        self.input_list.clear()
        self.output_list.clear()

        files = sorted(os.listdir(self.cache_path))  # 保证顺序
        print(f"[AudioDataset] Loading {len(files)} samples from {self.cache_path}...")

        for file in tqdm(files, desc="Loading samples", unit="file"):
            filepath = os.path.join(self.cache_path, file)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.input_list.append(data['input'])
                self.output_list.append(data['output'])
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as e:
                print(f"[Warning] Skipping corrupted file {file}: {e}")


if __name__ == '__main__':
    ''' '''
    processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("./wav2vec2-base-960h")


    myAudioDataset = AudioDataset(processor=processor,model=model)
    myAudioDataset.generateSample()

    loader = DataLoader(
        myAudioDataset,
        batch_size=32,
        shuffle=True
    )

    for audio_feat, exp in loader:
        print(audio_feat.device)
        print(audio_feat.shape, len(exp))
        print(np.array(exp).shape)


