import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from transformers import pipeline

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置中文字体
PROJECT_ROOT = Path(__file__).resolve().parent.parent

font_path = PROJECT_ROOT / "font" / "NotoSansCJK-Regular.ttc"

fm.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False


class TransformerSentimentAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.input_file = self.base_dir / "data" / "processed" / "structured_interviews.jsonl"
        self.output_dir = self.base_dir 

        logging.info("正在加载 Transformer 预训练情感模型...")
        self.sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
            device=-1 # -1 表示使用 CPU，如果有 NVIDIA GPU 可改为 0
        )
        logging.info("[INFO] 模型加载完毕！")

    def load_data(self):
        """读取清洗好的黄金语料"""
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                # 过滤掉太短的无意义短句（小于5个字通常无法判断情感）
                if len(record['text']) >= 5:
                    data.append(record)
        df = pd.DataFrame(data)
        return df

    def analyze_sentiment(self):
        df = self.load_data()
        logging.info(f"[INFO] 开始对 {len(df)} 条访谈语句进行深度语义情感测算...")

        scores = []
        labels = []
        
        # 遍历测算每一句话的情感
        for idx, text in enumerate(df['text']):
            if idx % 50 == 0 and idx > 0:
                logging.info(f"已处理 {idx} 条...")
                
            try:
                # 截断超长文本 
                truncated_text = text[:500] 
                result = self.sentiment_pipeline(truncated_text)[0]
                
                # 该模型返回标签 'Positive' 或 'Negative'，以及对应的置信度 score
                label = result['label']
                confidence = result['score']

                if label == 'Positive':
                    favorability = 50 + (confidence * 50)
                else:
                    favorability = 50 - (confidence * 50)
                    
                scores.append(round(favorability, 2))
                labels.append(label)
                
            except Exception as e:
                logging.error(f"解析失败: {text[:20]}... 错误: {e}")
                scores.append(50.0) # 失败则给中性分
                labels.append('Neutral')

        df['sentiment_score'] = scores
        df['sentiment_label'] = labels
        
        # 保存带有情感得分的详细数据表
        csv_path = self.output_dir / "data" / "processed" / "sentiment_detailed_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"[INFO] 情感测算完成！详细得分已保存至: {csv_path.name}")
        
        self.plot_academic_distribution(df)
        self.generate_statistical_report(df)

    def plot_academic_distribution(self, df):
        """
        绘制小提琴图 (Violin Plot)
        它不仅能展示均值，还能展示情感分布的方差和两极分化程度
        """
        plt.figure(figsize=(10, 6), dpi=300)
        
        sns.violinplot(
            x="location", 
            y="sentiment_score", 
            data=df, 
            palette="muted", 
            inner="quartile", # 显示四分位数
            cut=0
        )

        sns.swarmplot(
            x="location", 
            y="sentiment_score", 
            data=df, 
            color="black", 
            alpha=0.4, 
            size=3
        )

        plt.title("不同地域对AI数字人赋能模式的情感倾向（好感度）核密度分布", fontsize=15, pad=15)
        plt.xlabel("地域 / 访谈主体", fontsize=12)
        plt.ylabel("情感好感度得分 (0=极度排斥, 100=极度拥护)", fontsize=12)
        plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='中性基准线 (50分)')
        plt.legend()
        plt.tight_layout()
        
        plot_path = self.output_dir / "imgs"/ "sentiment_violin_plot.png"
        plt.savefig(plot_path)
        logging.info(f"[INFO] 核密度分布图已生成: {plot_path.name}")

    def generate_statistical_report(self, df):
        """生成用于论文的描述性统计表格"""
        report = df.groupby('location')['sentiment_score'].agg(
            样本量='count',
            平均好感度='mean',
            中位数='median',
            情感方差='std',
            最高分='max',
            最低分='min'
        ).round(2).reset_index()
        
        report_path = self.output_dir / "sentiment_statistical_summary.csv"
        report.to_csv(report_path, index=False, encoding='utf-8-sig')
        logging.info(f"[INFO] 统计学摘要表已生成: {report_path.name}")
        
        print("\n" + "="*50)
        print("[INFO] 区域情感(好感度)核心统计摘要")
        print("="*50)
        print(report.to_string(index=False))
        print("="*50)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    analyzer = TransformerSentimentAnalyzer(base_dir=PROJECT_ROOT)
    analyzer.analyze_sentiment()