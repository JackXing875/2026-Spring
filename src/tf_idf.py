import json
import logging
import jieba
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

font_path = PROJECT_ROOT / "font" / "NotoSansCJK-Regular.ttc"

fm.fontManager.addfont(str(font_path))
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class TfidfAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.input_file = self.base_dir / "data" / "processed" / "structured_interviews.jsonl"
        self.output_dir = self.base_dir / "imgs"
        
        # 停用词表
        self.stop_words = {
            "我们", "你们", "他们", "这个", "那个", "然后", "就是", "什么", "怎么", 
            "可以", "可能", "觉得", "因为", "所以", "如果", "一个", "没有", "现在",
            "自己", "的话", "还是", "那些", "时候", "出来", "知道", "一样", "一些",
            "其实", "大家", "比较", "很多", "做", "说", "去", "这", "那", "就", "也", "在", "的"
        }

    def load_data(self):
        """读取 JSONL 数据并转为 DataFrame"""
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        logging.info(f"成功加载数据，共 {len(df)} 条发言。")
        return df

    def preprocess_text(self, text):
        """中文分词与停用词过滤"""
        # 使用 jieba 进行精确分词
        words = jieba.cut(text, cut_all=False)
        # 过滤单字词（通常意义不大）和停用词
        cleaned_words = [w for w in words if len(w) > 1 and w not in self.stop_words]
        return " ".join(cleaned_words)

    def run_tfidf(self, top_n=15):
        df = self.load_data()
        
        # 1. 按照地点 (location) 将所有文本合并
        # 这意味着我们将分析“不同村庄/企业”各自独有的高频特征词
        grouped_df = df.groupby('location')['text'].apply(lambda x: ' '.join(x)).reset_index()
        
        logging.info("[INFO] 正在进行中文分词处理...")
        grouped_df['processed_text'] = grouped_df['text'].apply(self.preprocess_text)
        
        # 2. 计算 TF-IDF
        logging.info("[INFO] 正在计算 TF-IDF 矩阵...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(grouped_df['processed_text'])
        
        feature_names = vectorizer.get_feature_names_out()
        
        # 3. 提取每个地点的 Top N 关键词
        results = []
        for i, row in grouped_df.iterrows():
            location = row['location']
            # 获取该地点的 tf-idf 向量
            tfidf_scores = tfidf_matrix[i].toarray().flatten()
            
            # 将词和分数打包，并按分数倒序排列
            word_scores = list(zip(feature_names, tfidf_scores))
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            
            top_words = word_scores[:top_n]
            for rank, (word, score) in enumerate(top_words, 1):
                results.append({
                    "Location": location,
                    "Rank": rank,
                    "Word": word,
                    "TF-IDF_Score": round(score, 4)
                })
        
        # 4. 保存为 CSV 给论文做表格用
        results_df = pd.DataFrame(results)
        csv_path = self.output_dir / "tfidf_top_words_by_location.csv"
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig 防止 Excel 打开乱码
        logging.info(f"[INFO] TF-IDF 特征词提取完毕！已保存至 {csv_path.name}")
        
        # 5. 可视化：生成对比条形图
        self.plot_results(results_df, top_n)

    def plot_results(self, results_df, top_n):
        """为每个地点绘制柱状图"""
        locations = results_df['Location'].unique()
        num_locs = len(locations)
        
        fig, axes = plt.subplots(num_locs, 1, figsize=(10, 5 * num_locs))
        if num_locs == 1:
            axes = [axes] # 兼容只有一个地点的情况
            
        for ax, location in zip(axes, locations):
            loc_data = results_df[results_df['Location'] == location].sort_values(by="TF-IDF_Score", ascending=True)
            
            ax.barh(loc_data['Word'], loc_data['TF-IDF_Score'], color='#4C72B0')
            ax.set_title(f"[{location}] Top {top_n} 特征词", fontsize=14)
            ax.set_xlabel("TF-IDF 权重")
            ax.set_ylabel("特征词")
                        
        plt.tight_layout()
        plot_path = self.output_dir / "tfidf_visualization.png"
        plt.savefig(plot_path, dpi=300)
        logging.info(f"[INFO] 可视化图表已生成并保存至 {plot_path.name}")

if __name__ == "__main__":
    analyzer = TfidfAnalyzer(base_dir=PROJECT_ROOT)
    analyzer.run_tfidf()