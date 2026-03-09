import json
import pandas as pd
import jieba
import jieba.posseg as pseg
import re
from pathlib import Path
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
import warnings

warnings.filterwarnings("ignore")

def get_interview_stopwords():
    """
    针对深度访谈及“数字人助农”课题定制的停用词表
    经历了三轮极限降噪，靶向清除所有口语冗余词
    """
    return set([
        '的', '了', '在', '是', '和', '就', '不', '都', '一', '上', '也', '很', '到', 
        '说', '去', '会', '着', '没有', '看', '好', '自己', '这', '那', '然后', '就是', 
        '什么', '怎么', '可以', '可能', '觉得', '因为', '所以', '如果', '一个', '现在',
        '的话', '还是', '那些', '时候', '出来', '知道', '一样', '一些', '其实', '大家', 
        '比较', '很多', '做', '对', '哎', '啊', '嗯', '哦', '吧', '呢', '嘛', '的话',
        '这个', '那个', '咱们', '我们', '你们', '他们', '当时', '后来', '基本上',
        '直播', '数字', '直播间', '东西', '情况', '方面', '问题', '里头', '应该', '进行',
        '有', '人', '能', '还', '来', '要', '还有', '包括', '需要', '不会', 
        '想', '把', '给', '让', '跟', '它', '但', '但是', '只', '又',
        '像', '问', '肯定', '一般', '直接', '一直', '弄', '比如', '走', '带', '提供',
        '非常', '特别', '这种', '那种', '一下', '一点', '一定', '可能', '其实', '或者',
        '开始', '发现', '产生', '作为', '买', '卖', '村', '已经', '算', '再', '人家'
    ])

def clean_and_tokenize_interview(text, stopwords, allowed_pos=('n', 'v', 'vn', 'a', 'ad', 'd')):
    """
    针对访谈文本的分词与词性过滤
    """
    if not isinstance(text, str) or text.strip() == '' or text.strip().lower() == 'nan':
        return []

    # 剔除标点、数字及英文字符
    text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    
    words = pseg.cut(text)
    valid_words = []
    
    for word, flag in words:
        # 放宽单字词限制：部分单字动词/形容词在访谈中极具表意性（如“贵”、“差”、“好”、“赚”）
        if word not in stopwords and len(word) >= 1 and flag.startswith(allowed_pos):
            valid_words.append(word)
            
    return valid_words


def analyze_interview_topics(jsonl_path, num_topics=3, num_words=6):
    stopwords = get_interview_stopwords()
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    df = pd.DataFrame(data)
    # 按地点 (location) 进行分组，对比河北、宁夏和企业端的不同主题
    grouped = df.groupby('location')
    results = {}

    print(f"[INFO] 开始质性访谈 LDA 深度分析，共发现 {len(grouped)} 个调研地/主体...\n")

    for location, group_df in grouped:
        print(f"========== 正在挖掘: {location} ==========")
        contents = group_df['text'].tolist()
        
        tokenized_docs = [clean_and_tokenize_interview(text, stopwords) for text in contents]
        tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 1] # 过滤掉分词后极短的无效句
        num_docs_valid = len(tokenized_docs)
        
        if num_docs_valid < 3:
            print(f"[DEBUG] [{location}] 有效对话过少(仅 {num_docs_valid} 句)，跳过 LDA 分析。\n")
            continue

        # 构建 Bigram 模型
        bigram = Phrases(tokenized_docs, min_count=2, threshold=5)
        bigram_mod = models.phrases.Phraser(bigram)
        docs_with_bigrams = [bigram_mod[doc] for doc in tokenized_docs]

        dictionary = corpora.Dictionary(docs_with_bigrams)
        
        if num_docs_valid >= 20:
            dictionary.filter_extremes(no_below=2, no_above=0.6)
        elif num_docs_valid >= 5:
            dictionary.filter_extremes(no_below=1, no_above=0.8)
            
        if len(dictionary) == 0:
            print(f"[DEBUG] [{location}] 过滤后词典为空，说明话语同质化极高或数据量太小，跳过。\n")
            continue

        corpus = [dictionary.doc2bow(doc) for doc in docs_with_bigrams]
        
        # 动态调整主题数：访谈文本如果较少，强行聚3类会拆散语义
        actual_num_topics = min(num_topics, max(1, len(dictionary) // 15))
        if actual_num_topics == 0: actual_num_topics = 1

        # 针对短文本/访谈记录，增加 passes 迭代次数以求模型充分收敛
        lda_model = models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=actual_num_topics,
            random_state=42,
            passes=30,          
            iterations=200,
            alpha='asymmetric', 
            workers=3           
        )
        
        if actual_num_topics > 1:
            coherence_model_lda = CoherenceModel(model=lda_model, texts=docs_with_bigrams, dictionary=dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print(f"[INFO] 模型主题一致性评分 (C_v Score): {coherence_lda:.4f}")
        else:
            coherence_lda = None
            print("[INFO] 语料高度集中，仅生成 1 个核心主题。")
        
        topics = []
        for idx, topic in lda_model.print_topics(num_words=num_words):
            topic_words = []
            for word_prob in topic.split(' + '):
                prob, word = word_prob.split('*"')
                word = word.replace('"', '')
                topic_words.append(f"{word}({float(prob):.4f})")
            
            topic_str = ", ".join(topic_words)
            topics.append(topic_str)
            print(f"主题 {idx+1}: {topic_str}")
        
        results[location] = {
            'coherence_score': coherence_lda,
            'topics': topics
        }
        print("\n")

    return results

if __name__ == "__main__":
    # 自动定位项目根目录，适配你的 2026-Spring 架构
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    input_path = PROJECT_ROOT / "data" / "processed" / "structured_interviews.jsonl"
    output_path = PROJECT_ROOT / "data" / "processed" / "lda_analysis_results.json"
    
    # 提取最具代表性的核心主题词
    final_results = analyze_interview_topics(input_path, num_topics=3, num_words=6)
    
    if final_results:
        print(f"[INFO] 正在将分析结果保存至: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        print("[INFO] 结果保存成功！你可以打开 JSON 文件查看不同地点的核心诉求差异。")
    else:
        print("[DEBUG] 未生成有效结果。")