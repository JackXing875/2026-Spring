import json
from pathlib import Path

class QuoteExtractor:
    def __init__(self, base_dir: str):
        self.input_file = Path(base_dir) / "data" / "processed" / "structured_interviews.jsonl"
        self.output_file = Path(base_dir) / "data" / "processed" / "golden_quotes.md"

        self.target_themes = {
            "宁夏银川保南村": [
                {"theme_name": "村企合作与集体经济", "keywords": ["公司", "集体", "联系"]},
                {"theme_name": "产业期望与变现", "keywords": ["产业", "价格", "钱"]},
                {"theme_name": "账号运营实操", "keywords": ["账号", "直播", "了解"]}
            ],
            "河北行唐龙洞村": [
                {"theme_name": "平台生态与产品合作", "keywords": ["平台", "产品", "合作"]},
                {"theme_name": "技术降维与实操", "keywords": ["脚本", "对接", "账号"]},
                {"theme_name": "数字包容与弱势赋能", "keywords": ["残疾人", "乡村_振兴", "农村", "快手"]}
            ]
        }

    def load_data(self):
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def extract(self):
        records = self.load_data()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("# 📝 论文实证引用金句集锦\n\n")
            
            for location, themes in self.target_themes.items():
                f.write(f"## 📍 {location}\n\n")
                # 过滤出该地点的记录
                loc_records = [r for r in records if location[:2] in r['location'] or r['location'] in location]
                
                for theme in themes:
                    f.write(f"### 主题：{theme['theme_name']}\n")
                    f.write(f"> 追踪关键词: {', '.join(theme['keywords'])}\n\n")
                    
                    found_quotes = []
                    for r in loc_records:
                        text = r['text']
                        # 只要包含任意一个关键词，且句子长度适中（>15字，排除太短的废话），就提取出来
                        # 把 "快_手" 还原为 "快手" 匹配
                        clean_keywords = [kw.replace('_', '') for kw in theme['keywords']]
                        if any(kw in text for kw in clean_keywords) and len(text) > 15:
                            found_quotes.append(f"**{r['speaker']}** (来自 `{r['source_file']}`):\n“{text}”\n")
                    
                    # 每个主题最多只挑最长/信息量最大的前 5 句
                    found_quotes.sort(key=len, reverse=True)
                    for quote in found_quotes[:5]:
                        f.write(quote + "\n")
                    
                    if not found_quotes:
                        f.write("*未找到合适长度的原话。*\n\n")

        print(f"金句提取完成！快去打开 {self.output_file.name} 挑选写进论文里吧！")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    extractor = QuoteExtractor(base_dir=PROJECT_ROOT)
    extractor.extract()