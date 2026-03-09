import re
import json
import logging
from pathlib import Path
from docx import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class PerfectInterviewETL:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.output_file = self.processed_dir / "structured_interviews.jsonl"
        self.processed_dir.mkdir(parents=True, exist_ok=True)


        self.pat_time = re.compile(r'^(.+?)\s+(\d{2}:\d{2}(?::\d{2})?)\s*(.*)$')

        self.pat_colon = re.compile(r'^([^:：]{1,15})[:：]\s*(.*)$')

        known_roles = [
            r'\d+号讲话人(?:\s+[^\s]+)?', 
            r'团队', r'农户', r'书记', r'负责人', r'技术人员', r'园长', r'主播\d*',
            r'[赵王吴徐邢杨]\s*[赵王吴徐邢杨]?' 
        ]
        roles_pattern = '|'.join(known_roles)
        self.pat_space = re.compile(rf'^({roles_pattern})\s+(.+)$')

    def read_docx(self, file_path: Path):
        try:
            doc = Document(file_path)
            return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        except Exception as e:
            logger.error(f"[DEBUG] 读取 {file_path.name} 失败: {e}")
            return []

    def infer_location(self, filename: str) -> str:
        if any(kw in filename for kw in ["保南村", "农户", "书记", "哈校长"]): return "宁夏保南村"
        if any(kw in filename for kw in ["河北", "行唐", "双创园", "园长"]): return "河北省龙洞村"
        if any(kw in filename for kw in ["鲲之益", "技术人员", "负责人"]): return "企业端"
        return "贵州省西安村(或未知)"

    def parse_text(self, paragraphs: list, filename: str, location: str):
        parsed_data = []
        current_speaker = "文档背景/研究员"
        current_text = []

        if "梳理" in filename or ("整理" in filename and "访谈整理" not in filename):
            for p in paragraphs:
                if p: parsed_data.append({"source_file": filename, "location": location, "speaker": "总结归纳", "text": p})
            return parsed_data

        for line in paragraphs:
            # 过滤纯分割线
            if set(line) == {'-'} or "---" in line:
                continue

            # 匹配时间戳
            match_time = self.pat_time.match(line)
            if match_time:
                self._save_record(parsed_data, filename, location, current_speaker, current_text)
                current_speaker = match_time.group(1).strip()
                current_text = [match_time.group(3).strip()]
                continue

            match_colon = self.pat_colon.match(line)
            if match_colon and len(match_colon.group(1)) <= 15 and " " not in match_colon.group(1).strip():
                self._save_record(parsed_data, filename, location, current_speaker, current_text)
                current_speaker = match_colon.group(1).strip()
                current_text = [match_colon.group(2).strip()]
                continue

            match_space = self.pat_space.match(line)
            if match_space:
                self._save_record(parsed_data, filename, location, current_speaker, current_text)
                current_speaker = match_space.group(1).strip()
                current_text = [match_space.group(2).strip()]
                continue
            current_text.append(line)

        self._save_record(parsed_data, filename, location, current_speaker, current_text)
        return parsed_data

    def _save_record(self, container, filename, location, speaker, text_list):
        text = " ".join(text_list).strip()
        if text:
            container.append({
                "source_file": filename,
                "location": location,
                "speaker": speaker,
                "text": text
            })
            text_list.clear()

    def run(self):
        docx_files = list(self.raw_dir.glob("*.docx"))
        logger.info(f"[INFO] 开始解析，共发现 {len(docx_files)} 个访谈文档...")
        
        total_records = 0
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for file_path in docx_files:
                loc = self.infer_location(file_path.name)
                paragraphs = self.read_docx(file_path)
                records = self.parse_text(paragraphs, file_path.name, loc)
                
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    total_records += 1
                logger.info(f"[INFO] [{file_path.name}] 解析完毕，提取 {len(records)} 条对话。")
                
        logger.info(f"[INFO] 提取了 {total_records} 条结构化数据！")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    etl = PerfectInterviewETL(base_dir=PROJECT_ROOT)
    etl.run()