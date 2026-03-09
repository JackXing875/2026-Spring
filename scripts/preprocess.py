import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from docx import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewETL:
    def __init__(self, base_dir: str):
        """
        初始化项目路径
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.output_file = self.processed_dir / "structured_interviews.jsonl"
        
        # 确保输出目录存在
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 预编译正则表达式以提升大规模文本的解析性能
        # 模式1: 姓名 时间戳 内容 (例如: 赵  00:32:08 谁会在直播间里看你...)
        self.pat_timestamp = re.compile(r'^([^\d\s\:\：]+)\s*(?:[\d号讲话人]*\s*)?\d{2}:\d{2}:\d{2}\s*(.*)$')
        # 模式2: 角色:内容 或 角色：内容 (例如: 团队 那我们先问一下... 或 园长：不，因为...)
        self.pat_qa = re.compile(r'^([^\:：\s]{1,5})[\:：\s]+(.*)$')

    def read_docx(self, file_path: Path) -> List[str]:
        """
        读取 docx 文件，返回清洗过空行的段落列表
        """
        try:
            doc = Document(file_path)
            # 剔除纯空格或空段落
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return paragraphs
        except Exception as e:
            logger.error(f"读取文件失败 {file_path.name}: {e}")
            return []

    def infer_metadata(self, filename: str) -> Dict[str, str]:
        """
        根据文件名推断元数据（地点、大致角色分类）
        """
        metadata = {"location": "未知", "doc_type": "访谈记录"}
        
        if "保南村" in filename or "农户" in filename or "书记" in filename:
            metadata["location"] = "宁夏保南村"
        elif "河北" in filename or "行唐" in filename or "双创园" in filename or "园长" in filename:
            metadata["location"] = "河北省龙洞村/行唐县"
        elif "鲲之益" in filename or "哈校长" in filename or "技术人员" in filename or "负责人" in filename:
            metadata["location"] = "企业端(上海/银川)"
            
        if "梳理" in filename or "整理" in filename and "访谈" not in filename:
            metadata["doc_type"] = "结构化总结"
            
        return metadata

    def parse_paragraphs(self, paragraphs: List[str], metadata: Dict[str, str], filename: str) -> List[Dict[str, Any]]:
        """
        将段落解析为结构化的对话字典
        """
        parsed_data = []
        current_speaker = "文档摘要/背景"
        current_text = []

        # 针对已经是总结性质的文档，直接按段落存储
        if metadata["doc_type"] == "结构化总结":
            for p in paragraphs:
                parsed_data.append({
                    "source_file": filename,
                    "location": metadata["location"],
                    "speaker": "研究员",
                    "text": p
                })
            return parsed_data

        # 针对对话性质的文档进行正则切分
        for line in paragraphs:
            # 过滤掉无意义的分割线
            if set(line) == {'-'} or "---" in line:
                continue
                
            match_ts = self.pat_timestamp.match(line)
            match_qa = self.pat_qa.match(line)

            if match_ts:
                if current_text:
                    parsed_data.append(self._build_record(filename, metadata, current_speaker, current_text))
                current_speaker = match_ts.group(1).strip()
                current_text = [match_ts.group(2).strip()]
                
            elif match_qa:
                if current_text:
                    parsed_data.append(self._build_record(filename, metadata, current_speaker, current_text))
                current_speaker = match_qa.group(1).strip()
                current_text = [match_qa.group(2).strip()]
                
            else:
                # 若未匹配到说话人，视为上一段对话的延续
                current_text.append(line)

        # 保存最后一段
        if current_text:
            parsed_data.append(self._build_record(filename, metadata, current_speaker, current_text))

        return parsed_data

    def _build_record(self, filename: str, metadata: Dict[str, str], speaker: str, text_list: List[str]) -> Dict[str, Any]:
        """
        构建单条 JSONL 记录
        """
        return {
            "source_file": filename,
            "location": metadata["location"],
            "speaker": speaker,
            "text": " ".join(text_list)
        }

    def run(self):
        """
        执行 ETL 管道
        """
        docx_files = list(self.raw_dir.glob("*.docx"))
        if not docx_files:
            logger.warning(f"在 {self.raw_dir} 中未找到任何 .docx 文件，请检查路径！")
            return

        logger.info(f"找到 {len(docx_files)} 个 Docx 文件，开始处理...")
        
        total_records = 0
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for file_path in docx_files:
                logger.info(f"正在解析: {file_path.name}")
                
                # 1. Extract
                paragraphs = self.read_docx(file_path)
                
                # 2. Transform
                metadata = self.infer_metadata(file_path.name)
                parsed_records = self.parse_paragraphs(paragraphs, metadata, file_path.name)
                
                # 3. Load
                for record in parsed_records:
                    if record["text"].strip(): # 剔除空文本
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        total_records += 1
                        
        logger.info(f"🎉 ETL 处理完成！共提取 {total_records} 条结构化数据。")
        logger.info(f"📁 输出文件路径: {self.output_file.absolute()}")


if __name__ == "__main__":
    # 假设你的终端当前在项目的根目录 (2026-Spring) 运行此脚本
    # 获取项目根目录的绝对路径
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    etl = InterviewETL(base_dir=PROJECT_ROOT)
    etl.run()