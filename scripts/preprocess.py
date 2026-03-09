import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class StrictOneLineETL:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.output_file = self.processed_dir / "structured_interviews.jsonl"
        self.error_file = self.processed_dir / "error_log.jsonl" # 用于存放不合规的行
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def parse_strict_txt(self, file_path: Path):
        parsed_data = []
        error_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if not lines:
            return [], []

        # 1. 提取全局元数据：地点
        location = "未知地点"
        first_line = lines[0].strip()
        if first_line.startswith('#'):
            location = first_line.lstrip('#').strip()
            lines = lines[1:] # 剔除第一行，留下正文

        # 2. 严格的单行解析
        for line_num, line in enumerate(lines, start=2):
            line = line.strip()
            
            # 忽略空行
            if not line:
                continue
                
            # 兼容全角和半角冒号
            colon_index = line.find('：') if '：' in line else line.find(':')
            
            # 严格校验：必须有冒号，且冒号前的发言人名字不能长得离谱（设定阈值15）
            if colon_index != -1 and colon_index <= 15:
                speaker = line[:colon_index].strip()
                text = line[colon_index + 1:].strip()
                
                parsed_data.append({
                    "source_file": file_path.name,
                    "location": location,
                    "speaker": speaker,
                    "text": text
                })
            else:
                # 任何没有冒号，或者冒号位置不对的行，直接打入错误日志
                error_data.append({
                    "source_file": file_path.name,
                    "line_number": line_num,
                    "error_reason": "未找到有效冒号或发言人名字过长",
                    "original_text": line
                })

        return parsed_data, error_data

    def run(self):
        txt_files = list(self.raw_dir.glob("*.txt"))
        if not txt_files:
            logger.error(f"[DEBUG] 在 {self.raw_dir} 中没有找到 .txt 文件！")
            return
            
        logger.info(f"[INFO] 发现 {len(txt_files)} 个严格格式 TXT 文档，开始提取...")
        
        total_records = 0
        total_errors = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as f_out, \
             open(self.error_file, 'w', encoding='utf-8') as f_err:
             
            for file_path in txt_files:
                records, errors = self.parse_strict_txt(file_path)
                
                for r in records:
                    f_out.write(json.dumps(r, ensure_ascii=False) + '\n')
                for e in errors:
                    f_err.write(json.dumps(e, ensure_ascii=False) + '\n')
                    
                total_records += len(records)
                total_errors += len(errors)
                
                loc_name = records[0]["location"] if records else "空文件"
                logger.info(f"[INFO] [{file_path.name}] 提取 {len(records)} 条。地点: {loc_name}。异常: {len(errors)} 条。")
                
        logger.info(f"[INFO] 生成 {total_records} 条数据")
        if total_errors > 0:
            logger.warning(f"[DEBUG] 发现了 {total_errors} 条格式不合规的行，已存入 {self.error_file.name}，请检查是否漏打冒号。")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    etl = StrictOneLineETL(base_dir=PROJECT_ROOT)
    etl.run()