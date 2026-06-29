import os
import tomllib
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

SRC_PATH = Path(__file__).resolve().parent.parent
CONFIG_PATH = SRC_PATH / "config.toml"

load_dotenv()

class MysqlConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    
class RedisConfig(BaseModel):
    host: str
    port: int
    password: str
    db: int

class MilvusConfig(BaseModel):
    host: str
    port: int
    db_name: str
    collection_name: str

class ElasticsearchConfig(BaseModel):
    host: str
    port: int
    index_name: str

class LlmConfig(BaseModel):
    model: str = os.getenv("DEEPSEEK_MODEL")
    api_key: str = os.getenv("DEEPSEEK_API_KEY")
    base_url: str = os.getenv("DEEPSEEK_API_URL")

class RetrievalConfig(BaseModel):
    parent_chunk_size: int
    child_chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    candidate_m: int

class LoggerConfig(BaseModel):
    log_file: str

class AppConfig(BaseModel):
    valid_sources: list[str]
    customer_service_phone: str

class BertConfig(BaseModel):
    model_dir: str
    tokenizer_dir: str
    backend: str = "auto"
    provider: str = "cuda"
    label_map: dict[str, int] = {}

class Config(BaseModel):
    bert: BertConfig
    mysql: MysqlConfig
    redis: RedisConfig
    milvus: MilvusConfig
    elasticsearch: ElasticsearchConfig
    llm: LlmConfig
    retrieval: RetrievalConfig
    logger: LoggerConfig
    app: AppConfig

def load_config(path: Path = CONFIG_PATH) -> Config:
    config_toml = tomllib.loads(path.read_text(encoding="utf-8"))
    return Config.model_validate(config_toml)


if __name__ == '__main__':
    config = load_config()
    print(config.app.valid_sources)
    print(config.llm.base_url)