# Elasticsearch IK 分词器安装指南

## 前置条件

- Docker + Docker Compose 已安装
- ES 版本：8.19.17（IK 版本必须完全一致）

## 方案：挂载预下载插件（推荐）

不改镜像、不用 Dockerfile、纯 docker-compose.yml。

### 1. 下载并解压 IK 插件

```bash
# 创建插件目录
mkdir -p es-plugins

# 下载（版本必须和 ES 一致）
wget https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-8.19.17.zip

# 解压到 es-plugins/ik 目录
mkdir -p es-plugins/ik
unzip elasticsearch-analysis-ik-8.19.17.zip -d es-plugins/ik

# 可选：清理 zip 包
rm elasticsearch-analysis-ik-8.19.17.zip
```

### 2. docker-compose.yml

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.19.17
    container_name: es-local
    mem_limit: 2g
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
      - xpack.security.enabled=false
      - bootstrap.memory_lock=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - my_es_data:/usr/share/elasticsearch/data
      - ./es-plugins/ik:/usr/share/elasticsearch/plugins/analysis-ik  # IK 插件
    ports:
      - "9200:9200"
      - "9300:9300"
    networks:
      - search-net

volumes:
  my_es_data:

networks:
  search-net:
    driver: bridge
```

### 3. 启动

```bash
docker compose up -d
```

## 验证

```bash
# 检查插件是否加载
curl http://localhost:9200/_cat/plugins
# 预期输出：es-local analysis-ik 8.19.17

# 测试中文分词
curl -X POST "http://localhost:9200/_analyze" -H 'Content-Type: application/json' -d'
{
  "analyzer": "ik_max_word",
  "text": "高血压患者能吃党参吗"
}'
```

## IK 两种分词模式

| 模式 | 用途 | 示例（"高血压治疗"） |
|------|------|----------------------|
| `ik_max_word` | 索引时用，细粒度，召回率高 | 高血压 / 治疗 / 高血压治疗 |
| `ik_smart` | 搜索时用，粗粒度，精准度高 | 高血压治疗 |

## 项目结构

```
medical-RAG/
├── docker-compose.yml
├── es-plugins/
│   └── ik/                  # IK 插件文件（挂载到容器内）
│       ├── elasticsearch-analysis-ik-8.19.17.jar
│       ├── plugin-descriptor.properties
│       └── config/
│           ├── IKAnalyzer.cfg.xml
│           ├── main.dic      # 主词典（可自定义）
│           └── ...
└── data/                    # 医疗数据
```

## 常见问题

### 版本不匹配

ES 启动会报错：`Plugin [analysis-ik] was built for Elasticsearch version x.x.x`

**解决**：IK 版本必须和 ES 完全一致，去 https://release.infinilabs.com/analysis-ik/stable/ 找对应版本。

### 自定义词典

编辑 `es-plugins/ik/config/IKAnalyzer.cfg.xml` 可添加自定义词典，重启 ES 生效。

### 国内下载慢

Infinilabs 官方源在国内有 CDN，一般比 GitHub 快。如果仍慢，可提前下好 zip 备用。
