# Anthropic バッチリクエストユーティリティ

[日本語](README.ja.md) | [English](README.md)

## 概要
分散処理とモニタリングのためにCeleryワーカーを使用して、Anthropic APIへのバッチリクエストを行うユーティリティです。自動リトライ、進捗モニタリング、結果管理などの機能を備え、大規模なリクエストを効率的に処理することができます。

## 必要条件
- Python 3.12以上
- Docker及びDocker Compose（推奨）
- Redisサーバー（オプショナル。デフォルトではDocker内のRedisを使用）
- Anthropic APIキー
- Dev Containers拡張機能がインストールされたVS Code

## インストール

### Dev Containerを使用する方法（推奨）
1. リポジトリをクローン
2. Dev Containers拡張機能がインストールされたVS Codeで開く
3. プロンプトが表示されたら「コンテナーで再度開く」をクリック
4. コンテナのビルドと初期化が完了するまで待機
5. テンプレートから`.env`ファイルを作成し、APIキーを設定：
   ```bash
   cp .env.back .env
   # .envを編集してAnthropic APIキーを設定：
   # ANTHROPIC_API_KEY=your_api_key_here
   ```

### 手動インストール
1. Python 3.12以上をインストール
2. Redisサーバーをインストール
3. リポジトリをクローン
4. 依存関係をインストール：
   ```bash
   pip install -r requirements.txt
   ```
5. 上記の説明に従って`.env`ファイルを作成・設定

## 設定
`src/python/anthropic_batch_request_util/config/settings.yaml`を編集してツールを設定：

### API設定
```yaml
api:
  version: "2023-06-01"                # Anthropic APIバージョン
  beta_features:                       # 有効化するベータ機能
    - "message-batches-2024-09-24"
    - "prompt-caching-2024-07-31"
  timeout: 300                         # リクエストタイムアウト（秒）
  max_payload_size: 104857600          # 最大ペイロードサイズ（100MB）
```

### モデル設定
```yaml
model:
  name: "claude-3-5-haiku-20241022"    # モデル識別子
  display_name: "haiku"                # ログ表示用の名前
  max_tokens: 8192                     # 最大出力トークン数
  temperature: 0.0                     # 生成時の温度（0-1）
  response_format: "text"              # レスポンス形式（text/json）
```

### バッチ処理設定
```yaml
batch:
  max_size: 10000                      # バッチあたりの最大リクエスト数
  chunk_size: 1000                     # 大規模バッチの分割サイズ
  enable_prompt_cache: false           # プロンプトキャッシュの有効化
  cache_type: "ephemeral"              # キャッシュタイプ（現在はephemeralのみ）
```

### ストレージ設定
```yaml
storage:
  base_dir: "output"                   # 出力のベースディレクトリ
  subdirs:
    requests: "batch_records"          # リクエスト記録の保存場所
    results: "results"                 # 結果の保存場所
    logs: "logs"                       # ログファイルの保存場所
```

## 使用方法

### ワーカー管理
以下のmakeコマンドでCeleryワーカーを管理：

```bash
# バックグラウンドモードでワーカーを起動
make start-worker

# デバッグモードでコンソール出力付きでワーカーを起動
make start-worker-debug

# 実行中のワーカーを停止
make stop-worker
```

### バッチ処理

1. バッチリクエストJSONファイルを作成（examples/batch_request.jsonを参照）：
```json
{
  "system_prompt": "You are a helpful AI assistant. Please provide clear and concise answers.",
  "messages_list": [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is the capital of France?"}]
  ],
  "custom_id_prefix": "example_batch"
}
```

2. バッチリクエストを実行：
```bash
make run-batch json_file=path/to/request.json
```

### テスト
サンプルリクエストでテストスイートを実行：
```bash
make test
```

### その他のコマンド
```bash
# 一時ファイルとログを削除
make clean

# 利用可能なすべてのコマンドを表示
make help
```

結果とログは以下の場所に保存されます：
- バッチ記録：`output/batch_records/`
- 結果：`output/results/`
- ワーカーログ：`logs/celery.log` 