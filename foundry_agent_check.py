from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

load_dotenv()

# 環境変数から読み込み
endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")  # https://<account>.services.ai.azure.com
agent_id = "asst_kZkROkiG479HS8xp6SMimMQ5"  # 上で取得したID
# api_key = os.getenv("AZURE_AI_FOUNDRY_API_KEY")

# AIProjectClientを作成
client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())

# プロジェクトの一覧を取得（疎通確認用）
projects = client.agents.list_agents()
print("=== Projects ===")
for project in projects:
    print(project.name, project.id)

# エージェント呼び出し
agent = client.agents.get_agent(agent_id=agent_id)
print(f"agent ID: {agent.id}")
print(f"agent_name: {agent.name}")

thread = client.agents.threads.create()
msg = client.agents.messages.create(thread_id=thread.id, role="user", content="株式会社ヘッドウォータースの所在地を教えてください。")
run = client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
# result = agent.run(input="テストです。AI Foundryにつながっていますか？")

# 実行完了待機と取得
while run.status in ["queued", "in_progress", "requires_action"]:
    import time; time.sleep(1)
    run = client.agents.runs.get(thread_id=thread.id, run_id=run.id)

print("=== Response ===")
# 結果確認
for message in client.agents.messages.list(thread_id=thread.id):
    print(message.role, message.content)