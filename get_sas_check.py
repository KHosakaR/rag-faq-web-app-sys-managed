from datetime import datetime, timedelta, timezone
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas

account = "hosakastorageeastus"  # ストレージアカウント名
account_url = f"https://{account}.blob.core.windows.net"
container = "faq"
blob = "DLハンズオン_マスタースライド_p04_講義内容_全日程.pdf"

cred = DefaultAzureCredential()
svc = BlobServiceClient(account_url=account_url, credential=cred)

now = datetime.now(timezone.utc)
st = now - timedelta(minutes=5)
se = now + timedelta(minutes=10)

udk = svc.get_user_delegation_key(key_start_time=st, key_expiry_time=se)

sas = generate_blob_sas(
    account_name=account,
    container_name=container,
    blob_name=blob,
    permission=BlobSasPermissions(read=True),
    start=st,
    expiry=se,
    user_delegation_key=udk,
    version="2023-08-03",
)
print(f"{account_url}/{container}/{blob}?{sas}")