# app/services/blob_sas_service.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple, Optional
from urllib.parse import urlparse, quote

from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
)
from azure.identity import DefaultAzureCredential  # optional import

from app.config import settings


@dataclass
class SasResult:
    url: str
    expires_at: str  # ISO8601 (UTC)


class BlobSasService:
    """
    Blob の SAS URL を発行するサービス（マネージドID専用 / User Delegation SAS）
    """

    def __init__(self):
        self.account_url = settings.blob_account_url
        if not self.account_url:
            raise RuntimeError("BLOB_ACCOUNT_URL is required (e.g., https://<account>.blob.core.windows.net)")
        self.default_ttl_min = settings.blob_sas_ttl_min

        self._cred = DefaultAzureCredential()
        self._blob_svc = BlobServiceClient(account_url=self.account_url, credential=self._cred)

        # UDK（User Delegation Key）の簡易キャッシュ
        self._udk = None
        self._udk_exp: Optional[datetime] = None

    # ---------- public API ----------
    def get_sas_url(self, path_or_url: str, ttl_min: Optional[int] = None) -> SasResult:
        """
        source_path（フルURL or 'container/blob'）から Read-Only SAS を発行して返す
        """
        ttl = ttl_min or self.default_ttl_min
        account_url, container, blob, is_full_url = self._parse_container_blob(path_or_url)

        now = datetime.utcnow()
        # 時計ずれ対策として -1min のバッファを付ける
        starts = now - timedelta(minutes=1)
        expires = now + timedelta(minutes=ttl)

        # アカウント不一致に防御（将来マルチアカウント運用のため）
        parsed_account = urlparse(account_url).netloc
        current_account = urlparse(self.account_url).netloc
        if parsed_account != current_account:
            # 別アカウントのURLに対しては、その場で臨時クライアントを生成
            temp_svc = BlobServiceClient(account_url=account_url, credential=self._cred)
            udk = self._get_udk(temp_svc, starts, expires)
            account_name = parsed_account.split(".")[0]
        else:
            # 既定アカウントのUDKをキャッシュして使う
            udk = self._get_udk(self._blob_svc, starts, expires)
            account_name = urlparse(self.account_url).netloc.split(".")[0]

        try:
            sas = generate_blob_sas(
                account_name=account_name,
                container_name=container,
                blob_name=blob,
                permission=BlobSasPermissions(read=True),
                expiry=expires,
                user_delegation_key=udk,
                protocol="https",  # HTTPSを強制
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to generate SAS. Verify that the Web App's managed identity has "
                "'Storage Blob Data Delegator' role on the storage account and network access is allowed."
            ) from e

        # container/blob 形式入力の場合は blob をURLエンコード（フルURLなら二重エンコード回避）
        encoded_blob = blob if is_full_url else quote(blob)
        sas_url = f"{account_url}/{container}/{encoded_blob}?{sas}"
        return SasResult(url=sas_url, expires_at=expires.isoformat() + "Z")

    # ---------- helpers ----------
    def _get_udk(self, svc: BlobServiceClient, starts: datetime, expires: datetime):
        """User Delegation Key を簡易キャッシュ。残り1分を切ったら再取得。"""
        try:
            if self._udk and self._udk_exp and (self._udk_exp - datetime.utcnow()) > timedelta(minutes=1):
                return self._udk
            udk = svc.get_user_delegation_key(starts_on=starts, expires_on=expires)
            self._udk = udk
            self._udk_exp = expires
            return udk
        except Exception as e:
            raise RuntimeError(
                "Failed to get User Delegation Key. Ensure the managed identity has "
                "'Storage Blob Data Delegator' role and network access is allowed."
            ) from e

    def _parse_container_blob(self, path_or_url: str) -> Tuple[str, str, str, bool]:
        """
        'https://.../container/blob' または 'container/blob' を
        (account_url, container, blob, is_full_url) に分解
        """
        if path_or_url.startswith("http"):
            u = urlparse(path_or_url)
            parts = [p for p in u.path.split("/") if p]
            if len(parts) < 2:
                raise ValueError("Invalid blob url.")
            container = parts[0]
            blob = "/".join(parts[1:])
            account_url = f"{u.scheme}://{u.netloc}"
            return account_url, container, blob, True
        else:
            if not self.account_url:
                raise ValueError("BLOB_ACCOUNT_URL is required to parse 'container/blob' style path.")
            parts = path_or_url.split("/", 1)
            if len(parts) != 2:
                raise ValueError("Invalid source_path (expect 'container/blob').")
            return self.account_url, parts[0], parts[1], False


# Create singleton instance
blob_sas_service = BlobSasService()