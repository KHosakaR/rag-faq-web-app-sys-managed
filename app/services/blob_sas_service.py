# app/services/blob_sas_service.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional
from urllib.parse import urlparse, unquote

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

    # ---------- public API ----------
    def get_sas_url(self, path_or_url: str, ttl_min: Optional[int] = None) -> SasResult:
        """
        source_path（フルURL or 'container/blob'）から Read-Only SAS を発行して返す
        """
        if ttl_min is None:
            ttl_min = self.default_ttl_min
        account_url, container, blob, is_full_url = self._parse_container_blob(path_or_url)
        
        # すべて timezone-aware (UTC) 
        now = datetime.now(timezone.utc)
        # 時計ずれ吸収：開始は 5分 前にしておく
        starts = now - timedelta(minutes=5)
        expires = now + timedelta(minutes=ttl_min)

        account_name = urlparse(account_url).netloc.split(".")[0]
        
        # --- User Delegation Key を取得（SDK の互換対応）---
        udk = self._get_udk(self._blob_svc, now, expires)
        
        sas = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob,
            permission=BlobSasPermissions(read=True),
            start=starts,
            expiry=expires,
            user_delegation_key=udk,
            version="2023-08-03",
        )

        sas_url = f"{account_url}/{container}/{blob}?{sas}"
        return SasResult(url=sas_url, expires_at=expires.isoformat())

    # ---------- helpers ----------
    def _get_udk(self, svc: BlobServiceClient, starts: datetime, expires: datetime):
        """User Delegation Key を簡易キャッシュ。残り1分を切ったら再取得。"""
        # SDK バージョン差を吸収
        try:
             # 新しめの SDK（キーワード引数OK）
            return svc.get_user_delegation_key(key_start_time=starts, key_expiry_time=expires)
        except TypeError:
            # 古めの SDK（位置引数のみ）
            return svc.get_user_delegation_key(starts, expires)  
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
            blob = unquote("/".join(parts[1:]))
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