"""
StorageService — abstract file storage with S3 and local filesystem backends.

Uses S3 (via boto3) in production, falls back to local filesystem for development.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

LOCAL_STORAGE_ROOT = Path(os.getenv("LOCAL_STORAGE_ROOT", "/tmp/jurisai_storage"))


class StorageService:
    """Upload, download, and delete files from S3 or local filesystem."""

    def __init__(self) -> None:
        self._s3_client = None
        self._use_s3 = bool(settings.S3_ENDPOINT and settings.S3_ACCESS_KEY)

    def _get_s3(self):
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT or None,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name="us-east-1",
            )
        return self._s3_client

    def _generate_key(
        self, tenant_id: str, filename: str, prefix: str = "documents"
    ) -> str:
        ext = Path(filename).suffix
        unique = uuid.uuid4().hex[:12]
        return f"{prefix}/{tenant_id}/{unique}{ext}"

    async def upload(
        self,
        data: bytes,
        filename: str,
        tenant_id: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload file and return the storage key."""
        key = self._generate_key(tenant_id, filename)

        if self._use_s3:
            try:
                s3 = self._get_s3()
                s3.put_object(
                    Bucket=settings.S3_BUCKET,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )
                logger.info("Uploaded %s to S3 bucket %s", key, settings.S3_BUCKET)
                return key
            except Exception as e:
                logger.error("S3 upload failed, falling back to local: %s", e)

        local_path = LOCAL_STORAGE_ROOT / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.info("Uploaded %s to local storage", key)
        return key

    async def download(self, key: str) -> Optional[bytes]:
        """Download file by storage key."""
        if self._use_s3:
            try:
                s3 = self._get_s3()
                resp = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
                return resp["Body"].read()
            except Exception as e:
                logger.error("S3 download failed: %s", e)

        local_path = LOCAL_STORAGE_ROOT / key
        if local_path.exists():
            return local_path.read_bytes()

        return None

    async def delete(self, key: str) -> bool:
        """Delete file from storage."""
        if self._use_s3:
            try:
                s3 = self._get_s3()
                s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)
                return True
            except Exception as e:
                logger.error("S3 delete failed: %s", e)

        local_path = LOCAL_STORAGE_ROOT / key
        if local_path.exists():
            local_path.unlink()
            return True

        return False

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL (S3) or a local path reference."""
        if self._use_s3:
            try:
                s3 = self._get_s3()
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.S3_BUCKET, "Key": key},
                    ExpiresIn=expires_in,
                )
                return url
            except Exception as e:
                logger.error("Presigned URL generation failed: %s", e)

        return f"/api/v1/documents/file/{key}"
