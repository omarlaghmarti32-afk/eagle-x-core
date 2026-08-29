"""AES-256-GCM + Ed25519 integrity."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import KEY_PATH


class CryptoEngine:
    def __init__(self, key_path: Path | None = None) -> None:
        self.key_path = Path(key_path or KEY_PATH)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self._aes_key, self._priv, self._pub = self._load_or_create()

    def _load_or_create(self):
        if self.key_path.exists():
            raw = self.key_path.read_bytes()
            aes_key = raw[:32]
            priv = Ed25519PrivateKey.from_private_bytes(raw[32:64])
        else:
            aes_key = os.urandom(32)
            priv = Ed25519PrivateKey.generate()
            blob = aes_key + priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            self.key_path.write_bytes(blob)
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
        pub = priv.public_key()
        return aes_key, priv, pub

    def encrypt(self, plaintext: str | bytes) -> str:
        data = plaintext.encode() if isinstance(plaintext, str) else plaintext
        nonce = os.urandom(12)
        ct = AESGCM(self._aes_key).encrypt(nonce, data, None)
        return base64.urlsafe_b64encode(nonce + ct).decode()

    def decrypt(self, token: str) -> str:
        raw = base64.urlsafe_b64decode(token.encode())
        nonce, ct = raw[:12], raw[12:]
        pt = AESGCM(self._aes_key).decrypt(nonce, ct, None)
        return pt.decode()

    def sign(self, message: str | bytes) -> str:
        data = message.encode() if isinstance(message, str) else message
        sig = self._priv.sign(data)
        return base64.urlsafe_b64encode(sig).decode()

    def verify(self, message: str | bytes, signature: str) -> bool:
        data = message.encode() if isinstance(message, str) else message
        try:
            self._pub.verify(base64.urlsafe_b64decode(signature.encode()), data)
            return True
        except Exception:
            return False

    def seal(self, payload: dict[str, Any]) -> dict[str, str]:
        body = json.dumps(payload, sort_keys=True, default=str)
        return {
            "ciphertext": self.encrypt(body),
            "signature": self.sign(body),
            "alg": "AES-256-GCM+Ed25519",
        }

    def public_key_b64(self) -> str:
        raw = self._pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.urlsafe_b64encode(raw).decode()
