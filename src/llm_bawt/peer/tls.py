"""TLS for the peer listener, pinned to each install's identity key (TASK-2517).

The listener's certificate is self-signed with the install's Ed25519 identity
key and built deterministically, so it never needs storing or rotating and
its key is exactly the one the invite fingerprint names. A joining install
reads the certificate, checks its key against the invite, and keeps it; the
inviting install gets the joiner's certificate in the signed pairing request.
Every later connection trusts only that one stored certificate, with hostname
checks off: the pin, not a CA or a DNS name, identifies the peer.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import ssl
import tempfile
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.x509.oid import NameOID

from . import identity

PROBE_TIMEOUT = 10.0
_NAME = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bawthub-peer")])
_NOT_BEFORE = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
_NOT_AFTER = datetime.datetime(9999, 12, 31, tzinfo=datetime.UTC)


class CertificateError(Exception):
    pass


def certificate_pem(own: identity.Identity) -> str:
    """This install's listener certificate; the same bytes on every call."""
    cert = (x509.CertificateBuilder()
            .subject_name(_NAME).issuer_name(_NAME)
            .public_key(own.private_key.public_key())
            .serial_number(int(own.key_id, 16) or 1)
            .not_valid_before(_NOT_BEFORE).not_valid_after(_NOT_AFTER)
            .sign(own.private_key, None))
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def public_key_of(cert_pem: str) -> str:
    """The identity public key a peer certificate carries, in identity's text form."""
    try:
        key = x509.load_pem_x509_certificate(cert_pem.encode()).public_key()
    except (ValueError, TypeError) as exc:
        raise CertificateError("not a certificate") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise CertificateError("peer certificate is not an install identity key")
    return identity.public_key_text(key)


def require_key(cert_pem: str, public_key: str) -> None:
    if public_key_of(cert_pem) != public_key:
        raise CertificateError("peer certificate is for another install key")


def client_context(cert_pem: str) -> ssl.SSLContext:
    """Trust exactly this certificate and nothing else."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.load_verify_locations(cadata=cert_pem)
    return ctx


def write_server_files(own: identity.Identity) -> tuple[str, str]:
    """Certificate and key files for uvicorn, in a private temporary directory."""
    folder = tempfile.mkdtemp(prefix="llm-bawt-peer-tls-")
    cert_path, key_path = os.path.join(folder, "cert.pem"), os.path.join(folder, "key.pem")
    key = own.private_key.private_bytes(serialization.Encoding.PEM,
                                        serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption())
    for path, data in ((cert_path, certificate_pem(own).encode()), (key_path, key)):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    return cert_path, key_path


async def fetch_certificate(base_url: str) -> str:
    """Read a peer's certificate without trusting it; the caller pins it."""
    parts = urlsplit(base_url)
    port = parts.port or 443
    return await asyncio.to_thread(ssl.get_server_certificate, (parts.hostname, port),
                                   timeout=PROBE_TIMEOUT)
