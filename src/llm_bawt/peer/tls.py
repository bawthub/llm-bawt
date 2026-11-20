"""TLS for the peer listener, pinned to each install's identity key (TASK-2517).

The listener's certificate is self-signed with the install's Ed25519 identity
key and built deterministically, so it never needs storing or rotating and
its key is exactly the one the invite fingerprint names. A joining install
reads the certificate, checks its key against the invite, and keeps it; the
inviting install gets the joiner's certificate in the signed pairing request.
Every later connection trusts only that one stored certificate, with hostname
checks off: the pin, not a CA or a DNS name, identifies the peer.

Key custody: the identity key stays encrypted at rest in the database. TLS
needs it in plaintext only for the moment uvicorn loads its context (see
``load_server_tls``); after that it lives only in process memory, as it
already does for signing assertions. Both ends require TLS 1.3.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import shutil
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


def canonical(cert_pem: str) -> str:
    """Exactly one PEM certificate, re-encoded; anything else is refused.

    A stored pin is loaded as the whole trust store, so a second certificate
    smuggled after the checked one must never reach it.
    """
    text_value = str(cert_pem or "")
    if text_value.count("-----BEGIN") != 1:
        raise CertificateError("expected exactly one certificate")
    try:
        cert = x509.load_pem_x509_certificate(text_value.encode())
    except (ValueError, TypeError) as exc:
        raise CertificateError("not a certificate") from exc
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def public_key_of(cert_pem: str) -> str:
    """The identity public key a peer certificate carries, in identity's text form."""
    key = x509.load_pem_x509_certificate(canonical(cert_pem).encode()).public_key()
    if not isinstance(key, Ed25519PublicKey):
        raise CertificateError("peer certificate is not an install identity key")
    return identity.public_key_text(key)


def require_key(cert_pem: str, public_key: str) -> str:
    """The canonical certificate, if it carries exactly ``public_key``."""
    pem = canonical(cert_pem)
    if public_key_of(pem) != public_key:
        raise CertificateError("peer certificate is for another install key")
    return pem


def client_context(cert_pem: str) -> ssl.SSLContext:
    """Trust exactly this certificate and nothing else, over TLS 1.3."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.check_hostname = False
    ctx.load_verify_locations(cadata=canonical(cert_pem))
    return ctx


def load_server_tls(config, own: identity.Identity) -> None:
    """Load a uvicorn config's TLS context, then delete the key file at once.

    uvicorn reads the key only from a file. It is written 0600 to a private
    directory on tmpfs where available and removed as soon as the context is
    loaded, on failure too, so the identity key does not stay on disk.
    """
    shm = "/dev/shm"
    folder = tempfile.mkdtemp(prefix="llm-bawt-peer-tls-",
                              dir=shm if os.path.isdir(shm) else None)
    try:
        cert_path, key_path = os.path.join(folder, "cert.pem"), os.path.join(folder, "key.pem")
        key = own.private_key.private_bytes(serialization.Encoding.PEM,
                                            serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption())
        for path, data in ((cert_path, certificate_pem(own).encode()), (key_path, key)):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        config.ssl_certfile, config.ssl_keyfile = cert_path, key_path
        config.load()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    config.ssl.minimum_version = ssl.TLSVersion.TLSv1_3


async def fetch_certificate(base_url: str) -> str:
    """Read a peer's certificate without trusting it; the caller pins it."""
    parts = urlsplit(base_url)
    port = parts.port or 443
    return await asyncio.to_thread(ssl.get_server_certificate, (parts.hostname, port),
                                   timeout=PROBE_TIMEOUT)
