"""Run the ANPR demo over local HTTPS for phone camera access.

Phone browsers usually block getUserMedia on plain http://<laptop-ip>.
This script creates a local self-signed certificate with your machine IPs in
the SAN list, then starts the existing FastAPI app with Uvicorn over HTTPS.
"""
import datetime
import ipaddress
import socket
import sys
from pathlib import Path

try:
    import uvicorn
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError as exc:
    raise SystemExit(
        "Missing HTTPS demo dependency. Run: pip install uvicorn cryptography"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
CERT_DIR = BACKEND_DIR / "certs"
CERT_FILE = CERT_DIR / "anpr-local.crt"
KEY_FILE = CERT_DIR / "anpr-local.key"


def local_ip_addresses():
    addresses = {"127.0.0.1"}
    hostname = socket.gethostname()
    for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
        if family == socket.AF_INET:
            addresses.add(sockaddr[0])
    return sorted(addresses)


def ensure_certificate():
    if CERT_FILE.exists() and KEY_FILE.exists():
        return

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names = [
        x509.DNSName("localhost"),
        x509.DNSName(socket.gethostname()),
    ]
    names.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in local_ip_addresses())
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "ANPR Local Demo"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def main():
    ensure_certificate()
    sys.path.insert(0, str(BACKEND_DIR))
    ips = ", ".join(f"https://{ip}:8443/scan.html" for ip in local_ip_addresses())
    print("Open scanner on phone:")
    print(ips)
    print("Open dashboard:")
    print("https://localhost:8443/dashboard.html")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8443,
        ssl_certfile=str(CERT_FILE),
        ssl_keyfile=str(KEY_FILE),
    )


if __name__ == "__main__":
    main()
