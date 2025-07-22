import socket
import sys

PORTS = [8000, 3000]


def check(port, host='localhost', timeout=2.0):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception as e:
        print(f"Port {port} not reachable: {e}")
        return False


def main():
    ok = True
    for port in PORTS:
        if check(port):
            print(f"Port {port} is open")
        else:
            ok = False
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
