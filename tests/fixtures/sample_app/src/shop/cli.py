import argparse

from .repo import build_repo, ping


def cmd_serve(args) -> int:
    build_repo()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="shopctl", description="Shop control")
    parser.add_argument("-v", "--verbose", action="store_true", help="Chatty output")
    sub = parser.add_subparsers(dest="cmd")
    p_serve = sub.add_parser("serve", help="Run the server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--mode", choices=["dev", "prod"], required=True)
    p_serve.set_defaults(func=cmd_serve)
    p_ping = sub.add_parser("ping")
    p_ping.add_argument("hosts", nargs="+")
    args = parser.parse_args()
    ping()
    return args.func(args)


if __name__ == "__main__":
    main()
