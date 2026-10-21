import os

import pika


def main():
    channel = pika.BlockingConnection(pika.ConnectionParameters(host=os.environ["BROKER_HOST"])).channel()
    channel.basic_publish(exchange="", routing_key="jobs", body=b"")


if __name__ == "__main__":
    main()
