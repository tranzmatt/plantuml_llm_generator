import json

import pika


def handle(ch, method, properties, body):
    job = json.loads(body)
    result = {"request_id": job["request_id"], "answer": job["text"].upper()}
    ch.basic_publish(exchange="results", routing_key=job["request_id"], body=json.dumps(result))
    ch.basic_publish(exchange="", routing_key="audit", body=json.dumps({"request_id": job["request_id"]}))
    ch.basic_ack(method.delivery_tag)


def main():
    channel = pika.BlockingConnection(pika.ConnectionParameters(host="broker")).channel()
    channel.queue_declare(queue="jobs", durable=True)
    channel.basic_consume(queue="jobs", on_message_callback=handle, auto_ack=False)
    channel.start_consuming()
