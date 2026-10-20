import json
import uuid

import gradio as gr
import pika


def submit(text):
    request_id = str(uuid.uuid4())
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="broker"))
    channel = connection.channel()
    channel.exchange_declare(exchange="results", exchange_type="direct", durable=True)
    reply = channel.queue_declare(queue="", exclusive=True)
    channel.queue_bind(exchange="results", queue=reply.method.queue, routing_key=request_id)
    message = {"request_id": request_id, "text": text}
    channel.basic_publish(exchange="", routing_key="jobs", body=json.dumps(message),
                          properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"))
    method, _, body = channel.basic_get(queue=reply.method.queue, auto_ack=True)
    event = json.loads(body)
    return event["answer"], event.get("score")


with gr.Blocks(title="Job Console") as demo:
    box = gr.Textbox(label="Question")
    answer = gr.Textbox(label="Answer")
    score = gr.Number(label="Score")
    go = gr.Button("Ask")
    go.click(fn=submit, inputs=box, outputs=[answer, score])

if __name__ == "__main__":
    demo.launch(server_port=7861)
