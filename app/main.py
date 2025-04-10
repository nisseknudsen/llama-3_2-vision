import base64
from typing import Literal

from make87_messages.core.header_pb2 import Header
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.text.text_plain_pb2 import PlainText
import make87
from ollama import chat, Client
from pydantic import BaseModel


# Define the schema for image objects
class Object(BaseModel):
    name: str
    confidence: float
    attributes: str


class ImageDescription(BaseModel):
    summary: str
    objects: list[Object]
    scene: str
    colors: list[str]
    time_of_day: Literal["Morning", "Afternoon", "Evening", "Night"]
    setting: Literal["Indoor", "Outdoor", "Unknown"]
    text_content: str | None = None


def main():
    make87.initialize()
    endpoint = make87.get_provider(
        name="IMAGE_TO_TEXT", requester_message_type=ImageJPEG, provider_message_type=PlainText
    )

    print("Loading Model...")
    client = Client()
    response = client.create(
        model="m87",
        from_="llama3.2-vision",
        system="You are describing random images with their contents to explain them to a human audience.",
        stream=False,
    )
    print(response.status)

    print("Waiting for model to load...")
    chat(model="m87", messages=[])

    print("Model loaded.")

    def callback(message: ImageJPEG) -> PlainText:
        image64 = base64.b64encode(message.data).decode("utf-8")
        response = client.chat(
            model="m87",
            format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
            messages=[
                {
                    "role": "user",
                    "content": "Analyze this image and return a detailed JSON description including objects, scene, colors and any text detected. If you cannot determine certain details, leave those fields empty.",
                    "images": [image64],
                },
            ],
            options={"temperature": 0},  # Set temperature to 0 for more deterministic output
        )

        image_analysis = ImageDescription.model_validate_json(response.message.content)
        print(f"Image analysis: {image_analysis}")

        return PlainText(
            header=make87.header_from_message(Header, message=message, append_entity_path="response"),
            body=image_analysis.model_dump_json(),
        )

    endpoint.provide(callback)
    make87.loop()


if __name__ == "__main__":
    main()
