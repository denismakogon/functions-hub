import cv2
import asyncio
import uvloop
import os


def split_video_to_frames(video_path, loop=None):
    if not os.path.exists(video_path):
        raise OSError("Video file {} not found".format(video_path))

    frames, tasks = [], []
    if not loop:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        loop = asyncio.get_event_loop()

    async def split_to_frames():
        success, image = capture.read()
        if not success:
            return
        frames.append(image)

    capture = cv2.VideoCapture(video_path)
    for _ in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
        tasks.append(split_to_frames())
    loop.run_until_complete(asyncio.wait(tasks))
    return frames
