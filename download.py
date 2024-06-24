import cv2
import os
from pytube import YouTube
from skimage.metrics import structural_similarity as ssim
import numpy as np


def download_video(youtube_url, output_path='video.mp4'):
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(file_extension='mp4', res="1080p").first()
        if stream is None:
            stream = yt.streams.filter(file_extension='mp4').first()
        stream.download(filename=output_path)
        print(f"Video downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


def extract_frames(video_path, output_folder='frames', interval=5, similarity_threshold=0.6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Extracting frames to {output_folder}")

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    success, prev_image = cap.read()
    saved_frame_count = 0

    while success:
        success, image = cap.read()
        if not success:
            break

        if count % (frame_rate * interval) == 0:
            if prev_image is not None:
                # Convert images to grayscale
                prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Calculate SSIM between previous and current frame
                score, _ = ssim(prev_gray, curr_gray, full=True)
                if score < similarity_threshold:
                    frame_file = os.path.join(output_folder, f'frame_{saved_frame_count}.jpg')
                    cv2.imwrite(frame_file, image)
                    print(f"Saved frame to {frame_file}")
                    saved_frame_count += 1

            prev_image = image

        count += 1

    cap.release()


def main():
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "YouTubeDownloads")
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)

    video_output_path = os.path.join(desktop_path, 'video.mp4')
    frames_output_folder = os.path.join(desktop_path, 'frames')

    youtube_url = input("유튜브 링크를 입력하세요: ")
    video_path = download_video(youtube_url, output_path=video_output_path)
    if video_path:
        extract_frames(video_path, output_folder=frames_output_folder, interval=5,
                       similarity_threshold=0.6)  # interval을 5초로 설정
    else:
        print("Failed to download video.")


if __name__ == '__main__':
    main()
