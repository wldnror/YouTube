import cv2
import os
from yt_dlp import YoutubeDL
from skimage.metrics import structural_similarity as ssim
import numpy as np

def download_video(youtube_url, output_path='video.mp4', resolution="1080p"):
    ydl_opts = {
        'format': f'bestvideo[height<={resolution[:-1]}]+bestaudio/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
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
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    saved_frame_count = 0
    prev_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % int(frame_rate * interval) == 0:
            if prev_image is not None:
                prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score, _ = ssim(prev_gray, curr_gray, full=True)
                if score < similarity_threshold:
                    frame_file = os.path.join(output_folder, f'frame_{saved_frame_count}.jpg')
                    cv2.imwrite(frame_file, frame)
                    print(f"Saved frame to {frame_file} (SSIM: {score:.2f})")
                    saved_frame_count += 1
            prev_image = frame

        count += 1

    cap.release()
    print(f"Total {saved_frame_count} frames saved.")

def main():
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "YouTubeDownloads")
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)

    video_output_path = os.path.join(desktop_path, 'video.mp4')
    frames_output_folder = os.path.join(desktop_path, 'frames')

    youtube_url = input("유튜브 링크를 입력하세요: ")
    resolution = input("다운로드할 해상도를 입력하세요 (예: 1080p, 기본값: 1080p): ") or "1080p"
    video_path = download_video(youtube_url, output_path=video_output_path, resolution=resolution)
    if video_path:
        try:
            interval = float(input("프레임 추출 간격(초)을 입력하세요 (기본값: 5): ") or 5)
            similarity_threshold = float(input("유사도 임계값을 입력하세요 (0~1, 기본값: 0.6): ") or 0.6)
            extract_frames(video_path, output_folder=frames_output_folder, interval=interval,
                           similarity_threshold=similarity_threshold)
        except ValueError:
            print("잘못된 입력 값입니다. 기본값을 사용합니다.")
            extract_frames(video_path, output_folder=frames_output_folder, interval=5,
                           similarity_threshold=0.6)
    else:
        print("Failed to download video.")

if __name__ == '__main__':
    main()
