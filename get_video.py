import cv2
import os

def main():    
    root_dir = f"/home/public/b509/code/dataset"
    video_name = "2021-1-16_32_video"
    input_dir = f"{root_dir}/csi_out/20210116/{video_name}"
    output_dir = f'{root_dir}/csi_out/out_video/{video_name}.avi'
    fps = 20          # 视频帧率
    size = (1280, 720) # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    for root, dirs, files in os.walk(input_dir):       
        for i in range(len(files)):
            file_path = os.path.join(input_dir, "csi_"+str(i)+".png")
            img = cv2.imread(file_path)
            print(file_path)
            video.write(img)
    
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()