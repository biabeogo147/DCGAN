import cv2
import dlib
import numpy as np


def get_landmarks(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('D:/DS-AI/data/shape_predictor_68_face_landmarks.dat')

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        print(f"Không phát hiện được khuôn mặt trong ảnh {image_path}")
        return None, None

    face = faces[0]

    landmarks = predictor(gray, face)

    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    landmarks_array = np.array(landmarks_points, dtype=np.float32)

    return img, landmarks_array


def compute_landmark_loss(landmarks_real, landmarks_gen):
    loss = np.mean(np.sum((landmarks_real - landmarks_gen) ** 2, axis=1))
    return loss


def draw_landmarks(img, landmarks, window_name):
    for (x, y) in landmarks:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imshow(window_name, img)


if __name__ == '__main__':
    # image_path_real = 'real_image.jpg'
    # img_real, landmarks_real = get_landmarks(image_path_real)
    # draw_landmarks(img_real, landmarks_real, 'Real Image Landmarks')

    image_path_gen = 'image.png'
    img_gen, landmarks_gen = get_landmarks(image_path_gen)
    draw_landmarks(img_gen, landmarks_gen, 'Generated Image Landmarks')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
