import cv2
import numpy as np
import scipy
import face_recognition

# GLOBAL VARIABLES
MAX_FEATURES = 8000  # Upperbound limit to the number of features to find on single picture
GOOD_MATCH_PERCENT = 0.08
PADDING_PIXELS = 30
MAX_PICTURE_HEIGHT = 1080
MAX_KEYPOINT_DISTANCE = 80

# 8000 / 0.08 / 30 / 1080 / 80


# Resize Img WithAspectRatio
def resize_with_aspect_ratio_and_add_border(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized_img = cv2.resize(image, dim, interpolation=inter)

    # Create an empty black_img and centering the resized

    w = dim[0] + PADDING_PIXELS * 2
    h = dim[1] + PADDING_PIXELS * 2

    blank_image = np.zeros((h, w, 3), np.uint8)

    for y in range(h):
        for x in range(w):

            x2 = x - PADDING_PIXELS
            y2 = y - PADDING_PIXELS

            if x2 < 0 or x2 >= dim[0]:
                x2 = None

            if y2 < 0 or y2 >= dim[1]:
                y2 = None

            if x2 is None or y2 is None:
                rgb_pixel = [0, 0, 0]
            else:
                rgb_pixel = resized_img[y2 - 1][x2 - 1]

            blank_image[y, x] = rgb_pixel

    # cv2.imshow("BlackCanvas", blank_image)
    # key = cv2.waitKey(0)

    return blank_image


def rgb_to_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_wrapped_image(filename, template):
    # MAIN ACTIVITIES
    # Load the template picture
    template = cv2.imread(template)
    url = "data/" + filename + ".jpeg"
    print(url)
    original_test = cv2.imread(url)  # Load test picture
    test = original_test = resize_with_aspect_ratio_and_add_border(
        original_test, height=None)  # Resize test picture
    # Convert picture to gray scale color palette
    test = rgb_to_gray_scale(test)

    # FEATURES
    # Scelgo il metodo di ricerca dei KeyPoints
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(template, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test, None)

    # FEATURES MATCHING
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)  # Chose the matcher
    matches = matcher.match(descriptors1, descriptors2, None)  # Make the match

    matches = list(matches)

    # FIRST ALGORYTHM
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Calculate how many matches to keep
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]  # Remove not so good matches

    newMatches = list()
    # matches = filter(checkDistance, matches)
    for match in matches:
        if (match.distance > MAX_KEYPOINT_DISTANCE):
            break
        # print("Distance:", match.distance)
        newMatches.append(match)

    imMatches = cv2.drawMatches(
        template, keypoints1, test, keypoints2, newMatches, None)
    saving_url = "./matches/" + filename + ".jpg"
    # cv2.imwrite(saving_url, imMatches)

    template = cv2.drawKeypoints(template, keypoints1, None)
    test = cv2.drawKeypoints(test, keypoints2, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints2[match.trainIdx].pt
        points2[i, :] = keypoints1[match.queryIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = template.shape
    im2Reg = cv2.warpPerspective(original_test, h, (width, height))

    # Loading Mask
    mask = cv2.imread("template/mask.png")  # Load the template picture
    mask = rgb_to_gray_scale(mask)

    # Applying Mask Template pixel per pixel
    (h, w) = im2Reg.shape[:2]
    for y in range(h):
        for x in range(w):
            v = mask[y - 1][x - 1]
            if v < 200:
                im2Reg[y - 1][x - 1] = [v, v, v]

    saving_url = "./results/" + filename + ".jpg"
    #cv2.imwrite(saving_url, im2Reg)

    return im2Reg


def rotate_selfie(selfie):
    # Function performs selfie rotation until a face is found, then break and return the turned picture
    for i in range(1, 5):
        # Convert selfie in gray scale
        selfie_gray = rgb_to_gray_scale(selfie)
        # Looking for a face in the picture
        selfie_faces = FACE_CASCADE.detectMultiScale(selfie_gray, 1.5, 8)
        # Check if at least one face was found
        if len(selfie_faces):
            break
        selfie = cv2.rotate(selfie, cv2.ROTATE_90_CLOCKWISE)
        # INTEGRARE IL SISTEMA DI ROTAZIONE IN CASO DI NESSUN VOLTO TROVATO
    return selfie

# Load the cascade1080
FACE_CASCADE = cv2.CascadeClassifier(
    'template/haarcascade_frontalface_default.xml')

TEST_SET = ["federico", "lapo", "federico", "matteo", "alessio", "riccardo",
            3655, 3713, 3967, 5325, 5446]

for user in TEST_SET:
    # Generate Wrapped Front image
    filename = str(user) + "_doc_fronte"
    front = get_wrapped_image(
        filename=filename, template="template/template-empty-front.png")

    # Generate Wrapped Back Image
    filename = str(user) + "_doc_retro"
    back = get_wrapped_image(
        filename=filename, template="template/template-empty-back.png")

    # Stacking vertically the 2 pictures
    collage = np.vstack((front, back))
    # Standardize the image to match the collage heigth
    collage = resize_with_aspect_ratio_and_add_border(
        collage, height=MAX_PICTURE_HEIGHT)

    # Loading Selfie
    selfie = cv2.imread("data/"+str(user)+"_doc_selfie.jpeg")
    selfie = rotate_selfie(selfie)
    # Resize the image
    selfie = resize_with_aspect_ratio_and_add_border(
        selfie, height=MAX_PICTURE_HEIGHT)

    # Stacking collage with the selfie horizontally
    collage = np.hstack((collage, selfie))

    saving_url = "./results/" + str(user) + ".jpg"
    cv2.imwrite(saving_url, resize_with_aspect_ratio_and_add_border(
        collage, width=1080))
    # Convert into grayscale
    gray = rgb_to_gray_scale(collage)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 8)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(collage, (x, y), (x + w, y + h), (0, 255, 255), 3)

    def match_faces(selfie_url, front_url):

        selfie_image = face_recognition.load_image_file(selfie_url)
        front_doc_image = face_recognition.load_image_file(front_url)
        cv2.imshow("Selfie", selfie_image)
        cv2.imshow("Front", front_doc_image)

        key = cv2.waitKey(0)

        selfie_encoding = face_recognition.face_encodings(selfie_image)
        front_doc_encoding = face_recognition.face_encodings(front_doc_image)
        if len(selfie_encoding) and len(front_doc_encoding):
            front_doc_encoding = front_doc_encoding[0]
            selfie_encoding = selfie_encoding[0]
            results = face_recognition.compare_faces(
                [selfie_encoding], front_doc_encoding)
            return results
        return False

    # Save selfie picture
    selfie_url = "./selfie/"+str(user)+".jpeg"
    cv2.imwrite(selfie_url, selfie)
    # Save front picture
    front_url = "./front/"+str(user)+".jpeg"
    cv2.imwrite(front_url, front)

    print(match_faces(selfie_url, front_url))


#cv2.imshow("ResultMerge", resize_with_aspect_ratio_and_add_border(collage, width=1080))
