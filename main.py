import cv2
import numpy as np
import scipy

# GLOBAL VARIABLES
MAX_FEATURES = 6000  # Upperbound limit to the number of features to find on single picture
GOOD_MATCH_PERCENT = 0.3
PADDING_PIXELS = 60

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

    w = dim[0] + PADDING_PIXELS*2
    h = dim[1] + PADDING_PIXELS*2

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
                rgb_pixel = resized_img[y2-1][x2-1]

            blank_image[y, x] = rgb_pixel

    # cv2.imshow("BlackCanvas", blank_image)
    #key = cv2.waitKey(0)

    return blank_image


def convert_to_gray_scale(image):
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_wrapped_image(filename, template):
    # MAIN ACTIVITIES
    # Load the template picture
    template = cv2.imread(template)
    url = "data/" + filename + ".jpeg"
    print(url)
    original_test = cv2.imread(url)  # Load test picture
    test = original_test = resize_with_aspect_ratio_and_add_border(
        original_test, height=1080)  # Resize test picture
    # Convert picture to gray scale color palette
    test = convert_to_gray_scale(test)

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
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Calculate how many matches to keep
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]  # Remove not so good matches

    imMatches = cv2.drawMatches(
        template, keypoints1, test, keypoints2, matches, None)
    saving_url = "./matches/" + filename + ".jpg"
    cv2.imwrite(saving_url, imMatches)

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
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Applying Mask
    (h, w) = im2Reg.shape[:2]
    for y in range(h):
        for x in range(w):
            v = mask[y-1][x-1]
            if v < 200:
                im2Reg[y-1][x-1] = [v, v, v]

    saving_url = "./results/" + filename + ".jpg"
    cv2.imwrite(saving_url, im2Reg)

    return im2Reg

    # cv2.imshow("TemplateView", template)
    # cv2.imshow("TestView", test)
    # cv2.imshow("MatchesView", imMatches)


TEST_SET = ["alessio", "lapo", "matteo", "riccardo",
            "federico", 3655, 3713, 3967, 4847, 5279, 5325, 5446, 5451]

for user in TEST_SET:
    # Generate Wrapped Front image
    filename = str(user)+"_doc_fronte"
    front = get_wrapped_image(filename=filename, template="template/template-empty-front.png")
    # Generate Wrapped Back Image
    filename = str(user)+"_doc_retro"
    back = get_wrapped_image(filename=filename, template="template/template-empty-back.png")

    cv2.imshow("ResultFront", front)
    cv2.imshow("ResultBack", back)
    key = cv2.waitKey(0)