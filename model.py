import os
import cv2
import string
import torch
import paddle
import imutils  
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt



# Mapping for special characters
special_char_mapping = {'-', '*', '=', '/', '@', '&', '.', '~', ',', ';', ':', ' ', '_'}


def remove_special_characters(text):
    """
    Remove special characters and replace them with their closest valid equivalent.
    
    Args:
        text (str): The input text.

    Returns:
        str: Text after removing special characters.
    """
    cleaned_text = ''
    for char in text:
        if char in special_char_mapping:
            continue
        else:
            cleaned_text += char
    return cleaned_text




# Define color ranges
color_ranges = {
    "yellow": ((20, 100, 100), (30, 255, 255)),
    "white": ((0, 0, 200), (180, 30, 255)),
    "blue": ((100, 150, 0), (140, 255, 255)),
    "red": ((0, 50, 50), (10, 255, 255)),  # Lower red
    "red2": ((170, 50, 50), (180, 255, 255)),  # Upper red
    "black": ((0, 0, 0), (180, 255, 50)),
    "green": ((40, 50, 50), (80, 255, 255)),
}



def detect_color(hsv_image):
    color_pixels = {}
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        color_pixels[color] = cv2.countNonZero(mask)
    
    # Merge red ranges
    if "red" in color_pixels and "red2" in color_pixels:
        color_pixels["red"] += color_pixels.pop("red2")
    
    # Determine the color with the maximum pixel count
    detected_color = max(color_pixels, key=color_pixels.get)

    return detected_color


def detect_car_colors(image, x1, y1, x2, y2):
    
    # h = y2 - y1
    # y1 += 0.3*h
    # y2 -= 0.25*h

    car_crop = image[int(y1):int(y2), int(x1):int(x2)]
    car_crop_hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    color = detect_color(car_crop_hsv)

    return color


def brand_extraction():
    pass

def get_img():

    print("Number of GPUs: ", paddle.device.cuda.device_count())

    # print("Number of GPU: ", torch.cuda.device_count())
    # if torch.cuda.is_available():
    #     print("GPU Name: ", torch.cuda.get_device_name())

    # torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    # torch.cuda.empty_cache()  # Clear cache before starting training


    folder_path = os.path.join(".", "test_images")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.splitext(file)[1].lower() in image_extensions
    ]

    image_names = [
        os.path.splitext(file)[0].lower()
        for file in os.listdir(folder_path)
        if os.path.splitext(file)[1].lower() in image_extensions
    ]

    return image_paths, image_names



def model_car_detection(image_paths, image_names):

    output_folder = os.path.join(".", "images")
    yolo_model = YOLO("yolov8n.pt")

    model_path = os.path.join(".", "models", "license_plate_detector.pt")
    license_plate_detector = YOLO(model_path)

    ocr = PaddleOCR(use_angle_cls=True, lang='en') 
    

    detect_car = []
    # img_car_id = []
    detect_plate = []
    # img_plate_id = []
    texts = []
    txt = []
    img_text_id = []

    for i,img in enumerate(image_paths):
        if i > 50:
            break
        # if i < 40:
        #     continue
        # elif i > 55:
        #     break

        initial_img = cv2.imread(img)

        dets = license_plate_detector(img)

 
        for det in dets[0].boxes.data.tolist():
            x1p, y1p, x2p, y2p, scorep, class_idp = det
            detect_plate.append([x1p, y1p, x2p, y2p, scorep])


            CONFIDENCE_THRESHOLD = 0.85

        
            # Original image OCR
            image = initial_img[int(y1p):int(y2p), int(x1p):int(x2p)]
            results = ocr.ocr(image, det=True, rec=True)

            # Variables to hold final text and confidence
            final_texts = []
            final_confs = []
            use_mirror = False

            # Check OCR result for original image
            if results[0]:
                for line in results[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence >= CONFIDENCE_THRESHOLD:  # Accept if confidence is high enough
                        final_texts.append(text)
                        final_confs.append(confidence)
                    else:
                        use_mirror = True  # Flag to check mirror image if confidence is low
            else:
                use_mirror = True  # Flag to check mirror image if no results

            # If confidence is low or no result, process mirrored image
            if use_mirror:
                mirrored_image = cv2.flip(image, 1)  # Flip the image horizontally
                results_mirror = ocr.ocr(mirrored_image, det=True, rec=True)

                if results_mirror[0]:
                    for line in results_mirror[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence >= CONFIDENCE_THRESHOLD:  # Only add high-confidence results
                            final_texts.append(text)
                            final_confs.append(confidence)

            
            # If valid text is detected, append to final results
            if final_texts:
                # plate_id.append(img_id[i])
                # is_det_plate.append(i)
                
                texts.append(" ".join(final_texts))  # Combine all detected lines into one string
                img_text_id.append(i)
                # conf.append(final_confs)  # Append confidence values

            final_plate_text = remove_special_characters(texts[len(texts)-1])
            txt.append(final_plate_text)
        
        # most_characters = calculate_most_frequent_characters(texts)

            color = (0, 255, 0)  
            thickness = 2
            x1p, y1p, x2p, y2p = int(x1p), int(y1p), int(x2p), int(y2p)
            cv2.rectangle(initial_img, (x1p, y1p), (x2p, y2p), color, thickness)


            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 0.6
            # font_thickness = 2
            # text_color = (0, 255, 0)  
            # text_position = (x1p, y1p-5)
            # cv2.putText(initial_img, final_plate_text, text_position, font, font_scale, text_color, font_thickness)

            hp = y2p - y1p
            wp = x2p - x1p

            color = (0, 0, 255)  
            thickness = 2
            x1p, y1p, x2p, y2p = int(x1p), int(y1p-2.5*hp), int(x2p), int(y2p-0.4*hp)
            cv2.rectangle(initial_img, (x1p, y1p), (x2p, y2p), color, thickness)

        dets = yolo_model(img)
        # plate.append(dets)
        for det in dets[0].boxes.data.tolist():
            x1c, y1c, x2c, y2c, scorec, class_idc = det
            detect_car.append([x1c, y1c, x2c, y2c, scorec])
            # img_car_id.append(i)
            # break
        # j = j+1
            x1c, y1c, x2c, y2c = int(x1c), int(y1c), int(x2c), int(y2c)
            color = (255, 0, 0)  
            thickness = 2

            h = y2c - y1c
            y1c += 0.3*h
            y2c -= 0.2*h

            y1c = int(y1c)
            y2c = int(y2c)

            clr = detect_car_colors(initial_img, x1c, y1c, x2c, y2c)

            cv2.rectangle(initial_img, (x1c, y1c), (x2c, y2c), color, thickness)


            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 2
            text_color = (0, 0, 255)  
            text_position = (x1c, y1c+15)
            cv2.putText(initial_img, clr, text_position, font, font_scale, text_color, font_thickness)
       
        output_path = os.path.join(output_folder, os.path.basename(img)) 
        cv2.imwrite(output_path, initial_img)
        print(f"Processed and saved: {output_path}")

        initial_img = initial_img[int(y1p):int(y2p), int(x1p):int(x2p)]



        # Read the image
        image = initial_img
        ratio = image.shape[0] / 500.0  # Resize based on height
        orig = image.copy()
        image = imutils.resize(image, height=500)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding to handle varying lighting
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

        # Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # Perform Canny edge detection
        edged = cv2.Canny(gray, 65, 200)

        # Apply morphological operations to close gaps
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # edged = cv2.dilate(edged, kernel, iterations=1)
        # edged = cv2.erode(edged, kernel, iterations=1)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Optionally, filter out small contours or those with low area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # Adjust threshold

        # Draw contours on the image
        cv2.drawContours(image, contours, -1, (0, 255, 255), 2)

        # Show the results
        # cv2.imshow("Contours", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        

        output_path = os.path.join(output_folder, ("(1)" + os.path.basename(img))) 
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {output_path}")
    

    write_csv(txt, img_text_id, "D:/project/plate_detection/test.csv", image_names)
    # save_imgs(image_paths, plate_id, is_det_plate, det_plate, texts)
            

        


def write_csv(text, car_id, output_path, image_names):

    with open(output_path, 'w') as f:
        f.write('{},{}\n'.format('image_name', 'license_number'))


        for i,res in enumerate(car_id):
            f.write('{},{}\n'.format(image_names[car_id[i]], text[i]))

        f.close()


image_paths,image_names = get_img()
model_car_detection(image_paths,image_names)



# def extrect_text():

#     invalid_first = ["-", "~", "_", "=", "E"]
    
#     ocr = PaddleOCR(use_angle_cls=True, lang='en') 

#     image_paths = get_img()
#     croped_img, img_id, det_plate = model_run(image_paths)
#     texts =[]
#     conf =[]
#     is_det_plate =[]
#     plate_id =[]

#     # croped_img = process_croped_images(croped_img)


#     # for idx, image in enumerate(image_array):
#     #     # If image is a file path, load it
#     #     if isinstance(image, str):  # File path
#     #         img = Image.open(image).convert("RGB")
#     #     else:  # Image is a numpy array
#     #         img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     #     # Preprocess and recognize text
#     #     pixel_values = processor(images=img, return_tensors="pt").pixel_values
#     #     generated_ids = model.generate(pixel_values)
#     #     recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     #     results[f"image_{idx}"] = recognized_text



#     # for i,img in enumerate(croped_img):
#     #     image = img
        
#     #     results = ocr.ocr(image, det=True, rec=True)

#     #     # print(type(results[0]))

#     #     if not results[0]:
#     #         continue

#     #     else:
#     #         plate_id.append(img_id[i])
#     #         is_det_plate.append(i)
        

#     #     for line in results[0]:
#     #         # if line[1][0][0] in invalid_first:
#     #         #     print(f"Text: {line[1][0][1:]}, Confidence: {line[1][1]}")
#     #         #     print("error-----------------------------------------------------")
#     #         # else:
#     #         #     print(f"Text: {line[1][0]}, Confidence: {line[1][1]}")

   
            
#     #         texts.append(line[1][0])
#     #         conf.append(line[1][1])
#     #         break

#     CONFIDENCE_THRESHOLD = 0.85

#     plate_id = []
#     is_det_plate = []
#     texts = []
#     conf = []

#     for i, img in enumerate(croped_img):
#         # Original image OCR
#         image = img
#         results = ocr.ocr(image, det=True, rec=True)

#         # Variables to hold final text and confidence
#         final_texts = []
#         final_confs = []
#         use_mirror = False

#         # Check OCR result for original image
#         if results[0]:
#             for line in results[0]:
#                 text = line[1][0]
#                 confidence = line[1][1]
#                 if confidence >= CONFIDENCE_THRESHOLD:  # Accept if confidence is high enough
#                     final_texts.append(text)
#                     final_confs.append(confidence)
#                 else:
#                     use_mirror = True  # Flag to check mirror image if confidence is low
#         else:
#             use_mirror = True  # Flag to check mirror image if no results

#         # If confidence is low or no result, process mirrored image
#         if use_mirror:
#             mirrored_image = cv2.flip(image, 1)  # Flip the image horizontally
#             results_mirror = ocr.ocr(mirrored_image, det=True, rec=True)

#             if results_mirror[0]:
#                 for line in results_mirror[0]:
#                     text = line[1][0]
#                     confidence = line[1][1]
#                     if confidence >= CONFIDENCE_THRESHOLD:  # Only add high-confidence results
#                         final_texts.append(text)
#                         final_confs.append(confidence)

#         # If valid text is detected, append to final results
#         if final_texts:
#             plate_id.append(img_id[i])
#             is_det_plate.append(i)
#             texts.append(" ".join(final_texts))  # Combine all detected lines into one string
#             conf.append(final_confs)  # Append confidence values
    
    
#     # most_characters = calculate_most_frequent_characters(texts)

#     for i,text in enumerate(texts):
#         texts[i] = remove_special_characters(text)
#         # texts[i] = correct_text(text, most_characters)
#     #     texts[i] = format_license(texts[i])

#     write_csv(texts, conf, plate_id, "C:/Users/VICTUS/Desktop/project/venv1/test.csv")
#     save_imgs(image_paths, plate_id, is_det_plate, det_plate, texts)
        