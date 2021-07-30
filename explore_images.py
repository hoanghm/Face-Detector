import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


link_dataset = 'C:/Users/hoang/Downloads/CelebA'
image_folder = link_dataset + '/' + 'img_align_celeba/img_align_celeba'
os.chdir(link_dataset)

attribs_df = pd.read_csv('list_attr_celeba.csv')

def show_img(id):
    img = cv2.imread(image_folder + '/' + id)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.rectangle()
    cv2.imshow(id,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_img_with_labels(id=None):
    print("Showing images with labels...")
    print("Press Enter to show another random image.")
    print("Press Esc to quit.")
    if id is None:
        id = '{:06d}'.format(np.random.randint(0,200000)) + '.jpg'
        
    Done = False
    while not Done:
        img_link = image_folder + '/' + id    
        img = cv2.imread(img_link)
        img_height, img_width = img.shape[:2]
        img_with_padding = cv2.copyMakeBorder(img, 0, 0, 0, 800, cv2.BORDER_CONSTANT) # 300px right-padding with black color
        
        # considered_attribs = ['Attractive', 'Bald', 'Black_Hair', 'Blond_Hair', 'Chubby', 'Eyeglasses']
        considered_attribs = attribs_df.columns[1:]
        img_attribs = attribs_df.loc[attribs_df['image_id'] == id].squeeze()
        
        # position where the first will be put, these values will be updated as more text is displayed
        text_x = img_width + 10
        text_y = 20
        
        for attrib in considered_attribs:
            text = '{}: {}'.format(attrib, 'yes' if img_attribs[attrib] > 0 else 'no')
            cv2.putText(img_with_padding, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
            text_y += 30
            if text_y > img_height:
                text_y = 20
                text_x += 300
        
        cv2.imshow(img_link[-10:], img_with_padding)
        if cv2.waitKey(0) == 27: # 27 is ESC
            Done = True
        else:
            id = '{:06d}'.format(np.random.randint(0,200000)) + '.jpg'
        
        cv2.destroyAllWindows()
    
    
    


