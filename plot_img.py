def  plot_img():
        img_o = cv2.imread(filename)
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_o = img_o[y:y+h, x:x+w]


        #print img_o[82, 200]
        #print img[182, 200]
        #color = cv2.cvtColor(np.uint8([[[213,31,150]]]),cv2.COLOR_RGB2HSV)
        #print color[0][0]
        #temp = [0, 0, 0]
        #temp[0] = int(color[0][0][0]) * 360.0 / 176.0
        #temp[1] = int(color[0][0][1]) * 100.0 / 255.0
        #temp[2] = int(color[0][0][2]) * 100.0 / 255.0

        #print temp
        plt.imshow(img_o)
        #plt.show()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        #plt.show()
        break

        mask = np.zeros(img.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
         
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        zero_indx = np.where(thresh==0)
        img[zero_indx] = [0, 0, 0]

        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        freq = itemfreq(label)
        dominant_color = center[max(freq, key=lambda x: x[1])[0]]
        #print '\n\ndominant color:\t', dominant_color, '\ncenter\t', center
        fig, ax = plt.subplots(nrows=1, ncols=3)
        fig.suptitle(filename)

        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(original)

        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(img)
        
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(res2)
        
        plt.show()
