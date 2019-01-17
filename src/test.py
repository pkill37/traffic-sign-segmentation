import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import data
import helpers

if __name__ == '__main__':
    helpers.seed()

    pwd = os.path.realpath(__file__)
    out_dir = os.path.abspath(os.path.join(pwd, '../../out/')) + '/'
    data_dir = os.path.abspath(os.path.join(pwd, '../../data/')) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=out_dir+'models/best.hdf5')
    parser.add_argument('--images_path', type=str, default=data_dir+'testImages/')
    parser.add_argument('--labels_path', type=str, default=data_dir+'testLabels/')
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    plt.ion()
    
    #lists with percentages for every image
    truePositives = list()
    falsePositives = list()
    falseNegatives = list()

    for x_batch, y_batch in data.MaskedImageSequence(images_path=args.images_path, labels_path=args.labels_path, img_height=args.img_height, img_width=args.img_width, batch_size=args.batch_size, augment=False):
        y_pred = model.predict_on_batch(x_batch)
        #y_pred = (y_pred > 0.5).astype(np.uint8)

        #pixel count
        truePositiveCount = 0
        falsePositiveCount = 0
        falseNegativeCount = 0
        totalCountOfSignPixels = 0

        for i in range(len(x_batch)):
            # i represents image in batch
            for row in range(224):
                for column in range(224):
                    # for sign pixels we take original ones
                    if(y_batch[i,row,column,0] == 255):
                        totalCountOfSignPixels = totalCountOfSignPixels + 1

                    if(y_batch[i,row,column,0] == 255 and y_pred[i,row,column,0] >= 200):
                        truePositiveCount = truePositiveCount + 1

                    elif(y_batch[i,row,column,0] == 255 and y_pred[i,row,column,0] < 200):
                        falseNegativeCount = falseNegativeCount + 1

                    elif(y_batch[i,row,column,0] == 0 and y_pred[i,row,column,0] == 255):
                        falsePositiveCount = falsePositiveCount + 1
            
           
            truePositives.append(truePositiveCount / totalCountOfSignPixels)
            falsePositives.append(falsePositiveCount / totalCountOfSignPixels)
            falseNegatives.append(falseNegativeCount / totalCountOfSignPixels)

            plt.imshow(x_batch[i,:,:,:])
            plt.imshow(y_pred[i,:,:,0], alpha=0.7)
            input('Press [Enter] to predict another mini-batch...')
            plt.close()

        # average for all images
        sumTruePositives = 0
        sumFalsePositives = 0
        sumFalseNegatives = 0
        totalImages = len(truePositives)

        for value in range(totalImages):
            sumTruePositives = sumTruePositives + truePositives[value]
            sumFalsePositives = sumFalsePositives + falsePositives[value]
            sumFalseNegatives = sumFalseNegatives + falseNegatives[value]

        print("average true positive %f\n", sumTruePositives/totalImages * 100)
        print("average false positive %f\n", sumFalsePositives/totalImages * 100)
        print("average false negative %f\n", sumFalseNegatives/totalImages * 100)



