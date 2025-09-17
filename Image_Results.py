import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Image_Results():
    I = [60, 80, 90, 110, 120]
    Images = np.load('Images.npy', allow_pickle=True)
    GT = np.load('GT.npy', allow_pickle=True)
    UNET = np.load('Segment3.npy', allow_pickle=True)
    RESUNET = np.load('Segment4.npy', allow_pickle=True)
    TRANSUNET = np.load('Segment2.npy', allow_pickle=True)
    ASUnet = np.load('Segment1.npy', allow_pickle=True)
    PROPOSED = np.load('Segment5.npy', allow_pickle=True)
    for i in range(len(I)):
        # print(i)
        plt.subplot(2, 4, 1)
        plt.title('Original')
        plt.imshow(Images[I[i]])
        plt.subplot(2, 4, 2)
        plt.title('GroundTruth')
        plt.imshow(GT[I[i]])
        plt.subplot(2, 4, 3)
        plt.title('UNET')
        plt.imshow(UNET[I[i]])
        plt.subplot(2, 4, 4)
        plt.title('RESUNET')
        plt.imshow(RESUNET[I[i]])
        plt.subplot(2, 4, 5)
        plt.title('TRANSUNET')
        plt.imshow(TRANSUNET[I[i]])
        plt.subplot(2, 4, 6)
        plt.title('ARNet')
        plt.imshow(ASUnet[I[i]])
        plt.subplot(2, 4, 7)
        plt.title('PROPOSED')
        plt.imshow(PROPOSED[I[i]])
        plt.tight_layout()
        # path = "./Results/Image_Results/Image_%s_%s.png" % (n + 1, i + 1)
        # plt.savefig(path)
        plt.show()
        cv.imwrite('./Results/Image_Results/orig-' + str(i + 1) + '.png', Images[I[i]])

        cv.imwrite('./Results/Image_Results/gt-' + str(i + 1) + '.png', GT[I[i]])
        cv.imwrite('./Results/Image_Results/unet-' + str(i + 1) + '.png', UNET[I[i]])
        cv.imwrite('./Results/Image_Results/resunet-' + str(i + 1) + '.png',
                   RESUNET[I[i]])
        cv.imwrite('./Results/Image_Results/Transunet-' + str(i + 1) + '.png',
                   TRANSUNET[I[i]])
        cv.imwrite('./Results/Image_Results/ARNet-' + str(i + 1) + '.png',
                   ASUnet[I[i]])
        cv.imwrite('./Results/Image_Results/proposed-' + str(i + 1) + '.png',
                   PROPOSED[I[i]])


def Sample_Images():
    Orig = np.load('Images.npy', allow_pickle=True)
    ind = [550, 571, 582, 563, 539, 534]
    fig, ax = plt.subplots(2, 3)
    plt.suptitle("Sample Images from Dataset")
    plt.subplot(2, 3, 1)
    plt.title('Image-1')
    plt.imshow(Orig[ind[0]])
    plt.subplot(2, 3, 2)
    plt.title('Image-2')
    plt.imshow(Orig[ind[1]])
    plt.subplot(2, 3, 3)
    plt.title('Image-3')
    plt.imshow(Orig[ind[2]])
    plt.subplot(2, 3, 4)
    plt.title('Image-4')
    plt.imshow(Orig[ind[3]])
    plt.subplot(2, 3, 5)
    plt.title('Image-5')
    plt.imshow(Orig[ind[4]])
    plt.subplot(2, 3, 6)
    plt.title('Image-6')
    plt.imshow(Orig[ind[5]])
    plt.show()
    cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(0 + 1) + '.png', Orig[ind[0]])
    cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(1 + 1) + '.png', Orig[ind[1]])
    cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(2 + 1) + '.png', Orig[ind[2]])
    cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(3 + 1) + '.png', Orig[ind[3]])
    cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(4 + 1) + '.png', Orig[ind[4]])
    cv.imwrite('./Results/Sample_Images/Abnormal-img-' + str(5 + 1) + '.png', Orig[ind[5]])


if __name__ == '__main__':
    Image_Results()
