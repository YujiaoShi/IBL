import cv2
import random
import numpy as np
# load the yaw, pitch angles for the street-view images and yaw angles for the aerial view
import scipy.io as sio


class InputData:
    # the path of your CVACT dataset

    img_root = '../../Data/CVACT/'

    # yaw_pitch_grd = sio.loadmat('./CVACT_orientations/yaw_pitch_grd_CVACT.mat')
    # yaw_sat = sio.loadmat('./CVACT_orientations/yaw_radius_sat_CVACT.mat')

    posDistThr = 25
    posDistSqThr = posDistThr * posDistThr

    panoCropPixels = int(832 / 2)

    panoRows = 128

    panoCols = 512

    satSize = 256

    def __init__(self, polar=1):
        self.polar = polar

        self.allDataList = './OriNet_CVACT/CVACT_orientations/ACT_data.mat'
        print('InputData::__init__: load %s' % self.allDataList)
        print('load data from CVACT dataset==========================')

        self.__cur_allid = 0  # for training
        self.id_alllist = []
        self.id_idx_alllist = []

        # load the mat

        anuData = sio.loadmat(self.allDataList)

        idx = 0
        for i in range(0, len(anuData['panoIds'])):
            grd = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'

            sat = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.png'

            g2a = self.img_root + 'g2a/' + anuData['panoIds'][i] + '_grdView.png'

            a2g = self.img_root + 'a2g/' + anuData['panoIds'][i] + '_satView_polish.png'

            sat_polar = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'

            self.id_alllist.append([grd, sat, g2a, a2g, sat_polar, anuData['utm'][i][0], anuData['utm'][i][1]])

            self.id_idx_alllist.append(idx)
            idx += 1
        self.all_data_size = len(self.id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', self.all_data_size)

        # partion the images into cells

        self.utms_all = np.zeros([2, self.all_data_size], dtype=np.float32)
        for i in range(0, self.all_data_size):
            self.utms_all[0, i] = self.id_alllist[i][5]
            self.utms_all[1, i] = self.id_alllist[i][6]

        self.training_inds = anuData['trainSet']['trainInd'][0][0] - 1

        self.trainNum = len(self.training_inds)

        self.trainList = []
        self.trainIdList = []
        self.trainUTM = np.zeros([2, self.trainNum], dtype=np.float32)
        for k in range(self.trainNum):
            self.trainList.append(self.id_alllist[self.training_inds[k][0]])
            self.trainUTM[:, k] = self.utms_all[:, self.training_inds[k][0]]
            self.trainIdList.append(k)

        self.__cur_id = 0  # for training

        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)

        self.valList = []
        self.valUTM = np.zeros([2, self.valNum], dtype=np.float32)
        for k in range(self.valNum):
            self.valList.append(self.id_alllist[self.val_inds[k][0]])
            self.valUTM[:, k] = self.utms_all[:, self.val_inds[k][0]]
        # cur validation index
        self.__cur_test_id = 0

    def next_batch_scan(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_test_id >= self.valNum:
            self.__cur_test_id = 0
            return None, None, None, None, None, None, None
        elif self.__cur_test_id + batch_size >= self.valNum:
            batch_size = self.valNum - self.__cur_test_id

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_g2a = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        grd_width = int(FOV / 360 * 512)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        batch_a2g = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        grd_shift = np.zeros([batch_size], dtype=np.int)

        # the utm coordinates are used to define the positive sample and negative samples
        batch_utm = np.zeros([batch_size, 2], dtype=np.float32)
        batch_dis_utm = np.zeros([batch_size, batch_size, 1], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # ground
            img = cv2.imread(self.valList[img_idx][0])

            if img is None or img.shape[0] * 2 != img.shape[1]:
                print('InputData::next_pair_batch: read fail ground: %s, %d, ' % (self.valList[img_idx][0], i))
                continue
            img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            img = img[64:-64, :, :]
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            a = np.random.rand()
            random_shift = int(a * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img_dup

            # grd_shift[i] = random_shift
            grd_shift[i] = (((1 - a) * 360) + FOV / 2) % 360 - 180

            # a2g
            img = cv2.imread(self.valList[img_idx][3])

            if img is None or img.shape[0] * 2 != img.shape[1]:
                print('InputData::next_pair_batch: read fail a2g: %s, %d, ' % (self.valList[img_idx][3], i))
                continue
            img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            img = img[64:-64, :, :]
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            # j = np.arange(0, 512)
            # img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_a2g[i, :, :, :] = img

            # satellite
            img = cv2.imread(self.valList[img_idx][1])
            # img = cv2.resize(img, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail satellite: %s, %d, ' % (self.valList[img_idx][1], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # g2a
            img = cv2.imread(self.valList[img_idx][2])
            # print(self.valList[img_idx][2])
            # img = cv2.resize(img, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail g2a: %s, %d, ' % (self.valList[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_g2a[i, :, :, :] = img

            # polar satellite
            img = cv2.imread(self.valList[img_idx][4])

            if img is None or img.shape[0] != self.panoRows or img.shape[1] != self.panoCols:
                print('InputData::next_pair_batch: read fail polar satellite: %s, %d, ' % (self.valList[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat_polar[i, :, :, :] = img

            batch_utm[i, 0] = self.valUTM[0, img_idx]
            batch_utm[i, 1] = self.valUTM[1, img_idx]

        self.__cur_test_id += batch_size

        # compute the batch gps distance
        for ih in range(batch_size):
            for jh in range(batch_size):
                batch_dis_utm[ih, jh, 0] = (batch_utm[ih, 0] - batch_utm[jh, 0]) * (
                        batch_utm[ih, 0] - batch_utm[jh, 0]) + (batch_utm[ih, 1] - batch_utm[jh, 1]) * (
                                                   batch_utm[ih, 1] - batch_utm[jh, 1])

        return batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, batch_dis_utm, grd_shift # (np.around(((512 - grd_shift) / 512 * 64) % 64)).astype(np.int)

    def next_pair_batch(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.trainIdList)

        if self.__cur_id + batch_size + 2 >= self.trainNum:
            self.__cur_id = 0
            return None, None, None, None, None, None, None

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_g2a = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        grd_width = int(FOV / 360 * 512)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        batch_a2g = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        grd_shift = np.zeros([batch_size], dtype=np.int)

        # the utm coordinates are used to define the positive sample and negative samples
        batch_utm = np.zeros([batch_size, 2], dtype=np.float32)
        batch_dis_utm = np.zeros([batch_size, batch_size, 1], dtype=np.float32)

        i = 0
        batch_idx = 0
        while True:

            if batch_idx >= batch_size or self.__cur_id + i >= self.trainNum:
                break

            img_idx = self.trainIdList[self.__cur_id + i]
            i += 1

            # ground
            img = cv2.imread(self.trainList[img_idx][0])

            if img is None or img.shape[0] * 2 != img.shape[1]:
                print('InputData::next_pair_batch: read fail ground: %s, %d, ' % (self.trainList[img_idx][0], i))
                continue
            img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            img = img[64:-64, :, :]
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            a = np.random.rand()
            random_shift = int(a * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img_dup

            # grd_shift[batch_idx] = random_shift
            grd_shift[batch_idx] = (((1 - a) * 360) + FOV / 2) % 360 - 180

            # a2g
            img = cv2.imread(self.trainList[img_idx][3])

            if img is None or img.shape[0] * 2 != img.shape[1]:
                print('InputData::next_pair_batch: read fail a2g: %s, %d, ' % (self.trainList[img_idx][3], i))
                continue
            img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            img = img[64:-64, :, :]
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            # j = np.arange(0, 512)
            # img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_a2g[batch_idx, :, :, :] = img

            # satellite
            img = cv2.imread(self.trainList[img_idx][1])
            # img = cv2.resize(img, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail satellite: %s, %d, ' % (self.trainList[img_idx][1], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img

            # g2a
            img = cv2.imread(self.trainList[img_idx][2])
            # img = cv2.resize(img, (self.satSize, self.satSize), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail g2a: %s, %d, ' % (self.trainList[img_idx][2], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_g2a[batch_idx, :, :, :] = img

            # polar satellite
            img = cv2.imread(self.trainList[img_idx][4])

            if img is None or img.shape[0] != self.panoRows or img.shape[1] != self.panoCols:
                print('InputData::next_pair_batch: read fail polar: %s, %d, ' % (self.trainList[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat_polar[batch_idx, :, :, :] = img

            batch_utm[batch_idx, 0] = self.trainUTM[0, img_idx]
            batch_utm[batch_idx, 1] = self.trainUTM[1, img_idx]

            batch_idx += 1

        # compute the batch gps distance
        for ih in range(batch_size):
            for jh in range(batch_size):
                batch_dis_utm[ih, jh, 0] = (batch_utm[ih, 0] - batch_utm[jh, 0]) * (
                        batch_utm[ih, 0] - batch_utm[jh, 0]) + (batch_utm[ih, 1] - batch_utm[jh, 1]) * (
                                                   batch_utm[ih, 1] - batch_utm[jh, 1])

        self.__cur_id += i
        # return batch_sat, batch_grd, batch_dis_utm
        return batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, batch_dis_utm, grd_shift #(np.around(((512 - grd_shift) / 512 * 64) % 64)).astype(np.int)

    def get_dataset_size(self):
        return self.trainNum

    #
    def get_test_dataset_size(self):
        return self.valNum

    #
    def reset_scan(self):
        self.__cur_test_id = 0


if __name__ == '__main__':
    input_data = InputData()
    batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, batch_dis_utm, _ = input_data.next_pair_batch(12)
