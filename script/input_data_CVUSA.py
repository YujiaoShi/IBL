import cv2
import random
import numpy as np
# load the yaw, pitch angles for the street-view images and yaw angles for the aerial view
import scipy.io as sio


class InputData:
    # the path of your CVACT dataset
    def __init__(self):
        self.img_root = '../../Data/CVUSA/'
        self.train_list = self.img_root + 'splits/train-19zl.csv'
        self.test_list = self.img_root + 'splits/val-19zl.csv'

        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]

                grd = self.img_root + data[1]
                sat = self.img_root + data[0]
                g2a = self.img_root + 'g2a/' + pano_id + '.png'
                a2g = self.img_root + 'a2g/' + pano_id + '.png'
                polar = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')

                self.id_list.append([grd, sat, g2a, a2g, polar])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                grd = self.img_root + data[1]
                sat = self.img_root + data[0]
                g2a = self.img_root + 'g2a/' + pano_id + '.png'
                a2g = self.img_root + 'a2g/' + pano_id + '.png'
                polar = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')

                self.id_test_list.append([grd, sat, g2a, a2g, polar])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)

    def next_batch_scan(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None, None, None, None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_g2a = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        grd_width = int(FOV / 360 * 512)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        batch_a2g = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        grd_shift = np.zeros([batch_size], dtype=np.int)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # ground
            img = cv2.imread(self.id_test_list[img_idx][0])

            if img is None:
                print('InputData::next_pair_batch: read fail ground: %s, %d, ' % (self.id_test_list[img_idx][0], i))
                continue
            # img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            # img = img[64:-64, :, :]
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
            grd_shift[i] = (((1-a)*360)+FOV/2)%360-180

            # a2g
            img = cv2.imread(self.id_test_list[img_idx][3])

            if img is None or img.shape[0] * 4 != img.shape[1]:
                print('InputData::next_pair_batch: read fail a2g: %s, %d, ' % (self.id_test_list[img_idx][3], i))
                continue
            #img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            #img = img[128:-64, :, :]
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            # j = np.arange(0, 512)
            # img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_a2g[i, :, :, :] = img

            # satellite
            img = cv2.imread(self.id_test_list[img_idx][1])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail satellite: %s, %d, ' % (self.id_test_list[img_idx][1], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # g2a
            img = cv2.imread(self.id_test_list[img_idx][2])
            # print(self.valList[img_idx][2])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail g2a: %s, %d, ' % (self.id_test_list[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_g2a[i, :, :, :] = img

            # polar satellite
            img = cv2.imread(self.id_test_list[img_idx][4])

            if img is None:
                print('InputData::next_pair_batch: read fail polar satellite: %s, %d, ' % (self.id_test_list[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat_polar[i, :, :, :] = img
            
        self.__cur_test_id += batch_size

        return batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, grd_shift #(np.around(((512 - grd_shift) / 512 * 64) % 64)).astype(np.int)

    def next_pair_batch(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None, None, None, None, None

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_g2a = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)

        grd_width = int(FOV / 360 * 512)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        batch_a2g = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)

        grd_shift = np.zeros([batch_size], dtype=np.int)

        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # ground
            img = cv2.imread(self.id_list[img_idx][0])

            if img is None:
                print('InputData::next_pair_batch: read fail ground: %s, %d, ' % (self.id_list[img_idx][0], i))
                continue
            # img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            # img = img[64:-64, :, :]
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
            grd_shift[batch_idx] = (((1-a)*360)+FOV/2)%360-180

            # a2g
            img = cv2.imread(self.id_list[img_idx][3])

            if img is None or img.shape[0] * 4 != img.shape[1]:
                print('InputData::next_pair_batch: read fail a2g: %s, %d, ' % (self.id_list[img_idx][3], i))
                continue
            #img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            #img = img[64:-64, :, :]
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            # j = np.arange(0, 512)
            # img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_a2g[batch_idx, :, :, :] = img

            # satellite
            img = cv2.imread(self.id_list[img_idx][1])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail satellite: %s, %d, ' % (self.id_list[img_idx][1], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img

            # g2a
            img = cv2.imread(self.id_list[img_idx][2])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail g2a: %s, %d, ' % (self.id_list[img_idx][2], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_g2a[batch_idx, :, :, :] = img

            # polar satellite
            img = cv2.imread(self.id_list[img_idx][4])

            if img is None:
                print('InputData::next_pair_batch: read fail polar: %s, %d, ' % (self.id_list[img_idx][4], i))
                continue

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat_polar[batch_idx, :, :, :] = img

            batch_idx += 1

        self.__cur_id += i
        # return batch_sat, batch_grd, batch_dis_utm
        return batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, grd_shift # (np.around(((512 - grd_shift) / 512 * 64) % 64)).astype(np.int)

    def get_dataset_size(self):
        return self.data_size

    #
    def get_test_dataset_size(self):
        return self.test_data_size

    #
    def reset_scan(self):
        self.__cur_test_id = 0


if __name__ == '__main__':
    input_data = InputData()
    batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, _ = input_data.next_batch_scan(12)
