import numpy as np
import cv2
from hog_fitures import HOG_fitures
from utils import *

fhog = HOG_fitures()

class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self.lambda_reg          = 0.0001  # регуляризация
        self.padding             = 2.5     # дополнительная область вокруг цели
        self.output_sigma_factor = 0.125   # полоса пропускания гауссовой цели

        if(hog):  # HOG
            # VOT
            self.interp_factor = 0.012     # коэффициент линейной интерполяции для адаптации
            self.sigma = 0.6               # пропускная способность гауссовского ядра
            self.cell_size = 4             # HOG размер ячейки
            self._hogfeatures = True
        else:                            
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        if(multiscale):
            self.template_size = 96        # размер шаблона
            self.scale_step = 1.05         # шаг шкалы для многомасштабной оценки
            self.scale_weight = 0.96       # для снижения показателей обнаружения других шкал для дополнительной стабильности
        elif(fixed_window):
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0, 0]
        self._roi = [0., 0., 0., 0.]
        self.size_patch = [0, 0, 0]
        self._scale = 1.
        self._alphaf = None
        self._prob = None
        self._tmpl = None
        self.hann = None

    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if(self._hogfeatures):
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh)**2, (x - sxh)**2
        res = np.exp(mult * (y + x))
        return fftd(res)

    def gaussianCorrelation(self, x1, x2):
        if(self._hogfeatures):
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in range(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                #caux = rearrange(caux)
                c += caux
            c = rearrange(c)
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)   # 'conjB=' is necessary!
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if(x1.ndim == 3 and x2.ndim == 3):
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif(x1.ndim == 2 and x2.ndim == 2):
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    def getFeatures(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]
        cx = self._roi[0] + self._roi[2] / 2
        cy = self._roi[1] + self._roi[3] / 2

        if(inithann):
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if(self.template_size > 1):
                if(padded_w >= padded_h):
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if(self._hogfeatures):
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if(z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]):
            z = cv2.resize(z, tuple(self._tmpl_sz))

        if(self._hogfeatures):
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
            FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T
        else:
            if(z.ndim == 3 and z.shape[2] == 3):
                FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            elif(z.ndim == 2):
                FeaturesMap = z
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if(inithann):
            self.createHanningMats()

        FeaturesMap = self.hann * FeaturesMap
        return FeaturesMap

    def detect(self, z, x):
        k = self.gaussianCorrelation(x, z)
        res = real(fftd(complex_multiplication(self._alphaf, fftd(k)), True))

        _, pv, _, pi = cv2.minMaxLoc(res)
        p = [float(pi[0]), float(pi[1])]

        if(pi[0] > 0 and pi[0] < res.shape[1] - 1):
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if(pi[1] > 0 and pi[1] < res.shape[0] - 1):
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv

    def train(self, x, train_interp_factor):
        k = self.gaussianCorrelation(x, x)
        alphaf = complex_division(self._prob, fftd(k) + self.lambda_reg)

        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    def init(self, roi, image):
        self._roi = list(map(float, roi))
        assert(roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.getFeatures(image, 1)
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
        self.train(self._tmpl, 1.0)

    def update(self, image):
        if(self._roi[0] + self._roi[2] <= 0):
            self._roi[0] = -self._roi[2] + 1
        if(self._roi[1] + self._roi[3] <= 0):
            self._roi[1] = -self._roi[2] + 1
        if(self._roi[0] >= image.shape[1] - 1):
            self._roi[0] = image.shape[1] - 2
        if(self._roi[1] >= image.shape[0] - 1):
            self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        if(self.scale_step != 1):
            # Маленький scale
            new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
            # Большой scale
            new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

            if(self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif(self.scale_weight * new_peak_value2 > peak_value):
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step

        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if(self._roi[0] >= image.shape[1] - 1):
            self._roi[0] = image.shape[1] - 1
        if(self._roi[1] >= image.shape[0] - 1):
            self._roi[1] = image.shape[0] - 1
        if(self._roi[0] + self._roi[2] <= 0):
            self._roi[0] = -self._roi[2] + 2
        if(self._roi[1] + self._roi[3] <= 0):
            self._roi[1] = -self._roi[3] + 2
        assert(self._roi[2] > 0 and self._roi[3] > 0)

        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self._roi
