# -*- coding: utf-8 -*-
# 获取屏幕信息，并通过视觉方法标定手牌与按钮位置，仿真鼠标点击操作输出
import os
import time
from typing import List, Tuple
import io
import functools
import inspect
import re

import cv2
import numpy as np

from .classifier import Classify
from ..sdk import Operation

from PIL import Image
import time

from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By

browser : webdriver.Chrome = None

DEBUG = False               # 是否显示检测中间结果
PRINT_LOG = True  # whether print args when enter handler

def dump_args(func):
    #Decorator to print function call details - parameters names and effective values.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if PRINT_LOG:
            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            func_args_str = ', '.join('{} = {!r}'.format(*item)
                                      for item in func_args.items())
            func_args_str = re.sub(r' *self.*?=.*?, *', '', func_args_str)
            #print(f'{func.__module__}.{func.__qualname__} ( {func_args_str} )')
            print(f'{func.__name__} ({func_args_str})')
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        time_used = end - start
        if PRINT_LOG:
            print(f'{func.__name__} time_used = {time_used}')
        return r
    return wrapper


def PosTransfer(pos, M: np.ndarray) -> np.ndarray:
    assert(len(pos) == 2)
    return cv2.perspectiveTransform(np.float32([[pos]]), M)[0][0]


def Similarity(img1: np.ndarray, img2: np.ndarray):
    assert(len(img1.shape) == len(img2.shape) == 3)
    if img1.shape[0] < img2.shape[0]:
        img1, img2 = img2, img1
    n, m, c = img2.shape
    img1 = cv2.resize(img1, (m, n))
    if DEBUG:
        cv2.imshow('diff', np.uint8(np.abs(np.float32(img1)-np.float32(img2))))
        cv2.waitKey(1)
    ksize = max(1, min(n, m)//50)
    img1 = cv2.blur(img1, ksize=(ksize, ksize))
    img2 = cv2.blur(img2, ksize=(ksize, ksize))
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    if DEBUG:
        cv2.imshow('bit', np.uint8((np.abs(img1-img2) < 30).sum(2) == 3)*255)
        cv2.waitKey(1)
    return ((np.abs(img1-img2) < 30).sum(2) == 3).sum()/(n*m)


def ObjectLocalization(objImg: np.ndarray, targetImg: np.ndarray) -> np.ndarray:
    """
    https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    Feature based object detection
    return: Homography matrix M (objImg->targetImg), if not found return None
    """
    img1 = objImg
    img2 = targetImg
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=5000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # store all the good matches as per Lowe's ratio test.
    good = []
    for i, j in enumerate(matches):
        if len(j) == 2:
            m, n = j
            if m.distance < 0.7*n.distance:
                good.append(m)
                matchesMask[i] = [1, 0]
    print('  Number of good matches:', len(good))
    if DEBUG:
        # draw
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, matches, None, **draw_params)
        img3 = cv2.pyrDown(img3)
        cv2.imshow('ORB match', img3)
        cv2.waitKey(1)
    # Homography
    MIN_MATCH_COUNT = 50
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if DEBUG:
            # draw
            matchesMask = mask.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, (0, 0, 255), 10, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                                   good, None, **draw_params)
            img3 = cv2.pyrDown(img3)
            cv2.imshow('Homography match', img3)
            cv2.waitKey(1)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = None
    assert(type(M) == type(None) or (
        type(M) == np.ndarray and M.shape == (3, 3)))
    return M


def getHomographyMatrix(img1, img2, threshold=0.0):
    # if similarity>threshold return M
    # else return None
    M = ObjectLocalization(img1, img2)
    if type(M) != type(None):
        print('  Homography Matrix:', M)
        n, m, c = img1.shape
        x0, y0 = np.int32(PosTransfer([0, 0], M))
        x1, y1 = np.int32(PosTransfer([m, n], M))
        sub_img = img2[y0:y1, x0:x1, :]
        S = Similarity(img1, sub_img)
        print('Similarity:', S)
        if S > threshold:
            return M
    return None

_width = None

@dump_args
def screenShot(calibration=False):
    global _width
    if calibration:
        body = browser.find_element(by=By.XPATH, value="/html")
        print(body.size)
        _width = int(body.size['width'])
    screenshot_png = browser.get_screenshot_as_png()
    nparr = np.fromstring(screenshot_png, np.uint8)
    screenshot = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h0, w0, _ = screenshot.shape
    screenshot = cv2.resize(screenshot,(_width, int(_width * h0 / w0)))
    # screenshot = Image.open(io.BytesIO(screenshot_png))
    # img = np.asarray(screenshot)
    # print(img.dtype)
    # print(img.shape)
    # return cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    return screenshot

@dump_args
def clickAt(x, y):
    print("Point to click", x, y)
    actionChains = ActionChains(browser)
    body = browser.find_element(by=By.XPATH, value="/html")
    actionChains.move_to_element_with_offset(body, x, y)
    actionChains.pause(0.1)
    actionChains.click()
    actionChains.perform()

def moveTo(x, y):
    actionChains = ActionChains(browser)
    body = browser.find_element(by=By.XPATH, value="/html")
    actionChains.move_to_element_with_offset(body, x, y)
    actionChains.perform()



class Layout:
    size = (1920, 1080)                                     # 界面长宽
    duanWeiChang = (1348, 321)                              # 段位场按钮
    menuButtons = [(1382, 406), (1382, 573), (1382, 740),
                   (1383, 885), (1393, 813)]   # 铜/银/金之间按钮
    tileSize = (95, 152)                                     # 自己牌的大小


class GUIInterface:

    def __init__(self, chrome_arguments=[]):
        self.startWebDriver(chrome_arguments)
        self.M = None  # Homography matrix from (1920,1080) to real window
        # load template imgs
        join = os.path.join
        root = os.path.dirname(__file__)
        def load(name): return cv2.imread(join(root, 'template', name))
        self.menuImg = load('menu.png')         # 初始菜单界面
        if (type(self.menuImg)==type(None)):
            raise FileNotFoundError("menu.png not found, please check the Chinese path")
        assert(self.menuImg.shape == (1080, 1920, 3))
        self.chiImg = load('chi.png')
        self.pengImg = load('peng.png')
        self.gangImg = load('gang.png')
        self.huImg = load('hu.png')
        self.zimoImg = load('zimo.png')
        self.tiaoguoImg = load('tiaoguo.png')
        self.liqiImg = load('liqi.png')
        # tile position cache
        self.is_tile_position_cache_valid = False
        self.hand_tile_pos = []
        # load classify model
        self.classify = Classify()
        self.tiaoguo_pos = None

    def startWebDriver(self, chrome_arguments):
        global browser
        chrome_options = Options()
        chrome_options.add_argument('--proxy-server=127.0.0.1:8080')
        chrome_options.add_argument('--ignore-certificate-errors')
        for arg in chrome_arguments:
            chrome_options.add_argument(arg)
        # chrome_options.add_argument('--user-data-dir=ChromeUserData')
        # chrome_options.add_argument('--profile-directory=Majsoul')
        browser = webdriver.Chrome(options=chrome_options)
        # browser.get('https://game.maj-soul.com/1/')   

    @dump_args
    def forceTiaoGuo(self):
        # 如果跳过按钮在屏幕上则强制点跳过，否则NoEffect
        self.clickButton(self.tiaoguoImg, similarityThreshold=0.7)

    @dump_args
    def actionDiscardTile(self, tile: str, pos=-1):
        retry = 10
        is_retry = False
        while retry > 0:
            if pos >= 0 and self.is_tile_position_cache_valid:
                x, y = self.hand_tile_pos[pos]
                clickAt(x, y)
                return True
            L = self._getHandTiles(retry=is_retry)
            print("Hand tiles", L)
            print("tile to play", tile)
            for t, (x, y) in L:
                if t == tile:
                    clickAt(x, y)
                    return True
            retry -= 1
            is_retry = True
        raise Exception(
            'GUIInterface.discardTile tile not found. L:', L, 'tile:', tile)
        return False

    @dump_args
    def actionChiPengGang(self, type_: Operation, tiles: List[str]):
        if type_ == Operation.NoEffect:
            self.clickButton(self.tiaoguoImg)
        elif type_ == Operation.Chi:
            self.clickButton(self.chiImg)
        elif type_ == Operation.Peng:
            self.clickButton(self.pengImg)
        elif type_ in (Operation.MingGang, Operation.JiaGang):
            self.clickButton(self.gangImg)

    @dump_args
    def actionLiqi(self, tile: str):
        self.clickButton(self.liqiImg)
        time.sleep(0.5)
        self.actionDiscardTile(tile)

    @dump_args
    def actionHu(self):
        self.clickButton(self.huImg)

    @dump_args
    def actionZimo(self):
        self.clickButton(self.zimoImg)

    def calibrateMenu(self):
        # if the browser is on the initial menu, set self.M and return to True
        # if not return False
        try:
            self.M = getHomographyMatrix(self.menuImg, screenShot(calibration=True), threshold=0.7)
        except Exception as e:
            print(e)
        result = type(self.M) != type(None)
        if result:
            self.waitPos = np.int32(PosTransfer([100, 100], self.M))
        return result

    @dump_args
    def _getHandTiles(self, retry=False) -> List[Tuple[str, Tuple[int, int]]]:
        # return a list of my tiles' position
        result = []
        assert(type(self.M) != type(None))
        screen_img1 = screenShot()
        # time.sleep(0.5)
        screen_img = screen_img1
        if retry:
            screen_img2 = screenShot()
            screen_img = np.minimum(screen_img, screen_img2)  # 消除高光动画
        img = screen_img.copy()     # for calculation
        start = np.int32(PosTransfer([235, 1002], self.M))
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 110
        tileThreshold = np.int32(0.7*(PosTransfer(Layout.tileSize, self.M)-O))
        fail = 0
        maxFail = np.int32(PosTransfer([100, 0], self.M)-O)[0]
        i = 0
        while fail < maxFail:
            x, y = start[0]+i, start[1]
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = screen_img[y:y+dy, x:x+dx, :]
                    tileStr = self.classify(tile_img)
                    result.append((tileStr, (x+dx//2, y+dy//2)))
                    i = x+dx-start[0]
            else:
                fail += 1
            i += 1
        # update cache
        max_y = 0
        for t, (x, y) in result:
            max_y = max(y, max_y)
        self.hand_tile_pos = []
        for t, (x, y) in result:
            self.hand_tile_pos.append((x, max_y))
        self.is_tile_position_cache_valid = True
        return result

    def clickButton(self, buttonImg, similarityThreshold=0.0):
        # 如果是跳过则点击保存的位置
        if buttonImg is self.tiaoguoImg:
            print("try tiaoguo cache")
            try:
                x, y = self.tiaoguo_pos
                print("use tiaoguo cache", x, y)
                clickAt(x, y)
                return
            except:
                print("tiaoguo cache is not valid")
        # 点击吃碰杠胡立直自摸
        x0, y0 = np.int32(PosTransfer([0, 0], self.M))
        x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
        zoom = (x1-x0)/Layout.size[0]
        n, m, _ = buttonImg.shape
        n = int(n*zoom)
        m = int(m*zoom)
        templ = cv2.resize(buttonImg, (m, n))
        x0, y0 = np.int32(PosTransfer([595, 557], self.M))
        x1, y1 = np.int32(PosTransfer([1508, 912], self.M))
        img = screenShot()[y0:y1, x0:x1, :]
        cv2.imshow('screenshot', img)
        cv2.waitKey(50)
        T = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF, mask=templ.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        if DEBUG:
            T = np.exp((1-T/T.max())*10)
            T = T/T.max()
            cv2.imshow('T', T)
            cv2.waitKey(0)
        dst = img[y:y+n, x:x+m].copy()
        dst[templ == 0] = 0
        if Similarity(templ, dst) >= similarityThreshold:
            clickAt(x+x0+m//2, y+y0+n//2)
            if buttonImg is self.tiaoguoImg:
                self.tiaoguo_pos = (x+x0+m//2, y+y0+n//2)
                print("set tiaoguo cache", x+x0+m//2, y+y0+n//2)

    def clickCandidateMeld(self, tiles: List[str]):
        # 有多种不同的吃碰方法，二次点击选择
        assert(len(tiles) == 2)
        # find all combination tiles
        result = []
        assert(type(self.M) != type(None))
        screen_img = screenShot()
        img = screen_img.copy()     # for calculation
        start = np.int32(PosTransfer([960, 753], self.M))
        leftBound = rightBound = start[0]
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 200
        tileThreshold = np.int32(0.7*(PosTransfer((78, 106), self.M)-O))
        maxFail = np.int32(PosTransfer([60, 0], self.M)-O)[0]
        for offset in [-1, 1]:
            #从中间向左右两个方向扫描
            i = 0
            while True:
                x, y = start[0]+i*offset, start[1]
                if offset == -1 and x < leftBound-maxFail:
                    break
                if offset == 1 and x > rightBound+maxFail:
                    break
                if all(img[y, x, :] > colorThreshold):
                    img[y, x, :] = colorThreshold
                    retval, image, mask, rect = cv2.floodFill(
                        image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                        loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                    x, y, dx, dy = rect
                    if dx > tileThreshold[0] and dy > tileThreshold[1]:
                        tile_img = screen_img[y:y+dy, x:x+dx, :]
                        tileStr = self.classify(tile_img)
                        result.append((tileStr, (x+dx//2, y+dy//2)))
                        leftBound = min(leftBound, x)
                        rightBound = max(rightBound, x+dx)
                i += 1
        result = sorted(result, key=lambda x: x[1][0])
        if len(result) == 0:
            return True  # 其他人先抢先Meld了！
        print('clickCandidateMeld tiles:', result)
        assert(len(result) % 2 == 0)
        for i in range(0, len(result), 2):
            x, y = result[i][1]
            if tuple(sorted([result[i][0], result[i+1][0]])) == tiles:
                clickAt(x, y)
                return True
        raise Exception('combination not found, tiles:',
                        tiles, ' combination:', result)
        return False

    def actionReturnToMenu(self):
        # 在终局以后点击确定跳转回菜单主界面
        x, y = np.int32(PosTransfer((1785, 1003), self.M))  # 终局确认按钮
        while True:
            time.sleep(5)
            x0, y0 = np.int32(PosTransfer([0, 0], self.M))
            x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
            img = screenShot()
            S = Similarity(self.menuImg, img[y0:y1, x0:x1, :])
            if S > 0.5:
                return True
            else:
                print('Similarity:', S)
                clickAt(x, y)

    def actionBeginGame(self, level: int):
        # 从开始界面点击匹配对局, level=0~4 (铜/银/金/玉/王座之间)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.duanWeiChang, self.M))
        clickAt(x, y)
        time.sleep(2)
        if level == 4:
            # 王座之间在屏幕外面需要先拖一下
            # TODO: selenium drag support for 王座之间
            raise NotImplementedError()
            # x, y = np.int32(PosTransfer(Layout.menuButtons[2], self.M))
            # pyautogui.moveTo(x, y)
            # time.sleep(1.5)
            # x, y = np.int32(PosTransfer(Layout.menuButtons[0], self.M))
            # pyautogui.dragTo(x, y)
            # time.sleep(1.5)
        x, y = np.int32(PosTransfer(Layout.menuButtons[level], self.M))
        clickAt(x, y)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.menuButtons[0], self.M))  # 四人东
        clickAt(x, y)
