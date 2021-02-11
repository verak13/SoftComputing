import pyautogui

from projekat.detect_record_module import real_time_detection
from pyautogui import *
import cv2
import numpy as np
import threading

from projekat.fingerprinting import do_hash
from projekat.read_text_module import OCR


class SongPop2Solver:
    def __init__(self, detect_model, ocr_model):
        self.detect_model = detect_model
        self.ocr_model = ocr_model
        self.ocr = OCR(self.ocr_model)

        # TODO implementirati hash
        self.hash = None

    def handle(self):
        # TODO spojiti hash sa ocr
        answer = do_hash()
        self.ocr.click_closest_answer(answer)

        # TODO zameniti
        # pyautogui.click(x=727, y=718)

    def thread_record(self):
        real_time_detection(self.detect_model, self, single_color=True, start_delay=0)

    def recognize(self, path):
        template = cv2.imread(path)
        large_image = screenshot()
        dirname = os.path.dirname(__file__)
        large_image.save(os.path.join(dirname, 'slicice\slika.png'))
        img_rgb = cv2.imread('slicice\slika.png')
        w, h = template.shape[:-1]

        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
        threshold = .8
        loc = np.where(res >= threshold)
        location = []
        for pt in zip(*loc[::-1]):
            if not location:
                location.append(pt[0])
                location.append(pt[1])
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        cv2.imwrite('result.png', img_rgb)

        if location:
            location[0] += h / 2
            location[1] += w / 2
        return location

    def play(self):

        detection = threading.Thread(target=self.thread_record)
        detection.setDaemon(True)
        detection.start()

        res = self.recognize("slicice/songpop2.PNG")
        if res:
            click(res)

        while True:
            error = False
            sleep(2)
            xDugme = self.recognize("slicice/xDugme.PNG")
            print("xDugme")
            if xDugme:
                print("xDugme -> clicked")

                click(xDugme)

            xDugme2 = self.recognize("slicice/xDugme2.PNG")
            print("xDugme2")

            if xDugme2:
                print("xDugme2 -> clicked")

                click(xDugme2)
            sleep(3)
            newGame = self.recognize("slicice/newGame.PNG")
            print("newGame")
            if newGame:
                print(newGame)
                click(newGame)
                while True:
                    sleep(1)
                    print("random")

                    random = self.recognize("slicice/rng.PNG")
                    if random:
                        print("random -> clicked")
                        click(random)
                        sleep(2)

                        choose = self.recognize("slicice/choose.png")
                        if choose:
                            click(choose)
                            sleep(2)

                        playlist = self.recognize("slicice/todaysHits.PNG")
                        if playlist:
                            click(playlist)
                            sleep(8)
                            xDugme2 = self.recognize("slicice/xDugme2.PNG")
                            if xDugme2:
                                click(xDugme2)
                            tooMuchGames = self.recognize("slicice/GameError.PNG")
                            if tooMuchGames:
                                ok = self.recognize("slicice/ok.PNG")
                                click(ok)
                                error = True
                            break
                        else:
                            back = self.recognize("slicice/nazad.PNG")
                            if back:
                                click(back)
            if error:
                continue

            smer = 1
            while True:
                home = self.recognize("slicice/Home.PNG")
                if home:
                    click(home)
                    break
                # else:
                #     if smer == 1:
                #         smer = 0
                #         moveRel(-100, -100)
                #     else:
                #         smer = 1
                #         moveRel(100, 100)
                #     sleep(3)


if __name__ == '__main__':
    songpop2 = SongPop2Solver("./DETECT_MODEL.h5", "./TEXT_MODEL.h5")
    songpop2.play()
